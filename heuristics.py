# value_ledger/heuristics.py
"""
Heuristic scorers with EMBEDDING-BASED NOVELTY.
Now understands semantic similarity, not just word overlap.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from .core import ValueVector

# Embedding support — lazy import to allow graceful degradation
try:
    from sentence_transformers import SentenceTransformer
    import torch
    _EMBEDDING_MODEL = None  # Loaded on first use
    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False


def get_embedding_model():
    """Singleton loader for the embedding model"""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        if not _HAS_EMBEDDINGS:
            raise RuntimeError("sentence-transformers not available")
        # Lightweight, high-quality, multilingual model
        _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        # Move to GPU if available
        if torch.cuda.is_available():
            _EMBEDDING_MODEL = _EMBEDDING_MODEL.to('cuda')
    return _EMBEDDING_MODEL


@dataclass
class ScoringContext:
    intent_id: str
    start_time: float
    end_time: Optional[float] = None
    interruptions: int = 0
    keystrokes: Optional[int] = None
    memory_content: Optional[str] = None
    memory_hash: Optional[str] = None
    previous_memories: Optional[List[Tuple[str, float, Optional[str]]]] = None
    outcome_tags: Optional[List[str]] = None
    risk_level: Optional[float] = None
    user_override: Optional[Dict[str, float]] = None


class HeuristicScorer:
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        raise NotImplementedError


# ==================== Scorers (unchanged except Novelty) ====================

class TimeScorer(HeuristicScorer):
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        if not ctx.end_time:
            return ValueVector(t=0.0)
        duration_hours = (ctx.end_time - ctx.start_time) / 3600.0
        if duration_hours <= 0:
            return ValueVector(t=0.1)
        t = 8.0 * math.log1p(duration_hours * 2)
        return ValueVector(t=max(0.5, min(15.0, t)))


class EffortScorer(HeuristicScorer):
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        base_time = TimeScorer()(ctx).t
        if base_time < 0.5:
            return ValueVector(e=0.0)
        interruption_factor = 1.0 + (ctx.interruptions * 0.35)
        if ctx.interruptions > 10:
            interruption_factor += (ctx.interruptions - 10) * 0.1
        density_bonus = 1.0
        if ctx.keystrokes and ctx.end_time:
            duration_hours = (ctx.end_time - ctx.start_time) / 3600.0
            kph = ctx.keystrokes / max(duration_hours, 0.1)
            if kph > 1000:
                density_bonus = 1.4
            elif kph > 600:
                density_bonus = 1.2
            elif kph < 200:
                density_bonus = 0.8
        e = base_time * interruption_factor * density_bonus
        return ValueVector(e=min(22.0, e))


class NoveltyScorer(HeuristicScorer):
    """
    EMBEDDING-BASED NOVELTY
    Uses semantic embeddings to compare current thought against personal history.
    Much more accurate than word overlap.
    """
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        content = ctx.memory_content
        prev = ctx.previous_memories or []

        # Case 1: No content or no access → high novelty (safe default)
        if not content or not prev:
            return ValueVector(n=8.5)

        # Case 2: Embeddings unavailable → fall back to Jaccard
        if not _HAS_EMBEDDINGS:
            return self._fallback_jaccard(ctx)

        try:
            model = get_embedding_model()
            current_embedding = model.encode(content, normalize_embeddings=True)

            similarities = []
            valid_prev_contents = [pc for _, _, pc in prev if pc and len(pc.strip()) > 10]

            if not valid_prev_contents:
                return ValueVector(n=9.0)

            prev_embeddings = model.encode(
                valid_prev_contents,
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=False
            )

            # Cosine similarity = dot product (since normalized)
            if len(prev_embeddings.shape) == 2:
                cos_sims = prev_embeddings @ current_embedding
                similarities = cos_sims.tolist()
            else:
                similarities = [float(prev_embeddings @ current_embedding)]

            # Use top-k most similar to avoid noise from distant memories
            k = min(10, len(similarities))
            top_similarities = sorted(similarities, reverse=True)[:k]
            avg_top_similarity = sum(top_similarities) / k

            # Non-linear novelty curve: resistant to moderate similarity
            novelty = 10.0 * (1.0 - math.pow(avg_top_similarity, 1.2))
            return ValueVector(n=max(1.0, novelty))

        except Exception as e:
            print(f"[NoveltyScorer] Embedding failed ({e}), falling back to Jaccard")
            return self._fallback_jaccard(ctx)

    def _fallback_jaccard(self, ctx: ScoringContext) -> ValueVector:
        content = ctx.memory_content
        prev = ctx.previous_memories or []
        if not content or not prev:
            return ValueVector(n=8.0)

        current_words = set(content.lower().split())
        if len(current_words) < 5:
            return ValueVector(n=7.0)

        similarities = []
        for _, _, prev_content in prev:
            if not prev_content:
                continue
            prev_words = set(prev_content.lower().split())
            if not prev_words:
                continue
            jaccard = len(current_words & prev_words) / len(current_words | prev_words)
            similarities.append(jaccard)

        if not similarities:
            return ValueVector(n=9.0)

        avg_similarity = sum(sorted(similarities, reverse=True)[:10]) / min(10, len(similarities))
        novelty = 10.0 * (1.0 - math.pow(avg_similarity, 0.8))
        return ValueVector(n=max(1.0, novelty))


class FailureScorer(HeuristicScorer):
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        tags = {t.lower() for t in (ctx.outcome_tags or [])}
        if "dead_end" in tags or "failure" in tags:
            return ValueVector(f=9.0)
        if "partial" in tags or "stuck" in tags:
            return ValueVector(f=6.0)
        if "breakthrough" in tags or "insight" in tags:
            return ValueVector(f=4.0)
        return ValueVector(f=1.5)


class RiskScorer(HeuristicScorer):
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        explicit = ctx.risk_level or 0.0
        inferred = 0.0
        if ctx.memory_content:
            content = ctx.memory_content.lower()
            high_risk_phrases = ["all in", "bet everything", "existential", "critical path", "do or die"]
            medium_risk = ["gamble", "risky", "danger", "bold move"]
            if any(phrase in content for phrase in high_risk_phrases):
                inferred = 0.9
            elif any(word in content for word in medium_risk):
                inferred = 0.6
        r = 10.0 * max(explicit, inferred)
        return ValueVector(r=min(10.0, r))


class StrategyScorer(HeuristicScorer):
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        content = ctx.memory_content or ""
        lower = content.lower()
        base = 3.0
        indicators = {
            "architecture": 6.0, "framework": 5.5, "system design": 6.5,
            "long-term": 4.0, "leverage": 5.0, "second-order": 8.0,
            "meta": 7.0, "counterfactual": 7.5, "plan b": 4.5,
        }
        score = base
        for phrase, boost in indicators.items():
            if phrase in lower:
                score = max(score, boost)
        if len(content) > 1500:
            score += 2.0
        if content.count("\n") > 20 or content.count("- ") > 10:
            score += 1.5
        if content.count("?") > 8:
            score += 1.0
        return ValueVector(s=min(16.0, score))


# ==================== Engine ====================

class HeuristicEngine:
    def __init__(self):
        self.scorers = [
            TimeScorer(),
            EffortScorer(),
            NoveltyScorer(),   # ← Now embedding-powered!
            FailureScorer(),
            RiskScorer(),
            StrategyScorer(),
        ]

    def score(self, ctx: ScoringContext) -> ValueVector:
        total = ValueVector()
        for scorer in self.scorers:
            contribution = scorer(ctx)
            for field in total.__dict__:
                setattr(total, field, getattr(total, field) + getattr(contribution, field))

        if total.total() > 70.0:
            scale = 70.0 / total.total()
            total = ValueVector(**{k: v * scale for k, v in total.dict().items()})

        if ctx.user_override:
            for k, delta in ctx.user_override.items():
                if k in total.__dict__:
                    current = getattr(total, k)
                    setattr(total, k, max(0.0, current + delta))

        return total
