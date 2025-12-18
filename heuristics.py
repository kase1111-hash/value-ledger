# value_ledger/heuristics.py
"""
Heuristic scorers for the Value Ledger.
Each scorer returns a partial or full ValueVector contribution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Any, Callable
from pathlib import Path

from .core import ValueVector


@dataclass
class ScoringContext:
    """Everything a scorer might need from other modules"""
    intent_id: str
    start_time: float
    end_time: Optional[float] = None
    interruptions: int = 0                    # From Boundary Daemon
    keystrokes: Optional[int] = None          # From input monitoring
    memory_hash: Optional[str] = None
    memory_content: Optional[str] = None      # Only if explicitly allowed by Learning Contracts
    previous_memories: Optional[list] = None  # List of (hash, timestamp, content) tuples
    outcome_tags: Optional[list[str]] = None  # e.g., ["success", "partial", "dead_end"]
    risk_level: Optional[float] = None       # 0.0–1.0 from Learning Contracts or human input
    user_override: Optional[Dict[str, float]] = None


class HeuristicScorer:
    """Base class – makes it easy to add new scorers"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        raise NotImplementedError


# ————————————————————————
# Individual Heuristic Scorers
# ————————————————————————

class TimeScorer(HeuristicScorer):
    """T = raw duration with diminishing returns"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        if not ctx.end_time:
            return ValueVector(t=0.0)
        duration_hours = (ctx.end_time - ctx.start_time) / 3600.0
        # Logarithmic scaling: first hour worth more than 10th hour
        t = min(10.0, 2.0 * (duration_hours ** 0.6))
        return ValueVector(t=max(0.1, t))  # minimum 0.1 even for tiny sessions


class EffortScorer(HeuristicScorer):
    """E = time × interruption penalty + input density"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        base = TimeScorer()(ctx).t
        if base == 0:
            return ValueVector(e=0.0)

        # Interruption penalty (context switching is expensive)
        interruption_penalty = 1.0 + (ctx.interruptions * 0.4)
        
        # Input density bonus (keystrokes per hour)
        density_bonus = 1.0
        if ctx.keystrokes and ctx.end_time:
            duration_hours = (ctx.end_time - ctx.start_time) / 3600.0
            kph = ctx.keystrokes / max(duration_hours, 0.1)
            if kph > 800:
                density_bonus = 1.3
            elif kph > 400:
                density_bonus = 1.15

        e = base * interruption_penalty * density_bonus
        return ValueVector(e=min(20.0, e))


class NoveltyScorer(HeuristicScorer):
    """N = inverse of similarity to existing memory corpus"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        if not ctx.previous_memories or ctx.memory_content is None:
            return ValueVector(n=5.0)  # default medium novelty if no data

        # Simple TF-IDF-like novelty (replace with embeddings later)
        content = ctx.memory_content.lower()
        similarities = []
        for _, _, prev_content in ctx.previous_memories[-50:]:  # recent context window
            if not prev_content:
                continue
            overlap = len(set(content.split()) & set(prev_content.lower().split()))
            union = len(set(content.split()) | set(prev_content.lower().split()))
            similarities.append(overlap / union if union else 0.0)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        novelty = 10.0 * (1.0 - avg_similarity)
        return ValueVector(n=max(0.5, novelty))


class FailureScorer(HeuristicScorer):
    """F = higher when outcome is dead-end or partial"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        if not ctx.outcome_tags:
            return ValueVector(f=0.0)

        tags = {t.lower() for t in ctx.outcome_tags}
        if "dead_end" in tags or "failure" in tags:
            return ValueVector(f=8.0)
        if "partial" in tags:
            return ValueVector(f=4.0)
        return ValueVector(f=1.0)  # even success has tiny learning value


class RiskScorer(HeuristicScorer):
    """R = explicit risk level + inferred from language"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        explicit = ctx.risk_level or 0.0

        # Simple keyword inference if no explicit value
        inferred = 0.0
        if ctx.memory_content:
            risk_words = {"danger", "bet", "gamble", "all-in", "critical", "existential"}
            content = ctx.memory_content.lower()
            if any(word in content for word in risk_words):
                inferred = 0.6

        r = 10.0 * max(explicit, inferred)
        return ValueVector(r=max(0.0, min(10.0, r)))


class StrategyScorer(HeuristicScorer):
    """S = depth of planning, abstraction, meta-reasoning"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        if not ctx.memory_content:
            return ValueVector(s=2.0)

        content = ctx.memory_content.lower()
        strategy_indicators = {
            "plan": 3.0,
            "framework": 4.0,
            "architecture": 5.0,
            "if i were": 6.0,
            "counterfactual": 6.0,
            "long-term": 4.0,
            "leverage": 5.0,
            "second-order": 8.0,
            "meta": 7.0,
        }

        score = 2.0  # base
        for word, boost in strategy_indicators.items():
            if word in content:
                score = max(score, boost)

        # Bonus for length + complexity (proxy for depth)
        if len(content) > 1000:
            score += 2.0
        if content.count("\n") > 15:
            score += 1.5

        return ValueVector(s=min(15.0, score))


# ————————————————————————
# Scoring Engine (combines all)
# ————————————————————————

class HeuristicEngine:
    def __init__(self):
        self.scorers: list[HeuristicScorer] = [
            TimeScorer(),
            EffortScorer(),
            NoveltyScorer(),
            FailureScorer(),
            RiskScorer(),
            StrategyScorer(),
        ]

    def score(self, ctx: ScoringContext) -> ValueVector:
        """Run all scorers and merge results"""
        total = ValueVector()
        contributions = {}

        for scorer in self.scorers:
            vec = scorer(ctx)
            for field in vec.dict():
                if getattr(vec, field) > 0:
                    current = getattr(total, field)
                    new_val = current + getattr(vec, field)
                    setattr(total, field, new_val)
                    contributions.setdefault(field, 0.0)
                    contributions[field] += getattr(vec, field)

        # Optional: cap total per entry to prevent runaway scoring
        if total.total() > 60.0:
            scale = 60.0 / total.total()
            total = ValueVector(**{k: v * scale for k, v in total.dict().items()})

        return total
