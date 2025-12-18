# value_ledger/heuristics.py
"""
Advanced heuristic scorers for the Value Ledger.
Each scorer contributes to the T/E/N/F/R/S vector based on cognitive effort signals.
Now fully integrated with Memory Vault for accurate novelty assessment.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from .core import ValueVector


@dataclass
class ScoringContext:
    """Rich context gathered from IntentLog, Memory Vault, Boundary Daemon, etc."""
    intent_id: str
    start_time: float
    end_time: Optional[float] = None
    interruptions: int = 0
    keystrokes: Optional[int] = None
    memory_content: Optional[str] = None          # Combined human + agent text
    memory_hash: Optional[str] = None
    previous_memories: Optional[List[Tuple[str, float, Optional[str]]]] = None  # (hash, ts, content)
    outcome_tags: Optional[List[str]] = None
    risk_level: Optional[float] = None            # 0.0–1.0 explicit
    user_override: Optional[Dict[str, float]] = None  # Human tweaks (+/-)


class HeuristicScorer:
    """Base class for all scorers"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        raise NotImplementedError


# ==================== Individual Scorers ====================

class TimeScorer(HeuristicScorer):
    """T: Duration with diminishing returns — values sustained focus"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        if not ctx.end_time:
            return ValueVector(t=0.0)

        duration_hours = (ctx.end_time - ctx.start_time) / 3600.0
        if duration_hours <= 0:
            return ValueVector(t=0.1)  # minimum for instant intents

        # Concave scaling: first hour most valuable
        t = 8.0 * math.log1p(duration_hours * 2)   # ~8 at 1h, ~12 at 4h, caps naturally
        return ValueVector(t=max(0.5, min(15.0, t)))


class EffortScorer(HeuristicScorer):
    """E: Intensity — combines interruptions, input density, and flow breaks"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        base_time = TimeScorer()(ctx).t
        if base_time < 0.5:
            return ValueVector(e=0.0)

        # Interruption cost (context switching tax)
        interruption_factor = 1.0 + (ctx.interruptions * 0.35)
        if ctx.interruptions > 10:
            interruption_factor += (ctx.interruptions - 10) * 0.1  # escalating

        # Input density bonus (productive typing)
        density_bonus = 1.0
        if ctx.keystrokes and ctx.end_time:
            duration_hours = (ctx.end_time - ctx.start_time) / 3600.0
            kph = ctx.keystrokes / max(duration_hours, 0.1)
            if kph > 1000:
                density_bonus = 1.4
            elif kph > 600:
                density_bonus = 1.2
            elif kph < 200:
                density_bonus = 0.8  # slow = possibly stuck/thinking hard

        e = base_time * interruption_factor * density_bonus
        return ValueVector(e=min(22.0, e))


class NoveltyScorer(HeuristicScorer):
    """N: True uniqueness against personal memory corpus — now Memory Vault powered"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        prev = ctx.previous_memories or []
        content = ctx.memory_content

        # No content access (consent denied or empty) → conservatively high novelty
        if not content or not prev:
            return ValueVector(n=8.5)

        current_words = set(content.lower().split())
        if len(current_words) < 5:  # too short to assess
            return ValueVector(n=7.0)

        similarities = []
        for _, _, prev_content in prev:
            if not prev_content:
                continue
            prev_words = set(prev_content.lower().split())
            if not prev_words:
                continue
            # Jaccard similarity with smoothing
            intersection = len(current_words & prev_words)
            union = len(current_words | prev_words)
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)

        if not similarities:
            return ValueVector(n=9.0)

        avg_similarity = sum(similarities) / len(similarities)
        # Non-linear: small overlaps don't kill novelty, high overlap does
        novelty = 10.0 * (1.0 - math.pow(avg_similarity, 0.8))
        return ValueVector(n=max(1.0, novelty))


class FailureScorer(HeuristicScorer):
    """F: Learning value from dead ends and partial outcomes"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        tags = {t.lower() for t in (ctx.outcome_tags or [])}

        if "dead_end" in tags or "failure" in tags:
            return ValueVector(f=9.0)
        if "partial" in tags or "stuck" in tags:
            return ValueVector(f=6.0)
        if "breakthrough" in tags or "insight" in tags:
            return ValueVector(f=4.0)  # failure preceded success
        return ValueVector(f=1.5)  # baseline learning even in success


class RiskScorer(HeuristicScorer):
    """R: Exposure to downside — explicit + inferred from language"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        explicit = ctx.risk_level or 0.0

        inferred = 0.0
        if ctx.memory_content:
            content = ctx.memory_content.lower()
            high_risk_phrases = [
                "all in", "bet everything", "existential", "critical path",
                "point of no return", "last chance", "do or die"
            ]
            medium_risk = ["gamble", "risky", "danger", "bold move", "leap"]
            if any(phrase in content for phrase in high_risk_phrases):
                inferred = 0.9
            elif any(word in content for word in medium_risk):
                inferred = 0.6

        r = 10.0 * max(explicit, inferred)
        return ValueVector(r=min(10.0, r))


class StrategyScorer(HeuristicScorer):
    """S: Depth of planning, abstraction, and meta-cognition"""
    def __call__(self, ctx: ScoringContext) -> ValueVector:
        content = ctx.memory_content or ""
        lower = content.lower()

        base = 3.0

        # Hierarchical strategy indicators
        indicators = {
            "architecture": 6.0,
            "framework": 5.5,
            "system design": 6.5,
            "long-term": 4.0,
            "leverage": 5.0,
            "second-order": 8.0,
            "meta": 7.0,
            "counterfactual": 7.5,
            "if i were": 6.0,
            "plan b": 4.5,
            "fallback": 4.0,
            "trade-off": 5.0,
        }

        score = base
        for phrase, boost in indicators.items():
            if phrase in lower:
                score = max(score, boost)

        # Depth proxies
        if len(content) > 1500:
            score += 2.0
        if content.count("\n") > 20 or content.count("- ") > 10:
            score += 1.5
        if content.count("?") > 8:
            score += 1.0  # deep questioning

        return ValueVector(s=min(16.0, score))


# ==================== Scoring Engine ====================

class HeuristicEngine:
    """Combines all scorers with optional global caps and normalization"""
    def __init__(self):
        self.scorers: List[HeuristicScorer] = [
            TimeScorer(),
            EffortScorer(),
            NoveltyScorer(),
            FailureScorer(),
            RiskScorer(),
            StrategyScorer(),
        ]

    def score(self, ctx: ScoringContext) -> ValueVector:
        total = ValueVector()

        for scorer in self.scorers:
            contribution = scorer(ctx)
            for field in total.__dict__:
                current = getattr(total, field)
                added = getattr(contribution, field)
                setattr(total, field, current + added)

        # Optional: soft cap per entry to prevent explosion on epic sessions
        total_value = total.total()
        if total_value > 70.0:
            scale = 70.0 / total_value
            total = ValueVector(**{k: v * scale for k, v in total.dict().items()})

        # Apply human overrides last (e.g., "I feel this was more novel +2")
        if ctx.user_override:
            for k, delta in ctx.user_override.items():
                if k in total.__dict__:
                    current = getattr(total, k)
                    setattr(total, k, max(0.0, current + delta))

        return total
            scale = 60.0 / total.total()
            total = ValueVector(**{k: v * scale for k, v in total.dict().items()})

        return total
