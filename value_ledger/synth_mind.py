# value_ledger/synth_mind.py
"""
Synth-Mind Integration for Value Ledger.

This module integrates with the Synth-Mind cognitive architecture to:
- Track cognitive tier usage (System1, System2, etc.)
- Value different cognitive processing depths
- Score metacognition and tier switching

Synth-Mind Architecture Tiers:
- Tier 1 (System1): Fast, intuitive, pattern-matching
- Tier 2 (System2): Deliberate, analytical, logical
- Tier 3 (Meta): Self-reflection, strategy evaluation
- Tier 4 (Executive): Goal management, resource allocation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol, Callable
from enum import IntEnum

from .core import ValueVector


class CognitiveTier(IntEnum):
    """Cognitive processing tiers from Synth-Mind architecture."""
    SYSTEM1 = 1  # Fast, intuitive
    SYSTEM2 = 2  # Deliberate, analytical
    META = 3     # Self-reflection
    EXECUTIVE = 4  # Goal management


@dataclass
class TierChangeEvent:
    """Event emitted when cognitive tier changes."""
    timestamp: float
    from_tier: Optional[int]
    to_tier: int
    trigger: str  # What caused the tier change
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveTierContext:
    """
    Extended context for cognitive tier tracking.

    Can be used standalone or merged with ScoringContext.
    """
    # Tier tracking
    initial_tier: int = CognitiveTier.SYSTEM1
    final_tier: int = CognitiveTier.SYSTEM1
    tier_history: List[TierChangeEvent] = field(default_factory=list)

    # Derived metrics
    tier_switches: int = 0
    time_per_tier: Dict[int, float] = field(default_factory=dict)
    metacognition_depth: int = 0  # Number of meta-level reflections

    # Processing characteristics
    deliberation_ratio: float = 0.0  # Time in Tier2+ / total time
    meta_interventions: int = 0  # Times meta-tier corrected lower tiers

    def add_tier_change(self, event: TierChangeEvent):
        """Record a tier change event."""
        self.tier_history.append(event)
        self.tier_switches += 1
        self.final_tier = event.to_tier

        if event.to_tier >= CognitiveTier.META:
            self.metacognition_depth += 1

        if event.from_tier and event.from_tier < CognitiveTier.META <= event.to_tier:
            self.meta_interventions += 1

    def compute_time_distribution(self, total_duration: float):
        """Compute time spent in each tier."""
        if not self.tier_history:
            self.time_per_tier[self.initial_tier] = total_duration
            return

        # Sort events by timestamp
        events = sorted(self.tier_history, key=lambda e: e.timestamp)
        start_time = events[0].timestamp - 60  # Assume started 60s before first change

        current_tier = self.initial_tier
        current_start = start_time

        for event in events:
            duration = event.timestamp - current_start
            self.time_per_tier[current_tier] = self.time_per_tier.get(current_tier, 0) + duration
            current_tier = event.to_tier
            current_start = event.timestamp

        # Add remaining time for final tier
        final_duration = (events[-1].timestamp + 60) - current_start
        self.time_per_tier[current_tier] = self.time_per_tier.get(current_tier, 0) + final_duration

        # Compute deliberation ratio
        total = sum(self.time_per_tier.values())
        if total > 0:
            deliberate_time = sum(t for tier, t in self.time_per_tier.items() if tier >= CognitiveTier.SYSTEM2)
            self.deliberation_ratio = deliberate_time / total


class CognitiveTierScorer:
    """
    Score cognitive effort based on tier usage patterns.

    Scoring principles:
    - Higher tiers indicate more strategic/complex processing
    - Tier switches indicate effort in context shifting
    - Metacognition adds value across all dimensions
    - Time in deliberate mode (Tier2+) indicates careful work
    """

    # Base multipliers for each tier
    TIER_EFFORT_MULTIPLIERS = {
        CognitiveTier.SYSTEM1: 0.5,   # Fast intuition - lower effort
        CognitiveTier.SYSTEM2: 1.0,   # Deliberate - baseline
        CognitiveTier.META: 1.5,      # Reflection - higher effort
        CognitiveTier.EXECUTIVE: 2.0,  # Executive - highest effort
    }

    # Strategic value by tier
    TIER_STRATEGY_WEIGHTS = {
        CognitiveTier.SYSTEM1: 0.2,
        CognitiveTier.SYSTEM2: 0.5,
        CognitiveTier.META: 0.8,
        CognitiveTier.EXECUTIVE: 1.0,
    }

    def __init__(
        self,
        switch_effort_bonus: float = 0.1,
        meta_global_bonus: float = 0.05,
        deliberation_threshold: float = 0.3,
    ):
        """
        Initialize scorer.

        Args:
            switch_effort_bonus: Effort bonus per tier switch
            meta_global_bonus: Bonus applied to all dimensions per meta-intervention
            deliberation_threshold: Min deliberation ratio for strategic bonus
        """
        self.switch_effort_bonus = switch_effort_bonus
        self.meta_global_bonus = meta_global_bonus
        self.deliberation_threshold = deliberation_threshold

    def score(
        self,
        tier_context: CognitiveTierContext,
        base_vector: Optional[ValueVector] = None,
    ) -> ValueVector:
        """
        Score cognitive effort from tier context.

        Args:
            tier_context: Cognitive tier tracking data
            base_vector: Optional base vector to enhance

        Returns:
            ValueVector with tier-based scoring
        """
        # Start with base or zero
        if base_vector:
            t = base_vector.t
            e = base_vector.e
            n = base_vector.n
            f = base_vector.f
            r = base_vector.r
            s = base_vector.s
        else:
            t = e = n = f = r = s = 0.0

        # Compute time distribution if not done
        if not tier_context.time_per_tier:
            total_time = sum(
                e.timestamp for e in tier_context.tier_history
            ) if tier_context.tier_history else 0
            tier_context.compute_time_distribution(total_time or 60)

        # 1. Effort from tier complexity
        tier_effort = 0.0
        total_tier_time = sum(tier_context.time_per_tier.values())
        if total_tier_time > 0:
            for tier, tier_time in tier_context.time_per_tier.items():
                weight = tier_time / total_tier_time
                tier_effort += weight * self.TIER_EFFORT_MULTIPLIERS.get(tier, 1.0)
        e += tier_effort

        # 2. Effort from tier switches
        e += tier_context.tier_switches * self.switch_effort_bonus

        # 3. Strategic value from tier usage
        if total_tier_time > 0:
            for tier, tier_time in tier_context.time_per_tier.items():
                weight = tier_time / total_tier_time
                s += weight * self.TIER_STRATEGY_WEIGHTS.get(tier, 0.5)

        # 4. Metacognition bonus (applies to all dimensions)
        meta_bonus = tier_context.meta_interventions * self.meta_global_bonus
        t += meta_bonus
        e += meta_bonus
        n += meta_bonus
        s += meta_bonus

        # 5. Deliberation bonus for strategy
        if tier_context.deliberation_ratio >= self.deliberation_threshold:
            s += 0.2 * tier_context.deliberation_ratio

        # 6. Risk awareness from executive tier usage
        exec_ratio = tier_context.time_per_tier.get(CognitiveTier.EXECUTIVE, 0) / max(total_tier_time, 1)
        r += exec_ratio * 0.5

        return ValueVector(
            t=max(0, t),
            e=max(0, e),
            n=max(0, n),
            f=max(0, f),
            r=max(0, r),
            s=max(0, s),
        )


class SynthMindHook:
    """
    Hook for integrating with Synth-Mind cognitive events.

    Usage:
        hook = SynthMindHook(ledger)
        synth_mind.register_tier_listener(hook.on_tier_change)
    """

    def __init__(
        self,
        scorer: Optional[CognitiveTierScorer] = None,
        auto_score: bool = True,
    ):
        """
        Initialize hook.

        Args:
            scorer: Optional custom scorer
            auto_score: If True, auto-compute scores on session end
        """
        self.scorer = scorer or CognitiveTierScorer()
        self.auto_score = auto_score
        self.active_contexts: Dict[str, CognitiveTierContext] = {}
        self.completed_sessions: List[Dict[str, Any]] = []

    def start_session(self, session_id: str, initial_tier: int = CognitiveTier.SYSTEM1):
        """Start tracking a new cognitive session."""
        self.active_contexts[session_id] = CognitiveTierContext(
            initial_tier=initial_tier,
            final_tier=initial_tier,
        )

    def on_tier_change(
        self,
        session_id: str,
        from_tier: Optional[int],
        to_tier: int,
        trigger: str = "unknown",
        context: Optional[Dict] = None,
    ):
        """
        Handle tier change event from Synth-Mind.

        Args:
            session_id: Session identifier
            from_tier: Previous tier (None if first)
            to_tier: New tier
            trigger: What caused the change
            context: Additional context
        """
        if session_id not in self.active_contexts:
            self.start_session(session_id, to_tier)
            return

        event = TierChangeEvent(
            timestamp=time.time(),
            from_tier=from_tier,
            to_tier=to_tier,
            trigger=trigger,
            context=context or {},
        )
        self.active_contexts[session_id].add_tier_change(event)

    def end_session(
        self,
        session_id: str,
        total_duration: Optional[float] = None,
    ) -> Optional[ValueVector]:
        """
        End a cognitive session and optionally compute score.

        Args:
            session_id: Session to end
            total_duration: Total session duration (auto-computed if not provided)

        Returns:
            ValueVector if auto_score is True, None otherwise
        """
        if session_id not in self.active_contexts:
            return None

        ctx = self.active_contexts.pop(session_id)

        # Compute time distribution
        if total_duration:
            ctx.compute_time_distribution(total_duration)
        elif ctx.tier_history:
            duration = ctx.tier_history[-1].timestamp - ctx.tier_history[0].timestamp + 120
            ctx.compute_time_distribution(duration)

        # Store completed session
        session_data = {
            "session_id": session_id,
            "ended_at": time.time(),
            "tier_switches": ctx.tier_switches,
            "metacognition_depth": ctx.metacognition_depth,
            "deliberation_ratio": ctx.deliberation_ratio,
            "time_per_tier": ctx.time_per_tier.copy(),
        }

        if self.auto_score:
            score = self.scorer.score(ctx)
            session_data["score"] = score.dict()
            self.completed_sessions.append(session_data)
            return score

        self.completed_sessions.append(session_data)
        return None

    def get_session_context(self, session_id: str) -> Optional[CognitiveTierContext]:
        """Get context for an active session."""
        return self.active_contexts.get(session_id)

    def get_completed_sessions(self) -> List[Dict[str, Any]]:
        """Get all completed session data."""
        return self.completed_sessions.copy()


class TierEventEmitter(Protocol):
    """Protocol for tier event sources (for testing/mocking)."""

    def register_listener(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Register a listener for tier change events."""
        ...

    def emit_tier_change(
        self,
        session_id: str,
        from_tier: int,
        to_tier: int,
        trigger: str,
    ) -> None:
        """Emit a tier change event."""
        ...


class MockTierEmitter:
    """Mock emitter for testing Synth-Mind integration."""

    def __init__(self):
        self.listeners: List[Callable] = []

    def register_listener(self, callback: Callable) -> None:
        self.listeners.append(callback)

    def emit_tier_change(
        self,
        session_id: str,
        from_tier: Optional[int],
        to_tier: int,
        trigger: str = "test",
    ) -> None:
        for listener in self.listeners:
            listener(session_id, from_tier, to_tier, trigger)


def integrate_with_scoring_context(
    tier_context: CognitiveTierContext,
    scoring_context: "ScoringContext",
) -> None:
    """
    Merge cognitive tier data into a ScoringContext.

    Modifies scoring_context in place to add tier-related fields.
    """
    # Add tier tracking fields to metadata
    if not hasattr(scoring_context, "metadata"):
        scoring_context.metadata = {}

    scoring_context.metadata["cognitive_tier"] = tier_context.final_tier
    scoring_context.metadata["tier_switches"] = tier_context.tier_switches
    scoring_context.metadata["metacognition_depth"] = tier_context.metacognition_depth
    scoring_context.metadata["deliberation_ratio"] = tier_context.deliberation_ratio
    scoring_context.metadata["time_per_tier"] = tier_context.time_per_tier
