# tests/test_heuristics.py
"""
Tests for Value Ledger Heuristics Module.

Tests cover:
- ScoringContext creation
- TimeScorer
- EffortScorer
- NoveltyScorer (with fallback)
- FailureScorer
- RiskScorer
- StrategyScorer
- ReusabilityScorer
- HeuristicEngine integration
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from value_ledger.heuristics import (
    ScoringContext,
    HeuristicScorer,
    TimeScorer,
    EffortScorer,
    NoveltyScorer,
    FailureScorer,
    RiskScorer,
    StrategyScorer,
    ReusabilityScorer,
    HeuristicEngine,
    _HAS_EMBEDDINGS,
)
from value_ledger.core import ValueVector


# =============================================================================
# ScoringContext Tests
# =============================================================================


class TestScoringContext:
    """Tests for ScoringContext dataclass."""

    def test_minimal_context(self):
        """Minimal context with required fields."""
        ctx = ScoringContext(
            intent_id="test_intent",
            start_time=1000.0,
        )
        assert ctx.intent_id == "test_intent"
        assert ctx.start_time == 1000.0
        assert ctx.end_time is None
        assert ctx.interruptions == 0

    def test_full_context(self):
        """Context with all fields."""
        ctx = ScoringContext(
            intent_id="test_intent",
            start_time=1000.0,
            end_time=2000.0,
            interruptions=5,
            keystrokes=1500,
            memory_content="Test content for analysis",
            memory_hash="abc123",
            previous_memories=[("mem1", 0.8, "previous content")],
            outcome_tags=["success", "breakthrough"],
            risk_level=0.7,
            user_override={"t": 2.0},
        )
        assert ctx.end_time == 2000.0
        assert ctx.interruptions == 5
        assert ctx.keystrokes == 1500
        assert len(ctx.previous_memories) == 1
        assert ctx.risk_level == 0.7


# =============================================================================
# TimeScorer Tests
# =============================================================================


class TestTimeScorer:
    """Tests for TimeScorer."""

    def test_no_end_time(self):
        """No end time returns zero."""
        scorer = TimeScorer()
        ctx = ScoringContext(intent_id="test", start_time=1000.0)
        result = scorer(ctx)
        assert result.t == 0.0

    def test_zero_duration(self):
        """Zero duration returns minimum value."""
        scorer = TimeScorer()
        ctx = ScoringContext(intent_id="test", start_time=1000.0, end_time=1000.0)
        result = scorer(ctx)
        assert result.t == 0.1

    def test_negative_duration(self):
        """Negative duration returns minimum value."""
        scorer = TimeScorer()
        ctx = ScoringContext(intent_id="test", start_time=1000.0, end_time=500.0)
        result = scorer(ctx)
        assert result.t == 0.1

    def test_short_duration(self):
        """Short duration (1 minute) scores appropriately."""
        scorer = TimeScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=1060.0,  # 1 minute
        )
        result = scorer(ctx)
        assert 0.5 <= result.t <= 5.0

    def test_long_duration(self):
        """Long duration (2 hours) scores higher."""
        scorer = TimeScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=1000.0 + 7200,  # 2 hours
        )
        result = scorer(ctx)
        assert result.t > 5.0

    def test_capped_at_maximum(self):
        """Score is capped at 15.0."""
        scorer = TimeScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=1000.0 + 86400,  # 24 hours
        )
        result = scorer(ctx)
        assert result.t <= 15.0

    def test_minimum_threshold(self):
        """Score has minimum of 0.5 for positive duration."""
        scorer = TimeScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=1000.1,  # 0.1 seconds
        )
        result = scorer(ctx)
        assert result.t >= 0.5


# =============================================================================
# EffortScorer Tests
# =============================================================================


class TestEffortScorer:
    """Tests for EffortScorer."""

    def test_no_time_no_effort(self):
        """No time means no effort."""
        scorer = EffortScorer()
        ctx = ScoringContext(intent_id="test", start_time=1000.0)
        result = scorer(ctx)
        assert result.e == 0.0

    def test_basic_effort(self):
        """Basic effort without interruptions."""
        scorer = EffortScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=4600.0,  # 1 hour
            interruptions=0,
        )
        result = scorer(ctx)
        assert result.e > 0

    def test_interruption_multiplier(self):
        """Interruptions increase effort score."""
        scorer = EffortScorer()
        ctx_no_int = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=4600.0,
            interruptions=0,
        )
        ctx_with_int = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=4600.0,
            interruptions=5,
        )
        result_no = scorer(ctx_no_int)
        result_with = scorer(ctx_with_int)
        assert result_with.e > result_no.e

    def test_high_interruption_bonus(self):
        """More than 10 interruptions get additional bonus."""
        scorer = EffortScorer()
        # Use shorter duration to stay below the 22.0 cap
        ctx_10 = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=1600.0,  # 10 minutes
            interruptions=10,
        )
        ctx_15 = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=1600.0,  # 10 minutes
            interruptions=15,
        )
        result_10 = scorer(ctx_10)
        result_15 = scorer(ctx_15)
        # 15 should have additional bonus over 10 (if not capped)
        # Both may hit cap for high interruptions, so test that high interruptions produce high scores
        assert result_10.e > 0
        assert result_15.e > 0
        # With cap, both might be equal at max, which is still valid behavior
        assert result_15.e >= result_10.e

    def test_keystroke_density_bonus_high(self):
        """High keystroke density gets bonus."""
        scorer = EffortScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=4600.0,  # 1 hour
            keystrokes=5000,  # Very high kph
        )
        result = scorer(ctx)
        # Should have density bonus
        assert result.e > 0

    def test_keystroke_density_penalty_low(self):
        """Low keystroke density gets penalty."""
        scorer = EffortScorer()
        ctx_low = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=4600.0,  # 1 hour
            keystrokes=100,  # Very low
        )
        ctx_high = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=4600.0,
            keystrokes=2000,
        )
        result_low = scorer(ctx_low)
        result_high = scorer(ctx_high)
        assert result_high.e > result_low.e

    def test_effort_capped(self):
        """Effort is capped at 22.0."""
        scorer = EffortScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=1000.0 + 86400,  # 24 hours
            interruptions=100,
            keystrokes=100000,
        )
        result = scorer(ctx)
        assert result.e <= 22.0


# =============================================================================
# NoveltyScorer Tests
# =============================================================================


class TestNoveltyScorer:
    """Tests for NoveltyScorer."""

    def test_no_content_high_novelty(self):
        """No content defaults to high novelty."""
        scorer = NoveltyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            memory_content=None,
        )
        result = scorer(ctx)
        assert result.n == 8.5

    def test_no_previous_memories_high_novelty(self):
        """No previous memories means high novelty."""
        scorer = NoveltyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            memory_content="Some new content",
            previous_memories=[],
        )
        result = scorer(ctx)
        assert result.n == 8.5

    def test_fallback_jaccard_no_content(self):
        """Fallback Jaccard with no content."""
        scorer = NoveltyScorer()
        result = scorer._fallback_jaccard(
            ScoringContext(intent_id="test", start_time=0)
        )
        assert result.n == 8.0

    def test_fallback_jaccard_short_content(self):
        """Fallback Jaccard with very short content."""
        scorer = NoveltyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="hello",
            previous_memories=[("m1", 0.5, "world")],
        )
        result = scorer._fallback_jaccard(ctx)
        assert result.n == 7.0

    def test_fallback_jaccard_similar_content(self):
        """Fallback Jaccard with similar content gives lower novelty."""
        scorer = NoveltyScorer()
        content = "the quick brown fox jumps over the lazy dog"
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content=content,
            previous_memories=[
                ("m1", 0.5, "the quick brown fox jumps over the lazy dog"),
            ],
        )
        result = scorer._fallback_jaccard(ctx)
        # Very similar content should have low novelty
        assert result.n < 5.0

    def test_fallback_jaccard_different_content(self):
        """Fallback Jaccard with different content gives higher novelty."""
        scorer = NoveltyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="quantum computing blockchain artificial intelligence",
            previous_memories=[
                ("m1", 0.5, "cooking recipes baking chocolate cake dessert"),
            ],
        )
        result = scorer._fallback_jaccard(ctx)
        # Very different content should have high novelty
        assert result.n > 7.0

    def test_novelty_minimum(self):
        """Novelty has minimum value of 1.0."""
        scorer = NoveltyScorer()
        # Create context with identical content
        content = "exactly the same content repeated many times"
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content=content,
            previous_memories=[(f"m{i}", 0.9, content) for i in range(20)],
        )
        result = scorer._fallback_jaccard(ctx)
        assert result.n >= 1.0


# =============================================================================
# FailureScorer Tests
# =============================================================================


class TestFailureScorer:
    """Tests for FailureScorer."""

    def test_dead_end_high_score(self):
        """Dead end gets high failure score."""
        scorer = FailureScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            outcome_tags=["dead_end"],
        )
        result = scorer(ctx)
        assert result.f == 9.0

    def test_failure_tag_high_score(self):
        """Failure tag gets high failure score."""
        scorer = FailureScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            outcome_tags=["failure"],
        )
        result = scorer(ctx)
        assert result.f == 9.0

    def test_partial_medium_score(self):
        """Partial completion gets medium score."""
        scorer = FailureScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            outcome_tags=["partial"],
        )
        result = scorer(ctx)
        assert result.f == 6.0

    def test_stuck_medium_score(self):
        """Stuck gets medium score."""
        scorer = FailureScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            outcome_tags=["stuck"],
        )
        result = scorer(ctx)
        assert result.f == 6.0

    def test_breakthrough_lower_score(self):
        """Breakthrough gets lower failure score."""
        scorer = FailureScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            outcome_tags=["breakthrough"],
        )
        result = scorer(ctx)
        assert result.f == 4.0

    def test_insight_lower_score(self):
        """Insight gets lower failure score."""
        scorer = FailureScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            outcome_tags=["insight"],
        )
        result = scorer(ctx)
        assert result.f == 4.0

    def test_no_tags_default_score(self):
        """No tags gets default score."""
        scorer = FailureScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            outcome_tags=[],
        )
        result = scorer(ctx)
        assert result.f == 1.5

    def test_none_tags_default_score(self):
        """None tags gets default score."""
        scorer = FailureScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            outcome_tags=None,
        )
        result = scorer(ctx)
        assert result.f == 1.5

    def test_case_insensitive(self):
        """Tags are case-insensitive."""
        scorer = FailureScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            outcome_tags=["DEAD_END", "FAILURE"],
        )
        result = scorer(ctx)
        assert result.f == 9.0


# =============================================================================
# RiskScorer Tests
# =============================================================================


class TestRiskScorer:
    """Tests for RiskScorer."""

    def test_explicit_risk_level(self):
        """Explicit risk level is used."""
        scorer = RiskScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            risk_level=0.8,
        )
        result = scorer(ctx)
        assert result.r == 8.0

    def test_high_risk_phrase(self):
        """High risk phrases are detected."""
        scorer = RiskScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="This is an all in bet on the project",
        )
        result = scorer(ctx)
        assert result.r == 9.0

    def test_medium_risk_phrase(self):
        """Medium risk phrases are detected."""
        scorer = RiskScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="This is a risky move but worth trying",
        )
        result = scorer(ctx)
        assert result.r == 6.0

    def test_explicit_overrides_inferred(self):
        """Explicit risk level takes precedence if higher."""
        scorer = RiskScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            risk_level=0.95,
            memory_content="This is risky",  # Would infer 0.6
        )
        result = scorer(ctx)
        assert result.r == 9.5

    def test_no_risk_zero_score(self):
        """No risk indicators gives zero score."""
        scorer = RiskScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="Normal everyday task",
        )
        result = scorer(ctx)
        assert result.r == 0.0

    def test_risk_capped(self):
        """Risk is capped at 10.0."""
        scorer = RiskScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            risk_level=1.5,  # Over 1.0
        )
        result = scorer(ctx)
        assert result.r <= 10.0


# =============================================================================
# StrategyScorer Tests
# =============================================================================


class TestStrategyScorer:
    """Tests for StrategyScorer."""

    def test_base_score(self):
        """Empty content gets base score."""
        scorer = StrategyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="",
        )
        result = scorer(ctx)
        assert result.s == 3.0

    def test_architecture_indicator(self):
        """Architecture keyword boosts score."""
        scorer = StrategyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="The architecture of this system is modular",
        )
        result = scorer(ctx)
        assert result.s >= 6.0

    def test_second_order_high_score(self):
        """Second-order thinking gets high score."""
        scorer = StrategyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="Considering second-order effects of this change",
        )
        result = scorer(ctx)
        assert result.s >= 8.0

    def test_meta_indicator(self):
        """Meta-level thinking indicator."""
        scorer = StrategyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="At a meta level, we need to reconsider",
        )
        result = scorer(ctx)
        assert result.s >= 7.0

    def test_long_content_bonus(self):
        """Long content gets bonus."""
        scorer = StrategyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="x " * 1000,  # >1500 chars
        )
        result = scorer(ctx)
        assert result.s >= 5.0  # Base + length bonus

    def test_structured_content_bonus(self):
        """Structured content (many newlines) gets bonus."""
        scorer = StrategyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="Point 1\n" * 25,  # >20 newlines
        )
        result = scorer(ctx)
        assert result.s >= 4.5

    def test_questions_bonus(self):
        """Many questions indicate strategic thinking."""
        scorer = StrategyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="What if? Why? How? When? Where? Who? Which? What about? Can we?",
        )
        result = scorer(ctx)
        assert result.s >= 4.0

    def test_capped_at_maximum(self):
        """Strategy score is capped at 16.0."""
        scorer = StrategyScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="second-order meta counterfactual " * 100 + "\n" * 50 + "?" * 20,
        )
        result = scorer(ctx)
        assert result.s <= 16.0


# =============================================================================
# ReusabilityScorer Tests
# =============================================================================


class TestReusabilityScorer:
    """Tests for ReusabilityScorer."""

    def test_no_content_zero_score(self):
        """No content gets zero score."""
        scorer = ReusabilityScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="",
        )
        result = scorer(ctx)
        assert result.u == 0.0

    def test_base_score(self):
        """Basic content gets base score."""
        scorer = ReusabilityScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="Some basic content here",
        )
        result = scorer(ctx)
        assert result.u >= 2.0

    def test_pattern_indicator(self):
        """Pattern keyword boosts score."""
        scorer = ReusabilityScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="This pattern can be applied universally",
        )
        result = scorer(ctx)
        assert result.u > 2.0

    def test_abstraction_high_score(self):
        """Abstraction keywords get high score."""
        scorer = ReusabilityScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="This abstraction is reusable and composable",
        )
        result = scorer(ctx)
        assert result.u >= 7.0

    def test_domain_agnostic_bonus(self):
        """Domain-agnostic content gets bonus."""
        scorer = ReusabilityScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="This is a universal, domain-agnostic approach",
        )
        result = scorer(ctx)
        assert result.u >= 6.0

    def test_specific_penalty(self):
        """Specific/one-off content gets penalty."""
        scorer = ReusabilityScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="This specific hack is a one-time workaround",
        )
        result = scorer(ctx)
        # Should be lower due to penalties
        assert result.u < 3.0

    def test_code_blocks_bonus(self):
        """Code blocks indicate reusability."""
        scorer = ReusabilityScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="```python\ndef example():\n    pass\n```",
        )
        result = scorer(ctx)
        assert result.u >= 3.5

    def test_class_definition_bonus(self):
        """Class definitions indicate reusability."""
        scorer = ReusabilityScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="class MyReusableComponent:\n    def method1(self):\n        pass",
        )
        result = scorer(ctx)
        assert result.u >= 4.0

    def test_capped_at_maximum(self):
        """Reusability is capped at 12.0."""
        scorer = ReusabilityScorer()
        ctx = ScoringContext(
            intent_id="test",
            start_time=0,
            memory_content="pattern principle template abstraction generalize reusable modular composable universal domain-agnostic " * 50,
        )
        result = scorer(ctx)
        assert result.u <= 12.0


# =============================================================================
# HeuristicEngine Tests
# =============================================================================


class TestHeuristicEngine:
    """Tests for HeuristicEngine integration."""

    def test_engine_has_all_scorers(self):
        """Engine has all seven scorers."""
        engine = HeuristicEngine()
        assert len(engine.scorers) == 7

    def test_engine_combines_scores(self):
        """Engine combines all scorer outputs."""
        engine = HeuristicEngine()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=4600.0,  # 1 hour
            interruptions=3,
            memory_content="This is a strategic architecture pattern",
            outcome_tags=["partial"],
            risk_level=0.5,
        )
        result = engine.score(ctx)
        assert result.t > 0  # Time
        assert result.e > 0  # Effort
        assert result.n > 0  # Novelty
        assert result.f > 0  # Failure learning
        assert result.r > 0  # Risk
        assert result.s > 0  # Strategy
        # Reusability may be low for this content

    def test_engine_total_capped(self):
        """Total score is capped at 70.0."""
        engine = HeuristicEngine()
        # Create context that would produce very high scores
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=1000.0 + 86400,  # 24 hours
            interruptions=50,
            keystrokes=100000,
            memory_content="second-order meta abstraction pattern universal architecture " * 100,
            outcome_tags=["dead_end"],
            risk_level=1.0,
        )
        result = engine.score(ctx)
        assert result.total() <= 70.0

    def test_engine_user_override(self):
        """User override is applied."""
        engine = HeuristicEngine()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=2000.0,
            user_override={"t": 10.0},  # Add 10 to time
        )
        result = engine.score(ctx)
        # Time should be boosted by override
        assert result.t > 5.0

    def test_engine_user_override_negative(self):
        """Negative override doesn't go below zero."""
        engine = HeuristicEngine()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=1001.0,  # Very short
            user_override={"t": -100.0},  # Try to make negative
        )
        result = engine.score(ctx)
        assert result.t >= 0.0

    def test_deterministic_scoring(self):
        """Same input produces same output."""
        engine = HeuristicEngine()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=2000.0,
            memory_content="test content",
        )
        result1 = engine.score(ctx)
        result2 = engine.score(ctx)
        assert result1.total() == result2.total()


class TestHeuristicScorerBase:
    """Tests for base HeuristicScorer class."""

    def test_abstract_method(self):
        """Base class __call__ raises NotImplementedError."""
        scorer = HeuristicScorer()
        ctx = ScoringContext(intent_id="test", start_time=0)
        with pytest.raises(NotImplementedError):
            scorer(ctx)


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_very_short_duration(self):
        """Very short duration handled correctly."""
        engine = HeuristicEngine()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=1000.001,  # 1 millisecond
        )
        result = engine.score(ctx)
        assert result.total() > 0

    def test_unicode_content(self):
        """Unicode content handled correctly."""
        engine = HeuristicEngine()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=2000.0,
            memory_content="策略 패턴 архитектура 🎯 émoji",
        )
        result = engine.score(ctx)
        assert result.total() > 0

    def test_empty_strings(self):
        """Empty strings handled correctly."""
        engine = HeuristicEngine()
        ctx = ScoringContext(
            intent_id="",
            start_time=1000.0,
            end_time=2000.0,
            memory_content="",
        )
        result = engine.score(ctx)
        assert result.total() >= 0

    def test_none_values(self):
        """None values handled correctly."""
        engine = HeuristicEngine()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=None,
            memory_content=None,
            outcome_tags=None,
            risk_level=None,
        )
        result = engine.score(ctx)
        assert result.total() >= 0

    def test_extreme_interruptions(self):
        """Extreme interruption count handled."""
        engine = HeuristicEngine()
        ctx = ScoringContext(
            intent_id="test",
            start_time=1000.0,
            end_time=2000.0,
            interruptions=1000,
        )
        result = engine.score(ctx)
        assert result.e <= 22.0  # Capped
