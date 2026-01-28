# tests/test_integration.py
"""
Tests for IntentLog integration module.

Tests cover:
- RateLimiter class
- IntentLogConnector with rate limiting
- Event handling
- Stale intent cleanup
"""

import os
import time
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from value_ledger.integration import (
    RateLimiter,
    IntentEvent,
    IntentLogConnector,
    create_intentlog_listener,
    _DEFAULT_RATE_LIMIT,
    _DEFAULT_RATE_WINDOW,
)
from value_ledger.core import ValueLedger
from value_ledger.security import RateLimitError


# =============================================================================
# RateLimiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_default_settings(self):
        """Default settings are correct."""
        limiter = RateLimiter()
        assert limiter.max_events == _DEFAULT_RATE_LIMIT
        assert limiter.window_seconds == _DEFAULT_RATE_WINDOW

    def test_custom_settings(self):
        """Custom settings are applied."""
        limiter = RateLimiter(max_events=10, window_seconds=5.0)
        assert limiter.max_events == 10
        assert limiter.window_seconds == 5.0

    def test_check_allows_under_limit(self):
        """Events under limit are allowed."""
        limiter = RateLimiter(max_events=5, window_seconds=60.0)
        for _ in range(5):
            assert limiter.check() is True

    def test_check_raises_over_limit(self):
        """Events over limit raise RateLimitError."""
        limiter = RateLimiter(max_events=3, window_seconds=60.0)
        limiter.check()
        limiter.check()
        limiter.check()
        with pytest.raises(RateLimitError):
            limiter.check()

    def test_check_returns_false_no_raise(self):
        """Events over limit return False when raise_on_limit=False."""
        limiter = RateLimiter(max_events=2, window_seconds=60.0)
        assert limiter.check(raise_on_limit=False) is True
        assert limiter.check(raise_on_limit=False) is True
        assert limiter.check(raise_on_limit=False) is False

    def test_window_expiration(self):
        """Old events expire from the window."""
        limiter = RateLimiter(max_events=2, window_seconds=0.1)  # Very short window

        # Fill up the limit
        limiter.check()
        limiter.check()

        # Wait for window to expire
        time.sleep(0.15)

        # Should be allowed again
        assert limiter.check() is True

    def test_reset(self):
        """Reset clears the event history."""
        limiter = RateLimiter(max_events=2, window_seconds=60.0)
        limiter.check()
        limiter.check()

        # At limit
        assert limiter.check(raise_on_limit=False) is False

        # Reset
        limiter.reset()

        # Should be allowed again
        assert limiter.check() is True

    def test_current_count(self):
        """Current count tracks events correctly."""
        limiter = RateLimiter(max_events=10, window_seconds=60.0)
        assert limiter.current_count == 0

        limiter.check()
        assert limiter.current_count == 1

        limiter.check()
        limiter.check()
        assert limiter.current_count == 3

    def test_rate_limit_error_details(self):
        """RateLimitError contains useful details."""
        limiter = RateLimiter(max_events=1, window_seconds=30.0)
        limiter.check()

        try:
            limiter.check()
            assert False, "Should have raised RateLimitError"
        except RateLimitError as e:
            assert e.details["limit"] == 1
            assert e.details["window_seconds"] == 30.0
            assert e.details["current_count"] == 1


# =============================================================================
# IntentEvent Tests
# =============================================================================


class TestIntentEvent:
    """Tests for IntentEvent dataclass."""

    def test_minimal_event(self):
        """Minimal event with required fields."""
        event = IntentEvent(
            event_type="intent_started",
            intent_id="test-123",
            timestamp=time.time(),
        )
        assert event.event_type == "intent_started"
        assert event.intent_id == "test-123"
        assert event.interruptions == 0
        assert event.human_reasoning is None

    def test_full_event(self):
        """Full event with all fields."""
        event = IntentEvent(
            event_type="intent_completed",
            intent_id="test-456",
            timestamp=1000.0,
            human_reasoning="I want to do X",
            agent_output="Here is X done",
            memory_hash="sha256:abc",
            interruptions=3,
            keystrokes=500,
            outcome_tags=["success"],
            risk_level=0.5,
            metadata={"key": "value"},
        )
        assert event.agent_output == "Here is X done"
        assert event.interruptions == 3
        assert event.outcome_tags == ["success"]


# =============================================================================
# IntentLogConnector Tests
# =============================================================================


class TestIntentLogConnector:
    """Tests for IntentLogConnector class."""

    @pytest.fixture
    def temp_ledger(self):
        """Create a temporary ledger."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        yield ValueLedger(path)
        if os.path.exists(path):
            os.remove(path)

    def test_default_rate_limit(self, temp_ledger):
        """Default rate limit is enabled."""
        connector = IntentLogConnector(temp_ledger)
        assert connector._rate_limiter is not None
        assert connector._rate_limiter.max_events == _DEFAULT_RATE_LIMIT

    def test_custom_rate_limit(self, temp_ledger):
        """Custom rate limit is applied."""
        connector = IntentLogConnector(temp_ledger, rate_limit=50, rate_window=30.0)
        assert connector._rate_limiter.max_events == 50
        assert connector._rate_limiter.window_seconds == 30.0

    def test_disable_rate_limit(self, temp_ledger):
        """Rate limiting can be disabled."""
        connector = IntentLogConnector(temp_ledger, rate_limit=None)
        assert connector._rate_limiter is None

    def test_handle_event_rate_limited(self, temp_ledger):
        """Events are rate limited."""
        connector = IntentLogConnector(temp_ledger, rate_limit=2, rate_window=60.0)

        event = IntentEvent(
            event_type="intent_started",
            intent_id="test",
            timestamp=time.time(),
        )

        # First two should succeed
        connector.handle_event(event)
        connector.handle_event(event)

        # Third should be rate limited
        with pytest.raises(RateLimitError):
            connector.handle_event(event)

        assert connector.rate_limited_count == 1

    def test_handle_event_no_raise(self, temp_ledger):
        """Rate limiting can silently drop events."""
        connector = IntentLogConnector(temp_ledger, rate_limit=1, rate_window=60.0)

        event = IntentEvent(
            event_type="intent_started",
            intent_id="test",
            timestamp=time.time(),
        )

        connector.handle_event(event)
        # This should not raise
        connector.handle_event(event, raise_on_rate_limit=False)

    def test_handle_event_dict(self, temp_ledger):
        """Events can be passed as dictionaries."""
        connector = IntentLogConnector(temp_ledger)

        event_dict = {
            "event_type": "intent_started",
            "intent_id": "test-dict",
            "timestamp": time.time(),
        }

        connector.handle_event(event_dict)
        assert "test-dict" in connector.active_intents

    def test_intent_started_tracked(self, temp_ledger):
        """intent_started events are tracked."""
        connector = IntentLogConnector(temp_ledger)

        event = IntentEvent(
            event_type="intent_started",
            intent_id="track-test",
            timestamp=1000.0,
        )

        connector.handle_event(event)
        assert "track-test" in connector.active_intents
        assert connector.active_intents["track-test"] == 1000.0


class TestIntentLogConnectorStaleCleanup:
    """Tests for stale intent cleanup."""

    @pytest.fixture
    def temp_ledger(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        yield ValueLedger(path)
        if os.path.exists(path):
            os.remove(path)

    def test_cleanup_stale_intents(self, temp_ledger):
        """Stale intents are cleaned up."""
        connector = IntentLogConnector(temp_ledger, max_intent_age=1.0)

        # Add some intents with old timestamps
        connector.active_intents["old-1"] = time.time() - 10.0
        connector.active_intents["old-2"] = time.time() - 5.0
        connector.active_intents["new-1"] = time.time()

        removed = connector.cleanup_stale_intents()

        assert removed == 2
        assert "old-1" not in connector.active_intents
        assert "old-2" not in connector.active_intents
        assert "new-1" in connector.active_intents

    def test_cleanup_triggered_automatically(self, temp_ledger):
        """Cleanup is triggered when active_intents exceeds 100."""
        connector = IntentLogConnector(temp_ledger, max_intent_age=1.0)

        # Add 101 stale intents
        for i in range(101):
            connector.active_intents[f"stale-{i}"] = time.time() - 10.0

        # Add a new event - should trigger cleanup
        event = IntentEvent(
            event_type="intent_started",
            intent_id="trigger",
            timestamp=time.time(),
        )
        connector.handle_event(event)

        # All stale intents should be cleaned up
        assert len(connector.active_intents) == 1
        assert "trigger" in connector.active_intents


class TestCreateIntentLogListener:
    """Tests for factory function."""

    def test_create_listener(self):
        """Factory creates a connector."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            connector = create_intentlog_listener(path)
            assert isinstance(connector, IntentLogConnector)
            assert connector._rate_limiter is not None
        finally:
            if os.path.exists(path):
                os.remove(path)
