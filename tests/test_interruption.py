# tests/test_interruption.py
"""
Tests for Boundary Daemon integration (interruption tracking).
"""

import time
import pytest

from value_ledger.interruption import (
    InterruptionType,
    InterruptionEvent,
    InterruptionSummary,
    InterruptionTracker,
    BoundaryDaemonListener,
    MockBoundaryEmitter,
    calculate_effort_factor,
    create_boundary_daemon_hook,
    integrate_with_scoring_context,
)
from value_ledger.heuristics import ScoringContext


class TestInterruptionType:
    """Tests for InterruptionType enum."""

    def test_all_types_exist(self):
        """Verify all expected interruption types exist."""
        assert InterruptionType.EXTERNAL.value == "external"
        assert InterruptionType.CONTEXT_SWITCH.value == "context_switch"
        assert InterruptionType.SELF.value == "self"
        assert InterruptionType.FOCUS_LOST.value == "focus_lost"
        assert InterruptionType.FOCUS_REGAINED.value == "focus_regained"


class TestInterruptionEvent:
    """Tests for InterruptionEvent dataclass."""

    def test_basic_event(self):
        """Test creating a basic event."""
        event = InterruptionEvent(
            intent_id="test-123",
            timestamp=1000.0,
            interruption_type=InterruptionType.EXTERNAL,
        )
        assert event.intent_id == "test-123"
        assert event.timestamp == 1000.0
        assert event.interruption_type == InterruptionType.EXTERNAL
        assert event.source is None
        assert event.duration is None
        assert event.metadata == {}

    def test_full_event(self):
        """Test creating an event with all fields."""
        event = InterruptionEvent(
            intent_id="test-456",
            timestamp=2000.0,
            interruption_type=InterruptionType.CONTEXT_SWITCH,
            source="slack",
            duration=5.5,
            metadata={"app": "test"},
        )
        assert event.source == "slack"
        assert event.duration == 5.5
        assert event.metadata == {"app": "test"}


class TestInterruptionTracker:
    """Tests for InterruptionTracker class."""

    def test_start_session(self):
        """Test starting a tracking session."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")
        assert tracker.is_session_active("intent-1")
        assert "intent-1" in tracker.get_active_session_ids()

    def test_record_interruption(self):
        """Test recording interruptions."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")

        event = InterruptionEvent(
            intent_id="intent-1",
            timestamp=time.time(),
            interruption_type=InterruptionType.EXTERNAL,
            source="notification",
        )
        tracker.record_interruption(event)

        assert tracker.get_interruption_count("intent-1") == 1

    def test_auto_start_session_on_record(self):
        """Test that recording to unknown session auto-starts it."""
        tracker = InterruptionTracker()

        event = InterruptionEvent(
            intent_id="auto-start",
            timestamp=time.time(),
            interruption_type=InterruptionType.EXTERNAL,
        )
        tracker.record_interruption(event)

        assert tracker.is_session_active("auto-start")
        assert tracker.get_interruption_count("auto-start") == 1

    def test_weighted_interruptions(self):
        """Test weighted interruption calculation."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")

        # Add different types of interruptions
        events = [
            InterruptionEvent(
                intent_id="intent-1",
                timestamp=time.time(),
                interruption_type=InterruptionType.EXTERNAL,  # weight 1.0
            ),
            InterruptionEvent(
                intent_id="intent-1",
                timestamp=time.time(),
                interruption_type=InterruptionType.CONTEXT_SWITCH,  # weight 0.7
            ),
            InterruptionEvent(
                intent_id="intent-1",
                timestamp=time.time(),
                interruption_type=InterruptionType.SELF,  # weight 0.3
            ),
            InterruptionEvent(
                intent_id="intent-1",
                timestamp=time.time(),
                interruption_type=InterruptionType.FOCUS_REGAINED,  # weight 0.0
            ),
        ]
        for event in events:
            tracker.record_interruption(event)

        weighted = tracker.get_weighted_interruptions("intent-1")
        assert weighted == pytest.approx(2.0, abs=0.01)  # 1.0 + 0.7 + 0.3 + 0.0

    def test_focus_regained_not_counted(self):
        """Test that focus regained events don't count as interruptions."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")

        tracker.record_interruption(InterruptionEvent(
            intent_id="intent-1",
            timestamp=time.time(),
            interruption_type=InterruptionType.FOCUS_REGAINED,
        ))

        # Focus regained should not count as an interruption
        assert tracker.get_interruption_count("intent-1") == 0

    def test_end_session_returns_summary(self):
        """Test ending a session returns a summary."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")

        tracker.record_interruption(InterruptionEvent(
            intent_id="intent-1",
            timestamp=time.time(),
            interruption_type=InterruptionType.EXTERNAL,
            source="test",
            duration=3.0,
        ))

        summary = tracker.end_session("intent-1")

        assert isinstance(summary, InterruptionSummary)
        assert summary.intent_id == "intent-1"
        assert summary.total_count == 1
        assert summary.weighted_count == pytest.approx(1.0)
        assert "external" in summary.by_type
        assert summary.total_interruption_time == pytest.approx(3.0)
        assert len(summary.events) == 1
        assert not tracker.is_session_active("intent-1")

    def test_end_session_unknown_returns_empty_summary(self):
        """Test ending unknown session returns empty summary."""
        tracker = InterruptionTracker()
        summary = tracker.end_session("unknown")

        assert summary.total_count == 0
        assert summary.weighted_count == 0.0
        assert len(summary.events) == 0

    def test_completed_summaries_stored(self):
        """Test that completed summaries are stored."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")
        tracker.end_session("intent-1")

        summaries = tracker.get_completed_summaries()
        assert len(summaries) == 1
        assert summaries[0].intent_id == "intent-1"

    def test_summary_to_dict(self):
        """Test summary serialization."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")
        summary = tracker.end_session("intent-1")

        d = summary.to_dict()
        assert "intent_id" in d
        assert "total_count" in d
        assert "weighted_count" in d
        assert "by_type" in d
        assert "session_start" in d
        assert "session_end" in d


class TestBoundaryDaemonListener:
    """Tests for BoundaryDaemonListener class."""

    def test_handle_notification_event(self):
        """Test handling notification events."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")
        listener = BoundaryDaemonListener(tracker)

        result = listener.handle_boundary_event({
            "type": "notification_received",
            "intent_id": "intent-1",
            "source": "slack",
            "timestamp": time.time(),
        })

        assert result is not None
        assert result.interruption_type == InterruptionType.EXTERNAL
        assert result.source == "slack"
        assert tracker.get_interruption_count("intent-1") == 1

    def test_handle_context_switch(self):
        """Test handling context switch events."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")
        listener = BoundaryDaemonListener(tracker)

        listener.handle_boundary_event({
            "type": "context_switch",
            "intent_id": "intent-1",
            "timestamp": time.time(),
        })

        assert tracker.get_interruption_count("intent-1") == 1

    def test_handle_focus_events(self):
        """Test handling focus lost/regained events."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")
        listener = BoundaryDaemonListener(tracker)

        listener.handle_boundary_event({
            "type": "focus_lost",
            "intent_id": "intent-1",
            "timestamp": time.time(),
        })

        listener.handle_boundary_event({
            "type": "focus_regained",
            "intent_id": "intent-1",
            "duration": 5.0,
            "timestamp": time.time(),
        })

        # Only focus_lost counts as interruption
        assert tracker.get_interruption_count("intent-1") == 1

    def test_unknown_event_type_ignored(self):
        """Test that unknown event types are ignored."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")
        listener = BoundaryDaemonListener(tracker)

        result = listener.handle_boundary_event({
            "type": "unknown_event",
            "intent_id": "intent-1",
            "timestamp": time.time(),
        })

        assert result is None
        assert tracker.get_interruption_count("intent-1") == 0

    def test_missing_intent_id_with_resolver(self):
        """Test using active_intent_resolver when intent_id missing."""
        tracker = InterruptionTracker()
        tracker.start_session("resolved-intent")

        def resolver():
            return "resolved-intent"

        listener = BoundaryDaemonListener(tracker, active_intent_resolver=resolver)

        result = listener.handle_boundary_event({
            "type": "notification_received",
            "timestamp": time.time(),
        })

        assert result is not None
        assert result.intent_id == "resolved-intent"

    def test_missing_intent_id_no_resolver(self):
        """Test that events without intent_id and no resolver are ignored."""
        tracker = InterruptionTracker()
        listener = BoundaryDaemonListener(tracker)

        result = listener.handle_boundary_event({
            "type": "notification_received",
            "timestamp": time.time(),
        })

        assert result is None

    def test_connect_disconnect(self):
        """Test connect/disconnect to daemon."""
        tracker = InterruptionTracker()
        listener = BoundaryDaemonListener(tracker)

        assert not listener.is_connected

        result = listener.connect_to_daemon("unix:///test.sock")
        assert result is True
        assert listener.is_connected

        listener.disconnect()
        assert not listener.is_connected


class TestMockBoundaryEmitter:
    """Tests for MockBoundaryEmitter class."""

    def test_emit_notification(self):
        """Test emitting notification events."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")
        listener = BoundaryDaemonListener(tracker)
        emitter = MockBoundaryEmitter()

        emitter.register_listener(listener.handle_boundary_event)
        emitter.emit_notification("intent-1", source="test")

        assert tracker.get_interruption_count("intent-1") == 1

    def test_emit_context_switch(self):
        """Test emitting context switch events."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")
        listener = BoundaryDaemonListener(tracker)
        emitter = MockBoundaryEmitter()

        emitter.register_listener(listener.handle_boundary_event)
        emitter.emit_context_switch("intent-1", from_app="editor", to_app="browser")

        assert tracker.get_interruption_count("intent-1") == 1

    def test_emit_focus_sequence(self):
        """Test emitting focus lost/regained sequence."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")
        listener = BoundaryDaemonListener(tracker)
        emitter = MockBoundaryEmitter()

        emitter.register_listener(listener.handle_boundary_event)
        emitter.emit_focus_lost("intent-1")
        emitter.emit_focus_regained("intent-1", duration=10.0)

        # Only focus_lost counts
        assert tracker.get_interruption_count("intent-1") == 1


class TestCalculateEffortFactor:
    """Tests for calculate_effort_factor function."""

    def test_zero_interruptions(self):
        """Test effort factor with no interruptions."""
        factor = calculate_effort_factor(0.0)
        assert factor == pytest.approx(1.0)

    def test_weighted_interruptions(self):
        """Test effort factor calculation."""
        # 5 weighted interruptions: 1.0 + (5 * 0.35) = 2.75
        factor = calculate_effort_factor(5.0)
        assert factor == pytest.approx(2.75)

    def test_high_interruption_bonus(self):
        """Test bonus for high raw count."""
        # 15 raw, 10 weighted:
        # Base: 1.0 + (10 * 0.35) = 4.5
        # Bonus: (15 - 10) * 0.1 = 0.5
        # Total: 5.0
        factor = calculate_effort_factor(10.0, raw_count=15)
        assert factor == pytest.approx(5.0)

    def test_no_bonus_under_threshold(self):
        """Test no bonus when raw count <= 10."""
        factor = calculate_effort_factor(5.0, raw_count=10)
        assert factor == pytest.approx(2.75)  # Same as without raw_count


class TestIntegration:
    """Integration tests for the full interruption tracking flow."""

    def test_full_tracking_flow(self):
        """Test complete tracking flow from start to scoring."""
        # Create tracker and listener
        tracker, listener = create_boundary_daemon_hook()

        # Start session
        tracker.start_session("intent-1")

        # Simulate boundary events
        events = [
            {"type": "notification_received", "intent_id": "intent-1", "source": "email"},
            {"type": "context_switch", "intent_id": "intent-1"},
            {"type": "focus_lost", "intent_id": "intent-1"},
            {"type": "focus_regained", "intent_id": "intent-1", "duration": 30},
            {"type": "notification_received", "intent_id": "intent-1", "source": "slack"},
        ]

        for event in events:
            event["timestamp"] = time.time()
            listener.handle_boundary_event(event)

        # End session and get summary
        summary = tracker.end_session("intent-1")

        # Verify counts
        assert summary.total_count == 4  # focus_regained doesn't count
        assert summary.weighted_count > 0

        # Calculate effort factor
        factor = calculate_effort_factor(
            summary.weighted_count,
            raw_count=summary.total_count,
        )
        assert factor > 1.0

    def test_integrate_with_scoring_context(self):
        """Test integrating summary with ScoringContext."""
        tracker = InterruptionTracker()
        tracker.start_session("intent-1")

        # Add some interruptions
        tracker.record_interruption(InterruptionEvent(
            intent_id="intent-1",
            timestamp=time.time(),
            interruption_type=InterruptionType.EXTERNAL,
        ))
        tracker.record_interruption(InterruptionEvent(
            intent_id="intent-1",
            timestamp=time.time(),
            interruption_type=InterruptionType.CONTEXT_SWITCH,
        ))

        summary = tracker.end_session("intent-1")

        # Create scoring context
        ctx = ScoringContext(
            intent_id="intent-1",
            start_time=summary.session_start,
            end_time=summary.session_end,
            interruptions=0,  # Will be updated
        )

        # Integrate
        integrate_with_scoring_context(summary, ctx)

        # Verify integration
        assert ctx.interruptions == 2


class TestMultipleSessions:
    """Tests for handling multiple concurrent sessions."""

    def test_multiple_sessions(self):
        """Test tracking multiple sessions concurrently."""
        tracker = InterruptionTracker()

        # Start two sessions
        tracker.start_session("intent-1")
        tracker.start_session("intent-2")

        # Add interruptions to each
        tracker.record_interruption(InterruptionEvent(
            intent_id="intent-1",
            timestamp=time.time(),
            interruption_type=InterruptionType.EXTERNAL,
        ))
        tracker.record_interruption(InterruptionEvent(
            intent_id="intent-2",
            timestamp=time.time(),
            interruption_type=InterruptionType.EXTERNAL,
        ))
        tracker.record_interruption(InterruptionEvent(
            intent_id="intent-2",
            timestamp=time.time(),
            interruption_type=InterruptionType.EXTERNAL,
        ))

        # Verify separate counts
        assert tracker.get_interruption_count("intent-1") == 1
        assert tracker.get_interruption_count("intent-2") == 2

        # End sessions
        summary1 = tracker.end_session("intent-1")
        summary2 = tracker.end_session("intent-2")

        assert summary1.total_count == 1
        assert summary2.total_count == 2
