# value_ledger/interruption.py
"""
Boundary Daemon Integration for Value Ledger.

This module integrates with the Boundary Daemon to track interruptions
and context switches that affect effort scoring.

Boundary Daemon tracks:
- External interruptions (notifications, external events)
- Context switches (focus changes between tasks/applications)
- Self-interruptions (user-initiated breaks)
- Focus events (lost/regained focus)

Interruption Scoring:
Interruptions multiply effort value because maintaining focus despite
interruptions requires additional cognitive effort. The formula is:
    effort_factor = 1.0 + (weighted_interruptions * 0.35)
    if interruptions > 10: effort_factor += (interruptions - 10) * 0.1
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Protocol
from enum import Enum


class InterruptionType(Enum):
    """Types of interruptions tracked by Boundary Daemon."""
    EXTERNAL = "external"           # External notification, alert, etc.
    CONTEXT_SWITCH = "context_switch"  # Changed application/task focus
    SELF = "self"                   # User-initiated break
    FOCUS_LOST = "focus_lost"       # Lost focus (unfocused window, etc.)
    FOCUS_REGAINED = "focus_regained"  # Regained focus after interruption


@dataclass
class InterruptionEvent:
    """
    Represents a single interruption event from Boundary Daemon.

    Attributes:
        intent_id: The intent session this interruption belongs to
        timestamp: Unix timestamp when interruption occurred
        interruption_type: Category of interruption
        source: What caused the interruption (app name, notification source, etc.)
        duration: How long the interruption lasted (seconds), None if ongoing
        metadata: Additional context about the interruption
    """
    intent_id: str
    timestamp: float
    interruption_type: InterruptionType
    source: Optional[str] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterruptionSummary:
    """
    Summary of interruptions for an intent session.

    Returned when a session ends to provide scoring context.
    """
    intent_id: str
    total_count: int
    weighted_count: float
    by_type: Dict[str, int]
    total_interruption_time: float
    events: List[InterruptionEvent]
    session_start: float
    session_end: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intent_id": self.intent_id,
            "total_count": self.total_count,
            "weighted_count": self.weighted_count,
            "by_type": self.by_type,
            "total_interruption_time": self.total_interruption_time,
            "event_count": len(self.events),
            "session_start": self.session_start,
            "session_end": self.session_end,
        }


class InterruptionTracker:
    """
    Tracks interruptions per intent session for effort scoring.

    Usage:
        tracker = InterruptionTracker()
        tracker.start_session("intent-123")

        # Record interruptions as they occur
        tracker.record_interruption(InterruptionEvent(
            intent_id="intent-123",
            timestamp=time.time(),
            interruption_type=InterruptionType.EXTERNAL,
            source="slack",
        ))

        # When session ends, get summary for scoring
        summary = tracker.end_session("intent-123")
        effort_factor = 1.0 + (summary.weighted_count * 0.35)
    """

    # Weights for different interruption types
    # External interruptions are most disruptive, self-interruptions least
    TYPE_WEIGHTS = {
        InterruptionType.EXTERNAL: 1.0,
        InterruptionType.CONTEXT_SWITCH: 0.7,
        InterruptionType.FOCUS_LOST: 0.5,
        InterruptionType.SELF: 0.3,
        InterruptionType.FOCUS_REGAINED: 0.0,  # Positive event, no penalty
    }

    def __init__(self):
        """Initialize the interruption tracker."""
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.completed_summaries: List[InterruptionSummary] = []

    def start_session(self, intent_id: str) -> None:
        """
        Begin tracking interruptions for an intent.

        Args:
            intent_id: Unique identifier for the intent session
        """
        self.active_sessions[intent_id] = {
            "start_time": time.time(),
            "events": [],
        }

    def record_interruption(self, event: InterruptionEvent) -> None:
        """
        Record an interruption event.

        Args:
            event: The interruption event to record

        Raises:
            ValueError: If the intent session is not active
        """
        intent_id = event.intent_id

        # Auto-start session if not exists (graceful handling)
        if intent_id not in self.active_sessions:
            self.start_session(intent_id)

        self.active_sessions[intent_id]["events"].append(event)

    def get_interruption_count(self, intent_id: str) -> int:
        """
        Get total interruption count for an intent.

        Args:
            intent_id: Intent session identifier

        Returns:
            Total number of interruptions (excluding FOCUS_REGAINED)
        """
        if intent_id not in self.active_sessions:
            return 0

        events = self.active_sessions[intent_id]["events"]
        return sum(
            1 for e in events
            if e.interruption_type != InterruptionType.FOCUS_REGAINED
        )

    def get_weighted_interruptions(self, intent_id: str) -> float:
        """
        Calculate weighted interruption score for effort multiplier.

        Different interruption types have different weights:
        - External interruptions: 1.0 (most disruptive)
        - Context switches: 0.7
        - Focus lost: 0.5
        - Self-interruptions: 0.3 (least disruptive)
        - Focus regained: 0.0 (positive, no penalty)

        Args:
            intent_id: Intent session identifier

        Returns:
            Weighted sum of interruptions
        """
        if intent_id not in self.active_sessions:
            return 0.0

        events = self.active_sessions[intent_id]["events"]
        return sum(
            self.TYPE_WEIGHTS.get(e.interruption_type, 1.0)
            for e in events
        )

    def end_session(self, intent_id: str) -> InterruptionSummary:
        """
        Finalize session and return summary for effort scoring.

        Args:
            intent_id: Intent session identifier

        Returns:
            InterruptionSummary with all session data

        Raises:
            KeyError: If the intent session is not active
        """
        if intent_id not in self.active_sessions:
            # Return empty summary for unknown sessions
            return InterruptionSummary(
                intent_id=intent_id,
                total_count=0,
                weighted_count=0.0,
                by_type={},
                total_interruption_time=0.0,
                events=[],
                session_start=time.time(),
                session_end=time.time(),
            )

        session = self.active_sessions.pop(intent_id)
        events = session["events"]
        end_time = time.time()

        # Count by type
        by_type: Dict[str, int] = {}
        for event in events:
            type_name = event.interruption_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        # Calculate total interruption time
        total_time = sum(e.duration or 0.0 for e in events)

        summary = InterruptionSummary(
            intent_id=intent_id,
            total_count=sum(
                1 for e in events
                if e.interruption_type != InterruptionType.FOCUS_REGAINED
            ),
            weighted_count=sum(
                self.TYPE_WEIGHTS.get(e.interruption_type, 1.0)
                for e in events
            ),
            by_type=by_type,
            total_interruption_time=total_time,
            events=events,
            session_start=session["start_time"],
            session_end=end_time,
        )

        self.completed_summaries.append(summary)
        return summary

    def get_active_session_ids(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())

    def get_completed_summaries(self) -> List[InterruptionSummary]:
        """Get all completed session summaries."""
        return self.completed_summaries.copy()

    def is_session_active(self, intent_id: str) -> bool:
        """Check if a session is currently active."""
        return intent_id in self.active_sessions


class BoundaryDaemonListener:
    """
    Listens to events from Boundary Daemon and translates them to interruptions.

    The Boundary Daemon emits events like:
    - notification_received: External notification arrived
    - context_switch: User switched focus to different app/task
    - focus_lost: Window lost focus
    - focus_regained: Window regained focus
    - mode_changed: Boundary mode changed (OPEN, RESTRICTED, etc.)

    Usage:
        tracker = InterruptionTracker()
        listener = BoundaryDaemonListener(tracker)

        # Connect to daemon event stream
        listener.connect_to_daemon("unix:///var/run/boundary-daemon.sock")

        # Or manually feed events
        listener.handle_boundary_event({
            "type": "notification_received",
            "timestamp": time.time(),
            "source": "slack",
            "intent_id": "intent-123",
        })
    """

    # Map Boundary Daemon event types to InterruptionTypes
    EVENT_TYPE_MAP = {
        "notification_received": InterruptionType.EXTERNAL,
        "context_switch": InterruptionType.CONTEXT_SWITCH,
        "focus_lost": InterruptionType.FOCUS_LOST,
        "focus_regained": InterruptionType.FOCUS_REGAINED,
        "external_alert": InterruptionType.EXTERNAL,
        "user_break": InterruptionType.SELF,
    }

    def __init__(
        self,
        tracker: InterruptionTracker,
        active_intent_resolver: Optional[Callable[[], Optional[str]]] = None,
    ):
        """
        Initialize the listener.

        Args:
            tracker: InterruptionTracker to record events to
            active_intent_resolver: Optional function to get current active intent ID
                                   (used when event doesn't specify intent_id)
        """
        self.tracker = tracker
        self.active_intent_resolver = active_intent_resolver
        self._connected = False
        self._daemon_url: Optional[str] = None

    def handle_boundary_event(self, event: Dict[str, Any]) -> Optional[InterruptionEvent]:
        """
        Handle an event from Boundary Daemon.

        Expected event format:
            {
                "type": "notification_received",  # Event type
                "timestamp": 1703350000.0,        # Unix timestamp
                "source": "slack",                # Optional source
                "intent_id": "intent-123",        # Optional, uses resolver if missing
                "duration": 5.0,                  # Optional duration in seconds
                "metadata": {...},                # Optional additional data
            }

        Args:
            event: Dictionary containing boundary event data

        Returns:
            The created InterruptionEvent, or None if event was ignored
        """
        event_type = event.get("type", "")

        # Map to interruption type
        interruption_type = self.EVENT_TYPE_MAP.get(event_type)
        if interruption_type is None:
            # Unknown event type, ignore
            return None

        # Resolve intent_id
        intent_id = event.get("intent_id")
        if not intent_id and self.active_intent_resolver:
            intent_id = self.active_intent_resolver()

        if not intent_id:
            # No intent to associate with, ignore
            return None

        # Create interruption event
        interruption_event = InterruptionEvent(
            intent_id=intent_id,
            timestamp=event.get("timestamp", time.time()),
            interruption_type=interruption_type,
            source=event.get("source"),
            duration=event.get("duration"),
            metadata=event.get("metadata", {}),
        )

        # Record to tracker
        self.tracker.record_interruption(interruption_event)

        return interruption_event

    def connect_to_daemon(self, daemon_url: str) -> bool:
        """
        Subscribe to Boundary Daemon event stream.

        This is a stub implementation. In production, this would:
        1. Connect to the Unix socket at daemon_url
        2. Subscribe to relevant event types
        3. Start a background listener thread

        Args:
            daemon_url: Unix socket URL (e.g., "unix:///var/run/boundary-daemon.sock")

        Returns:
            True if connection successful, False otherwise
        """
        # Stub implementation - real implementation would connect to daemon
        self._daemon_url = daemon_url
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from Boundary Daemon."""
        self._connected = False
        self._daemon_url = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to daemon."""
        return self._connected


class BoundaryEventEmitter(Protocol):
    """Protocol for boundary event sources (for testing/mocking)."""

    def register_listener(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a listener for boundary events."""
        ...

    def emit_event(self, event: Dict[str, Any]) -> None:
        """Emit a boundary event."""
        ...


class MockBoundaryEmitter:
    """Mock emitter for testing Boundary Daemon integration."""

    def __init__(self):
        """Initialize mock emitter."""
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []

    def register_listener(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a listener."""
        self.listeners.append(callback)

    def emit_event(self, event: Dict[str, Any]) -> None:
        """Emit an event to all listeners."""
        for listener in self.listeners:
            listener(event)

    def emit_notification(
        self,
        intent_id: str,
        source: str = "test",
        timestamp: Optional[float] = None,
    ) -> None:
        """Convenience method to emit a notification event."""
        self.emit_event({
            "type": "notification_received",
            "intent_id": intent_id,
            "source": source,
            "timestamp": timestamp or time.time(),
        })

    def emit_context_switch(
        self,
        intent_id: str,
        from_app: str = "editor",
        to_app: str = "browser",
        timestamp: Optional[float] = None,
    ) -> None:
        """Convenience method to emit a context switch event."""
        self.emit_event({
            "type": "context_switch",
            "intent_id": intent_id,
            "source": f"{from_app} -> {to_app}",
            "timestamp": timestamp or time.time(),
            "metadata": {"from_app": from_app, "to_app": to_app},
        })

    def emit_focus_lost(
        self,
        intent_id: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """Convenience method to emit a focus lost event."""
        self.emit_event({
            "type": "focus_lost",
            "intent_id": intent_id,
            "timestamp": timestamp or time.time(),
        })

    def emit_focus_regained(
        self,
        intent_id: str,
        duration: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Convenience method to emit a focus regained event."""
        self.emit_event({
            "type": "focus_regained",
            "intent_id": intent_id,
            "duration": duration,
            "timestamp": timestamp or time.time(),
        })


def calculate_effort_factor(
    weighted_interruptions: float,
    raw_count: Optional[int] = None,
) -> float:
    """
    Calculate the effort multiplier from interruption data.

    This implements the formula from specs.md and heuristics.py:
        factor = 1.0 + (weighted_interruptions * 0.35)
        if raw_count > 10: factor += (raw_count - 10) * 0.1

    Args:
        weighted_interruptions: Weighted sum of interruptions
        raw_count: Optional raw count for high-interruption bonus

    Returns:
        Effort multiplication factor (>= 1.0)
    """
    factor = 1.0 + (weighted_interruptions * 0.35)

    if raw_count and raw_count > 10:
        factor += (raw_count - 10) * 0.1

    return factor


def integrate_with_scoring_context(
    summary: InterruptionSummary,
    scoring_context: Any,
) -> None:
    """
    Merge interruption data into a ScoringContext.

    Modifies scoring_context in place to update interruption fields.

    Args:
        summary: InterruptionSummary from tracker
        scoring_context: ScoringContext to update
    """
    # Update the interruptions count
    scoring_context.interruptions = summary.total_count

    # Add detailed data to metadata if it exists
    if hasattr(scoring_context, "metadata") and scoring_context.metadata is not None:
        scoring_context.metadata["interruption_summary"] = summary.to_dict()
        scoring_context.metadata["weighted_interruptions"] = summary.weighted_count
    elif hasattr(scoring_context, "user_override"):
        # Store in user_override as fallback
        pass


def create_boundary_daemon_hook(
    tracker: Optional[InterruptionTracker] = None,
    active_intent_resolver: Optional[Callable[[], Optional[str]]] = None,
) -> tuple[InterruptionTracker, BoundaryDaemonListener]:
    """
    Factory function to create connected tracker and listener.

    Usage:
        tracker, listener = create_boundary_daemon_hook()
        listener.connect_to_daemon("unix:///var/run/boundary-daemon.sock")

    Args:
        tracker: Optional existing tracker to use
        active_intent_resolver: Optional function to resolve current intent ID

    Returns:
        Tuple of (InterruptionTracker, BoundaryDaemonListener)
    """
    tracker = tracker or InterruptionTracker()
    listener = BoundaryDaemonListener(tracker, active_intent_resolver)
    return tracker, listener
