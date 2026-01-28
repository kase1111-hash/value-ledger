# value_ledger/integration.py
"""
Integration with IntentLog events.
Listens for intent lifecycle events and automatically accrues value using heuristics.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import logging

from .core import ValueLedger
from .heuristics import ScoringContext, HeuristicEngine
from .security import RateLimitError

logger = logging.getLogger(__name__)

# Maximum age (in seconds) for an active intent before it's considered stale
_MAX_INTENT_AGE = 24 * 60 * 60  # 24 hours

# Default rate limiting settings
_DEFAULT_RATE_LIMIT = 100  # events per window
_DEFAULT_RATE_WINDOW = 60.0  # seconds


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for event processing.

    Prevents DoS attacks and excessive resource consumption from
    runaway event streams.
    """

    max_events: int = _DEFAULT_RATE_LIMIT
    window_seconds: float = _DEFAULT_RATE_WINDOW
    _events: list = field(default_factory=list)

    def check(self, raise_on_limit: bool = True) -> bool:
        """
        Check if a new event is allowed under the rate limit.

        Args:
            raise_on_limit: If True, raise RateLimitError when limit exceeded

        Returns:
            True if event is allowed, False if rate limited

        Raises:
            RateLimitError: If raise_on_limit=True and limit is exceeded
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Remove old events outside the window
        self._events = [t for t in self._events if t > window_start]

        # Check if under limit
        if len(self._events) >= self.max_events:
            if raise_on_limit:
                raise RateLimitError(
                    f"Rate limit exceeded: {self.max_events} events per {self.window_seconds}s",
                    details={
                        "limit": self.max_events,
                        "window_seconds": self.window_seconds,
                        "current_count": len(self._events),
                    }
                )
            return False

        # Record this event
        self._events.append(now)
        return True

    def reset(self):
        """Reset the rate limiter state."""
        self._events = []

    @property
    def current_count(self) -> int:
        """Current number of events in the window."""
        now = time.time()
        window_start = now - self.window_seconds
        return len([t for t in self._events if t > window_start])


@dataclass
class IntentEvent:
    """Standardized event payload from IntentLog"""

    event_type: str  # "intent_started", "intent_updated", "intent_completed", "intent_abandoned"
    intent_id: str
    timestamp: float
    human_reasoning: Optional[str] = None  # Raw human intent description
    agent_output: Optional[str] = None  # Final AI response or result
    memory_hash: Optional[str] = None  # From Memory Vault after encryption
    interruptions: int = 0  # Tracked by Boundary Daemon
    keystrokes: Optional[int] = None
    outcome_tags: Optional[list[str]] = (
        None  # e.g., ["success", "failure", "partial", "breakthrough"]
    )
    risk_level: Optional[float] = None  # 0.0–1.0
    metadata: Optional[Dict[str, Any]] = None


class IntentLogConnector:
    """
    Connects ValueLedger to IntentLog event stream.
    Call .handle_event() whenever IntentLog emits an event.

    Rate limiting is enabled by default to prevent DoS and resource exhaustion.
    """

    def __init__(
        self,
        ledger: ValueLedger,
        max_intent_age: float = _MAX_INTENT_AGE,
        rate_limit: Optional[int] = _DEFAULT_RATE_LIMIT,
        rate_window: float = _DEFAULT_RATE_WINDOW,
    ):
        """
        Initialize the connector.

        Args:
            ledger: ValueLedger instance to accrue to
            max_intent_age: Maximum age (seconds) before intent is considered stale
            rate_limit: Maximum events per window (None to disable rate limiting)
            rate_window: Rate limit window in seconds
        """
        self.ledger = ledger
        self.engine = HeuristicEngine()
        self.active_intents: Dict[str, float] = {}  # intent_id -> start_time
        self._max_intent_age = max_intent_age

        # Rate limiting
        self._rate_limiter: Optional[RateLimiter] = None
        if rate_limit is not None:
            self._rate_limiter = RateLimiter(
                max_events=rate_limit,
                window_seconds=rate_window,
            )
        self._rate_limited_count = 0

    def cleanup_stale_intents(self) -> int:
        """
        Remove intents that have been active for longer than max_intent_age.
        Returns the number of stale intents removed.
        """
        current_time = time.time()
        stale_intents = [
            intent_id
            for intent_id, start_time in self.active_intents.items()
            if current_time - start_time > self._max_intent_age
        ]
        for intent_id in stale_intents:
            logger.warning(f"Removing stale intent (no completion event): {intent_id}")
            del self.active_intents[intent_id]
        return len(stale_intents)

    def handle_event(self, event: IntentEvent | Dict[str, Any], raise_on_rate_limit: bool = True):
        """
        Main entry point — called by IntentLog (or a message bus) on every event.

        Args:
            event: The intent event to process
            raise_on_rate_limit: If True, raise RateLimitError when rate limited.
                                 If False, silently drop the event.

        Raises:
            RateLimitError: If rate limit is exceeded and raise_on_rate_limit=True
        """
        if isinstance(event, dict):
            event = IntentEvent(**event)

        # Check rate limit before processing
        if self._rate_limiter is not None:
            try:
                self._rate_limiter.check(raise_on_limit=raise_on_rate_limit)
            except RateLimitError:
                self._rate_limited_count += 1
                logger.warning(
                    f"Rate limit exceeded, dropping event for intent {event.intent_id}"
                )
                raise

        # Periodically cleanup stale intents to prevent memory leak
        if len(self.active_intents) > 100:
            self.cleanup_stale_intents()

        if event.event_type == "intent_started":
            self._on_intent_started(event)

        elif event.event_type in {"intent_completed", "intent_abandoned"}:
            self._on_intent_completed(event)

        # "intent_updated" could trigger partial accruals in future
        # For now, we only accrue on completion/abandonment

    @property
    def rate_limited_count(self) -> int:
        """Number of events that were rate limited."""
        return self._rate_limited_count

    def _on_intent_started(self, event: IntentEvent):
        """Record start time for duration tracking"""
        self.active_intents[event.intent_id] = event.timestamp
        logger.info(f"Intent started: {event.intent_id}")

    def _on_intent_completed(self, event: IntentEvent):
        """Process completed/abandoned intent and accrue value"""
        # Calculate duration
        start_time = self.active_intents.get(event.intent_id, event.timestamp - 3600)
        end_time = event.timestamp

        # Build rich scoring context from event
        content_for_analysis = ""
        if event.human_reasoning:
            content_for_analysis += event.human_reasoning + "\n"
        if event.agent_output:
            content_for_analysis += event.agent_output

        # === Memory Vault integration for novelty scoring ===
        from .memory_vault_hook import MemoryVaultHook

        mv_hook = MemoryVaultHook()
        novelty_context = mv_hook.get_novelty_context(
            current_content=content_for_analysis,
            intent_id=event.intent_id,
            memory_hash=event.memory_hash,
        )

        # Build full scoring context
        ctx = ScoringContext(
            intent_id=event.intent_id,
            start_time=start_time,
            end_time=end_time,
            interruptions=event.interruptions,
            keystrokes=event.keystrokes,
            memory_content=content_for_analysis or None,
            memory_hash=event.memory_hash,
            outcome_tags=event.outcome_tags or [],
            risk_level=event.risk_level,
            previous_memories=novelty_context,  # Real data from Memory Vault
            user_override=None,  # Could allow human to tweak post-completion
        )

        # Store raw content temporarily for later reassessment (if permitted)
        metadata = {
            "event_type": event.event_type,
            "outcome_tags": event.outcome_tags,
            "source": "IntentLog",
            "raw_content_for_novelty": (
                content_for_analysis if mv_hook.can_access_content(event.intent_id) else None
            ),
            **(event.metadata or {}),
        }

        # Auto-accrue using full heuristic engine
        entry_id = self.ledger.accrue_with_heuristics(ctx=ctx, metadata=metadata)

        # Clean up tracking
        if event.intent_id in self.active_intents:
            del self.active_intents[event.intent_id]

        # Log results
        current_value = self.ledger.current_value_for_intent(event.intent_id)
        logger.info(f"Accrued value for {event.intent_id}")
        logger.debug(f"Entry: {entry_id[:8]}... | Total: {current_value.total():.1f}")
        logger.debug(f"Vector: {current_value.model_dump()}")


# ———————————————————————————————
# Hook for external systems (e.g., Agent-OS core)
# ———————————————————————————————


def create_intentlog_listener(ledger_path: str = "ledger.jsonl") -> IntentLogConnector:
    """
    Factory function — used by Agent-OS to get a ready listener
    """
    ledger = ValueLedger(ledger_path)
    return IntentLogConnector(ledger)
