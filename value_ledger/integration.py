# value_ledger/integration.py
"""
Integration with IntentLog events.
Listens for intent lifecycle events and automatically accrues value using heuristics.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

import logging

from .core import ValueLedger
from .heuristics import ScoringContext, HeuristicEngine

logger = logging.getLogger(__name__)

# Maximum age (in seconds) for an active intent before it's considered stale
_MAX_INTENT_AGE = 24 * 60 * 60  # 24 hours


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
    """

    def __init__(self, ledger: ValueLedger, max_intent_age: float = _MAX_INTENT_AGE):
        self.ledger = ledger
        self.engine = HeuristicEngine()
        self.active_intents: Dict[str, float] = {}  # intent_id -> start_time
        self._max_intent_age = max_intent_age

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

    def handle_event(self, event: IntentEvent | Dict[str, Any]):
        """
        Main entry point — called by IntentLog (or a message bus) on every event.
        """
        if isinstance(event, dict):
            event = IntentEvent(**event)

        # Periodically cleanup stale intents to prevent memory leak
        if len(self.active_intents) > 100:
            self.cleanup_stale_intents()

        if event.event_type == "intent_started":
            self._on_intent_started(event)

        elif event.event_type in {"intent_completed", "intent_abandoned"}:
            self._on_intent_completed(event)

        # "intent_updated" could trigger partial accruals in future
        # For now, we only accrue on completion/abandonment

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
