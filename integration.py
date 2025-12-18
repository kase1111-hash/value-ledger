# value_ledger/integration.py
"""
Integration with IntentLog events.
Listens for intent lifecycle events and automatically accrues value using heuristics.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .core import ValueLedger
from .heuristics import ScoringContext, HeuristicEngine


@dataclass
class IntentEvent:
    """Standardized event payload from IntentLog"""
    event_type: str  # "intent_started", "intent_updated", "intent_completed", "intent_abandoned"
    intent_id: str
    timestamp: float
    human_reasoning: Optional[str] = None          # Raw human intent description
    agent_output: Optional[str] = None             # Final AI response or result
    memory_hash: Optional[str] = None              # From Memory Vault after encryption
    interruptions: int = 0                         # Tracked by Boundary Daemon
    keystrokes: Optional[int] = None
    outcome_tags: Optional[list[str]] = None       # e.g., ["success", "failure", "partial", "breakthrough"]
    risk_level: Optional[float] = None             # 0.0–1.0
    metadata: Optional[Dict[str, Any]] = None


class IntentLogConnector:
    """
    Connects ValueLedger to IntentLog event stream.
    Call .handle_event() whenever IntentLog emits an event.
    """

    def __init__(self, ledger: ValueLedger):
        self.ledger = ledger
        self.engine = HeuristicEngine()
        self.active_intents: Dict[str, float] = {}  # intent_id -> start_time

    def handle_event(self, event: IntentEvent | Dict[str, Any]):
        """
        Main entry point — called by IntentLog (or a message bus) on every event.
        """
        if isinstance(event, dict):
            event = IntentEvent(**event)

        if event.event_type == "intent_started":
            self._on_intent_started(event)

        elif event.event_type in {"intent_completed", "intent_abandoned"}:
            self._on_intent_completed(event)

        # "intent_updated" could trigger partial accruals in future
        # For now, we only accrue on completion/abandonment

    def _on_intent_started(self, event: IntentEvent):
        """Record start time for duration tracking"""
        self.active_intents[event.intent_id] = event.timestamp
        print(f"[ValueLedger] Intent started: {event.intent_id}")

    def _on_intent_completed(self, event: IntentEvent):
        """Main accrual logic — triggered on completion or abandonment"""
        start_time = self.active_intents.pop(event.intent_id, event.timestamp)
        end_time = event.timestamp

        # Build rich scoring context from event
        content_for_analysis = ""
        if event.human_reasoning:
            content_for_analysis += event.human_reasoning + "\n"
        if event.agent_output:
            content_for_analysis += event.agent_output

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
            # previous_memories would come from Memory Vault query — stub for now
            previous_memories=None,
            user_override=None,  # Could allow human to tweak post-completion
        )

        # Auto-accrue using full heuristic engine
        entry_id = self.ledger.accrue_with_heuristics(
            ctx=ctx,
            metadata={
                "event_type": event.event_type,
                "outcome_tags": event.outcome_tags,
                "source": "IntentLog",
                **(event.metadata or {}),
            }
        )

        current_value = self.ledger.current_value_for_intent(event.intent_id)
        print(f"[ValueLedger] Accrued value for {event.intent_id}")
        print(f"           Entry: {entry_id[:8]}... | Total: {current_value.total():.1f}")
        print(f"           Vector: {current_value.dict()}")


# ———————————————————————————————
# Hook for external systems (e.g., Agent-OS core)
# ———————————————————————————————

def create_intentlog_listener(ledger_path: str = "ledger.jsonl") -> IntentLogConnector:
    """
    Factory function — used by Agent-OS to get a ready listener
    """
    ledger = ValueLedger(ledger_path)
    return IntentLogConnector(ledger)
