"""
Simple CLI to test the ledger
"""

import json
from value_ledger.core import ValueLedger, ValueVector

def full_demo_with_heuristics():
    from value_ledger.core import ValueLedger
    from value_ledger.heuristics import ScoringContext

    ledger = ValueLedger("smart_ledger.jsonl")
def live_intent_simulation():
    """Simulates a stream of IntentLog events"""
    from value_ledger.core import ValueLedger
    from value_ledger.integration import IntentLogConnector, IntentEvent
    import time

    ledger = ValueLedger("integrated_ledger.jsonl")
    connector = IntentLogConnector(ledger)

    print("=== Simulating IntentLog event stream ===\n")

    # Event 1: Start intent
    connector.handle_event(IntentEvent(
        event_type="intent_started",
        intent_id="sim-001",
        timestamp=time.time(),
        human_reasoning="Design a better way to track cognitive effort in AI agents. "
                        "Current systems undervalue failures and deep thinking.",
        interruptions=0,
    ))
    time.sleep(1)

    # Event 2: Complete with rich outcome
    connector.handle_event(IntentEvent(
        event_type="intent_completed",
        intent_id="sim-001",
        timestamp=time.time(),
        human_reasoning="Design a better way to track cognitive effort...",
        agent_output="""
        After exploring several dead ends with token-based valuation,
        I realized that proof-of-effort via heuristic vectors (T/E/N/F/R/S)
        preserves value even in failure. This breakthrough changes the architecture.
        """,
        memory_hash="sha256:deadbeef123...",
        interruptions=12,
        keystrokes=3800,
        outcome_tags=["breakthrough", "partial_failure"],
        risk_level=0.75,
        metadata={"session_focus": "architecture_design"},
    ))

    print("\n=== Ledger state after one full intent ===")
    for entry in ledger.entries:
        print(f"{entry.timestamp:.0f} | {entry.status:6} | Total: {entry.value_vector.total():.1f} | {entry.metadata.get('event_type')}")
    ctx = ScoringContext(
        intent_id="intent-042",
        start_time=time.time() - 7200,  # 2 hours ago
        end_time=time.time(),
        interruptions=7,
        keystrokes=2400,
        memory_content="""
        I spent all morning chasing a dead-end with the retrieval system.
        Realized halfway through that the embedding space was collapsing.
        High risk — this could have broken the whole agent loop.
        But now I see a much better architecture using dynamic context windows.
        """,
        previous_memories=[
            ("hash1", time.time() - 86400, "tried retrieval before, didn't work"),
        ],
        outcome_tags=["dead_end", "breakthrough"],
        risk_level=0.8,
    )

    print("Accruing with full heuristic engine...")
    entry_id = ledger.accrue_with_heuristics(ctx)
    print(f"Entry ID: {entry_id[:8]}...")

    current = ledger.current_value_for_intent("intent-042")
    print("Auto-scored vector:")
    print(current.dict())
def demo():
    ledger = ValueLedger("demo_ledger.jsonl")

    print("Accruing initial cognitive effort...")
    entry1 = ledger.accrue(
        intent_id="intent-001",
        initial_vector={"t": 8.0, "e": 6.5, "n": 7.0, "f": 3.0, "r": 2.0, "s": 5.0},
        memory_hash="sha256:abc123...",
        metadata={"task": "exploring novel architecture"}
    )
    print(f"Accrued entry: {entry1[:8]}...")

    print("\nSimulating later reflection: outcome success + reduced novelty")
    parent = ledger.get_entry(entry1)
    corrected_vector = ValueVector(
        t=parent.value_vector.t,
        e=parent.value_vector.e,
        n=parent.value_vector.n * 0.6,  # Less novel than thought
        f=parent.value_vector.f * 1.8,  # Failure led to breakthrough
        r=parent.value_vector.r,
        s=parent.value_vector.s * 1.4,
    )

    notes = {"outcome": "success", "novelty_reassessment": "similar patterns found"}
    ledger.aggregate_correction(
        parent_id=entry1,
        new_vector=corrected_vector,
        notes=notes,
        freeze_parent=False,
    )

    print("\nCurrent value for intent-001:")
    print(json.dumps(ledger.current_value_for_intent("intent-001").dict(), indent=2))

if __name__ == "__main__":
    demo()
