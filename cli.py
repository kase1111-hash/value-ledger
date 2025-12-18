"""
Simple CLI to test the ledger
"""

import json
from value_ledger.core import ValueLedger, ValueVector

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
