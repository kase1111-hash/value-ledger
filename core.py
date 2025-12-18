"""
value_ledger/core.py

Core implementation of the Value Ledger.
Handles accrual, aggregation, freezing, revocation, and self-correction.
"""

from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import os

# Placeholder for future imports from common utils
def generate_entry_id(data: str) -> str:
    """Deterministic ID generation (replace with common.utils if available)"""
    return hashlib.sha256(data.encode()).hexdigest()


class ValueVector(BaseModel):
    """The six value units: T/E/N/F/R/S"""
    t: float = Field(default=0.0, ge=0.0, description="Time spent")
    e: float = Field(default=0.0, ge=0.0, description="Effort (interruption-adjusted)")
    n: float = Field(default=0.0, ge=0.0, description="Novelty")
    f: float = Field(default=0.0, ge=0.0, description="Value from failure/learning")
    r: float = Field(default=0.0, ge=0.0, description="Risk exposure")
    s: float = Field(default=0.0, ge=0.0, description="Strategic insight")

    def total(self) -> float:
        return sum(self.dict().values())

    def apply_subjective_weights(self, weights: Dict[str, float]) -> "ValueVector":
        data = self.dict()
        for key in data:
            data[key] *= weights.get(key, 1.0)
        return ValueVector(**data)

    def adjust_for_supply(self, supply_factors: Dict[str, float]) -> "ValueVector":
        """Scale down based on abundance (higher factor = more common = less value)"""
        data = self.dict()
        for key in data:
            factor = max(supply_factors.get(key, 1.0), 1.0)
            data[key] /= factor
        return ValueVector(**data)


class LedgerEntry(BaseModel):
    id: str
    timestamp: float = Field(default_factory=time.time)
    intent_id: str
    memory_hash: Optional[str] = None  # Reference to encrypted memory in Memory Vault
    value_vector: ValueVector
    status: str = "active"  # active | frozen | revoked
    parent_id: Optional[str] = None  # For corrections/aggregations
    correction_notes: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("id", pre=True, always=True)
    def ensure_id(cls, v, values):
        if v:
            return v
        # Fallback deterministic ID
        data = f"{values.get('intent_id')}_{values.get('timestamp')}"
        return generate_entry_id(data)


class ValueLedger:
    """
    Main ledger class - append-only, local JSON storage.
    Designed for offline-first operation.
    """

    def __init__(self, storage_path: str | Path = "ledger.jsonl"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[LedgerEntry] = self._load_all()

    def _load_all(self) -> List[LedgerEntry]:
        if not self.storage_path.exists():
            return []
        entries = []
        with open(self.storage_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        entries.append(LedgerEntry(**data))
                    except Exception as e:
                        print(f"Warning: Failed to load ledger line: {e}")
        return entries

    def _append(self, entry: LedgerEntry):
        self.entries.append(entry)
        with open(self.storage_path, "a") as f:
            json.dump(entry.dict(), f)
            f.write("\n")

    def accrue_with_heuristics(
        self,
        ctx: ScoringContext,
        metadata: Optional[Dict] = None,
    ) -> str:
        """High-level API used by IntentLog / Agent-OS at intent completion"""
        from .heuristics import HeuristicEngine  # local import to avoid circular

        engine = HeuristicEngine()
        auto_vector = engine.score(ctx)

        # Allow human override to boost/nerf specific axes
        if ctx.user_override:
            for k, v in ctx.user_override.items():
                current = getattr(auto_vector, k)
                setattr(auto_vector, k, max(0.0, current + v))

        return self.accrue(
            intent_id=ctx.intent_id,
            initial_vector=auto_vector.dict(),
            memory_hash=ctx.memory_hash,
            metadata={
                "scoring_engine": "HeuristicEngine v1",
                "raw_duration_s": (ctx.end_time or time.time()) - ctx.start_time,
                "interruptions": ctx.interruptions,
                **(metadata or {}),
            },
        )
    def accrue(
        self,
        intent_id: str,
        initial_vector: Dict[str, float],
        memory_hash: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Accrue new value - primary entry point"""
        vector = ValueVector(**{k: v for k, v in initial_vector.items() if k in ValueVector.model_fields})
        entry = LedgerEntry(
            intent_id=intent_id,
            memory_hash=memory_hash,
            value_vector=vector,
            metadata=metadata or {},
        )
        self._append(entry)
        return entry.id

    def aggregate_correction(
        self,
        parent_id: str,
        new_vector: ValueVector,
        notes: Dict[str, Any],
        freeze_parent: bool = False,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Self-correct or refine an existing entry via aggregation"""
        parent = self.get_entry(parent_id)
        if not parent:
            raise ValueError(f"Parent entry {parent_id} not found")

        corrected_entry = LedgerEntry(
            intent_id=parent.intent_id,
            memory_hash=parent.memory_hash,
            value_vector=new_vector,
            parent_id=parent_id,
            correction_notes=notes,
            status="active",
            metadata=metadata or {},
        )
        self._append(corrected_entry)

        if freeze_parent:
            parent.status = "frozen"
            self._save_all()  # Overwrite to update status

        return corrected_entry.id

    def get_entry(self, entry_id: str) -> Optional[LedgerEntry]:
        return next((e for e in self.entries if e.id == entry_id), None)

    def get_chain(self, entry_id: str) -> List[LedgerEntry]:
        """Get full correction chain starting from an entry"""
        chain = []
        current = self.get_entry(entry_id)
        while current:
            chain.append(current)
            current = self.get_entry(current.parent_id) if current.parent_id else None
        return list(reversed(chain))  # Oldest first

    def current_value_for_intent(self, intent_id: str) -> ValueVector:
        """Aggregate latest active value for a given intent"""
        relevant = [e for e in self.entries if e.intent_id == intent_id and e.status == "active"]
        if not relevant:
            return ValueVector()

        # Simple latest-wins for now; can evolve to weighted merge
        latest = max(relevant, key=lambda e: e.timestamp)
        return latest.value_vector

    def _save_all(self):
        """Rewrite entire ledger (rarely used - only for status changes)"""
        with open(self.storage_path, "w") as f:
            for entry in self.entries:
                json.dump(entry.dict(), f)
                f.write("\n")
