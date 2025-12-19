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
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pydantic import BaseModel, Field, field_validator, model_validator
import os

if TYPE_CHECKING:
    from .heuristics import ScoringContext

# Placeholder for future imports from common utils
def generate_entry_id(data: str) -> str:
    """Deterministic ID generation (replace with common.utils if available)"""
    return hashlib.sha256(data.encode()).hexdigest()


def compute_content_hash(content: Optional[str]) -> Optional[str]:
    """Compute SHA-256 hash of content for proof system."""
    if not content:
        return None
    return hashlib.sha256(content.encode()).hexdigest()


def compute_timestamp_proof(timestamp: float, entry_id: str) -> str:
    """
    Generate a timestamp proof.
    In production, this could anchor to an external timestamping service.
    For now, creates a local signed proof.
    """
    proof_data = f"{timestamp}:{entry_id}"
    return hashlib.sha256(proof_data.encode()).hexdigest()


class MerkleTree:
    """Simple Merkle tree for ledger proof generation."""

    def __init__(self):
        self.leaves: List[str] = []
        self._root: Optional[str] = None

    def add_leaf(self, data: str) -> int:
        """Add a leaf and return its index."""
        leaf_hash = hashlib.sha256(data.encode()).hexdigest()
        self.leaves.append(leaf_hash)
        self._root = None  # Invalidate cached root
        return len(self.leaves) - 1

    def get_root(self) -> Optional[str]:
        """Compute and return Merkle root."""
        if not self.leaves:
            return None
        if self._root:
            return self._root

        # Build tree bottom-up
        current_level = self.leaves[:]
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha256((left + right).encode()).hexdigest()
                next_level.append(combined)
            current_level = next_level

        self._root = current_level[0]
        return self._root

    def get_proof(self, leaf_index: int) -> List[Dict[str, str]]:
        """Get Merkle proof for a leaf at given index."""
        if leaf_index >= len(self.leaves):
            return []

        proof = []
        current_level = self.leaves[:]
        idx = leaf_index

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left

                if i == idx or i + 1 == idx:
                    # This pair contains our target
                    if idx % 2 == 0:
                        proof.append({"position": "right", "hash": right})
                    else:
                        proof.append({"position": "left", "hash": left})

                combined = hashlib.sha256((left + right).encode()).hexdigest()
                next_level.append(combined)

            current_level = next_level
            idx = idx // 2

        return proof

    def verify_proof(self, leaf_hash: str, proof: List[Dict[str, str]], root: str) -> bool:
        """Verify a Merkle proof."""
        current = leaf_hash
        for step in proof:
            if step["position"] == "left":
                current = hashlib.sha256((step["hash"] + current).encode()).hexdigest()
            else:
                current = hashlib.sha256((current + step["hash"]).encode()).hexdigest()
        return current == root


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


class ProofData(BaseModel):
    """Cryptographic proof fields for third-party verification (MP-02 compatible)."""
    content_hash: Optional[str] = None  # SHA-256 hash of source content
    timestamp_proof: Optional[str] = None  # Signed timestamp or anchor reference
    merkle_ref: Optional[str] = None  # Reference to Merkle tree position


class LedgerEntry(BaseModel):
    id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    intent_id: str
    memory_hash: Optional[str] = None  # Reference to encrypted memory in Memory Vault
    value_vector: ValueVector
    status: str = "active"  # active | frozen | revoked
    parent_id: Optional[str] = None  # For corrections/aggregations
    correction_notes: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Proof fields (Phase 1 - 17.1)
    proof: ProofData = Field(default_factory=ProofData)

    # Revocation fields (Phase 1 - 17.9)
    revoked_at: Optional[float] = None
    revoked_by: Optional[str] = None
    revocation_reason: Optional[str] = None

    @model_validator(mode='after')
    def ensure_id(self):
        if not self.id:
            # Fallback deterministic ID generation
            data = f"{self.intent_id}_{self.timestamp}"
            self.id = generate_entry_id(data)
        return self


class ValueLedger:
    """
    Main ledger class - append-only, local JSON storage.
    Designed for offline-first operation.
    """

    def __init__(self, storage_path: str | Path = "ledger.jsonl"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[LedgerEntry] = self._load_all()
        self.merkle_tree = MerkleTree()
        self._rebuild_merkle_tree()

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

    def _rebuild_merkle_tree(self):
        """Rebuild Merkle tree from existing entries."""
        self.merkle_tree = MerkleTree()
        for entry in self.entries:
            self.merkle_tree.add_leaf(entry.id)

    def _append(self, entry: LedgerEntry):
        # Add to Merkle tree and set merkle_ref
        leaf_index = self.merkle_tree.add_leaf(entry.id)
        entry.proof.merkle_ref = f"leaf:{leaf_index}"

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
            content_for_proof=ctx.memory_content,  # Use memory content for proof hash
        )
    def accrue(
        self,
        intent_id: str,
        initial_vector: Dict[str, float],
        memory_hash: Optional[str] = None,
        metadata: Optional[Dict] = None,
        content_for_proof: Optional[str] = None,
    ) -> str:
        """Accrue new value - primary entry point"""
        vector = ValueVector(**{k: v for k, v in initial_vector.items() if k in ValueVector.model_fields})

        # Create entry first to get ID
        entry = LedgerEntry(
            intent_id=intent_id,
            memory_hash=memory_hash,
            value_vector=vector,
            metadata=metadata or {},
        )

        # Compute proof fields (Phase 1 - 17.1)
        entry.proof.content_hash = compute_content_hash(content_for_proof)
        entry.proof.timestamp_proof = compute_timestamp_proof(entry.timestamp, entry.id)

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

    # === Phase 1 - 17.9: Explicit Revocation ===

    def revoke(
        self,
        entry_id: str,
        reason: Optional[str] = None,
        revoked_by: Optional[str] = None,
        revoke_children: bool = False,
    ) -> bool:
        """
        Revoke an entry. Value remains provable but non-exploitable.

        Args:
            entry_id: Entry to revoke
            reason: Optional reason for audit trail
            revoked_by: Optional identifier of who revoked
            revoke_children: Also revoke derived entries

        Returns:
            True if revocation succeeded, False if entry not found
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False

        if entry.status == "revoked":
            return True  # Already revoked

        # Update entry status
        entry.status = "revoked"
        entry.revoked_at = time.time()
        entry.revoked_by = revoked_by
        entry.revocation_reason = reason

        # Optionally revoke children
        if revoke_children:
            children = [e for e in self.entries if e.parent_id == entry_id]
            for child in children:
                self.revoke(
                    child.id,
                    reason=f"Parent {entry_id[:8]} revoked: {reason}",
                    revoked_by=revoked_by,
                    revoke_children=True,
                )

        self._save_all()
        return True

    # === Phase 1 - 17.1: Proof System Methods ===

    def get_merkle_root(self) -> Optional[str]:
        """Get current Merkle root for the ledger."""
        return self.merkle_tree.get_root()

    def get_merkle_proof(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get Merkle proof for an entry.

        Returns:
            Dict with 'leaf_hash', 'proof', and 'root' for verification
        """
        entry = self.get_entry(entry_id)
        if not entry or not entry.proof.merkle_ref:
            return None

        # Parse leaf index from merkle_ref
        try:
            leaf_index = int(entry.proof.merkle_ref.split(":")[1])
        except (IndexError, ValueError):
            return None

        leaf_hash = hashlib.sha256(entry_id.encode()).hexdigest()
        proof = self.merkle_tree.get_proof(leaf_index)
        root = self.merkle_tree.get_root()

        return {
            "entry_id": entry_id,
            "leaf_hash": leaf_hash,
            "proof": proof,
            "root": root,
            "verified": self.merkle_tree.verify_proof(leaf_hash, proof, root) if root else False,
        }

    def verify_entry_proof(self, entry_id: str) -> bool:
        """Verify an entry's Merkle proof is valid."""
        proof_data = self.get_merkle_proof(entry_id)
        return proof_data.get("verified", False) if proof_data else False

    def export_existence_proof(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Export a complete existence proof for third-party verification.
        Does NOT include content - only proves the entry existed.
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return None

        merkle_proof = self.get_merkle_proof(entry_id)

        return {
            "entry_id": entry_id,
            "timestamp": entry.timestamp,
            "intent_id": entry.intent_id,
            "status": entry.status,
            "value_vector": entry.value_vector.dict(),
            "proof": {
                "content_hash": entry.proof.content_hash,
                "timestamp_proof": entry.proof.timestamp_proof,
                "merkle_ref": entry.proof.merkle_ref,
                "merkle_proof": merkle_proof,
            },
            "revocation": {
                "revoked": entry.status == "revoked",
                "revoked_at": entry.revoked_at,
                "revoked_by": entry.revoked_by,
                "reason": entry.revocation_reason,
            } if entry.status == "revoked" else None,
        }
