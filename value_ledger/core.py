"""
value_ledger/core.py

Core implementation of the Value Ledger.
Handles accrual, aggregation, freezing, revocation, and self-correction.
"""

from __future__ import annotations

import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from .heuristics import ScoringContext

# Phase 2 - 17.5: Failure Mode Handling classes moved to failure_modes.py
# Re-exported here for backwards compatibility
from .failure_modes import ClockMonitor, SourceValidator, FailureModeHandler  # noqa: E402

logger = logging.getLogger(__name__)


# Placeholder for future imports from common utils
def generate_entry_id(data: str) -> str:
    """Deterministic ID generation (replace with common.utils if available)"""
    return hashlib.sha256(data.encode()).hexdigest()


def _validate_ledger_path(path: str | Path) -> Path:
    """
    Validate ledger storage path to prevent path traversal attacks.

    Args:
        path: The path to validate

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path is invalid or attempts traversal
    """
    path_str = str(path)

    # Check for null bytes
    if "\x00" in path_str:
        raise ValueError("Invalid characters in path")

    # Resolve to absolute path
    resolved = Path(path_str).resolve()

    # Block sensitive system paths
    sensitive_patterns = [
        "/etc/",
        "/proc/",
        "/sys/",
        "/dev/",
        "/.ssh/",
        "/.aws/",
        "/.config/",
        "/passwd",
        "/shadow",
        "/id_rsa",
    ]
    resolved_str = str(resolved).lower()
    for pattern in sensitive_patterns:
        if pattern in resolved_str:
            raise ValueError(f"Access to sensitive path not allowed: {pattern}")

    # Ensure it's a file path, not trying to write to root
    if str(resolved) == "/":
        raise ValueError("Cannot use root as storage path")

    logger.debug(f"Ledger path validated: {resolved}")
    return resolved


def compute_content_hash(content: Optional[str]) -> Optional[str]:
    """Compute SHA-256 hash of content for proof system."""
    if not content:
        return None
    return hashlib.sha256(content.encode()).hexdigest()


def compute_timestamp_proof(timestamp: float, entry_id: str) -> str:
    """
    Generate a timestamp proof.
    TODO: Anchor to external timestamping service (RFC 3161) for production.
    For now, creates a local signed proof.
    """
    proof_data = f"{timestamp}:{entry_id}"
    return hashlib.sha256(proof_data.encode()).hexdigest()


class MerkleTree:
    """
    Simple Merkle tree for ledger proof generation.

    This implementation follows standard Merkle tree construction where:
    - Leaves are hashed using SHA-256
    - Internal nodes are computed by concatenating and hashing child pairs
    - The root provides a cryptographic commitment to all leaves

    Edge Case: Odd-Length Levels
    ----------------------------
    When a level has an odd number of nodes, the last node is paired with itself
    (duplicated) to create a complete binary tree. This is standard practice but
    has implications for proof verification:

    - For a tree with 3 leaves [A, B, C], the first level becomes [(A,B), (C,C)]
    - If C is at index 2 (even), the proof includes C as the "right" sibling
    - This self-pairing still produces a valid proof because:
        hash(C + C) = hash(left + right) where left == right

    This behavior is intentional and maintains proof validity. Third-party
    verifiers should be aware that a proof step where left == right indicates
    the original leaf was at an odd-indexed position at a level's end.
    """

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
        """
        Get Merkle proof for a leaf at given index.

        Args:
            leaf_index: Zero-based index of the leaf to get proof for

        Returns:
            List of proof steps, each with 'position' ('left' or 'right') and 'hash'.
            Empty list if index is out of bounds.

        Note:
            When the target leaf is at the end of an odd-length level, the proof
            will include a step where the sibling hash equals the leaf's own hash
            (self-pairing). This is correct behavior - see class docstring.
        """
        if leaf_index < 0 or leaf_index >= len(self.leaves):
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
    """The seven value units: T/E/N/F/R/S/U"""

    t: float = Field(default=0.0, ge=0.0, description="Time spent")
    e: float = Field(default=0.0, ge=0.0, description="Effort (interruption-adjusted)")
    n: float = Field(default=0.0, ge=0.0, description="Novelty")
    f: float = Field(default=0.0, ge=0.0, description="Value from failure/learning")
    r: float = Field(default=0.0, ge=0.0, description="Risk exposure")
    s: float = Field(default=0.0, ge=0.0, description="Strategic insight")
    u: float = Field(default=0.0, ge=0.0, description="Reusability (cross-domain applicability)")

    def total(self) -> float:
        return sum(self.model_dump().values())

    def apply_subjective_weights(self, weights: Dict[str, float]) -> "ValueVector":
        data = self.model_dump()
        for key in data:
            data[key] *= weights.get(key, 1.0)
        return ValueVector(**data)

    def adjust_for_supply(self, supply_factors: Dict[str, float]) -> "ValueVector":
        """Scale down based on abundance (higher factor = more common = less value)"""
        data = self.model_dump()
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
    id: Optional[str] = Field(default=None, max_length=128)
    timestamp: float = Field(default_factory=time.time)
    intent_id: str = Field(..., min_length=1, max_length=256)
    memory_hash: Optional[str] = Field(default=None, max_length=128)
    value_vector: ValueVector
    status: str = Field(default="active", pattern=r"^(active|frozen|revoked)$")
    parent_id: Optional[str] = Field(default=None, max_length=128)
    correction_notes: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Proof fields (Phase 1 - 17.1)
    proof: ProofData = Field(default_factory=ProofData)

    # Revocation fields (Phase 1 - 17.9)
    revoked_at: Optional[float] = None
    revoked_by: Optional[str] = Field(default=None, max_length=256)
    revocation_reason: Optional[str] = Field(default=None, max_length=1024)

    # Owner & Classification fields (Phase 2 - 17.3)
    owner: Optional[str] = Field(default=None, max_length=256)
    classification: int = Field(default=0, ge=0, le=5)  # 0=public, 5=most restricted
    contract_id: Optional[str] = Field(default=None, max_length=256)

    # Multi-parent aggregation fields (Phase 2 - 17.4)
    parent_ids: List[str] = Field(default_factory=list)  # Multiple parent entries
    aggregation_rule: Optional[str] = Field(default=None, pattern=r"^(sum|max|weighted)$")

    @model_validator(mode="after")
    def ensure_id(self):
        if not self.id:
            # Fallback deterministic ID generation
            data = f"{self.intent_id}_{self.timestamp}"
            self.id = generate_entry_id(data)
        return self

    @model_validator(mode="after")
    def sync_parent_fields(self):
        """Ensure parent_id and parent_ids are in sync."""
        if self.parent_id and self.parent_id not in self.parent_ids:
            self.parent_ids = [self.parent_id] + self.parent_ids
        if self.parent_ids and not self.parent_id:
            self.parent_id = self.parent_ids[0] if self.parent_ids else None
        return self


class ValueLedger:
    """
    Main ledger class - append-only, local JSON storage.
    Designed for offline-first operation.

    Security:
    - Storage paths are validated to prevent path traversal attacks
    """

    def __init__(
        self,
        storage_path: str | Path = "ledger.jsonl",
        max_entries: Optional[int] = None,
    ):
        # Validate storage path to prevent path traversal
        self.storage_path = _validate_ledger_path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entries = max_entries
        self.entries: List[LedgerEntry] = self._load_all()
        self.merkle_tree = MerkleTree()
        self._rebuild_merkle_tree()

    def _load_all(self) -> List[LedgerEntry]:
        if not self.storage_path.exists():
            return []
        entries = []
        error_count = 0
        line_num = 0
        with open(self.storage_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        entries.append(LedgerEntry(**data))
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        error_count += 1
                        logger.warning(f"Failed to load ledger line {line_num}: {e}")
        if error_count > 0:
            logger.warning(f"Skipped {error_count} malformed entries out of {line_num}")
        if self._max_entries and len(entries) > self._max_entries:
            logger.warning(
                f"Ledger has {len(entries)} entries, capping to {self._max_entries} most recent"
            )
            entries = sorted(entries, key=lambda e: e.timestamp)[-self._max_entries:]
        return entries

    @property
    def entry_count(self) -> int:
        """Total entries on disk including those not loaded."""
        if not self.storage_path.exists():
            return 0
        with open(self.storage_path, "r") as f:
            return sum(1 for line in f if line.strip())

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
            json.dump(entry.model_dump(), f)
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
            initial_vector=auto_vector.model_dump(),
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
        owner: Optional[str] = None,
        classification: int = 0,
        contract_id: Optional[str] = None,
    ) -> str:
        """Accrue new value - primary entry point"""
        if metadata:
            metadata_str = json.dumps(metadata)
            if len(metadata_str) > 65536:  # 64KB limit
                raise ValueError(f"Metadata too large: {len(metadata_str)} bytes (max 65536)")

        vector = ValueVector(
            **{k: v for k, v in initial_vector.items() if k in ValueVector.model_fields}
        )

        # Validate classification (Phase 2 - 17.3)
        if classification < 0 or classification > 5:
            raise ValueError(f"Classification must be 0-5, got {classification}")

        # Inject correlation ID if available
        entry_metadata = metadata.copy() if metadata else {}
        try:
            from .security import get_correlation_id, _correlation_id
            cid = _correlation_id.get()
            if cid is not None:
                entry_metadata["correlation_id"] = cid
        except ImportError:
            pass

        # Create entry first to get ID
        entry = LedgerEntry(
            intent_id=intent_id,
            memory_hash=memory_hash,
            value_vector=vector,
            metadata=entry_metadata,
            owner=owner,
            classification=classification,
            contract_id=contract_id,
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
                json.dump(entry.model_dump(), f)
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
            "value_vector": entry.value_vector.model_dump(),
            "proof": {
                "content_hash": entry.proof.content_hash,
                "timestamp_proof": entry.proof.timestamp_proof,
                "merkle_ref": entry.proof.merkle_ref,
                "merkle_proof": merkle_proof,
            },
            "revocation": (
                {
                    "revoked": entry.status == "revoked",
                    "revoked_at": entry.revoked_at,
                    "revoked_by": entry.revoked_by,
                    "reason": entry.revocation_reason,
                }
                if entry.status == "revoked"
                else None
            ),
        }

    # === Phase 2 - 17.3: Owner & Classification Access Control ===

    def get_entries_for_owner(
        self,
        owner: str,
        include_unowned: bool = False,
    ) -> List[LedgerEntry]:
        """
        Get all entries owned by a specific owner.

        Args:
            owner: Owner identifier to filter by
            include_unowned: If True, also include entries with no owner set

        Returns:
            List of matching entries
        """
        result = []
        for entry in self.entries:
            if entry.owner == owner:
                result.append(entry)
            elif include_unowned and entry.owner is None:
                result.append(entry)
        return result

    def filter_by_classification(
        self,
        max_classification: int,
        owner: Optional[str] = None,
    ) -> List[LedgerEntry]:
        """
        Get entries up to a maximum classification level.

        Args:
            max_classification: Maximum classification level (0-5) to include
            owner: Optional owner filter

        Returns:
            List of entries at or below the specified classification level
        """
        result = []
        for entry in self.entries:
            if entry.classification <= max_classification:
                if owner is None or entry.owner == owner:
                    result.append(entry)
        return result

    def check_access(
        self,
        entry_id: str,
        requester_owner: Optional[str] = None,
        requester_clearance: int = 0,
    ) -> bool:
        """
        Check if a requester has access to an entry.

        Rules:
        - Owner always has access to their own entries
        - Others need sufficient clearance level
        - Revoked entries still accessible for audit (read-only)

        Args:
            entry_id: Entry to check access for
            requester_owner: Owner ID of the requester
            requester_clearance: Clearance level of requester (0-5)

        Returns:
            True if access is allowed
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False

        # Owner always has access
        if entry.owner == requester_owner and requester_owner is not None:
            return True

        # Check classification level
        return requester_clearance >= entry.classification

    def transfer_ownership(
        self,
        entry_id: str,
        new_owner: str,
        current_owner: Optional[str] = None,
    ) -> bool:
        """
        Transfer ownership of an entry.

        Args:
            entry_id: Entry to transfer
            new_owner: New owner ID
            current_owner: Current owner (for verification)

        Returns:
            True if transfer succeeded
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False

        # Verify current owner if specified
        if current_owner is not None and entry.owner != current_owner:
            return False

        # Cannot transfer revoked entries
        if entry.status == "revoked":
            return False

        entry.owner = new_owner
        self._save_all()
        return True

    def update_classification(
        self,
        entry_id: str,
        new_classification: int,
        requester_owner: Optional[str] = None,
    ) -> bool:
        """
        Update classification level of an entry.

        Args:
            entry_id: Entry to update
            new_classification: New classification level (0-5)
            requester_owner: Owner requesting the change

        Returns:
            True if update succeeded
        """
        if new_classification < 0 or new_classification > 5:
            return False

        entry = self.get_entry(entry_id)
        if not entry:
            return False

        # Only owner can change classification
        if requester_owner is not None and entry.owner != requester_owner:
            return False

        entry.classification = new_classification
        self._save_all()
        return True

    # === Phase 2 - 17.4: Multi-Parent Aggregation ===

    def aggregate_entries(
        self,
        entry_ids: List[str],
        rule: str = "sum",
        weights: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict] = None,
        owner: Optional[str] = None,
        classification: Optional[int] = None,
    ) -> str:
        """
        Aggregate multiple entries into a single new entry.

        Args:
            entry_ids: List of entry IDs to aggregate
            rule: Aggregation rule - "sum", "max", or "weighted"
            weights: Per-entry weights for "weighted" rule {entry_id: weight}
            metadata: Additional metadata for the aggregated entry
            owner: Owner for the new entry (inherits from first parent if not set)
            classification: Classification level (inherits max from parents if not set)

        Returns:
            ID of the new aggregated entry

        Raises:
            ValueError: If entries not found or invalid rule
        """
        if not entry_ids:
            raise ValueError("Must provide at least one entry to aggregate")

        if rule not in ("sum", "max", "weighted"):
            raise ValueError(f"Invalid aggregation rule: {rule}. Use 'sum', 'max', or 'weighted'")

        # Gather parent entries
        parents = []
        for eid in entry_ids:
            entry = self.get_entry(eid)
            if not entry:
                raise ValueError(f"Entry not found: {eid}")
            if entry.status == "revoked":
                raise ValueError(f"Cannot aggregate revoked entry: {eid}")
            parents.append(entry)

        # Compute aggregated vector based on rule
        if rule == "sum":
            aggregated = self._aggregate_sum(parents)
        elif rule == "max":
            aggregated = self._aggregate_max(parents)
        elif rule == "weighted":
            if not weights:
                weights = {e.id: 1.0 for e in parents}
            aggregated = self._aggregate_weighted(parents, weights)

        # Determine inherited properties
        inherited_owner = owner or parents[0].owner
        inherited_classification = classification
        if inherited_classification is None:
            inherited_classification = max(p.classification for p in parents)

        # Use intent_id from first parent (or create combined)
        intent_ids = list(set(p.intent_id for p in parents))
        combined_intent = (
            intent_ids[0] if len(intent_ids) == 1 else f"aggregated:{','.join(intent_ids[:3])}"
        )

        # Create aggregated entry
        entry = LedgerEntry(
            intent_id=combined_intent,
            value_vector=aggregated,
            parent_ids=entry_ids,
            aggregation_rule=rule,
            status="active",
            owner=inherited_owner,
            classification=inherited_classification,
            metadata={
                "aggregation_source": "multi_parent",
                "parent_count": len(parents),
                "rule": rule,
                **(metadata or {}),
            },
        )

        self._append(entry)
        return entry.id

    def _aggregate_sum(self, entries: List[LedgerEntry]) -> ValueVector:
        """Sum all value vectors."""
        result = {"t": 0.0, "e": 0.0, "n": 0.0, "f": 0.0, "r": 0.0, "s": 0.0, "u": 0.0}
        for entry in entries:
            vec = entry.value_vector.model_dump()
            for k in result:
                result[k] += vec.get(k, 0.0)
        return ValueVector(**result)

    def _aggregate_max(self, entries: List[LedgerEntry]) -> ValueVector:
        """Take maximum value for each dimension."""
        result = {"t": 0.0, "e": 0.0, "n": 0.0, "f": 0.0, "r": 0.0, "s": 0.0, "u": 0.0}
        for entry in entries:
            vec = entry.value_vector.model_dump()
            for k in result:
                result[k] = max(result[k], vec.get(k, 0.0))
        return ValueVector(**result)

    def _aggregate_weighted(
        self,
        entries: List[LedgerEntry],
        weights: Dict[str, float],
    ) -> ValueVector:
        """Weighted sum of value vectors."""
        result = {"t": 0.0, "e": 0.0, "n": 0.0, "f": 0.0, "r": 0.0, "s": 0.0, "u": 0.0}
        total_weight = sum(weights.get(e.id, 1.0) for e in entries)

        for entry in entries:
            weight = weights.get(entry.id, 1.0) / total_weight
            vec = entry.value_vector.model_dump()
            for k in result:
                result[k] += vec.get(k, 0.0) * weight
        return ValueVector(**result)


# === Phase 2 - 17.5: Failure Mode Handling ===
# Classes exported at module level for backwards compatibility (imported at top of file)

__all__ = [
    "ClockMonitor",
    "SourceValidator",
    "FailureModeHandler",
]
