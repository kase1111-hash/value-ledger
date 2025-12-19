# value_ledger/memory_vault_hook.py
"""
Secure integration with Memory Vault for novelty scoring and self-correction.
Only accesses content if explicitly allowed by Learning Contracts.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time

# Placeholder imports — will be replaced with real ones when Memory Vault exists
# from memory_vault.core import MemoryVault, MemoryEntry
# from learning_contracts.engine import ConsentChecker

# Stub types for now (remove when real modules are available)
@dataclass
class MemoryEntryStub:
    hash: str
    timestamp: float
    content: Optional[str] = None  # None if not consented
    metadata: Dict[str, Any] = None

class MemoryVaultStub:
    def __init__(self, vault_path: str = "memory_vault/"):
        """Stub accepts vault_path for API compatibility with real MemoryVault."""
        self.vault_path = vault_path

    def query_similar(
        self,
        query_content: str,
        limit: int = 50,
        min_timestamp: Optional[float] = None,
    ) -> List[MemoryEntryStub]:
        # In real impl: vector similarity search
        # For now: return empty (novelty defaults high)
        return []

    def get_recent(self, limit: int = 50) -> List[MemoryEntryStub]:
        return []

# Temporary stubs — replace with real imports
MemoryVault = MemoryVaultStub
ConsentChecker = None  # Will have .check_access(intent_id, "novelty_scoring") -> bool


class MemoryVaultHook:
    """
    Secure bridge between ValueLedger and Memory Vault.
    Handles consent, content access, and similarity context.
    """

    def __init__(self, vault_path: str = "memory_vault/"):
        self.vault = MemoryVault(vault_path)
        self.consent_checker = ConsentChecker() if ConsentChecker else None

    def can_access_content(self, intent_id: str) -> bool:
        """Check Learning Contracts for permission to read raw memory content"""
        if self.consent_checker is None:
            return False  # Default deny until implemented
        return self.consent_checker.check_access(intent_id, purpose="novelty_scoring")

    def get_novelty_context(
        self,
        current_content: Optional[str],
        intent_id: str,
        memory_hash: Optional[str] = None,
        lookback_days: int = 90,
    ) -> List[Tuple[str, float, Optional[str]]]:
        """
        Returns list of (hash, timestamp, content) for previous memories.
        Content is None if no consent.
        Prioritizes recent + similar.
        """
        if not current_content or not self.can_access_content(intent_id):
            # No content access → can't compute novelty accurately
            return []

        min_ts = time.time() - (lookback_days * 86400)

        # First: try similarity search if content available
        similar = self.vault.query_similar(
            query_content=current_content,
            limit=30,
            min_timestamp=min_ts,
        )

        # Fallback/add: recent memories
        recent = self.vault.get_recent(limit=50)

        # Dedupe and merge, preferring similar ones first
        seen_hashes = set()
        context: List[Tuple[str, float, Optional[str]]] = []

        for entry in similar + recent:
            if entry.hash in seen_hashes:
                continue
            seen_hashes.add(entry.hash)
            # Only include content if consented (should be true here, but double-check)
            content = entry.content if self.can_access_content(intent_id) else None
            context.append((entry.hash, entry.timestamp, content))
            if len(context) >= 50:
                break

        return context

    def trigger_novelty_reassessment(self, ledger: "ValueLedger", intent_id: str):
        """
        Called by Memory Vault or Boundary Daemon when a new memory
        might reduce novelty of past entries.
        """
        # Find all entries for this intent_id
        relevant_entries = [
            e for e in ledger.entries
            if e.intent_id == intent_id and e.status == "active"
        ]
        if not relevant_entries:
            return

        # Get latest memory content (assume latest entry has it via metadata or re-query)
        latest = max(relevant_entries, key=lambda e: e.timestamp)
        current_content = latest.metadata.get("raw_content_for_novelty")

        if not current_content:
            return

        new_context = self.get_novelty_context(
            current_content=current_content,
            intent_id=intent_id,
        )

        # Re-run novelty scorer
        from .heuristics import NoveltyScorer, ScoringContext

        dummy_ctx = ScoringContext(
            intent_id=intent_id,
            start_time=0,
            end_time=0,
            memory_content=current_content,
            previous_memories=new_context,
        )
        new_novelty_vec = NoveltyScorer()(dummy_ctx)
        new_n = new_novelty_vec.n

        # If significantly changed, create correction entry
        for entry in relevant_entries:
            old_n = entry.value_vector.n
            if abs(old_n - new_n) > 1.0:  # threshold
                corrected_vector = entry.value_vector
                corrected_vector.n = new_n

                ledger.aggregate_correction(
                    parent_id=entry.id,
                    new_vector=corrected_vector,
                    notes={
                        "type": "novelty_reassessment",
                        "old_n": old_n,
                        "new_n": new_n,
                        "trigger": "new_memory_added",
                        "similar_memories": len(new_context),
                    },
                    freeze_parent=False,
                )
                print(f"[ValueLedger] Novelty corrected for {entry.id[:8]}...: {old_n:.1f} → {new_n:.1f}")
