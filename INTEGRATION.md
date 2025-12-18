# Cross-Repository Integration Guide

This document specifies how the Value Ledger integrates with the other 8 modules in the Agent-OS ecosystem.

## Module Ecosystem Overview

The Agent-OS ecosystem consists of 9 modules:

1. **Agent-OS** – Core natural-language-native OS for AI agents
2. **synth-mind** – Synthetic mind architecture with cognitive tiers
3. **IntentLog** – Captures human reasoning and intent behind actions
4. **memory-vault** – Encrypted storage (planned)
5. **learning-contracts** – Consent and boundary engine (planned)
6. **boundary-daemon** – System monitoring and safety daemon
7. **NatLangChain** – Natural language-native ledger concepts
8. **common** – Shared utilities
9. **value-ledger** (this module)

---

## 1. IntentLog Integration

### Status: ✅ IMPLEMENTED (with stubs)

### Purpose
Value Ledger listens to IntentLog events to automatically accrue value when cognitive work completes.

### Integration Points

#### Required IntentLog Event Structure

```python
@dataclass
class IntentEvent:
    event_type: str  # "intent_started", "intent_updated", "intent_completed", "intent_abandoned"
    intent_id: str
    timestamp: float
    human_reasoning: Optional[str] = None
    agent_output: Optional[str] = None
    memory_hash: Optional[str] = None
    interruptions: int = 0
    keystrokes: Optional[int] = None
    outcome_tags: Optional[list[str]] = None
    risk_level: Optional[float] = None  # 0.0–1.0
    metadata: Optional[Dict[str, Any]] = None
```

#### Event Types

- `intent_started` - Triggers start time tracking
- `intent_completed` - Triggers value accrual
- `intent_abandoned` - Also triggers value accrual (failures have value!)
- `intent_updated` - Reserved for future partial accruals

#### Required Fields

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `event_type` | str | ✅ | Determines handler behavior |
| `intent_id` | str | ✅ | Links to ledger entries |
| `timestamp` | float | ✅ | Unix timestamp for duration tracking |
| `human_reasoning` | str | ⚠️ Recommended | Used for novelty scoring |
| `agent_output` | str | ⚠️ Recommended | Used for novelty scoring |
| `memory_hash` | str | ⚠️ Recommended | Links to Memory Vault |
| `interruptions` | int | ⚠️ Recommended | Effort scoring multiplier |
| `keystrokes` | int | ⚠️ Recommended | Effort density calculation |
| `outcome_tags` | list[str] | ⚠️ Recommended | Failure/success scoring |
| `risk_level` | float | Optional | Risk value scoring |

#### Usage Example

```python
from value_ledger import create_intentlog_listener

# In IntentLog or Agent-OS core:
ledger_listener = create_intentlog_listener(ledger_path="ledger.jsonl")

# When an intent event occurs:
ledger_listener.handle_event(intent_event)
```

#### Expected Behavior

- IntentLog should emit events via a message bus or direct calls
- Value Ledger maintains no state about intent lifecycle (that's IntentLog's job)
- Value Ledger only tracks `start_time` temporarily in memory for duration calculation

### Compatibility Requirements

✅ **IntentLog MUST:**
- Emit events in the `IntentEvent` structure
- Include unique `intent_id` for each intent
- Provide accurate timestamps

⚠️ **IntentLog SHOULD:**
- Include `human_reasoning` and `agent_output` for accurate novelty scoring
- Track `interruptions` count (or get from Boundary Daemon)
- Tag outcomes as `["success"]`, `["failure"]`, `["breakthrough"]`, etc.

❌ **IntentLog MUST NOT:**
- Delete intents before Value Ledger processes them
- Modify timestamps after emission

---

## 2. Memory Vault Integration

### Status: ⚠️ STUBBED (awaiting Memory Vault implementation)

### Purpose
Memory Vault provides encrypted memory storage and similarity search for novelty scoring.

### Integration Points

#### Required Memory Vault API

```python
class MemoryVault:
    def query_similar(
        self,
        query_content: str,
        limit: int = 50,
        min_timestamp: Optional[float] = None,
    ) -> List[MemoryEntry]:
        """Return similar memories for novelty comparison"""
        pass

    def get_recent(self, limit: int = 50) -> List[MemoryEntry]:
        """Return recent memories (fallback for similarity)"""
        pass


@dataclass
class MemoryEntry:
    hash: str  # SHA-256 hash of encrypted content
    timestamp: float
    content: Optional[str] = None  # Decrypted content (if permitted by Learning Contracts)
    metadata: Dict[str, Any] = None
```

#### Current Implementation

File: `value_ledger/memory_vault_hook.py`

Uses stub implementations that return empty lists. This causes novelty scores to default HIGH (8.5-9.0) until real Memory Vault is available.

#### Novelty Scoring Process

1. **Without Memory Vault**: Novelty defaults to 8.5 (safe high estimate)
2. **With Memory Vault**:
   - Query similar memories using semantic embeddings
   - Compare current content against top-k most similar memories
   - Calculate novelty score: `10.0 * (1 - similarity^1.2)`

#### Usage Example

```python
from value_ledger.memory_vault_hook import MemoryVaultHook

mv_hook = MemoryVaultHook(vault_path="memory_vault/")

# Get context for novelty scoring
novelty_context = mv_hook.get_novelty_context(
    current_content=intent_content,
    intent_id=intent_id,
    memory_hash=memory_hash,
    lookback_days=90,
)
```

### Compatibility Requirements

✅ **Memory Vault MUST:**
- Provide `query_similar()` method with semantic similarity search
- Return `MemoryEntry` objects with hash and timestamp
- Store encrypted content with SHA-256 hashes

⚠️ **Memory Vault SHOULD:**
- Use embeddings for similarity (not just keyword matching)
- Support temporal filtering (`min_timestamp`)
- Provide decrypted `content` field when permitted by Learning Contracts

❌ **Memory Vault MUST NOT:**
- Return plaintext content without consent verification
- Modify memory hashes after creation

### Migration Path

When Memory Vault is implemented:

1. Replace `MemoryVaultStub` in `memory_vault_hook.py`:
   ```python
   from memory_vault.core import MemoryVault, MemoryEntry
   ```

2. Test novelty scoring with real data
3. Optional: Implement novelty reassessment triggers when new memories are added

---

## 3. Learning Contracts Integration

### Status: ⚠️ STUBBED (awaiting Learning Contracts implementation)

### Purpose
Learning Contracts control access permissions for reading memory content in novelty scoring.

### Integration Points

#### Required Learning Contracts API

```python
class ConsentChecker:
    def check_access(
        self,
        intent_id: str,
        purpose: str,  # e.g., "novelty_scoring"
    ) -> bool:
        """Check if access is permitted for this intent and purpose"""
        pass
```

#### Current Implementation

File: `value_ledger/memory_vault_hook.py`

Currently defaults to **DENY** (returns `False`) until Learning Contracts is implemented.

#### Access Control Flow

```python
def can_access_content(self, intent_id: str) -> bool:
    if self.consent_checker is None:
        return False  # Default deny
    return self.consent_checker.check_access(intent_id, purpose="novelty_scoring")
```

If access is denied:
- Memory content is not passed to novelty scorer
- Novelty defaults to high value (8.5)
- `raw_content_for_novelty` in metadata is set to `None`

### Compatibility Requirements

✅ **Learning Contracts MUST:**
- Provide `check_access(intent_id, purpose)` method
- Return boolean permission status
- Handle unknown intent_ids gracefully (return `False`)

⚠️ **Learning Contracts SHOULD:**
- Support `purpose="novelty_scoring"` as a valid access reason
- Allow per-intent access configuration
- Respect user privacy preferences

❌ **Learning Contracts MUST NOT:**
- Hang or block indefinitely on permission checks
- Leak information about consent status in error messages

### Migration Path

When Learning Contracts is implemented:

1. Replace stub in `memory_vault_hook.py`:
   ```python
   from learning_contracts.engine import ConsentChecker
   ```

2. Update default behavior from deny-all to contract-based
3. Test with various consent scenarios

---

## 4. Boundary Daemon Integration

### Status: ⚠️ PASSIVE (expects data from IntentLog)

### Purpose
Boundary Daemon tracks interruptions and context switches that affect effort scoring.

### Integration Points

#### Required Data

Value Ledger expects the `interruptions` count to be included in `IntentEvent` from IntentLog.

```python
IntentEvent(
    event_type="intent_completed",
    intent_id="...",
    interruptions=7,  # ← From Boundary Daemon
    # ...
)
```

#### Interruption Scoring

Interruptions multiply effort value:

```python
interruption_factor = 1.0 + (interruptions * 0.35)
if interruptions > 10:
    interruption_factor += (interruptions - 10) * 0.1
```

### Compatibility Requirements

✅ **Boundary Daemon MUST:**
- Track interruption count per intent session
- Report count to IntentLog for inclusion in events

⚠️ **Boundary Daemon SHOULD:**
- Define "interruption" consistently (context switch, notification, etc.)
- Reset count on new intent start
- Distinguish between external and self-interruptions

❌ **Boundary Daemon MUST NOT:**
- Report negative interruption counts
- Count normal pauses as interruptions

---

## 5. Agent-OS Core Integration

### Status: ✅ READY (provides factory function)

### Purpose
Agent-OS core orchestrates all modules and manages the event bus.

### Integration Points

#### Factory Function

```python
from value_ledger import create_intentlog_listener

# In Agent-OS startup:
ledger_listener = create_intentlog_listener(ledger_path="~/.agent-os/ledger.jsonl")

# Connect to event bus:
event_bus.subscribe("intent.*", ledger_listener.handle_event)
```

#### Expected Directory Structure

```
~/.agent-os/
├── ledger.jsonl          # Value Ledger storage
├── intent_log.jsonl      # IntentLog storage
├── memory_vault/         # Memory Vault storage
└── contracts/            # Learning Contracts storage
```

### Compatibility Requirements

✅ **Agent-OS MUST:**
- Initialize Value Ledger listener during startup
- Route IntentLog events to Value Ledger
- Ensure write permissions for ledger storage

⚠️ **Agent-OS SHOULD:**
- Configure ledger path via settings
- Handle Value Ledger errors gracefully (don't crash on ledger issues)
- Provide admin commands to query ledger state

---

## 6. Common Module Integration

### Status: ⚠️ STUBBED (using local implementation)

### Purpose
Common module provides shared utilities like ID generation, hashing, and logging.

### Integration Points

#### Expected Common Utilities

```python
# Expected in common.utils:
def generate_entry_id(data: str) -> str:
    """Deterministic ID generation from data"""
    return hashlib.sha256(data.encode()).hexdigest()
```

#### Current Implementation

File: `value_ledger/core.py`

```python
# Placeholder for future imports from common utils
def generate_entry_id(data: str) -> str:
    """Deterministic ID generation (replace with common.utils if available)"""
    return hashlib.sha256(data.encode()).hexdigest()
```

### Compatibility Requirements

✅ **Common MUST:**
- Provide deterministic ID generation (same input → same output)
- Use SHA-256 or equivalent cryptographic hash

⚠️ **Common SHOULD:**
- Include utility for timestamp validation
- Provide structured logging helpers

### Migration Path

When common module is available:

1. Replace local function:
   ```python
   from common.utils import generate_entry_id
   ```

2. Remove local implementation
3. Verify ID generation remains deterministic

---

## 7. Synth-Mind Integration

### Status: ⏸️ NO DIRECT INTEGRATION (future)

### Purpose
Synth-mind provides cognitive tier architecture for the agent's thinking process.

### Future Integration Ideas

- **Cognitive tier tracking**: Value different thinking tiers (System 1 vs System 2)
- **Metacognition value**: Extra value for self-reflection and strategy
- **Tier switching cost**: Account for cognitive gear-shifting

### No current integration requirements.

---

## 8. NatLangChain Integration

### Status: ⏸️ NO DIRECT INTEGRATION (conceptual alignment)

### Purpose
NatLangChain provides natural-language-native ledger concepts and potentially blockchain-like proofs.

### Conceptual Alignment

- **Proof generation**: Value Ledger already supports hash-based proofs
- **Merkle trees**: `proof.merkle_ref` field exists but not yet implemented
- **Append-only ledger**: Both use immutable ledger design

### Future Integration Ideas

- Export Value Ledger entries to NatLangChain format
- Use NatLangChain for third-party proof verification
- Timestamp anchoring via NatLangChain

### No current integration requirements.

---

## Integration Checklist

### For IntentLog Team

- [ ] Implement `IntentEvent` dataclass with all recommended fields
- [ ] Emit events on intent lifecycle: started, completed, abandoned
- [ ] Include `interruptions` count from Boundary Daemon
- [ ] Tag outcomes with meaningful labels (success, failure, breakthrough, etc.)
- [ ] Provide `memory_hash` from Memory Vault after storage

### For Memory Vault Team

- [ ] Implement `query_similar()` with semantic embedding search
- [ ] Implement `get_recent()` for fallback novelty context
- [ ] Return `MemoryEntry` objects with hash, timestamp, and (optionally) content
- [ ] Integrate with Learning Contracts for content access control
- [ ] Support SHA-256 hashes for all memories

### For Learning Contracts Team

- [ ] Implement `ConsentChecker.check_access(intent_id, purpose)`
- [ ] Support `purpose="novelty_scoring"` access type
- [ ] Default to deny-all until user configures permissions
- [ ] Handle edge cases (unknown intent_id, missing contract)

### For Boundary Daemon Team

- [ ] Track interruption count per intent session
- [ ] Report interruptions to IntentLog for event inclusion
- [ ] Define clear interruption criteria
- [ ] Reset count on new intent start

### For Agent-OS Core Team

- [ ] Initialize Value Ledger listener on startup
- [ ] Route IntentLog events to Value Ledger
- [ ] Configure storage paths via settings
- [ ] Add admin commands: `ledger-query`, `ledger-export`, `ledger-stats`
- [ ] Handle Value Ledger errors gracefully

### For Common Module Team

- [ ] Provide `generate_entry_id()` utility
- [ ] Ensure deterministic hash-based ID generation
- [ ] Optional: Provide timestamp validation utilities
- [ ] Optional: Provide structured logging helpers

---

## Testing Integration Points

### Mock Testing

Each module can test Value Ledger integration using mocks:

```python
# Example: Testing IntentLog → Value Ledger
from value_ledger import IntentLogConnector, IntentEvent
from value_ledger.core import ValueLedger
import tempfile

def test_intent_completion_accrues_value():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = ValueLedger(f"{tmpdir}/test.jsonl")
        connector = IntentLogConnector(ledger)

        # Simulate intent lifecycle
        connector.handle_event(IntentEvent(
            event_type="intent_started",
            intent_id="test-001",
            timestamp=1000.0,
        ))

        connector.handle_event(IntentEvent(
            event_type="intent_completed",
            intent_id="test-001",
            timestamp=1300.0,  # 5 minutes later
            human_reasoning="Explored a novel approach...",
            interruptions=3,
            outcome_tags=["success"],
        ))

        # Verify value was accrued
        value = ledger.current_value_for_intent("test-001")
        assert value.total() > 0
        assert value.t > 0  # Time value
        assert value.e > 0  # Effort value
```

### Integration Testing

When multiple modules are available:

```bash
# Full integration test
pytest tests/integration/test_cross_repo.py

# Specific integrations
pytest tests/integration/test_intentlog_integration.py
pytest tests/integration/test_memory_vault_integration.py
```

---

## Version Compatibility

| Value Ledger | IntentLog | Memory Vault | Learning Contracts | Agent-OS |
|--------------|-----------|--------------|-------------------|----------|
| 0.1.0        | ≥0.1.0    | ≥0.1.0 (stub) | ≥0.1.0 (stub)     | ≥0.1.0   |

---

## Contact & Issues

For integration questions or compatibility issues:

- **Repository**: https://github.com/kase1111-hash/value-ledger
- **Issues**: Report integration bugs with label `integration`
- **Spec Questions**: Reference `specs.md` for design decisions

---

## Summary

**Ready for Integration**: IntentLog, Agent-OS Core
**Waiting for Implementation**: Memory Vault, Learning Contracts
**Passive Integration**: Boundary Daemon (via IntentLog)
**Future Integration**: Synth-Mind, NatLangChain, Common

Value Ledger is designed to work gracefully even when dependent modules are stubbed, providing safe defaults (high novelty, deny-all consent) until real implementations are available.
