# Value Ledger Specification

## 1. Purpose

The Value Ledger is the **economic and evidentiary accounting layer** of the learning co-worker ecosystem. It assigns, accrues, and preserves value for cognitive work: ideas, failed paths, effort, novelty, and time.

If Memory Vault stores *what happened* and Learning Contracts govern *permission*, the Value Ledger answers:

> **"What is this worth, and how can I prove it existed?"**

Value is recorded even when outcomes fail.

---

## 2. Design Principles

1. **Effort Has Value** – Failure still accrues credit.
2. **Proof > Price** – Ledger proves existence, not market value.
3. **Non-Destructive Accounting** – Revocation freezes value, not history.
4. **Owner-Centric Valuation** – No external pricing oracle required.
5. **Audit Without Disclosure** – Value proofs without content leakage.
6. **Offline First** – No dependency on networks or blockchains.

---

## 3. What the Ledger Records

The ledger records **meta-value**, not raw content.

### Recordable Assets

* Time invested
* Cognitive difficulty
* Novelty
* Search space explored
* Failed paths eliminated
* Reusable heuristics created
* Strategic insights (when permitted)

---

## 4. Value Units

The ledger uses **abstract value units**, not currency.

### 4.1 Core Units

| Unit | Meaning          | Implementation Status |
| ---- | ---------------- | --------------------- |
| T    | Time (seconds)   | ✅ Implemented        |
| E    | Effort intensity | ✅ Implemented        |
| N    | Novelty          | ✅ Implemented        |
| F    | Failure density  | ✅ Implemented        |
| R    | Reusability      | ✅ Implemented (as Risk exposure) |
| S    | Strategic depth  | ✅ Implemented        |

Units are normalized to [0–1] except time.

**Note:** Implementation uses `r` as "Risk exposure" rather than "Reusability". Consider renaming or adding a separate reusability metric.

---

## 5. Core Ledger Object Schema

### 5.1 Specified Schema

```json
{
  "ledger_id": "uuid",
  "created_at": "timestamp",
  "owner": "human_id",
  "source": {
    "memory_id": "uuid|null",
    "intent_id": "uuid|null",
    "contract_id": "uuid"
  },
  "value_vector": {
    "time_seconds": 0,
    "effort": 0.0,
    "novelty": 0.0,
    "failure_density": 0.0,
    "reusability": 0.0,
    "strategic_depth": 0.0
  },
  "classification": 0-5,
  "status": "active|frozen|revoked",
  "derivation": {
    "parent_ledger_ids": ["uuid"],
    "aggregation_rule": "sum|max|weighted"
  },
  "proof": {
    "content_hash": "sha256",
    "timestamp_proof": "local_ts",
    "merkle_ref": "hash"
  }
}
```

### 5.2 Implementation Status

| Field | Status | Notes |
|-------|--------|-------|
| `ledger_id` | ✅ Implemented | As `id` in `LedgerEntry` |
| `created_at` | ✅ Implemented | As `timestamp` |
| `owner` | ❌ Not implemented | No owner tracking |
| `source.memory_id` | ✅ Partial | As `memory_hash` |
| `source.intent_id` | ✅ Implemented | As `intent_id` |
| `source.contract_id` | ❌ Not implemented | No contract linking |
| `value_vector` | ✅ Implemented | All 6 units in `ValueVector` |
| `classification` | ❌ Not implemented | No classification system |
| `status` | ✅ Implemented | active/frozen/revoked |
| `derivation.parent_ledger_ids` | ✅ Partial | Single `parent_id` only |
| `derivation.aggregation_rule` | ❌ Not implemented | No rule specification |
| `proof.content_hash` | ❌ Not implemented | |
| `proof.timestamp_proof` | ❌ Not implemented | |
| `proof.merkle_ref` | ❌ Not implemented | |

---

## 6. Ledger Operations

### 6.1 Accrual

**Status:** ✅ Implemented

* Triggered by:

  * Memory creation
  * Intent completion
  * Explicit owner marking
* Automatic scoring permitted
* Owner may override scores

**Implementation:** `ValueLedger.accrue()` and `ValueLedger.accrue_with_heuristics()`

### 6.2 Aggregation

**Status:** ✅ Partially Implemented

* Multiple ledger entries may be combined
* Aggregation never deletes originals
* Parent-child relationships preserved

**Implementation:** `ValueLedger.aggregate_correction()` - supports single parent; multi-parent aggregation not yet implemented.

### 6.3 Freezing

**Status:** ✅ Implemented

* Triggered by:

  * Contract expiration
  * Boundary downgrade
* Value retained, but no further accrual

**Implementation:** Status field set to "frozen" via `aggregate_correction(freeze_parent=True)`

### 6.4 Revocation

**Status:** ⚠️ Partial

* Triggered by owner
* Value remains provable
* Asset becomes non-exploitable

**Implementation:** Status field supports "revoked" but no explicit `revoke()` method exists.

---

## 7. Valuation Heuristics (Non-Binding)

**Status:** ✅ Implemented

The system may *suggest* value scores based on:

* Time × interruption resistance → `TimeScorer`, `EffortScorer`
* Novelty against prior memory corpus → `NoveltyScorer` (embedding-based)
* Failure density (paths ruled out per time) → `FailureScorer`
* Compression ratio (raw → heuristic) → Not implemented

Suggestions are advisory only.

**Implementation:** `HeuristicEngine` with 6 scorers in `heuristics.py`

---

## 8. Interaction with IntentLog

**Status:** ✅ Implemented

* Every IntentLog entry may generate ledger entries
* Ledger binds value to *intent*, not just outcome
* Failed intents accrue value proportional to search

**Implementation:** `IntentLogConnector` in `integration.py`

---

## 9. Interaction with Memory Vault

**Status:** ⚠️ Stubbed

* Ledger stores hashes, never plaintext
* Classification mirrors memory classification
* Vault deletion does NOT delete ledger proof

**Implementation:** `MemoryVaultHook` with stub implementations. Returns empty lists until Memory Vault is available.

---

## 10. Interaction with Learning Contracts

**Status:** ⚠️ Stubbed

* Ledger creation requires valid contract
* Contract caps strategic depth scoring
* Revoked contracts freeze derived value

**Implementation:** `ConsentChecker` stub defaults to deny-all.

---

## 11. Threat Model

### Threats

* Overvaluation inflation
* Silent value loss
* Retroactive theft
* External coercion

### Mitigations

| Threat    | Mitigation              | Status |
| --------- | ----------------------- | ------ |
| Inflation | Owner override + audits | ⚠️ Override only |
| Loss      | Append-only logs        | ✅ Implemented |
| Theft     | Hash-based proofs       | ❌ Not implemented |
| Coercion  | Offline proofs          | ❌ Not implemented |

---

## 12. Proof & Export

**Status:** ❌ Not Implemented

The ledger can export:

* Time-stamped proof of existence
* Aggregated value summaries
* Merkle proofs for third parties

Exports never include content.

---

## 13. Failure Modes

**Status:** ❌ Not Implemented

* Clock drift → pause accrual
* Ambiguous source → deny entry
* Classification mismatch → freeze

---

## 14. Non-Goals

* Real-time pricing
* Market speculation
* Token issuance
* Automatic monetization

---

## 15. Design Constraint

> The world pays for outcomes.
> The ledger preserves proof of effort.

The Value Ledger exists for when those diverge.

---

## 16. Unimplemented Features & Implementation Plans

This section documents features specified in the documentation but not yet implemented, along with detailed implementation plans.

### 16.1 Proof System (MP-02 Compatibility)

**Source:** specs.md §12, MP-02-spec.md

**Status:** Not implemented

**Description:** Cryptographic proof generation for third-party verification without content disclosure.

**Implementation Plan:**

1. **Add Proof Fields to LedgerEntry**
   - Add `content_hash: str` - SHA-256 hash of source content
   - Add `timestamp_proof: str` - Signed timestamp or anchor reference
   - Add `merkle_ref: Optional[str]` - Reference to Merkle tree position

2. **Create ProofGenerator Class** (`value_ledger/proof.py`)
   ```python
   class ProofGenerator:
       def generate_content_hash(content: str) -> str
       def generate_timestamp_proof(timestamp: float) -> str
       def build_merkle_tree(entries: List[LedgerEntry]) -> MerkleTree
       def get_merkle_proof(entry_id: str) -> MerkleProof
   ```

3. **Implement Export Functions**
   ```python
   def export_existence_proof(entry_id: str) -> ExistenceProof
   def export_value_summary(intent_id: str) -> ValueSummary
   def export_merkle_proof(entry_id: str) -> MerkleProof
   ```

4. **Integration Steps:**
   - Compute content_hash on every `accrue()` call
   - Build incremental Merkle tree on ledger updates
   - Store Merkle root in separate file for anchoring

---

### 16.2 Effort Receipt Protocol (MP-02)

**Source:** MP-02-spec.md

**Status:** Not implemented

**Description:** Full implementation of the Proof-of-Effort Receipt Protocol for NatLangChain compatibility.

**Implementation Plan:**

1. **Create Receipt Data Structures** (`value_ledger/receipt.py`)
   ```python
   @dataclass
   class EffortSignal:
       signal_type: str  # "voice", "text", "command", "tool"
       timestamp: float
       hash: str
       modality: str

   @dataclass
   class EffortSegment:
       segment_id: str
       start_time: float
       end_time: float
       signals: List[EffortSignal]
       segmentation_rule: str

   @dataclass
   class EffortReceipt:
       receipt_id: str
       time_bounds: Tuple[float, float]
       signal_hashes: List[str]
       effort_summary: str
       validation_metadata: ValidationMetadata
       observer_id: str
       validator_id: str
       prior_receipts: List[str]
   ```

2. **Create Observer Interface** (`value_ledger/observer.py`)
   ```python
   class Observer(Protocol):
       def capture_signal(signal: Any) -> EffortSignal
       def get_modality() -> str
       def get_observer_id() -> str
   ```

3. **Create Validator Interface** (`value_ledger/validator.py`)
   ```python
   class Validator:
       model_id: str
       model_version: str

       def validate_segment(segment: EffortSegment) -> ValidationResult
       def generate_summary(segment: EffortSegment) -> str
   ```

4. **Create Receipt Builder**
   ```python
   class ReceiptBuilder:
       def from_ledger_entry(entry: LedgerEntry) -> EffortReceipt
       def anchor_to_ledger(receipt: EffortReceipt) -> str
       def verify_receipt(receipt: EffortReceipt) -> VerificationResult
   ```

---

### 16.3 Owner & Classification System

**Source:** specs.md §5

**Status:** Not implemented

**Description:** Track ownership and classification levels (0-5) for entries.

**Implementation Plan:**

1. **Extend LedgerEntry Schema**
   ```python
   class LedgerEntry(BaseModel):
       # ... existing fields ...
       owner: Optional[str] = None
       classification: int = Field(default=0, ge=0, le=5)
       contract_id: Optional[str] = None
   ```

2. **Add Classification Enforcement**
   ```python
   def accrue(..., owner: str, classification: int = 0, contract_id: str = None):
       # Validate classification against contract
       # Check owner permissions
   ```

3. **Add Classification-Based Access Control**
   ```python
   def get_entries_for_owner(owner: str) -> List[LedgerEntry]
   def filter_by_classification(entries: List, max_level: int) -> List
   ```

---

### 16.4 Multi-Parent Aggregation

**Source:** specs.md §5, §6.2

**Status:** Partial (single parent only)

**Description:** Support aggregating multiple ledger entries with configurable rules.

**Implementation Plan:**

1. **Extend LedgerEntry for Multi-Parent**
   ```python
   class LedgerEntry(BaseModel):
       # ... existing ...
       parent_ids: List[str] = Field(default_factory=list)  # Replace parent_id
       aggregation_rule: str = "sum"  # sum | max | weighted
   ```

2. **Implement Aggregation Rules**
   ```python
   class AggregationEngine:
       def aggregate_sum(entries: List[LedgerEntry]) -> ValueVector
       def aggregate_max(entries: List[LedgerEntry]) -> ValueVector
       def aggregate_weighted(entries: List[LedgerEntry], weights: Dict) -> ValueVector
   ```

3. **Update aggregate_correction()**
   ```python
   def aggregate_entries(
       parent_ids: List[str],
       aggregation_rule: str = "sum",
       weights: Optional[Dict[str, float]] = None,
   ) -> str
   ```

---

### 16.5 Failure Mode Handling

**Source:** specs.md §13

**Status:** Not implemented

**Description:** Handle clock drift, ambiguous sources, and classification mismatches.

**Implementation Plan:**

1. **Clock Drift Detection** (`value_ledger/safety.py`)
   ```python
   class ClockMonitor:
       last_timestamp: float
       drift_threshold: float = 5.0  # seconds

       def check_clock(current_ts: float) -> ClockStatus
       def should_pause_accrual() -> bool
   ```

2. **Source Validation**
   ```python
   class SourceValidator:
       def validate_source(
           intent_id: Optional[str],
           memory_hash: Optional[str],
           contract_id: Optional[str]
       ) -> ValidationResult

       def is_ambiguous(result: ValidationResult) -> bool
   ```

3. **Classification Mismatch Handler**
   ```python
   def check_classification_consistency(
       entry: LedgerEntry,
       memory_classification: int,
       contract_max_classification: int
   ) -> MismatchResult

   def auto_freeze_on_mismatch(entry_id: str, reason: str)
   ```

---

### 16.6 Synth-Mind Integration

**Source:** INTEGRATION.md §7

**Status:** Not implemented

**Description:** Value different cognitive tiers from synth-mind architecture.

**Implementation Plan:**

1. **Add Cognitive Tier Tracking**
   ```python
   @dataclass
   class ScoringContext:
       # ... existing ...
       cognitive_tier: Optional[int] = None  # 1=System1, 2=System2, etc.
       tier_switches: int = 0
       metacognition_depth: int = 0
   ```

2. **Create TierScorer**
   ```python
   class CognitiveTierScorer(HeuristicScorer):
       def __call__(self, ctx: ScoringContext) -> ValueVector:
           # Higher tiers = more strategic value
           # Tier switches add effort value
           # Metacognition adds to all dimensions
   ```

3. **Integrate with synth-mind Events**
   ```python
   from synth_mind.events import TierChangeEvent

   def handle_tier_change(event: TierChangeEvent):
       # Track tier switches for effort scoring
   ```

---

### 16.7 NatLangChain Export

**Source:** INTEGRATION.md §8

**Status:** Not implemented

**Description:** Export ledger entries to NatLangChain format for blockchain anchoring.

**Implementation Plan:**

1. **Create NatLangChain Adapter** (`value_ledger/natlangchain.py`)
   ```python
   class NatLangChainExporter:
       def to_nlc_format(entry: LedgerEntry) -> NLCRecord
       def batch_export(entries: List[LedgerEntry]) -> List[NLCRecord]
       def anchor_to_chain(records: List[NLCRecord]) -> AnchorResult
   ```

2. **Implement Format Conversion**
   ```python
   @dataclass
   class NLCRecord:
       record_type: str = "effort_receipt"
       timestamp: float
       proof_hash: str
       value_summary: Dict
       prior_anchors: List[str]
   ```

3. **Add Verification Hooks**
   ```python
   def verify_anchor(entry_id: str, chain_ref: str) -> bool
   def get_chain_proof(entry_id: str) -> ChainProof
   ```

---

### 16.8 Admin CLI Commands

**Source:** INTEGRATION.md §5

**Status:** Not implemented

**Description:** Administrative commands for ledger management.

**Implementation Plan:**

1. **Extend CLI** (`value_ledger/cli.py`)
   ```python
   @click.group()
   def cli():
       pass

   @cli.command()
   @click.option('--intent-id', help='Filter by intent')
   @click.option('--status', help='Filter by status')
   @click.option('--since', help='Filter by timestamp')
   def query(intent_id, status, since):
       """Query ledger entries"""

   @cli.command()
   @click.option('--format', type=click.Choice(['json', 'csv', 'merkle']))
   @click.option('--output', type=click.Path())
   def export(format, output):
       """Export ledger data"""

   @cli.command()
   def stats():
       """Show ledger statistics"""
       # Total entries, by status, by intent, value distributions
   ```

2. **Add Stats Calculator**
   ```python
   class LedgerStats:
       def total_entries() -> int
       def entries_by_status() -> Dict[str, int]
       def value_distribution() -> ValueDistribution
       def top_intents_by_value(n: int) -> List[Tuple[str, float]]
   ```

---

### 16.9 Explicit Revocation Method

**Source:** specs.md §6.4

**Status:** Partial

**Description:** Add explicit `revoke()` method with proper handling.

**Implementation Plan:**

1. **Add revoke() Method**
   ```python
   def revoke(
       self,
       entry_id: str,
       reason: Optional[str] = None,
       revoke_children: bool = False
   ) -> bool:
       """
       Revoke an entry. Value remains provable but non-exploitable.

       Args:
           entry_id: Entry to revoke
           reason: Optional reason for audit trail
           revoke_children: Also revoke derived entries
       """
   ```

2. **Add Revocation Metadata**
   ```python
   class LedgerEntry:
       # ... existing ...
       revocation_reason: Optional[str] = None
       revoked_at: Optional[float] = None
       revoked_by: Optional[str] = None
   ```

---

### 16.10 Reusability Metric

**Source:** specs.md §4.1

**Status:** Not implemented (R is currently Risk)

**Description:** Track reusability of heuristics and patterns separately from risk.

**Implementation Plan:**

1. **Option A: Add 7th Dimension**
   ```python
   class ValueVector(BaseModel):
       t: float  # Time
       e: float  # Effort
       n: float  # Novelty
       f: float  # Failure
       r: float  # Risk (keep as is)
       s: float  # Strategy
       u: float  # Reusability (new)
   ```

2. **Option B: Rename R to Reusability, Add Risk Separately**
   - Requires migration of existing data
   - More breaking change

3. **Create ReusabilityScorer**
   ```python
   class ReusabilityScorer(HeuristicScorer):
       def __call__(self, ctx: ScoringContext) -> ValueVector:
           # Score based on:
           # - Pattern abstraction level
           # - Cross-domain applicability
           # - Reference frequency in future work
   ```

---

## 17. Implementation Priority

| Feature | Priority | Complexity | Dependencies |
|---------|----------|------------|--------------|
| Proof System (16.1) | High | Medium | None |
| Admin CLI (16.8) | High | Low | None |
| Explicit Revocation (16.9) | High | Low | None |
| Owner/Classification (16.3) | Medium | Medium | Learning Contracts |
| Failure Mode Handling (16.5) | Medium | Medium | None |
| Multi-Parent Aggregation (16.4) | Medium | Medium | None |
| Effort Receipt Protocol (16.2) | Medium | High | Proof System |
| NatLangChain Export (16.7) | Low | Medium | NatLangChain module |
| Synth-Mind Integration (16.6) | Low | Medium | Synth-Mind module |
| Reusability Metric (16.10) | Low | Low | Migration needed |

---

## 18. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024 | Initial specification |
| 0.1.1 | 2024 | Added implementation status and plans |
