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

## 16. NatLangChain Compatibility

**Source:** [NatLangChain Repository](https://github.com/kase1111-hash/NatLangChain)

**Status:** ⏸️ Designed but not implemented

### 16.1 Overview

NatLangChain is a prose-first distributed ledger where natural language entries form the immutable substrate. Value Ledger entries can be exported and anchored to NatLangChain for third-party verification.

### 16.2 Key Compatibility Points

| NatLangChain Concept | Value Ledger Mapping |
|---------------------|---------------------|
| Prose entries | Effort summaries from EffortReceipt |
| SHA-256 chaining | content_hash in proof field |
| Proof of Understanding | Validator summaries in MP-02 receipts |
| Append-only ledger | JSONL storage format |
| Deterministic validation | HeuristicEngine with reproducible scoring |

### 16.3 REST API Integration

NatLangChain exposes endpoints that Value Ledger can use for anchoring:

```python
# Target endpoints for integration:
POST /entry          # Submit effort receipt as prose entry
POST /entry/validate # Validate receipt before anchoring
GET  /chain          # Query anchored receipts
GET  /chain/narrative # Human-readable view of effort history
```

### 16.4 Proof of Understanding Alignment

NatLangChain uses "Proof of Understanding" where validators paraphrase entries to demonstrate comprehension. MP-02 validators similarly produce deterministic summaries. This alignment enables:

- Validator summaries can double as NatLangChain entry prose
- Multiple validators provide redundancy (semantic consensus)
- Ambiguity handling through NatLangChain clarification protocols

---

## 17. Unimplemented Features & Implementation Plans

This section documents features specified in the documentation but not yet implemented, along with detailed implementation plans.

### 17.1 Proof System (MP-02 Compatibility)

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

### 17.2 Effort Receipt Protocol (MP-02)

**Source:** MP-02-spec.md, [NatLangChain Repository](https://github.com/kase1111-hash/NatLangChain)

**Status:** Not implemented

**Description:** Full implementation of the Proof-of-Effort Receipt Protocol for NatLangChain compatibility. MP-02 asserts that effort occurred with traceable provenance, without asserting value, ownership, or compensation.

**Key MP-02 Principles:**
- Process Over Artifact — Effort is validated as a process unfolding over time
- Continuity Matters — Temporal progression is a primary signal of genuine work
- Receipts, Not Claims — The protocol records evidence, not conclusions about value
- Model Skepticism — LLM assessments are advisory and must be reproducible
- Partial Observability — Uncertainty is preserved, not collapsed

**Implementation Plan:**

1. **Create Receipt Data Structures** (`value_ledger/receipt.py`)
   ```python
   @dataclass
   class EffortSignal:
       signal_type: str  # "voice", "text", "command", "tool"
       timestamp: float
       hash: str  # SHA-256 hash of signal content
       modality: str

   @dataclass
   class EffortSegment:
       segment_id: str
       start_time: float
       end_time: float
       signals: List[EffortSignal]
       segmentation_rule: str  # Must be deterministic and disclosed

   @dataclass
   class ValidationMetadata:
       validator_id: str
       model_id: str
       model_version: str
       validation_timestamp: float
       coherence_score: Optional[float]
       progression_score: Optional[float]
       uncertainty_markers: List[str]  # Preserve ambiguity per MP-02

   @dataclass
   class EffortReceipt:
       receipt_id: str
       time_bounds: Tuple[float, float]
       signal_hashes: List[str]
       effort_summary: str  # Deterministic, suitable for NatLangChain prose
       validation_metadata: ValidationMetadata
       observer_id: str
       validator_id: str
       prior_receipts: List[str]

       # Failure mode tracking (MP-02 §11)
       observation_gaps: List[Tuple[float, float]]
       conflicting_validations: List[str]
       suspected_manipulation: bool
       is_incomplete: bool
   ```

2. **Create Observer Interface** (`value_ledger/observer.py`)
   ```python
   class Observer(Protocol):
       """
       Observers MUST:
       - Time-stamp all signals
       - Preserve ordering
       - Disclose capture modality

       Observers MUST NOT:
       - Alter raw signals
       - Infer intent beyond observed data
       """
       def capture_signal(signal: Any) -> EffortSignal
       def get_modality() -> str
       def get_observer_id() -> str
       def get_capture_mode() -> str  # "continuous" or "intermittent"
   ```

3. **Create Validator Interface** (`value_ledger/validator.py`)
   ```python
   class Validator:
       """
       Validators MUST:
       - Produce deterministic summaries
       - Disclose model identity and version
       - Preserve dissent and uncertainty

       Validators MUST NOT:
       - Declare effort as valuable
       - Assert originality or ownership
       - Collapse ambiguous signals into certainty
       """
       model_id: str
       model_version: str

       def validate_segment(segment: EffortSegment) -> ValidationResult
       def generate_summary(segment: EffortSegment) -> str
       def assess_coherence(segment: EffortSegment) -> float
       def assess_progression(segment: EffortSegment) -> float
       def detect_adversarial_patterns(segment: EffortSegment) -> List[str]
   ```

4. **Create Receipt Builder**
   ```python
   class ReceiptBuilder:
       def from_ledger_entry(entry: LedgerEntry) -> EffortReceipt
       def anchor_to_ledger(receipt: EffortReceipt) -> str
       def verify_receipt(receipt: EffortReceipt) -> VerificationResult
       def compute_receipt_hash(receipt: EffortReceipt) -> str  # For NLC anchoring

   def verify_third_party(receipt: EffortReceipt) -> bool:
       """
       Third party verification per MP-02 §10:
       - Recompute receipt hashes
       - Inspect validation metadata
       - Confirm ledger inclusion
       Trust in Observer or Validator is NOT required.
       """
   ```

5. **NatLangChain Anchoring Integration**
   ```python
   def anchor_receipt_to_nlc(receipt: EffortReceipt, nlc_client: NLCClient) -> str:
       """
       Convert EffortReceipt to NatLangChain prose entry and anchor.
       Uses effort_summary as the canonical prose for Proof of Understanding.
       """
   ```

---

### 17.3 Owner & Classification System

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

### 17.4 Multi-Parent Aggregation

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

### 17.5 Failure Mode Handling

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

### 17.6 Synth-Mind Integration

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

### 17.7 NatLangChain Export

**Source:** INTEGRATION.md §8, [NatLangChain Repository](https://github.com/kase1111-hash/NatLangChain)

**Status:** Not implemented

**Description:** Export ledger entries to NatLangChain format for blockchain anchoring. NatLangChain is a prose-first distributed ledger where natural language entries form the immutable substrate.

**Implementation Plan:**

1. **Create NatLangChain Adapter** (`value_ledger/natlangchain.py`)
   ```python
   class NatLangChainExporter:
       base_url: str = "http://localhost:5000"  # NatLangChain node

       def to_nlc_format(entry: LedgerEntry) -> NLCRecord
       def batch_export(entries: List[LedgerEntry]) -> List[NLCRecord]
       def anchor_to_chain(records: List[NLCRecord]) -> AnchorResult
       def validate_before_anchor(record: NLCRecord) -> ValidationResult
   ```

2. **Implement Format Conversion (Prose-First)**
   ```python
   @dataclass
   class NLCRecord:
       record_type: str = "effort_receipt"
       timestamp: float
       proof_hash: str  # SHA-256 compatible with NLC block chaining
       value_summary: Dict
       prior_anchors: List[str]

       # NatLangChain-specific fields
       prose_entry: str  # Human-readable narrative (required by NLC)
       author: str  # Owner/creator identifier
       intent_summary: str  # From validator deterministic summary

   def to_prose_entry(entry: LedgerEntry) -> str:
       """Convert ledger entry to human-readable prose for NatLangChain."""
       return f"""
       Effort Receipt #{entry.id[:8]}
       Time: {entry.timestamp}
       Intent: {entry.intent_id}
       Value Vector: T={entry.value_vector.t:.1f}, E={entry.value_vector.e:.1f},
                     N={entry.value_vector.n:.1f}, F={entry.value_vector.f:.1f},
                     R={entry.value_vector.r:.1f}, S={entry.value_vector.s:.1f}
       Status: {entry.status}
       """
   ```

3. **REST API Integration**
   ```python
   class NLCClient:
       def submit_entry(record: NLCRecord) -> str:
           """POST /entry - Returns entry hash"""

       def validate_entry(record: NLCRecord) -> ValidationResult:
           """POST /entry/validate - Dry-run validation"""

       def get_chain_narrative() -> str:
           """GET /chain/narrative - Human-readable ledger view"""

       def search_by_intent(intent: str) -> List[NLCRecord]:
           """GET /entries/search?intent=... - Semantic search"""
   ```

4. **Proof of Understanding Compatibility**
   ```python
   def generate_validator_summary(entry: LedgerEntry) -> str:
       """
       Generate deterministic summary suitable for NatLangChain's
       Proof of Understanding consensus mechanism.
       """
       # Summary must be reproducible by other validators
       # Used for semantic consensus across NLC nodes
   ```

5. **Add Verification Hooks**
   ```python
   def verify_anchor(entry_id: str, chain_ref: str) -> bool
   def get_chain_proof(entry_id: str) -> ChainProof
   def check_inclusion(entry_id: str) -> InclusionProof
   ```

---

### 17.8 Admin CLI Commands

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

### 17.9 Explicit Revocation Method

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

### 17.10 Reusability Metric

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

### 17.11 Boundary Daemon Integration

**Source:** INTEGRATION.md §4

**Status:** Not implemented (passive integration via IntentLog)

**Description:** Direct integration with Boundary Daemon for interruption tracking and context switch detection that affects effort scoring.

**Implementation Plan:**

1. **Create InterruptionTracker** (`value_ledger/interruption.py`)
   ```python
   @dataclass
   class InterruptionEvent:
       intent_id: str
       timestamp: float
       interruption_type: str  # "external", "self", "context_switch"
       source: Optional[str] = None  # What caused the interruption
       duration: Optional[float] = None  # How long the interruption lasted

   class InterruptionTracker:
       def __init__(self):
           self.active_sessions: Dict[str, List[InterruptionEvent]] = {}

       def start_session(self, intent_id: str) -> None:
           """Begin tracking interruptions for an intent"""

       def record_interruption(self, event: InterruptionEvent) -> None:
           """Record an interruption event"""

       def get_interruption_count(self, intent_id: str) -> int:
           """Get total interruptions for an intent"""

       def get_weighted_interruptions(self, intent_id: str) -> float:
           """
           Calculate weighted interruption score:
           - External interruptions: weight 1.0
           - Context switches: weight 0.7
           - Self-interruptions: weight 0.3
           """

       def end_session(self, intent_id: str) -> InterruptionSummary:
           """Finalize session and return summary for effort scoring"""
   ```

2. **Boundary Daemon Event Listener**
   ```python
   class BoundaryDaemonListener:
       def __init__(self, tracker: InterruptionTracker):
           self.tracker = tracker

       def handle_boundary_event(self, event: BoundaryEvent) -> None:
           """
           Handle events from Boundary Daemon:
           - "notification_received"
           - "context_switch"
           - "focus_lost"
           - "focus_regained"
           """

       def connect_to_daemon(self, daemon_url: str) -> None:
           """Subscribe to Boundary Daemon event stream"""
   ```

3. **Integration with Effort Scoring**
   ```python
   # Update EffortScorer to use InterruptionTracker
   class EffortScorer(HeuristicScorer):
       def __init__(self, interruption_tracker: Optional[InterruptionTracker] = None):
           self.tracker = interruption_tracker

       def calculate_interruption_factor(self, intent_id: str) -> float:
           if self.tracker is None:
               # Fall back to IntentLog-provided count
               return 1.0
           weighted = self.tracker.get_weighted_interruptions(intent_id)
           return 1.0 + (weighted * 0.35)
   ```

4. **Compatibility Requirements**

   ✅ **Boundary Daemon MUST:**
   - Track interruption count per intent session
   - Report count to IntentLog for inclusion in events
   - Emit events in real-time for direct integration

   ⚠️ **Boundary Daemon SHOULD:**
   - Define "interruption" consistently (context switch, notification, etc.)
   - Reset count on new intent start
   - Distinguish between external and self-interruptions

   ❌ **Boundary Daemon MUST NOT:**
   - Report negative interruption counts
   - Count normal pauses as interruptions

---

### 17.12 Common Module Integration

**Source:** INTEGRATION.md §6

**Status:** Not implemented (using local implementations)

**Description:** Integration with the Common module for shared utilities including ID generation, timestamp validation, and structured logging.

**Implementation Plan:**

1. **ID Generation Utilities**
   ```python
   # Current local implementation to be replaced:
   def generate_entry_id(data: str) -> str:
       """Deterministic ID generation (replace with common.utils if available)"""
       return hashlib.sha256(data.encode()).hexdigest()

   # Expected from common module:
   from common.utils import generate_entry_id, generate_uuid
   ```

2. **Timestamp Validation**
   ```python
   # Expected from common module:
   class TimestampValidator:
       def validate(self, timestamp: float) -> ValidationResult:
           """Validate timestamp is reasonable (not future, not too old)"""

       def check_drift(self, ts1: float, ts2: float, max_drift: float) -> bool:
           """Check if two timestamps are within acceptable drift"""

       def normalize(self, timestamp: float) -> float:
           """Normalize timestamp to consistent precision"""
   ```

3. **Structured Logging**
   ```python
   # Expected from common module:
   from common.logging import get_structured_logger, AuditLogger

   class ValueLedger:
       def __init__(self, storage_path: str):
           self.logger = get_structured_logger("value_ledger")
           self.audit = AuditLogger("value_ledger.audit")

       def accrue(self, ...):
           self.audit.log_event("accrue", entry_id=entry.id, value=entry.value_vector)
   ```

4. **Migration Path**
   ```python
   # When common module is available:
   try:
       from common.utils import generate_entry_id
       from common.validation import TimestampValidator
       from common.logging import get_structured_logger
   except ImportError:
       # Fall back to local implementations
       from value_ledger._compat import (
           generate_entry_id,
           TimestampValidator,
           get_structured_logger,
       )
   ```

5. **Compatibility Requirements**

   ✅ **Common MUST:**
   - Provide deterministic ID generation (same input → same output)
   - Use SHA-256 or equivalent cryptographic hash
   - Maintain backwards compatibility

   ⚠️ **Common SHOULD:**
   - Include utility for timestamp validation
   - Provide structured logging helpers
   - Support audit logging for compliance

---

### 17.13 MP-02 Privacy and Agency Controls

**Source:** MP-02-spec.md §12

**Status:** Not implemented

**Description:** Privacy controls and human agency features as specified in the MP-02 protocol for protecting raw signals while maintaining receipt verifiability.

**Implementation Plan:**

1. **Signal Encryption Layer** (`value_ledger/privacy.py`)
   ```python
   class SignalEncryption:
       """Encrypt raw signals while preserving hash verifiability"""

       def encrypt_signal(
           self,
           signal: EffortSignal,
           key: bytes,
       ) -> EncryptedSignal:
           """
           Encrypt signal content.
           Hash is computed BEFORE encryption so verification works.
           """

       def decrypt_signal(
           self,
           encrypted: EncryptedSignal,
           key: bytes,
       ) -> EffortSignal:
           """Decrypt signal for authorized access"""

       def verify_encrypted(
           self,
           encrypted: EncryptedSignal,
           expected_hash: str,
       ) -> bool:
           """Verify hash matches without decryption"""

   @dataclass
   class EncryptedSignal:
       hash: str  # Pre-encryption hash for verification
       encrypted_content: bytes
       encryption_metadata: Dict[str, Any]  # Algorithm, key_id, etc.
   ```

2. **Receipt Privacy Controls**
   ```python
   class ReceiptPrivacyManager:
       """Control what information is exposed in receipts"""

       def __init__(self, default_privacy: PrivacyLevel = PrivacyLevel.STANDARD):
           self.default_privacy = default_privacy

       def create_public_receipt(
           self,
           receipt: EffortReceipt,
           privacy_level: PrivacyLevel,
       ) -> PublicReceipt:
           """
           Create receipt for public/third-party consumption.

           Privacy levels:
           - MINIMAL: Only hashes and timestamps
           - STANDARD: Hashes, timestamps, summary (default)
           - DETAILED: Include validation metadata
           - FULL: Complete receipt (owner only)
           """

       def redact_receipt(
           self,
           receipt: EffortReceipt,
           fields_to_redact: List[str],
       ) -> RedactedReceipt:
           """Selectively redact specific fields"""

   class PrivacyLevel(Enum):
       MINIMAL = "minimal"
       STANDARD = "standard"
       DETAILED = "detailed"
       FULL = "full"
   ```

3. **Future Observation Revocation**
   ```python
   class ObservationConsent:
       """Manage human consent for future observation"""

       def grant_observation(
           self,
           human_id: str,
           scope: ObservationScope,
           duration: Optional[timedelta] = None,
       ) -> ConsentGrant:
           """Grant consent for future observation"""

       def revoke_observation(
           self,
           human_id: str,
           scope: Optional[ObservationScope] = None,
       ) -> RevocationRecord:
           """
           Revoke future observation consent.
           Past receipts remain immutable per MP-02 §12.
           """

       def check_consent(
           self,
           human_id: str,
           observation_type: str,
       ) -> bool:
           """Check if observation is currently permitted"""

   @dataclass
   class ObservationScope:
       modalities: List[str]  # ["voice", "text", "commands"]
       contexts: List[str]  # ["work", "personal"]
       purposes: List[str]  # ["effort_tracking", "novelty_scoring"]
   ```

4. **Immutability Enforcement**
   ```python
   class ReceiptImmutability:
       """Enforce immutability of past receipts"""

       def seal_receipt(self, receipt: EffortReceipt) -> SealedReceipt:
           """
           Seal receipt to prevent modification.
           Returns cryptographic seal for verification.
           """

       def verify_seal(self, sealed: SealedReceipt) -> bool:
           """Verify receipt has not been modified"""

       def create_amendment(
           self,
           original: SealedReceipt,
           correction: ReceiptCorrection,
       ) -> AmendmentRecord:
           """
           Create amendment record (original remains unchanged).
           Used for corrections without altering history.
           """
   ```

---

### 17.14 MP-02 External Protocol Compatibility

**Source:** MP-02-spec.md §14

**Status:** Not implemented

**Description:** Compatibility with external protocols including MP-01 Negotiation & Ratification, licensing modules, and external audit systems.

**Implementation Plan:**

1. **MP-01 Negotiation Protocol Adapter**
   ```python
   class MP01Adapter:
       """Adapter for MP-01 Negotiation & Ratification protocol"""

       def effort_to_negotiation_payload(
           self,
           receipt: EffortReceipt,
       ) -> NegotiationPayload:
           """
           Convert effort receipt to MP-01 negotiation format.
           Used when effort value is subject to negotiation.
           """

       def attach_to_contract(
           self,
           receipt: EffortReceipt,
           contract_id: str,
       ) -> ContractAttachment:
           """Attach effort receipt to MP-01 contract"""

       def verify_ratification(
           self,
           receipt_id: str,
           ratification: Ratification,
       ) -> bool:
           """Verify effort was ratified via MP-01"""
   ```

2. **Licensing Module Integration**
   ```python
   class LicensingAdapter:
       """Adapter for licensing and delegation modules"""

       def create_license(
           self,
           receipt: EffortReceipt,
           license_type: LicenseType,
           terms: LicenseTerms,
       ) -> EffortLicense:
           """
           Create license for effort receipt.
           Enables controlled sharing and delegation.
           """

       def delegate_effort(
           self,
           receipt: EffortReceipt,
           delegate_id: str,
           scope: DelegationScope,
       ) -> DelegationRecord:
           """Delegate effort value to another party"""

       def verify_license(
           self,
           receipt_id: str,
           license_id: str,
       ) -> LicenseVerification:
           """Verify license is valid for receipt"""

   class LicenseType(Enum):
       VIEW_ONLY = "view_only"
       DERIVATIVE = "derivative"
       COMMERCIAL = "commercial"
       FULL_TRANSFER = "full_transfer"
   ```

3. **External Audit System Interface**
   ```python
   class AuditInterface:
       """Interface for external audit systems"""

       def export_for_audit(
           self,
           entries: List[LedgerEntry],
           audit_format: AuditFormat,
       ) -> AuditExport:
           """
           Export ledger entries for external audit.
           Supports multiple audit standards.
           """

       def generate_audit_proof(
           self,
           entry_ids: List[str],
           audit_scope: AuditScope,
       ) -> AuditProof:
           """
           Generate cryptographic proof for auditors.
           Proves existence and integrity without full disclosure.
           """

       def respond_to_audit_query(
           self,
           query: AuditQuery,
           disclosure_level: DisclosureLevel,
       ) -> AuditResponse:
           """Respond to specific audit queries"""

   class AuditFormat(Enum):
       ISO_27001 = "iso_27001"
       SOC2 = "soc2"
       GDPR = "gdpr"
       CUSTOM = "custom"
   ```

4. **Protocol Version Management**
   ```python
   class ProtocolCompatibility:
       """Manage compatibility across protocol versions"""

       SUPPORTED_PROTOCOLS = {
           "mp-01": ["1.0", "1.1"],
           "mp-02": ["1.0"],
       }

       def check_compatibility(
           self,
           protocol: str,
           version: str,
       ) -> CompatibilityResult:
           """Check if protocol version is supported"""

       def upgrade_receipt(
           self,
           receipt: EffortReceipt,
           target_version: str,
       ) -> EffortReceipt:
           """Upgrade receipt to newer protocol version"""
   ```

---

### 17.15 Enhanced Validation Criteria

**Source:** MP-02-spec.md §7

**Status:** Not implemented

**Description:** Detailed validation criteria for effort segments including linguistic coherence, conceptual progression, internal consistency, and synthesis detection.

**Implementation Plan:**

1. **Linguistic Coherence Assessor**
   ```python
   class LinguisticCoherenceAssessor:
       """Assess linguistic coherence of effort segments"""

       def assess_coherence(
           self,
           segment: EffortSegment,
       ) -> CoherenceScore:
           """
           Assess linguistic coherence across signals.

           Checks:
           - Grammatical consistency
           - Vocabulary stability
           - Topic continuity
           - Register consistency
           """

       def detect_discontinuities(
           self,
           segment: EffortSegment,
       ) -> List[Discontinuity]:
           """Identify points of linguistic discontinuity"""

   @dataclass
   class CoherenceScore:
       overall: float  # 0.0 - 1.0
       grammatical: float
       vocabulary: float
       topic: float
       register: float
       confidence: float
       discontinuities: List[Discontinuity]
   ```

2. **Conceptual Progression Analyzer**
   ```python
   class ConceptualProgressionAnalyzer:
       """Analyze conceptual development over time"""

       def analyze_progression(
           self,
           segment: EffortSegment,
       ) -> ProgressionAnalysis:
           """
           Track how concepts develop through the segment.

           Metrics:
           - Concept introduction rate
           - Concept refinement patterns
           - Depth vs breadth ratio
           - Iteration patterns (cycles of revision)
           """

       def identify_breakthroughs(
           self,
           segment: EffortSegment,
       ) -> List[Breakthrough]:
           """Identify potential conceptual breakthroughs"""

       def map_concept_evolution(
           self,
           segment: EffortSegment,
       ) -> ConceptMap:
           """Create map of concept evolution over time"""

   @dataclass
   class ProgressionAnalysis:
       progression_score: float  # 0.0 - 1.0
       introduction_rate: float  # New concepts per minute
       refinement_ratio: float  # Refinements vs new concepts
       depth_breadth_ratio: float
       iteration_count: int
       breakthroughs: List[Breakthrough]
   ```

3. **Internal Consistency Checker**
   ```python
   class InternalConsistencyChecker:
       """Check internal consistency of effort segments"""

       def check_consistency(
           self,
           segment: EffortSegment,
       ) -> ConsistencyReport:
           """
           Check for internal contradictions and consistency.

           Checks:
           - Temporal consistency (no future references)
           - Logical consistency (no contradictions)
           - Reference consistency (valid back-references)
           - Style consistency (uniform approach)
           """

       def find_contradictions(
           self,
           segment: EffortSegment,
       ) -> List[Contradiction]:
           """Identify potential contradictions in content"""

   @dataclass
   class ConsistencyReport:
       is_consistent: bool
       consistency_score: float
       temporal_issues: List[TemporalIssue]
       logical_issues: List[LogicalIssue]
       reference_issues: List[ReferenceIssue]
       style_issues: List[StyleIssue]
   ```

4. **Synthesis vs Duplication Detector**
   ```python
   class SynthesisDetector:
       """Detect synthesis vs duplication in effort"""

       def classify_content(
           self,
           segment: EffortSegment,
           reference_corpus: Optional[List[str]] = None,
       ) -> SynthesisClassification:
           """
           Classify content as synthesis or duplication.

           Classifications:
           - ORIGINAL: Novel synthesis
           - DERIVATIVE: Building on existing work
           - COMPILATION: Aggregating existing content
           - DUPLICATION: Copy with minimal change
           - UNKNOWN: Insufficient data to classify
           """

       def calculate_originality(
           self,
           segment: EffortSegment,
           reference_corpus: List[str],
       ) -> float:
           """Calculate originality score against corpus"""

       def detect_source_patterns(
           self,
           segment: EffortSegment,
       ) -> List[SourcePattern]:
           """Detect patterns suggesting external sources"""

   class SynthesisClassification(Enum):
       ORIGINAL = "original"
       DERIVATIVE = "derivative"
       COMPILATION = "compilation"
       DUPLICATION = "duplication"
       UNKNOWN = "unknown"
   ```

5. **Integrated Validation Pipeline**
   ```python
   class EnhancedValidator:
       """Enhanced validator combining all assessment criteria"""

       def __init__(self):
           self.coherence = LinguisticCoherenceAssessor()
           self.progression = ConceptualProgressionAnalyzer()
           self.consistency = InternalConsistencyChecker()
           self.synthesis = SynthesisDetector()

       def validate_segment(
           self,
           segment: EffortSegment,
           validation_level: ValidationLevel = ValidationLevel.STANDARD,
       ) -> EnhancedValidationResult:
           """
           Run full validation pipeline.

           Validation levels:
           - QUICK: Coherence only
           - STANDARD: Coherence + Progression
           - THOROUGH: All assessments
           - FORENSIC: Deep analysis with source detection
           """

       def generate_validation_summary(
           self,
           result: EnhancedValidationResult,
       ) -> str:
           """Generate deterministic summary per MP-02 §7 requirement"""

   class ValidationLevel(Enum):
       QUICK = "quick"
       STANDARD = "standard"
       THOROUGH = "thorough"
       FORENSIC = "forensic"
   ```

---

## 18. Implementation Priority

| Feature | Priority | Complexity | Dependencies |
|---------|----------|------------|--------------|
| Proof System (17.1) | High | Medium | None |
| Admin CLI (17.8) | High | Low | None |
| Explicit Revocation (17.9) | High | Low | None |
| Owner/Classification (17.3) | Medium | Medium | Learning Contracts |
| Failure Mode Handling (17.5) | Medium | Medium | None |
| Multi-Parent Aggregation (17.4) | Medium | Medium | None |
| Effort Receipt Protocol (17.2) | Medium | High | Proof System (17.1) |
| NatLangChain Compatibility (§16) | Medium | Medium | Proof System (17.1), MP-02 (17.2) |
| Boundary Daemon Integration (17.11) | Medium | Medium | Boundary Daemon module |
| Common Module Integration (17.12) | Low | Low | Common module |
| MP-02 Privacy & Agency (17.13) | Medium | High | MP-02 (17.2) |
| MP-02 External Compatibility (17.14) | Low | High | MP-02 (17.2), External protocols |
| Enhanced Validation Criteria (17.15) | Low | High | MP-02 (17.2) |
| NatLangChain Export (17.7) | Low | Medium | NatLangChain module, NLC Compatibility (§16) |
| Synth-Mind Integration (17.6) | Low | Medium | Synth-Mind module |
| Reusability Metric (17.10) | Low | Low | Migration needed |

### Implementation Phases

**Phase 1: Foundation (High Priority)**
1. Proof System (17.1) — Required for all verification features
2. Admin CLI (17.8) — Enables debugging and management
3. Explicit Revocation (17.9) — Completes status lifecycle

**Phase 2: Core Features (Medium Priority)**
4. Owner/Classification (17.3) — Requires Learning Contracts dependency
5. Multi-Parent Aggregation (17.4) — Enhances value composition
6. Failure Mode Handling (17.5) — Production safety

**Phase 3: Protocol Integration (Medium Priority)**
7. Effort Receipt Protocol (17.2) — Full MP-02 implementation
8. NatLangChain Compatibility (§16) — Anchoring preparation
9. Boundary Daemon Integration (17.11) — Direct interruption tracking

**Phase 4: External Integration (Low Priority)**
10. NatLangChain Export (17.7) — Production anchoring
11. Synth-Mind Integration (17.6) — Cognitive tier tracking
12. Reusability Metric (17.10) — Schema extension
13. Common Module Integration (17.12) — Shared utilities

**Phase 5: Advanced Features (Low Priority)**
14. MP-02 Privacy & Agency (17.13) — Signal encryption and consent
15. MP-02 External Compatibility (17.14) — Protocol adapters
16. Enhanced Validation Criteria (17.15) — Advanced effort validation

---

## 19. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024 | Initial specification |
| 0.1.1 | 2024 | Added implementation status and plans |
| 0.2.0 | 2025-12-19 | Added NatLangChain compatibility section, updated section numbering, verified implementation status against codebase |
| 0.3.0 | 2025-12-23 | Added unimplemented features from INTEGRATION.md and MP-02-spec.md: Boundary Daemon Integration (17.11), Common Module Integration (17.12), MP-02 Privacy & Agency Controls (17.13), MP-02 External Protocol Compatibility (17.14), Enhanced Validation Criteria (17.15). Added Phase 5 to implementation roadmap. |
