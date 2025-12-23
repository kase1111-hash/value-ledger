# Value Ledger Complete Specification

This document consolidates all specifications for the Value Ledger project:
- Core Specification (formerly specs.md)
- Cross-Repository Integration Guide (formerly INTEGRATION.md)
- MP-02 Proof-of-Effort Receipt Protocol (formerly MP-02-spec.md)

---

# Part 1: Core Specification

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
| R    | Risk exposure    | ✅ Implemented        |
| S    | Strategic depth  | ✅ Implemented        |
| U    | Reusability      | ✅ Implemented        |

Units are normalized to [0–1] except time.

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
    "risk": 0.0,
    "strategic_depth": 0.0,
    "reusability": 0.0
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
| `owner` | ✅ Implemented | Owner tracking with access control |
| `source.memory_id` | ✅ Implemented | As `memory_hash` |
| `source.intent_id` | ✅ Implemented | As `intent_id` |
| `source.contract_id` | ✅ Implemented | As `contract_id` |
| `value_vector` | ✅ Implemented | 7 units (T/E/N/F/R/S/U) in `ValueVector` |
| `classification` | ✅ Implemented | 0-5 levels with access control |
| `status` | ✅ Implemented | active/frozen/revoked |
| `derivation.parent_ledger_ids` | ✅ Implemented | Multi-parent via `parent_ids` |
| `derivation.aggregation_rule` | ✅ Implemented | sum/max/weighted |
| `proof.content_hash` | ✅ Implemented | SHA-256 via `compute_content_hash()` |
| `proof.timestamp_proof` | ✅ Implemented | Via `compute_timestamp_proof()` |
| `proof.merkle_ref` | ✅ Implemented | Via `MerkleTree` class |

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

**Status:** ✅ Fully Implemented

* Multiple ledger entries may be combined
* Aggregation never deletes originals
* Parent-child relationships preserved

**Implementation:** `ValueLedger.aggregate_entries()` - supports multi-parent aggregation with `sum`, `max`, or `weighted` rules.

### 6.3 Freezing

**Status:** ✅ Implemented

* Triggered by:
  * Contract expiration
  * Boundary downgrade
* Value retained, but no further accrual

**Implementation:** Status field set to "frozen" via `aggregate_correction(freeze_parent=True)`

### 6.4 Revocation

**Status:** ✅ Fully Implemented

* Triggered by owner
* Value remains provable
* Asset becomes non-exploitable

**Implementation:** `ValueLedger.revoke()` method with `revoked_at`, `revoked_by`, `revocation_reason` fields, and optional cascade to children.

---

## 7. Valuation Heuristics (Non-Binding)

**Status:** ✅ Implemented

The system may *suggest* value scores based on:

* Time × interruption resistance → `TimeScorer`, `EffortScorer`
* Novelty against prior memory corpus → `NoveltyScorer` (embedding-based)
* Failure density (paths ruled out per time) → `FailureScorer`
* Reusability potential → `ReusabilityScorer`

Suggestions are advisory only.

**Implementation:** `HeuristicEngine` with 7 scorers in `heuristics.py`

---

## 8. Proof & Export

**Status:** ✅ Fully Implemented

The ledger can export:

* Time-stamped proof of existence
* Aggregated value summaries
* Merkle proofs for third parties

Exports never include content.

**Implementation:**
- `export_existence_proof()` - Complete proof with timestamps, Merkle proof, revocation info
- `get_merkle_proof()` - Returns leaf hash, proof path, and root
- `verify_entry_proof()` - Verification method
- CLI: `export` command with JSON/CSV/Merkle formats

---

## 9. Failure Modes

**Status:** ✅ Fully Implemented

* Clock drift → pause accrual
* Ambiguous source → deny entry
* Classification mismatch → freeze

**Implementation:**
- `ClockMonitor` - Detects clock drift, future timestamps, clock regression
- `SourceValidator` - Validates entries for integrity, classification compatibility
- `FailureModeHandler` - Unified handler with `safe_accrue()`, `safe_aggregate()`, health reporting

---

## 10. Threat Model

### Threats

* Overvaluation inflation
* Silent value loss
* Retroactive theft
* External coercion

### Mitigations

| Threat    | Mitigation              | Status |
| --------- | ----------------------- | ------ |
| Inflation | Owner override + audits | ✅ Implemented |
| Loss      | Append-only logs        | ✅ Implemented |
| Theft     | Hash-based proofs       | ✅ Implemented |
| Coercion  | Offline proofs          | ✅ Implemented |

---

## 11. Non-Goals

* Real-time pricing
* Market speculation
* Token issuance
* Automatic monetization

---

## 12. Design Constraint

> The world pays for outcomes.
> The ledger preserves proof of effort.

The Value Ledger exists for when those diverge.

---

# Part 2: Cross-Repository Integration Guide

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

## Integration Status Summary

| Module | Status | Implementation |
|--------|--------|----------------|
| IntentLog | ✅ Implemented | `integration.py` |
| Memory Vault | ⚠️ Stubbed | `memory_vault_hook.py` |
| Learning Contracts | ⚠️ Stubbed | Consent checking |
| Boundary Daemon | ✅ Implemented | `interruption.py` |
| Agent-OS Core | ✅ Ready | Factory functions |
| Common Module | ⚠️ Stubbed | Local implementations |
| Synth-Mind | ✅ Implemented | `synth_mind.py` |
| NatLangChain | ✅ Implemented | `natlangchain.py` |

---

## IntentLog Integration

### Required IntentLog Event Structure

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

### Usage Example

```python
from value_ledger import create_intentlog_listener

ledger_listener = create_intentlog_listener(ledger_path="ledger.jsonl")
ledger_listener.handle_event(intent_event)
```

---

## Boundary Daemon Integration

### Status: ✅ Implemented

Direct integration with Boundary Daemon for interruption tracking.

### Usage Example

```python
from value_ledger import create_boundary_daemon_hook

tracker, listener = create_boundary_daemon_hook()
tracker.start_session("intent-123")

listener.handle_boundary_event({
    "type": "notification_received",
    "intent_id": "intent-123",
    "source": "slack",
    "timestamp": time.time(),
})

summary = tracker.end_session("intent-123")
```

### Interruption Scoring

```python
interruption_factor = 1.0 + (weighted_interruptions * 0.35)
if interruptions > 10:
    interruption_factor += (interruptions - 10) * 0.1
```

---

## NatLangChain Integration

### Status: ✅ Implemented

Export ledger entries to NatLangChain format for blockchain anchoring.

### Key Classes

- `NLCRecord` - NatLangChain-compatible record format
- `NLCClient` - REST API client for NatLangChain nodes
- `NatLangChainExporter` - Converts ledger entries to NLC format
- `ProofOfUnderstandingValidator` - Generates validator summaries

### Usage Example

```python
from value_ledger import NatLangChainExporter, NLCClient

client = NLCClient(base_url="http://localhost:5000")
exporter = NatLangChainExporter(client)

record = exporter.to_nlc_format(ledger_entry)
result = exporter.anchor_to_chain(record)
```

---

# Part 3: MP-02 Proof-of-Effort Receipt Protocol

## Purpose

MP-02 defines the protocol by which human intellectual effort is observed, validated, and recorded as cryptographically verifiable receipts on NatLangChain.

The protocol establishes a primitive that is:

- Verifiable without trusting the issuer
- Human-readable over long time horizons
- Composable with negotiation, licensing, and settlement protocols

MP-02 does not assert value, ownership, or compensation. It asserts that effort occurred, with traceable provenance.

---

## Design Principles

1. **Process Over Artifact** — Effort is validated as a process unfolding over time, not a single output.
2. **Continuity Matters** — Temporal progression is a primary signal of genuine work.
3. **Receipts, Not Claims** — The protocol records evidence, not conclusions about value.
4. **Model Skepticism** — LLM assessments are advisory and must be reproducible.
5. **Partial Observability** — Uncertainty is preserved, not collapsed.

---

## Definitions

### Effort
A temporally continuous sequence of human cognitive activity directed toward an intelligible goal.

### Signal
A raw observable trace of effort, including but not limited to:
- Voice transcripts
- Text edits
- Command history
- Structured tool interaction

### Effort Segment
A bounded time slice of signals treated as a unit of analysis.

### Receipt
A cryptographic record attesting that a specific effort segment occurred, with references to its source signals and validation metadata.

---

## Actors

### Human Worker
The individual whose effort is being recorded.

### Observer
A system component responsible for capturing raw signals.

### Validator
An LLM-assisted process that analyzes effort segments for coherence and progression.

### Ledger
An append-only system that anchors receipts and their hashes.

---

## Receipt Structure

Each Effort Receipt MUST include:

- Receipt ID
- Time bounds
- Hashes of referenced signals
- Deterministic effort summary
- Validation metadata
- Observer and Validator identifiers

**Implementation:** `value_ledger/receipt.py`

```python
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
    observation_gaps: List[Tuple[float, float]]
    conflicting_validations: List[str]
    suspected_manipulation: bool
    is_incomplete: bool
```

---

## Observer Requirements

Observers MUST:
- Time-stamp all signals
- Preserve ordering
- Disclose capture modality

Observers MUST NOT:
- Alter raw signals
- Infer intent beyond observed data

---

## Validator Requirements

Validators MUST:
- Produce deterministic summaries
- Disclose model identity and version
- Preserve dissent and uncertainty

Validators MUST NOT:
- Declare effort as valuable
- Assert originality or ownership
- Collapse ambiguous signals into certainty

---

## Verification

A third party MUST be able to:

- Recompute receipt hashes
- Inspect validation metadata
- Confirm ledger inclusion

Trust in the Observer or Validator is not required.

---

## Privacy and Agency

- Raw signals MAY be encrypted or access-controlled
- Receipts MUST not expose raw content by default
- Humans MAY revoke future observation

Past receipts remain immutable.

---

## Canonical Rule

> If effort cannot be independently verified as having occurred over time, it must not be capitalized.

---

# Part 4: Implementation Status

## Completed Features (v0.3.1)

| Feature | Module | Status |
|---------|--------|--------|
| Proof System | `core.py` | ✅ Implemented |
| Effort Receipt Protocol | `receipt.py` | ✅ Implemented |
| Owner & Classification | `core.py` | ✅ Implemented |
| Multi-Parent Aggregation | `core.py` | ✅ Implemented |
| Failure Mode Handling | `core.py` | ✅ Implemented |
| Synth-Mind Integration | `synth_mind.py` | ✅ Implemented |
| NatLangChain Export | `natlangchain.py` | ✅ Implemented |
| Admin CLI | `cli.py` | ✅ Implemented |
| Explicit Revocation | `core.py` | ✅ Implemented |
| Reusability Metric | `heuristics.py` | ✅ Implemented |
| Boundary Daemon Integration | `interruption.py` | ✅ Implemented |
| MP-02 Privacy & Agency Controls | `privacy.py` | ✅ Implemented |
| MP-02 External Compatibility | `compatibility.py` | ✅ Implemented |

## Pending Features

| Feature | Priority | Status |
|---------|----------|--------|
| Common Module Integration | Low | Awaiting common module |
| Enhanced Validation Criteria | Low | Not implemented |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024 | Initial specification |
| 0.1.1 | 2024 | Added implementation status and plans |
| 0.2.0 | 2025-12-19 | Added NatLangChain compatibility section |
| 0.3.0 | 2025-12-23 | Added Boundary Daemon, Common Module, MP-02 Privacy, External Compatibility, Enhanced Validation |
| 0.3.1 | 2025-12-23 | Updated all implementation statuses to reflect actual code state |
| 0.4.0 | 2025-12-23 | Implemented MP-02 Privacy & Agency Controls (consent, encryption, filtering)
| 0.5.0 | 2025-12-23 | Implemented MP-02 External Compatibility (MP-01, licensing, audit formats)
