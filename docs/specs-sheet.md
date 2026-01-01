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

| Module | Status | Implementation | Lines |
|--------|--------|----------------|-------|
| Core Ledger | ✅ Implemented | `core.py` | 1,196 |
| CLI | ✅ Implemented | `cli.py` | 533 |
| Heuristics | ✅ Implemented | `heuristics.py` | 357 |
| IntentLog | ✅ Implemented | `integration.py` | 135 |
| Boundary Daemon | ✅ Implemented | `interruption.py` | 700 |
| Synth-Mind | ✅ Implemented | `synth_mind.py` | 403 |
| NatLangChain | ✅ Implemented | `natlangchain.py` | 721 |
| MP-02 Receipts | ✅ Implemented | `receipt.py` | 647 |
| Privacy & Agency | ✅ Implemented | `privacy.py` | 793 |
| Enhanced Validation | ✅ Implemented | `validation.py` | 994 |
| Compatibility | ✅ Implemented | `compatibility.py` | 966 |
| Memory Vault | ⚠️ Stubbed | `memory_vault_hook.py` | 165 |
| Learning Contracts | ⚠️ Stubbed | Consent checking | - |
| Agent-OS Core | ✅ Ready | Factory functions | - |

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

## Synth-Mind Integration

### Status: ✅ Implemented

Cognitive tier tracking for valuing different types of mental effort.

### Cognitive Tiers

| Tier | Name | Description |
|------|------|-------------|
| 1 | SYSTEM1 | Fast, intuitive, pattern-matching |
| 2 | SYSTEM2 | Deliberate, analytical, logical |
| 3 | META | Self-reflection, strategy evaluation |
| 4 | EXECUTIVE | Goal management, resource allocation |

### Key Classes

- `CognitiveTierContext` - Tier history and metrics
- `CognitiveTierScorer` - Values tier usage patterns
- `TierChangeEvent` - Tracks tier transitions
- `SynthMindHook` - Integration point for Synth-Mind module

### Usage Example

```python
from value_ledger import CognitiveTierContext, CognitiveTierScorer

context = CognitiveTierContext()
context.record_tier_change(CognitiveTier.SYSTEM1, CognitiveTier.SYSTEM2)

scorer = CognitiveTierScorer()
score = scorer.score(context)
```

---

## Privacy & Agency Controls

### Status: ✅ Implemented

Per MP-02 §12, implements privacy controls and human agency.

### Privacy Levels

| Level | Description |
|-------|-------------|
| PUBLIC | No restrictions |
| INTERNAL | Organization-only access |
| RESTRICTED | Need-to-know basis |
| CONFIDENTIAL | Encrypted, explicit consent required |

### Consent Management

| Status | Description |
|--------|-------------|
| GRANTED | Consent given (may have time bounds) |
| PENDING | Awaiting consent decision |
| REVOKED | Consent withdrawn |

### Revocation Scopes

| Scope | Effect |
|-------|--------|
| SINGLE | Revoke single receipt only |
| INTENT | Revoke all receipts for an intent |
| ALL_FUTURE | Revoke all future observation rights |

### Key Classes

- `PrivacyLevel` - Enumeration of privacy levels
- `ConsentStatus` - Consent state tracking
- `RevocationScope` - Scope of revocation actions
- `ObservationConsent` - Consent record with time bounds
- `SignalEncryptor` - Fernet-based encryption (PBKDF2HMAC key derivation)
- `PrivacyFilter` - Content filtering based on privacy levels
- `ConsentRegistry` - Manages consent records
- `AgencyController` - Controls revocation and data access

### Usage Example

```python
from value_ledger import SignalEncryptor, PrivacyFilter, ConsentRegistry

# Encrypt sensitive signals
encryptor = SignalEncryptor(password="secure_password")
encrypted = encryptor.encrypt("sensitive content")
decrypted = encryptor.decrypt(encrypted)

# Filter content by privacy level
filter = PrivacyFilter()
filtered = filter.filter_content(receipt, PrivacyLevel.INTERNAL)

# Manage consent
registry = ConsentRegistry()
registry.grant_consent(human_id="user-123", scope="observation")
```

---

## Enhanced Validation Criteria

### Status: ✅ Implemented

Per MP-02 §7, validators MAY assess multiple dimensions of effort.

### Validation Criteria

| Criterion | Description |
|-----------|-------------|
| Coherence | Linguistic consistency and structure |
| Progression | Conceptual development over time |
| Consistency | Internal agreement across signals |
| Authenticity | Indicators of synthesis vs duplication |
| Completeness | Coverage and comprehensiveness |
| Temporal | Time sequence validity |

### Key Classes

- `CoherenceScore` - Linguistic structure assessment
- `ProgressionScore` - Conceptual advancement detection
- `ConsistencyScore` - Internal agreement checking
- `AuthenticityScore` - Synthesis indicators
- `CompletenessScore` - Coverage assessment
- `EnhancedValidator` - Main validation orchestrator
- `ConsistencyChecker` - Signal agreement analysis
- `DuplicationDetector` - Plagiarism/synthesis detection
- `ConfidenceCalculator` - Confidence scoring

### Usage Example

```python
from value_ledger import EnhancedValidator

validator = EnhancedValidator()
result = validator.validate(effort_segment)

print(f"Coherence: {result.coherence.score}")
print(f"Progression: {result.progression.score}")
print(f"Authenticity: {result.authenticity.score}")
```

---

## MP-02 External Compatibility

### Status: ✅ Implemented

External protocol interoperability and licensing management.

### License Types

| Type | Description |
|------|-------------|
| EXCLUSIVE | Single licensee, no other grants |
| NON_EXCLUSIVE | Multiple licensees allowed |
| DELEGATION_ALLOWED | Licensee may sublicense (max depth: 3) |

### Audit Export Formats

| Format | Use Case |
|--------|----------|
| JSON-LD | Linked data applications |
| W3C Verifiable Credentials | Identity and credential systems |
| OpenTimestamps-style | Blockchain timestamping |
| Audit Logs | Compliance and auditing |

### Key Classes

- `MP01Proposal` - Negotiation proposal format
- `MP01Ratification` - Ratification method support
- `LicenseManager` - Grant, revoke, delegate licenses
- `LicenseType` - License type enumeration
- `DelegationRecord` - License delegation tracking
- `AuditExporter` - Export to multiple formats
- `ProtocolAdapter` - Cross-protocol interoperability

### Usage Example

```python
from value_ledger import LicenseManager, AuditExporter

# License management
manager = LicenseManager()
license_id = manager.grant_license(
    receipt_id="receipt-123",
    licensee="org-456",
    license_type=LicenseType.NON_EXCLUSIVE
)

# Export to audit formats
exporter = AuditExporter()
json_ld = exporter.to_json_ld(receipt)
w3c_vc = exporter.to_w3c_vc(receipt)
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

## Codebase Statistics

- **Total modules:** 13 Python files
- **Total code:** ~7,825 lines
- **Test files:** 5 test modules
- **Dependencies:** pydantic (required), cryptography (required), sentence-transformers (optional), torch (optional)

## Implemented Features (v0.5.0)

| Feature | Module | Status | Lines |
|---------|--------|--------|-------|
| Core Ledger & Proofs | `core.py` | ✅ Implemented | 1,196 |
| Command-Line Interface | `cli.py` | ✅ Implemented | 533 |
| Heuristic Scoring (7 scorers) | `heuristics.py` | ✅ Implemented | 357 |
| IntentLog Integration | `integration.py` | ✅ Implemented | 135 |
| Boundary Daemon Integration | `interruption.py` | ✅ Implemented | 700 |
| Effort Receipt Protocol | `receipt.py` | ✅ Implemented | 647 |
| Privacy & Agency Controls | `privacy.py` | ✅ Implemented | 793 |
| Enhanced Validation | `validation.py` | ✅ Implemented | 994 |
| External Compatibility | `compatibility.py` | ✅ Implemented | 966 |
| NatLangChain Export | `natlangchain.py` | ✅ Implemented | 721 |
| Synth-Mind Integration | `synth_mind.py` | ✅ Implemented | 403 |
| Memory Vault Hook | `memory_vault_hook.py` | ⚠️ Stubbed | 165 |

## Security Features

| Feature | Implementation | Status |
|---------|----------------|--------|
| Path Traversal Prevention | `_validate_ledger_path()` in `core.py` | ✅ Implemented |
| Null Byte Detection | Ledger path validation | ✅ Implemented |
| Sensitive Path Blocking | /etc, /proc, /sys, .ssh, .aws blocked | ✅ Implemented |
| SSRF Protection | `_validate_url()` in `natlangchain.py` | ✅ Implemented |
| Private IP Blocking | Loopback, private ranges, metadata services | ✅ Implemented |
| Signal Encryption | Fernet + PBKDF2HMAC in `privacy.py` | ✅ Implemented |

## Test Coverage

| Test File | Focus |
|-----------|-------|
| `test_validation.py` | Enhanced validation criteria |
| `test_compatibility.py` | MP-02 compatibility, licensing, audit |
| `test_interruption.py` | Interruption tracking |
| `test_privacy.py` | Privacy & consent controls |
| `test_e2e_simulation.py` | End-to-end workflows |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024 | Initial specification |
| 0.2.0 | 2024 | Added NatLangChain compatibility |
| 0.3.0 | 2024 | Added Boundary Daemon integration, MP-02 Privacy & Agency Controls |
| 0.4.0 | 2024 | Added MP-02 External Compatibility (MP-01, licensing, audit formats) |
| 0.5.0 | 2024 | Added Enhanced Validation Criteria, security hardening, consolidated spec docs |
