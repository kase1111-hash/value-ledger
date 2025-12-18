# Value Ledger Specification

## 1. Purpose

The Value Ledger is the **economic and evidentiary accounting layer** of the learning co-worker ecosystem. It assigns, accrues, and preserves value for cognitive work: ideas, failed paths, effort, novelty, and time.

If Memory Vault stores *what happened* and Learning Contracts govern *permission*, the Value Ledger answers:

> **“What is this worth, and how can I prove it existed?”**

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

| Unit | Meaning          |
| ---- | ---------------- |
| T    | Time (seconds)   |
| E    | Effort intensity |
| N    | Novelty          |
| F    | Failure density  |
| R    | Reusability      |
| S    | Strategic depth  |

Units are normalized to [0–1] except time.

---

## 5. Core Ledger Object Schema

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

---

## 6. Ledger Operations

### 6.1 Accrual

* Triggered by:

  * Memory creation
  * Intent completion
  * Explicit owner marking
* Automatic scoring permitted
* Owner may override scores

### 6.2 Aggregation

* Multiple ledger entries may be combined
* Aggregation never deletes originals
* Parent-child relationships preserved

### 6.3 Freezing

* Triggered by:

  * Contract expiration
  * Boundary downgrade
* Value retained, but no further accrual

### 6.4 Revocation

* Triggered by owner
* Value remains provable
* Asset becomes non-exploitable

---

## 7. Valuation Heuristics (Non-Binding)

The system may *suggest* value scores based on:

* Time × interruption resistance
* Novelty against prior memory corpus
* Failure density (paths ruled out per time)
* Compression ratio (raw → heuristic)

Suggestions are advisory only.

---

## 8. Interaction with IntentLog

* Every IntentLog entry may generate ledger entries
* Ledger binds value to *intent*, not just outcome
* Failed intents accrue value proportional to search

---

## 9. Interaction with Memory Vault

* Ledger stores hashes, never plaintext
* Classification mirrors memory classification
* Vault deletion does NOT delete ledger proof

---

## 10. Interaction with Learning Contracts

* Ledger creation requires valid contract
* Contract caps strategic depth scoring
* Revoked contracts freeze derived value

---

## 11. Threat Model

### Threats

* Overvaluation inflation
* Silent value loss
* Retroactive theft
* External coercion

### Mitigations

| Threat    | Mitigation              |
| --------- | ----------------------- |
| Inflation | Owner override + audits |
| Loss      | Append-only logs        |
| Theft     | Hash-based proofs       |
| Coercion  | Offline proofs          |

---

## 12. Proof & Export

The ledger can export:

* Time-stamped proof of existence
* Aggregated value summaries
* Merkle proofs for third parties

Exports never include content.

---

## 13. Failure Modes

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
