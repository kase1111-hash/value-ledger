# User Manual

## Overview

The Value Ledger is an accounting layer for cognitive effort. It records meta-value from cognitive work—ideas, effort, time, novelty, failures, and strategic insights—even when outcomes are unsuccessful.

## Command-Line Interface

### Basic Usage

```bash
python -m value_ledger.cli <command> [options]
```

### Global Options

| Option | Description |
|--------|-------------|
| `--ledger <path>` | Path to ledger file (default: `ledger.jsonl`) |
| `--help` | Show help message |

---

## Commands

### stats

Show ledger statistics.

```bash
python -m value_ledger.cli stats
```

Output includes:
- Total entries and unique intents
- Status breakdown (active/frozen/revoked)
- Value summary by dimension (T/E/N/F/R/S/U)
- Time range of entries
- Top intents by value

### query

Query ledger entries with filters.

```bash
python -m value_ledger.cli query [options]
```

| Option | Description |
|--------|-------------|
| `--intent-id <id>` | Filter by intent ID |
| `--status <status>` | Filter by status (active/frozen/revoked) |
| `--since <timestamp>` | Filter entries after Unix timestamp |
| `--limit <n>` | Limit results (default: 50) |

**Examples:**

```bash
# Show active entries
python -m value_ledger.cli query --status active

# Show entries for specific intent
python -m value_ledger.cli query --intent-id intent-123 --limit 10

# Show recent entries
python -m value_ledger.cli query --since 1703300000
```

### show

Show detailed information for a single entry.

```bash
python -m value_ledger.cli show <entry_id>
```

Displays:
- Basic info (intent ID, timestamp, status)
- Value vector breakdown
- Proof data (content hash, timestamp proof, Merkle reference)
- Revocation info (if revoked)
- Lineage (parent entries)
- Metadata

**Note:** You can use a prefix of the entry ID for convenience.

### export

Export ledger data to file.

```bash
python -m value_ledger.cli export <output_path> [options]
```

| Option | Description |
|--------|-------------|
| `--format <fmt>` | Format: json, csv, merkle (default: json) |
| `--intent-id <id>` | Filter by intent ID |
| `--include-proofs` | Include Merkle proofs in JSON export |

**Examples:**

```bash
# Export to JSON
python -m value_ledger.cli export backup.json

# Export to CSV
python -m value_ledger.cli export report.csv --format csv

# Export Merkle tree data
python -m value_ledger.cli export merkle.json --format merkle

# Export with proofs
python -m value_ledger.cli export proof.json --include-proofs
```

### revoke

Revoke an entry.

```bash
python -m value_ledger.cli revoke <entry_id> [options]
```

| Option | Description |
|--------|-------------|
| `--reason <text>` | Reason for revocation |
| `--by <identifier>` | Who is revoking |
| `--cascade` | Also revoke child entries |

**Example:**

```bash
python -m value_ledger.cli revoke abc123 --reason "Duplicate entry" --by admin
```

### proof

Export existence proof for an entry.

```bash
python -m value_ledger.cli proof <entry_id>
```

Returns JSON proof including:
- Entry ID and timestamp
- Content hash
- Merkle proof path
- Merkle root

### demo

Run demonstration with sample data.

```bash
python -m value_ledger.cli demo
```

Creates sample entries in `demo_ledger.jsonl` and displays statistics.

---

## Python API

### Creating a Ledger

```python
from value_ledger import ValueLedger

ledger = ValueLedger("my_ledger.jsonl")
```

### Accruing Value

```python
entry_id = ledger.accrue(
    intent_id="task-001",
    initial_vector={
        "t": 8.0,   # Time
        "e": 6.5,   # Effort
        "n": 7.0,   # Novelty
        "f": 3.0,   # Failure density
        "r": 2.0,   # Risk
        "s": 5.0,   # Strategic depth
    },
    memory_hash="sha256:abc123...",
    metadata={"task": "exploring architecture"},
    content_for_proof="Content for hashing",
)
```

### Accruing with Heuristics

Let the system auto-score based on context:

```python
from value_ledger.heuristics import ScoringContext

context = ScoringContext(
    duration_seconds=1800,
    interruptions=5,
    keystrokes=1200,
    outcome_success=True,
    risk_level=0.3,
    human_reasoning="Explored novel approach...",
)

entry_id = ledger.accrue_with_heuristics(
    intent_id="task-002",
    context=context,
)
```

### Aggregating Entries

Combine multiple entries:

```python
aggregate_id = ledger.aggregate_entries(
    parent_ids=["entry-1", "entry-2", "entry-3"],
    rule="weighted",
    weights=[0.5, 0.3, 0.2],
)
```

### Querying

```python
# Get all entries
entries = ledger.entries

# Get single entry
entry = ledger.get_entry(entry_id)

# Get entry chain (lineage)
chain = ledger.get_chain(entry_id)
```

### Exporting Proofs

```python
# Existence proof
proof = ledger.export_existence_proof(entry_id)

# Merkle proof
merkle_proof = ledger.get_merkle_proof(entry_id)

# Merkle root
root = ledger.get_merkle_root()
```

### Revoking Entries

```python
ledger.revoke(
    entry_id=entry_id,
    reason="No longer valid",
    revoked_by="admin",
    revoke_children=True,
)
```

---

## Value Dimensions

| Unit | Name | Range | Description |
|------|------|-------|-------------|
| T | Time | seconds | Duration of effort |
| E | Effort | 0-1 | Effort intensity |
| N | Novelty | 0-1 | Uniqueness vs prior work |
| F | Failure | 0-1 | Failed paths explored |
| R | Risk | 0-1 | Risk exposure |
| S | Strategy | 0-1 | Strategic depth |
| U | Reusability | 0-1 | Potential for reuse |

---

## IntentLog Integration

Connect to IntentLog events:

```python
from value_ledger import create_intentlog_listener

listener = create_intentlog_listener(ledger_path="ledger.jsonl")

# Handle events from IntentLog
listener.handle_event(intent_event)
```

---

## Boundary Daemon Integration

Track interruptions during work sessions:

```python
from value_ledger import create_boundary_daemon_hook

tracker, listener = create_boundary_daemon_hook()

# Start tracking
tracker.start_session("intent-123")

# Handle boundary events
listener.handle_boundary_event({
    "type": "notification_received",
    "intent_id": "intent-123",
    "source": "slack",
    "timestamp": time.time(),
})

# End session and get summary
summary = tracker.end_session("intent-123")
```

---

## NatLangChain Export

Export to NatLangChain format:

```python
from value_ledger import NatLangChainExporter, NLCClient

client = NLCClient(base_url="http://localhost:5000")
exporter = NatLangChainExporter(client)

record = exporter.to_nlc_format(ledger_entry)
result = exporter.anchor_to_chain(record)
```

---

## Synth-Mind Integration

Track cognitive tier changes during work sessions:

```python
from value_ledger import CognitiveTierContext, CognitiveTierScorer, CognitiveTier

# Create tier context
context = CognitiveTierContext()

# Record tier changes
context.record_tier_change(CognitiveTier.SYSTEM1, CognitiveTier.SYSTEM2)
context.record_tier_change(CognitiveTier.SYSTEM2, CognitiveTier.META)

# Score the cognitive effort
scorer = CognitiveTierScorer()
score = scorer.score(context)

# Access tier history
print(f"Tier changes: {len(context.tier_changes)}")
print(f"Time in META tier: {context.time_in_tier(CognitiveTier.META)}")
```

### Cognitive Tiers

| Tier | Name | Description |
|------|------|-------------|
| 1 | SYSTEM1 | Fast, intuitive, pattern-matching |
| 2 | SYSTEM2 | Deliberate, analytical, logical |
| 3 | META | Self-reflection, strategy evaluation |
| 4 | EXECUTIVE | Goal management, resource allocation |

---

## Privacy & Agency Controls

Manage privacy levels, consent, and encryption:

```python
from value_ledger import (
    SignalEncryptor,
    PrivacyFilter,
    ConsentRegistry,
    AgencyController,
    PrivacyLevel,
    RevocationScope,
)

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
registry.revoke_consent(human_id="user-123", scope="observation")

# Control agency
controller = AgencyController(registry)
controller.revoke_observation(
    human_id="user-123",
    scope=RevocationScope.ALL_FUTURE,
    reason="User requested full revocation"
)
```

### Privacy Levels

| Level | Description |
|-------|-------------|
| PUBLIC | No restrictions |
| INTERNAL | Organization-only access |
| RESTRICTED | Need-to-know basis |
| CONFIDENTIAL | Encrypted, explicit consent required |

---

## Enhanced Validation

Validate effort segments with multiple criteria:

```python
from value_ledger import (
    EnhancedValidator,
    ConsistencyChecker,
    DuplicationDetector,
    ConfidenceCalculator,
)

# Create validator
validator = EnhancedValidator()

# Validate effort segment
result = validator.validate(effort_segment)

# Access individual scores
print(f"Coherence: {result.coherence.score}")
print(f"Progression: {result.progression.score}")
print(f"Consistency: {result.consistency.score}")
print(f"Authenticity: {result.authenticity.score}")
print(f"Completeness: {result.completeness.score}")

# Check for duplicates
detector = DuplicationDetector()
is_duplicate = detector.check(signal_content)

# Calculate confidence
calculator = ConfidenceCalculator()
confidence = calculator.calculate(validation_result)
```

### Validation Criteria

| Criterion | Description |
|-----------|-------------|
| Coherence | Linguistic consistency and structure |
| Progression | Conceptual development over time |
| Consistency | Internal agreement across signals |
| Authenticity | Indicators of synthesis vs duplication |
| Completeness | Coverage and comprehensiveness |

---

## License Management

Manage licenses for effort receipts:

```python
from value_ledger import LicenseManager, LicenseType, AuditExporter

# Create license manager
manager = LicenseManager()

# Grant a license
license_id = manager.grant_license(
    receipt_id="receipt-123",
    licensee="org-456",
    license_type=LicenseType.NON_EXCLUSIVE,
)

# Delegate a license
delegation_id = manager.delegate_license(
    license_id=license_id,
    delegatee="sub-org-789",
)

# Revoke a license
manager.revoke_license(license_id, reason="Contract ended")

# Export to audit formats
exporter = AuditExporter()
json_ld = exporter.to_json_ld(receipt)
w3c_vc = exporter.to_w3c_vc(receipt)
opentimestamps = exporter.to_opentimestamps(receipt)
```

### License Types

| Type | Description |
|------|-------------|
| EXCLUSIVE | Single licensee, no other grants allowed |
| NON_EXCLUSIVE | Multiple licensees allowed |
| DELEGATION_ALLOWED | Licensee may sublicense (max depth: 3) |

---

## Effort Receipts (MP-02)

Create and manage effort receipts:

```python
from value_ledger import (
    ReceiptBuilder,
    DefaultObserver,
    DefaultValidator,
    EffortSignal,
    SignalType,
)

# Create observer and validator
observer = DefaultObserver(observer_id="obs-001")
validator = DefaultValidator(validator_id="val-001")

# Observe signals
signal = observer.observe(
    content="User typed code implementation",
    signal_type=SignalType.KEYSTROKE,
)

# Build receipt
builder = ReceiptBuilder()
receipt = (
    builder
    .add_signal(signal)
    .set_time_bounds(start=1700000000, end=1700003600)
    .set_observer(observer.observer_id)
    .set_validator(validator.validator_id)
    .build()
)

# Validate receipt
validation = validator.validate(receipt)
```

### Signal Types

| Type | Description |
|------|-------------|
| VOICE | Voice transcripts |
| TEXT | Text edits |
| COMMAND | Command history |
| TOOL | Structured tool interaction |
| KEYSTROKE | Keyboard input |
| NAVIGATION | App/page navigation |

---

## Best Practices

1. **Regular Backups**: Export ledger periodically using `export` command
2. **Use Intent IDs**: Always provide meaningful intent IDs for tracking
3. **Include Metadata**: Add relevant metadata for future reference
4. **Revoke Properly**: Use revocation with reasons, don't delete entries
5. **Verify Proofs**: Use `proof` command to verify entry existence
6. **Manage Consent**: Always check consent before observing signals
7. **Encrypt Sensitive Data**: Use SignalEncryptor for confidential content
8. **Validate Effort**: Use EnhancedValidator for receipt quality assurance

## Troubleshooting

### Empty Query Results

- Check if ledger file exists and has entries
- Verify filter criteria (status, intent-id)
- Try broader filters or remove limits

### Proof Verification Fails

- Ensure entry hasn't been modified
- Verify Merkle tree integrity with `export --format merkle`

### Performance Issues

- Large ledgers may be slow; use `--limit` in queries
- Consider periodic aggregation of related entries
