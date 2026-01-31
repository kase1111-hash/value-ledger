# Value Ledger

Evidentiary accounting layer for cognitive effort in the Agent-OS ecosystem. Records and preserves value from cognitive work (ideas, effort, time, novelty, failures, strategic insights) with cryptographic proofs.

## Quick Reference

```bash
# Install
pip install -e ".[dev]"          # Development
pip install -e ".[embeddings]"   # With ML novelty detection

# Test
pytest tests/ -v                 # Run all 211 tests
pytest tests/ --cov=value_ledger --cov-report=html  # With coverage (70% min)

# Code quality
black value_ledger/ tests/       # Format (line-length: 100)
ruff check value_ledger/         # Lint
mypy value_ledger/               # Type check
pre-commit run --all-files       # All hooks

# CLI
value-ledger stats               # Show statistics
value-ledger query --intent-id <id>  # Query entries
value-ledger export json --output ledger.json
value-ledger demo                # Interactive demo
```

## Architecture

### Core Concepts

- **ValueVector**: 7-dimensional value tracking (t=time, e=effort, n=novelty, f=failure, r=risk, s=strategy, u=reusability)
- **LedgerEntry**: Immutable record with Merkle proof, linked to IntentLog via `intent_id`
- **Append-Only JSONL**: Local storage in `ledger.jsonl`, no external dependencies

### Module Overview

| Module | Purpose |
|--------|---------|
| `core.py` | ValueLedger, ValueVector, LedgerEntry, MerkleTree |
| `cli.py` | 7 CLI commands (stats, query, show, export, revoke, proof, demo) |
| `heuristics.py` | 7 scoring algorithms with embedding-based novelty |
| `receipt.py` | MP-02 Effort Receipt Protocol |
| `privacy.py` | Consent management, Fernet encryption, signal filtering |
| `validation.py` | 6 validation criteria (coherence, progression, consistency, authenticity, completeness, temporal) |
| `security.py` | SIEM integration, rate limiting, `@protected_operation` decorator |
| `integration.py` | IntentLog event connector with rate limiting |
| `interruption.py` | Boundary Daemon integration for effort multipliers |
| `synth_mind.py` | Cognitive tier tracking (System1/2, Meta, Executive) |
| `natlangchain.py` | NatLangChain export with SSRF protection |
| `compatibility.py` | MP-01/MP-02 protocol interoperability |

### Data Flow

```
IntentLog Events → IntentLogConnector → HeuristicEngine → ValueVector
                                              ↓
    Boundary Daemon interruptions ──────→ effort multipliers
    Synth-Mind cognitive tiers ──────────→ tier weighting
                                              ↓
                               ValueLedger.accrue() → JSONL storage
                                              ↓
                               Export (JSON/CSV/MP-02 Receipt/NatLangChain)
```

## Key Patterns

### Exception Hierarchy
```python
ValueLedgerError (base)
├── ValidationError
├── StorageError
├── CryptographyError
├── IntegrationError
├── SecurityError
└── RateLimitError
```

Each exception includes: `message`, `code`, `details`, `timestamp`

### Security Patterns
- Path validation prevents traversal attacks (`_validate_ledger_path()`)
- SSRF protection in natlangchain.py (`_validate_url()`)
- Null byte injection checks
- Rate limiting via token bucket
- Security events logged in CEF format

### Integration Hooks
- `BoundaryDaemonListener` - interruption events
- `SynthMindHook` - cognitive tier changes
- `IntentLogConnector` - intent events with rate limiting

## Conventions

- **Line length**: 100 characters (Black)
- **Type hints**: Required throughout (mypy enforced)
- **Test coverage**: 70% minimum
- **Logging**: `logging` module with `value_ledger.*` namespace
- **Soft revocation**: Entries frozen, never deleted (audit compliance)
- **Lazy imports**: Optional deps (sentence-transformers) loaded on demand

## Docker

```bash
docker compose up dev         # Development with live reload
docker compose up test        # Run tests
docker compose run cli demo   # CLI
```

## Important Files

- `pyproject.toml` - Package config, dependencies, tool settings
- `docs/specs-sheet.md` - Complete specification
- `docs/user-manual.md` - CLI and API usage guide
- `.pre-commit-config.yaml` - Pre-commit hooks (Black, Ruff, Bandit, detect-secrets)
- `.github/workflows/ci.yml` - CI pipeline (Python 3.9-3.12 matrix)
