# Value Ledger

[![CI](https://github.com/kase1111-hash/value-ledger/actions/workflows/ci.yml/badge.svg)](https://github.com/kase1111-hash/value-ledger/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-0.1.0--alpha.1-orange.svg)](https://github.com/kase1111-hash/value-ledger/releases)

**Cognitive work accounting and effort attribution ledger for the Agent-OS ecosystem**

> ⚠️ **Alpha Release**: This is v0.1.0-alpha.1. APIs may change before stable release. Feedback welcome!

The Value Ledger is an idea value tracking system that records and preserves value for cognitive work — including ideas, effort, time, novelty, failures, and strategic insights — even when outcomes are unsuccessful. It enables thought work valuation and novelty measurement in human-AI collaboration contexts.

> *In a world that pays for results, the Value Ledger provides verifiable proof of effort and creative work attribution.*

## What Problem Does This Solve?

How do you value cognitive work? How do you track idea contributions and measure thinking effort? The Value Ledger answers these questions by providing:

- **Idea contribution accounting** — Record and track who contributed what ideas
- **Cognitive effort ledger** — Measure creative work value even when projects fail
- **Attribution for ideas** — Fair credit distribution in collaborative work
- **Intellectual labor economics** — A framework for valuing thought work

## Quick Start

```bash
# Install
pip install -e .

# Run demo
python -m value_ledger.cli demo

# Or use the CLI directly
value-ledger stats
```

### Docker

```bash
# Run with Docker
docker compose up value-ledger

# Run tests in container
docker compose up test
```

## Overview

This module is part of the Agent-OS ecosystem — a natural-language-native operating system for autonomous AI agents. The Value Ledger serves as the dedicated accounting layer, tracking meta-value derived from cognitive processes without storing sensitive content.

**Key Principles:**
- Prove existence and effort of cognitive work
- Support consent-based boundaries (via Learning Contracts)
- Bind value to human intent (via IntentLog)
- Reference encrypted memories (via Memory Vault)
- Enable offline-first, owner-centric valuation

**Important:** This ledger is not a cryptocurrency, token system, or speculative market. It is purely evidentiary — designed for proof, audit, and fair attribution in human-AI collaboration.

## Architecture

The Value Ledger comprises 14 Python modules (~12,000 lines of code):

| Module | Purpose |
|--------|---------|
| `core.py` | Main ValueLedger class, entry management, Merkle proofs |
| `cli.py` | Command-line interface (7 commands) |
| `heuristics.py` | 7 scoring algorithms with embedding-based novelty |
| `integration.py` | IntentLog event-driven integration |
| `interruption.py` | Boundary Daemon interruption tracking |
| `receipt.py` | MP-02 Effort Receipt Protocol |
| `privacy.py` | Privacy controls, consent management, encryption |
| `validation.py` | Enhanced validation criteria (6 assessment types) |
| `compatibility.py` | MP-01 negotiation, licensing, audit export |
| `natlangchain.py` | NatLangChain export with SSRF protection |
| `synth_mind.py` | Cognitive tier tracking (4 tiers) |
| `security.py` | Boundary-SIEM/Daemon integration, error handling |
| `memory_vault_hook.py` | Memory Vault integration (stubbed) |

## Features

| Category | Features |
|----------|----------|
| **Core** | 7-dimensional value vector (T/E/N/F/R/S/U), Merkle tree proofs, multi-parent aggregation |
| **Scoring** | 7 heuristic scorers including embedding-based novelty (`all-MiniLM-L6-v2`) |
| **CLI** | 7 commands: stats, query, show, export, revoke, proof, demo |
| **Security** | SSRF protection, path traversal prevention, Fernet encryption, SIEM integration |
| **Protocols** | MP-02 Effort Receipts, MP-01 negotiation, W3C Verifiable Credentials |
| **Integrations** | IntentLog, Boundary Daemon, Boundary-SIEM, Synth-Mind, NatLangChain |

## Installation

### Requirements

- Python >= 3.9
- pip (Python package manager)

### From Source

```bash
# Clone the repository
git clone https://github.com/kase1111-hash/value-ledger.git
cd value-ledger

# Install in editable mode
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With embedding support (for novelty scoring)
pip install -e ".[embeddings]"
```

### Windows

```batch
:: Run assembly script (creates venv, installs deps, runs tests)
assemble.bat

:: Start the CLI
start.bat demo
```

### Docker

```bash
# Production image
docker build --target production -t value-ledger .

# Development with live reload
docker compose up dev

# Run tests
docker compose up test
```

## Usage

### CLI

```bash
# Show ledger statistics
value-ledger stats

# Query entries
value-ledger query --intent "research-task"

# Export to JSON
value-ledger export json --output ledger.json

# Generate Merkle proof
value-ledger proof <entry-id>

# Run demo
value-ledger demo
```

### Python API

```python
from value_ledger import ValueLedger, ValueVector

# Create ledger
ledger = ValueLedger("ledger.jsonl")

# Add entry with value vector
entry = ledger.append(
    intent_id="research-task-001",
    value_vector=ValueVector(t=1.0, e=0.8, n=0.5, f=0.0, r=0.3, s=0.2, u=0.4),
    content_hash="sha256:...",
)

# Query entries
results = ledger.query(intent_id="research-task-001")

# Generate Merkle proof
proof = ledger.generate_merkle_proof(entry.id)
```

### Security Integration

```python
from value_ledger import init_security, protected_operation, SecurityEventType

# Initialize security (connects to Boundary-SIEM and Daemon)
security = init_security(
    siem_endpoint="http://siem:8080/api/v1/events",
    daemon_socket="/var/run/boundary-daemon/api.sock",
)

# Use decorator for protected operations
@protected_operation("ledger_write", resource_arg="path")
def write_entry(path: str, data: dict):
    ...

# Report security events
security.report_security_event(
    SecurityEventType.ENTRY_CREATED,
    "New ledger entry created",
)
```

## Documentation

| Document | Description |
|----------|-------------|
| [User Manual](docs/user-manual.md) | CLI commands and Python API usage |
| [Specs Sheet](docs/specs-sheet.md) | Complete specification and protocols |
| [Contributing](docs/contributing.md) | Development setup and guidelines |
| [Code of Conduct](docs/code-of-conduct.md) | Community standards |
| [Changelog](CHANGELOG.md) | Version history and changes |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=value_ledger

# Format code
black value_ledger/ tests/

# Lint
ruff check value_ledger/

# Install pre-commit hooks
pre-commit install
```

## Testing

The project includes 211 tests covering all modules:

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_security.py -v

# Run with coverage report
pytest tests/ --cov=value_ledger --cov-report=html
```

## Known Limitations (Alpha)

- Memory Vault integration is stubbed (awaiting external module)
- Learning Contracts consent checking is placeholder
- Embedding model downloads on first use (~90MB)
- Performance not yet optimized for large ledgers (>100K entries)

## Part of the Agent-OS Ecosystem

The Value Ledger is a core component of the Agent-OS ecosystem — a natural-language-native operating system for autonomous AI agents focused on digital sovereignty, intent preservation, and human-AI collaboration.

### Agent-OS Core

- [Agent-OS](https://github.com/kase1111-hash/Agent-OS) — Natural language operating system for AI agents
- [synth-mind](https://github.com/kase1111-hash/synth-mind) — NLOS-based agent with psychological modules for emergent continuity and empathy
- [boundary-daemon-](https://github.com/kase1111-hash/boundary-daemon-) — Trust enforcement layer defining cognition boundaries
- [memory-vault](https://github.com/kase1111-hash/memory-vault) — Secure, owner-sovereign storage for cognitive artifacts
- [learning-contracts](https://github.com/kase1111-hash/learning-contracts) — Safety protocols for AI learning and data management

### Security Infrastructure

- [Boundary-SIEM](https://github.com/kase1111-hash/Boundary-SIEM) — Security information and event management for AI systems

### NatLangChain Ecosystem

- [NatLangChain](https://github.com/kase1111-hash/NatLangChain) — Prose-first, intent-native blockchain protocol for natural language
- [IntentLog](https://github.com/kase1111-hash/IntentLog) — Git for human reasoning, tracks "why" changes happen via prose commits
- [RRA-Module](https://github.com/kase1111-hash/RRA-Module) — Revenant Repo Agent for abandoned repository monetization
- [mediator-node](https://github.com/kase1111-hash/mediator-node) — LLM mediation layer for matching, negotiation, and closure proposals
- [ILR-module](https://github.com/kase1111-hash/ILR-module) — IP & Licensing Reconciliation for dispute resolution
- [Finite-Intent-Executor](https://github.com/kase1111-hash/Finite-Intent-Executor) — Posthumous execution of predefined intent (Solidity smart contract)

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

---

*The world pays for outcomes. The Value Ledger preserves proof of effort.*
