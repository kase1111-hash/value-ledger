# Value Ledger

**Accounting layer for cognitive effort in the Agent-OS ecosystem**

The Value Ledger is the economic and evidentiary accounting system for AI-human co-learning environments. It records and preserves value for cognitive work — including ideas, effort, time, novelty, failures, and strategic insights — even when outcomes are unsuccessful.

> In a world that pays for results, the Value Ledger provides verifiable proof of effort.

## Overview

This module is part of a larger 9-module system building Agent-OS, a natural-language-native operating system for autonomous AI agents. The Value Ledger serves as the dedicated accounting layer, tracking meta-value derived from cognitive processes without storing sensitive content.

**Key focus:**

- Prove existence and effort of cognitive work
- Support consent-based boundaries (via Learning Contracts)
- Bind value to human intent (via IntentLog)
- Reference encrypted memories (via Memory Vault)
- Enable offline-first, owner-centric valuation

**Important:** This ledger is not a cryptocurrency, token system, or speculative market. It is purely evidentiary — designed for proof, audit, and fair attribution in human-AI collaboration.

## Architecture

The Value Ledger comprises 13 Python modules (~7,825 lines of code):

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
| `memory_vault_hook.py` | Memory Vault integration (stubbed) |

## Documentation

| Document | Description |
|----------|-------------|
| [User Manual](docs/user-manual.md) | CLI commands and Python API usage |
| [Specs Sheet](docs/specs-sheet.md) | Complete specification: design principles, value units, ledger schema, operations, integrations, and MP-02 protocol |
| [Contributing](docs/contributing.md) | Installation, setup, and contributor guidelines |
| [Code of Conduct](docs/code-of-conduct.md) | Community standards |

## Current Status

**Version 0.5.0** - Core implementation complete with advanced features.

### Implemented Features

| Category | Features |
|----------|----------|
| **Core** | 7-dimensional value vector (T/E/N/F/R/S/U), Merkle tree proofs, multi-parent aggregation |
| **Scoring** | 7 heuristic scorers including embedding-based novelty (`all-MiniLM-L6-v2`) |
| **Failure Handling** | Clock drift detection, source validation, unified failure mode handler |
| **CLI** | 7 commands: stats, query, show, export, revoke, proof, demo |
| **MP-02 Protocol** | Effort receipts, privacy/agency controls, enhanced validation |
| **Integrations** | IntentLog, Boundary Daemon, Synth-Mind, NatLangChain |
| **Security** | SSRF protection, path traversal prevention, Fernet encryption |
| **Export Formats** | JSON, CSV, Merkle, JSON-LD, W3C Verifiable Credentials, OpenTimestamps |

### Stubbed (Awaiting External Modules)

- Memory Vault integration - graceful degradation when unavailable
- Learning Contracts consent checking - placeholder for future integration

## Installation

### Requirements

- Python >= 3.9
- pip (Python package manager)

### Dependencies

**Required:**
- `pydantic>=2.0` - Data validation
- `cryptography>=41.0` - Cryptographic operations

**Optional (for embedding-based novelty scoring):**
- `sentence-transformers` - Semantic similarity embeddings
- `torch` - PyTorch for GPU acceleration

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/kase1111-hash/value-ledger.git
cd value-ledger

# Install in editable mode
pip install -e .

# Verify installation
python -m value_ledger.cli --help
```

### With Development Dependencies

For development and testing:

```bash
pip install -e ".[dev]"
```

This includes: `pytest`, `black`, `ruff`

### Storage Configuration

The ledger uses JSONL (JSON Lines) format. Default path: `ledger.jsonl`

When integrated with Agent-OS: `~/.agent-os/ledger.jsonl`

### Verifying Installation

```bash
# Run the CLI demo
python -m value_ledger.cli demo

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=value_ledger
```

### Troubleshooting

**Import Errors:** Ensure pydantic is installed: `pip install pydantic>=2.0`

**Permission Issues:** Ensure write permissions: `chmod 755 ~/.agent-os/`

**Embedding Model Issues:** If novelty scoring fails, install optional dependencies:
```bash
pip install sentence-transformers torch
```
The embedding model (`all-MiniLM-L6-v2`) downloads on first use (~90MB).

**Cryptography Issues:** If encryption fails, ensure cryptography is properly installed:
```bash
pip install cryptography>=41.0
```

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

---

*The world pays for outcomes. The Value Ledger preserves proof of effort.*
