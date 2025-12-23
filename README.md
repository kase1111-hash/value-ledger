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

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](docs/installation.md) | Setup and installation instructions |
| [User Manual](docs/user-manual.md) | CLI commands and Python API usage |
| [Specs Sheet](docs/specs-sheet.md) | Complete specification: design principles, value units, ledger schema, operations, integrations, and MP-02 protocol |
| [Contributing](docs/contributing.md) | Guidelines for contributors |
| [Code of Conduct](docs/code-of-conduct.md) | Community standards |

## Current Status

**Version 0.3.1** - Core implementation complete.

Implemented features:
- 7-dimensional value vector (T/E/N/F/R/S/U)
- Proof system with Merkle trees
- Multi-parent aggregation
- Failure mode handling
- Admin CLI (query, export, stats, revoke)
- IntentLog integration
- Boundary Daemon integration
- Synth-Mind integration
- NatLangChain export
- MP-02 Effort Receipts

## Quick Start (Development)

```bash
pip install -e .
python -m value_ledger.cli
```

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

---

*The world pays for outcomes. The Value Ledger preserves proof of effort.*
