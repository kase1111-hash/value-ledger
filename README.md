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
| [specs.md](specs.md) | Core specification: design principles, value units, ledger schema, operations, and threat model |
| [INTEGRATION.md](INTEGRATION.md) | Cross-repository integration guide with all 9 Agent-OS modules |
| [MP-02-spec.md](MP-02-spec.md) | Proof-of-Effort Receipt Protocol for NatLangChain compatibility |

## Current Status

This repository currently contains:

- Core specification ([specs.md](specs.md))
- Cross-repository integration guide ([INTEGRATION.md](INTEGRATION.md))
- Protocol specification ([MP-02-spec.md](MP-02-spec.md))
- GPL-3.0 license

Implementation in progress — Core code, storage backend, and module integrations are under development (Python, consistent with the ecosystem).

## Quick Start (Development)

```bash
pip install -e .
python -m value_ledger.cli
```

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

---

*The world pays for outcomes. The Value Ledger preserves proof of effort.*
