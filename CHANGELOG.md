# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.1] - 2026-01-01

First public alpha release of Value Ledger - the evidentiary accounting layer for cognitive effort in the Agent-OS ecosystem.

> ⚠️ **Alpha Release**: This is an early release intended for testing and feedback. APIs may change before the stable 1.0 release.

### Core Features

- **Value Vector System**
  - 7-dimensional value tracking: Time (T), Effort (E), Novelty (N), Failure (F), Reusability (R), Strategy (S), Understanding (U)
  - Merkle tree-based cryptographic proofs for tamper detection
  - Multi-parent entry aggregation for complex workflows
  - Append-only ledger with soft revocation (freezes value, preserves history)
  - JSONL storage with deterministic entry IDs

- **CLI Interface** (7 commands)
  - `stats` - Display ledger statistics and summaries
  - `query` - Query entries with flexible filters
  - `show` - Display detailed entry information
  - `export` - Export to JSON, CSV, or Merkle proof formats
  - `revoke` - Revoke/freeze entry value with audit trail
  - `proof` - Generate and verify Merkle proofs
  - `demo` - Run interactive demo workflow

- **Heuristic Scoring Engine**
  - 7 pluggable scoring algorithms
  - Embedding-based novelty detection (SentenceTransformer: all-MiniLM-L6-v2)
  - Clock drift detection for temporal integrity
  - Source validation for entry authenticity

### Protocol Support

- **MP-02 Effort Receipt Protocol**
  - Structured effort receipts with validation metadata
  - Privacy & consent controls (PrivacyLevel, ConsentStatus)
  - License management (grant, revoke, delegate)
  - Third-party verification support

- **Enhanced Validation** (6 assessment types)
  - Coherence, Progression, Consistency
  - Authenticity, Completeness, Confidence scoring

- **Export Formats**
  - JSON, CSV, Merkle proofs
  - JSON-LD for linked data
  - W3C Verifiable Credentials
  - OpenTimestamps compatibility

### Integrations

- **Boundary Daemon** - Interruption tracking and effort multipliers
- **Boundary-SIEM** - Security event reporting (JSON HTTP, CEF protocol)
- **IntentLog** - Event-driven intent binding
- **Synth-Mind** - Cognitive tier tracking (4 tiers)
- **NatLangChain** - Natural language chain export (with SSRF protection)
- **Memory Vault** - Encrypted memory references (stubbed, graceful degradation)

### Security

- **Security Integration Module** (`security.py`)
  - Custom exception hierarchy for structured error handling
  - Boundary-SIEM client with hash chain integrity
  - Boundary Daemon client for policy enforcement
  - `@protected_operation` decorator for audited operations
  - `security_context` context manager

- **Hardening**
  - SSRF protection for external requests
  - Path traversal prevention
  - Null byte injection checks
  - Fernet encryption for sensitive content
  - Clock manipulation detection

### Developer Experience

- **CI/CD Pipeline**
  - GitHub Actions with Python 3.9-3.12 matrix testing
  - Automated security scanning (Bandit, Safety)
  - Release workflow for PyPI publishing
  - Dependabot for dependency updates

- **Containerization**
  - Multi-stage Dockerfile (production, development, embeddings)
  - docker-compose.yml for local development
  - Non-root user, health checks

- **Code Quality**
  - Pre-commit hooks (Black, Ruff, Bandit, detect-secrets)
  - 211 tests with pytest
  - Coverage reporting with pytest-cov
  - Type hints throughout

- **Platform Support**
  - Windows batch files (`assemble.bat`, `start.bat`)
  - Unix/Linux/macOS support
  - Docker for cross-platform deployment

### Documentation

- Comprehensive specification sheet (`docs/specs-sheet.md`)
- User manual with CLI and API examples (`docs/user-manual.md`)
- Contributing guidelines (`docs/contributing.md`)
- Code of conduct (`docs/code-of-conduct.md`)

### Known Limitations (Alpha)

- Memory Vault integration is stubbed (awaiting external module)
- Learning Contracts consent checking is placeholder
- Embedding model downloads on first use (~90MB)
- Performance not yet optimized for large ledgers (>100K entries)

---

[0.1.0-alpha.1]: https://github.com/kase1111-hash/value-ledger/releases/tag/v0.1.0-alpha.1
