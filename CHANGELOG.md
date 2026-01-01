# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI/CD pipeline with multi-Python version testing (3.9-3.12)
- Automated security scanning with Bandit and Safety
- Docker containerization with multi-stage builds
- Pre-commit hooks configuration for code quality
- Dependabot for automated dependency updates
- Release workflow for PyPI publishing
- pytest-cov for coverage reporting
- CLI entry point script (`value-ledger` command)

### Changed
- Updated pyproject.toml with comprehensive tool configurations
- Applied Black formatting to all Python files
- Fixed 33 linting errors (unused imports, f-string issues)

## [0.5.0] - 2024-12-01

### Added
- **Core Ledger System**
  - 7-dimensional value vector (Time, Effort, Novelty, Failure, Reusability, Strategy, Understanding)
  - Merkle tree-based cryptographic proofs
  - Multi-parent entry aggregation
  - Append-only ledger with revocation (freezes, doesn't delete)
  - JSONL storage with deterministic IDs

- **CLI Interface** (7 commands)
  - `stats` - Show ledger statistics
  - `query` - Query entries with filters
  - `show` - Display specific entry details
  - `export` - Export to JSON/CSV/Merkle formats
  - `revoke` - Revoke/freeze entry value
  - `proof` - Generate Merkle proof
  - `demo` - Run demo workflow

- **Heuristic Scoring**
  - 7 heuristic scorers
  - Embedding-based novelty detection (SentenceTransformer: all-MiniLM-L6-v2)
  - Clock drift detection
  - Source validation

- **MP-02 Effort Receipt Protocol**
  - Effort receipts with validation metadata
  - Privacy & consent controls (PrivacyLevel, ConsentStatus)
  - External compatibility (MP-01 negotiation)
  - License management (grant, revoke, delegate)

- **Enhanced Validation** (6 assessment types)
  - Coherence assessment
  - Progression assessment
  - Consistency assessment
  - Authenticity assessment
  - Completeness assessment
  - Confidence calculation

- **Integrations**
  - IntentLog (event-driven)
  - Boundary Daemon (interruption tracking)
  - Synth-Mind (cognitive tier tracking)
  - NatLangChain (with SSRF protection)
  - Memory Vault (stubbed, graceful degradation)

- **Security**
  - SSRF protection
  - Path traversal prevention
  - Null byte injection checks
  - Fernet encryption for signal content

- **Export Formats**
  - JSON, CSV, Merkle proofs
  - JSON-LD
  - W3C Verifiable Credentials
  - OpenTimestamps

- **Documentation**
  - Comprehensive specs sheet
  - User manual with CLI and API usage
  - Contributing guidelines
  - Code of conduct

### Security
- Path traversal prevention in ledger path validation
- Sensitive system path blocking
- Clock manipulation detection

## [0.4.0] - 2024-11-15

### Added
- Initial implementation of Value Ledger core
- Basic value vector tracking
- JSONL persistence layer

## [0.3.0] - 2024-11-01

### Added
- Project structure and packaging
- Basic Pydantic models
- Initial test suite

---

[Unreleased]: https://github.com/kase1111-hash/value-ledger/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/kase1111-hash/value-ledger/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/kase1111-hash/value-ledger/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/kase1111-hash/value-ledger/releases/tag/v0.3.0
