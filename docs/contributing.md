# Contributing to Value Ledger

Thank you for your interest in contributing to Value Ledger! This document covers installation, setup, and contribution guidelines.

## Installation

### Requirements

- Python >= 3.9
- pip (Python package manager)
- Git

### Dependencies

**Required:**
- `pydantic>=2.0` - Data validation
- `cryptography>=41.0` - Cryptographic operations

**Optional (for embedding-based novelty scoring):**
- `sentence-transformers` - Semantic similarity embeddings
- `torch` - PyTorch for GPU acceleration

### From Source

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

```bash
pip install -e ".[dev]"
```

This includes:
- `pytest` - Testing framework
- `black` - Code formatter
- `ruff` - Linter

### Directory Structure

```
value-ledger/
├── value_ledger/              # Main package (~7,825 lines)
│   ├── __init__.py            # Package exports (100+ classes/functions)
│   ├── core.py                # Core ValueLedger class (1,196 lines)
│   ├── cli.py                 # Command-line interface (533 lines)
│   ├── heuristics.py          # 7 scoring heuristics (357 lines)
│   ├── integration.py         # IntentLog integration (135 lines)
│   ├── interruption.py        # Boundary Daemon integration (700 lines)
│   ├── memory_vault_hook.py   # Memory Vault hook (165 lines, stubbed)
│   ├── natlangchain.py        # NatLangChain export (721 lines)
│   ├── receipt.py             # MP-02 Effort Receipts (647 lines)
│   ├── privacy.py             # Privacy & consent controls (793 lines)
│   ├── validation.py          # Enhanced validation criteria (994 lines)
│   ├── compatibility.py       # MP-02 external compatibility (966 lines)
│   └── synth_mind.py          # Synth-Mind integration (403 lines)
├── tests/                     # Test suite (5 test files)
│   ├── test_validation.py     # Enhanced validation tests
│   ├── test_compatibility.py  # Compatibility & licensing tests
│   ├── test_interruption.py   # Interruption tracking tests
│   ├── test_privacy.py        # Privacy & consent tests
│   └── test_e2e_simulation.py # End-to-end workflow tests
├── docs/                      # Documentation
│   ├── specs-sheet.md         # Complete specification
│   ├── user-manual.md         # CLI and API usage
│   ├── contributing.md        # This file
│   └── code-of-conduct.md     # Community standards
├── LICENSE                    # GPL-3.0-only
├── README.md                  # Project overview
├── pyproject.toml             # Python project configuration
└── build.bat                  # Windows build script
```

### Storage Configuration

The ledger uses JSONL (JSON Lines) format for storage. Default path: `ledger.jsonl`

When integrated with Agent-OS:

```
~/.agent-os/
├── ledger.jsonl          # Value Ledger storage
├── intent_log.jsonl      # IntentLog storage
├── memory_vault/         # Memory Vault storage
└── contracts/            # Learning Contracts storage
```

### Troubleshooting

**Import Errors:** Ensure pydantic is installed: `pip install pydantic>=2.0`

**Permission Issues:** Ensure write permissions: `chmod 755 ~/.agent-os/`

---

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/value-ledger.git
   cd value-ledger
   ```

3. Follow the [Installation instructions](../README.md#installation) in the README
4. Install with dev dependencies: `pip install -e ".[dev]"`
5. Verify setup: `pytest tests/`

## Development Workflow

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring

Example: `feature/add-merkle-verification`

### Making Changes

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Format code:
   ```bash
   black value_ledger/
   ```

5. Lint code:
   ```bash
   ruff check value_ledger/
   ```

6. Commit your changes with clear messages:
   ```bash
   git commit -m "Add feature: description of what was added"
   ```

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues when applicable

Examples:
```
Add Merkle proof verification method
Fix clock drift detection in failure handler
Update documentation for CLI export command
Refactor heuristics module for better testability
```

## Code Style

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Document public functions with docstrings
- Keep functions focused and single-purpose

### Example

```python
def calculate_effort_factor(
    weighted_interruptions: float,
    raw_count: Optional[int] = None,
) -> float:
    """
    Calculate effort multiplier based on interruptions.

    Args:
        weighted_interruptions: Weighted sum of interruptions
        raw_count: Optional raw interruption count for additional scaling

    Returns:
        Effort factor multiplier (>= 1.0)
    """
    factor = 1.0 + (weighted_interruptions * 0.35)
    if raw_count and raw_count > 10:
        factor += (raw_count - 10) * 0.1
    return factor
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_interruption.py

# Run with coverage
pytest tests/ --cov=value_ledger
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Test edge cases and error conditions

Example:
```python
def test_interruption_tracker_calculates_weighted_sum():
    tracker = InterruptionTracker()
    tracker.start_session("test-intent")

    tracker.record_interruption(InterruptionEvent(
        type=InterruptionType.EXTERNAL,
        timestamp=100.0,
        intent_id="test-intent",
    ))

    weighted = tracker.get_weighted_interruptions("test-intent")
    assert weighted == 1.0  # EXTERNAL weight is 1.0
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG if applicable
5. Submit pull request with clear description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
Describe how you tested the changes

## Related Issues
Fixes #123
```

## Architecture Overview

### Core Modules

| Module | Purpose | Lines |
|--------|---------|-------|
| `core.py` | ValueLedger class, entry management, Merkle proofs | 1,196 |
| `cli.py` | Command-line interface (7 commands) | 533 |
| `heuristics.py` | 7 scoring algorithms with embedding-based novelty | 357 |
| `integration.py` | IntentLog event-driven connection | 135 |
| `interruption.py` | Boundary Daemon integration | 700 |
| `receipt.py` | MP-02 Effort Receipts | 647 |
| `privacy.py` | Privacy controls, consent, encryption | 793 |
| `validation.py` | Enhanced validation criteria (6 types) | 994 |
| `compatibility.py` | MP-01 negotiation, licensing, audit export | 966 |
| `natlangchain.py` | NatLangChain export with SSRF protection | 721 |
| `synth_mind.py` | Synth-Mind cognitive tier integration | 403 |
| `memory_vault_hook.py` | Memory Vault integration (stubbed) | 165 |

### Key Classes

- `ValueLedger` - Main ledger management
- `LedgerEntry` - Individual entry records
- `ValueVector` - 7-dimensional value representation
- `HeuristicEngine` - Auto-scoring system with 7 scorers
- `MerkleTree` - Cryptographic proofs
- `EffortReceipt` - MP-02 effort receipt structure
- `SignalEncryptor` - Fernet-based signal encryption
- `EnhancedValidator` - Multi-criteria validation
- `LicenseManager` - License grant/revoke/delegate
- `AuditExporter` - Export to JSON-LD, W3C-VC, OpenTimestamps

### Security Classes

- `ClockMonitor` - Detects clock drift and manipulation
- `SourceValidator` - Validates entry integrity
- `FailureModeHandler` - Unified failure handling
- `PrivacyFilter` - Content filtering by privacy level
- `ConsentRegistry` - Consent management

## Adding New Features

### Checklist

1. Review [specs-sheet.md](specs-sheet.md) for design guidelines
2. Check if feature aligns with design principles
3. Implement with tests
4. Update documentation
5. Add to `__init__.py` exports if public API

### Integration Features

When adding module integrations:

1. Create stub implementations first
2. Define clear interfaces
3. Handle graceful degradation when modules unavailable
4. Document expected interfaces in specs-sheet.md

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered
- Impact on existing functionality

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 license.

## Questions?

- Open an issue for technical questions
- Review existing issues before creating new ones
- Check documentation in `/docs` folder

Thank you for contributing to Value Ledger!
