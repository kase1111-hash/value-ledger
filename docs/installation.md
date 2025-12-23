# Installation Guide

## Requirements

- Python >= 3.9
- pip (Python package manager)

## Dependencies

The Value Ledger module requires:

- `pydantic>=2.0` - Data validation
- `cryptography>=41.0` - Cryptographic operations

## Installation Methods

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

This includes:
- `pytest` - Testing framework
- `black` - Code formatter
- `ruff` - Linter

## Directory Structure

After installation, your project should have:

```
value-ledger/
├── value_ledger/          # Main package
│   ├── __init__.py
│   ├── core.py            # Core ValueLedger class
│   ├── cli.py             # Command-line interface
│   ├── heuristics.py      # Scoring heuristics
│   ├── integration.py     # IntentLog integration
│   ├── interruption.py    # Boundary Daemon integration
│   ├── memory_vault_hook.py
│   ├── natlangchain.py    # NatLangChain export
│   ├── receipt.py         # MP-02 Effort Receipts
│   └── synth_mind.py      # Synth-Mind integration
├── tests/                 # Test suite
├── docs/                  # Documentation
├── LICENSE
├── README.md
└── pyproject.toml
```

## Storage Configuration

The ledger uses JSONL (JSON Lines) format for storage. Default path: `ledger.jsonl`

### Agent-OS Integration Path

When integrated with Agent-OS:

```
~/.agent-os/
├── ledger.jsonl          # Value Ledger storage
├── intent_log.jsonl      # IntentLog storage
├── memory_vault/         # Memory Vault storage
└── contracts/            # Learning Contracts storage
```

## Verifying Installation

```bash
# Run the CLI demo
python -m value_ledger.cli demo

# Run tests
pytest tests/
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure pydantic is installed:

```bash
pip install pydantic>=2.0
```

### Permission Issues

Ensure write permissions for the ledger storage directory:

```bash
chmod 755 ~/.agent-os/
```

## Next Steps

- Read the [User Manual](user-manual.md) for usage instructions
- Review the [Specs Sheet](specs-sheet.md) for technical details
