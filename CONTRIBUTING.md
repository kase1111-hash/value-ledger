# Contributing to Value Ledger

Thank you for your interest in contributing to Value Ledger!

For detailed contribution guidelines, please see our [full contributing guide](docs/contributing.md).

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/value-ledger.git`
3. Install with dev dependencies: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature-name`
5. Make your changes and run tests: `pytest tests/`
6. Submit a pull request

## Development Requirements

- Python >= 3.9
- pip (Python package manager)
- Git

## Code Quality

Before submitting, ensure:

```bash
# Run tests
pytest tests/

# Format code
black value_ledger/ tests/

# Lint code
ruff check value_ledger/
```

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 license.

## Questions?

- Open an issue for technical questions
- Review existing issues before creating new ones
- Check documentation in the `/docs` folder
