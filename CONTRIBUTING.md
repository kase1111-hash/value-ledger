# Contributing to Value Ledger

Thank you for your interest in contributing to Value Ledger!

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

**Required dependencies:**
- `pydantic>=2.0` — Data validation

**Optional (for embedding-based novelty scoring):**
- `sentence-transformers` — Semantic similarity embeddings
- `torch` — PyTorch for GPU acceleration

## Branch Naming

- `feature/` — New features
- `fix/` — Bug fixes
- `docs/` — Documentation updates
- `refactor/` — Code refactoring

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

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues when applicable

## Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Test edge cases and error conditions

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Submit pull request with clear description

## Adding New Features

1. Review [specs-sheet.md](docs/specs-sheet.md) for design guidelines
2. Check if feature aligns with design principles
3. Implement with tests
4. Handle graceful degradation when modules are unavailable
5. Add to `__init__.py` exports if public API

## Reporting Issues

**Bug reports** should include: Python version, OS, steps to reproduce, expected vs actual behavior, and error messages.

**Feature requests** should include: use case, proposed solution, and impact on existing functionality.

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 license.

## Questions?

- Open an issue for technical questions
- Review existing issues before creating new ones
- Check documentation in the `/docs` folder
