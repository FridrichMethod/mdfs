# Contributing

Thanks for your interest in `mdfs`!

## Setup

```bash
make venv          # creates .venv (Python 3.12) and installs everything
.venv/bin/python -c "import jax; print(jax.devices())"   # check GPU
```

## Development workflow

1. Branch off `main`.
2. Write tests first (the suite mirrors `src/mdfs/`; physics changes must keep
   `tests/test_params_vs_openmm.py` green).
3. Implement, keeping energy functions pure and units in the OpenMM convention
   (nm / ps / amu / kJ/mol / e -- see `src/mdfs/constants.py`).
4. Run the local gate before committing:
   ```bash
   make format lint mypy bandit test
   ```
   Slow / end-to-end tests: `make slow-test`.
5. Use conventional-commit messages (`feat:`, `fix:`, `refactor:`, `test:`,
   `docs:`, `chore:`). Never bypass hooks with `--no-verify`.
6. Open a PR once CI is green.

## Conventions

See [CLAUDE.md](CLAUDE.md) for the full list (absolute imports, `pathlib`,
`logging` over `print`, Google docstrings, type hints, immutable aggregates,
parameters sourced from OpenMM's resolved `System`).
