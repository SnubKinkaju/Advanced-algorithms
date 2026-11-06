# Minimum Set Cover

Advanced Algorithms project, Master DSC/MLDM, Université Jean Monnet.

Exact and approximate algorithms for the unweighted minimum set cover problem, compared on generated and benchmark instances.

## Installation

Python 3.10 or later is required. On macOS, the default `python3` may be 3.9: install a recent version with `brew install python@3.12` and use `python3.12` below.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Structure

- `src/setcover/`: code (instances, file readers, generator, bounds, methods)
- `tests/`: automated tests
- `data/benchmarks/`: benchmark instances, origin in `SOURCE.md`
- `doc/`: planning and other documents

To add a method, see `CONTRIBUTING.md`.