# Contributing to ErrorLens

Thanks for helping! ErrorLens values **statistical correctness over feature count**: a
contribution that makes a reported number more trustworthy is worth more than a new chart.

## Development setup

```bash
git clone https://github.com/errorlens/errorlens
cd errorlens
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Checks (the same ones CI runs)

```bash
pytest                       # full test suite (add -m "not slow" to skip the perf test)
ruff check .                 # linting
mypy                         # type checking (strict-ish, see pyproject.toml)
```

Run the examples after changes to discovery or reporting and look at the output:

```bash
python examples/classification_demo.py
python examples/pdf_report_demo.py       # writes examples/output/example_report.*
```

## Guidelines

* **Never imply causation** in user-facing text: "associated with", "higher error rate
  observed in" — not "causes".
* **Any new statistic needs a test** against a reference implementation (scipy) or a known
  value, and must state whether it is descriptive or inferential.
* **Discovery changes need a planted-pattern test** (the true failure region is known) *and*
  must keep the null-data test passing (pure noise ⇒ no patterns).
* Report content is decided only in `errorlens/reporting/report.py`; renderers
  (`html.py`, `pdf.py`, `markdown.py`) must not compute or omit analysis content.
* Optional dependencies (matplotlib, reportlab) are imported lazily through
  `errorlens.utils.optional.require`, which raises `MissingDependencyError` with install
  instructions.
* Keep the core dependency set minimal (numpy, pandas, scipy, scikit-learn, click).
* Don't swallow exceptions; wrap them in an `ErrorLensError` subclass with an actionable
  message and chain the original (`raise ... from exc`).

## Pull requests

1. Open an issue first for larger changes (new tasks, new discovery algorithms).
2. Add tests and update `CHANGELOG.md` under an "Unreleased" heading.
3. Make sure `pytest`, `ruff check .` and `mypy` pass.

## Releasing

```bash
python -m build
twine check dist/*
twine upload dist/*
```
