"""Generate the example reports in every format from ONE analysis result.

Writes examples/output/example_report.{html,pdf,md,json} plus a PNG overview figure.
PDF export requires:  pip install "errorlens[pdf]"

Run:  python examples/pdf_report_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from _data import credit_dataset
from errorlens import ErrorLens, MissingDependencyError

OUTPUT = Path(__file__).parent / "output"


def main() -> None:
    X_train, y_train, X_test, y_test = credit_dataset()
    model = make_pipeline(
        make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["region", "student"]),
                                remainder="passthrough"),
        HistGradientBoostingClassifier(random_state=0),
    ).fit(X_train, y_train)
    result = ErrorLens(model, X_test, y_test, max_patterns=10,
                       title="Credit default model — error analysis").analyze()

    OUTPUT.mkdir(exist_ok=True)
    for ext in ("html", "md", "json"):
        print("wrote", result.export(OUTPUT / f"example_report.{ext}"))
    try:
        print("wrote", result.to_pdf(OUTPUT / "example_report.pdf"))
    except MissingDependencyError as exc:
        print(exc)

    fig = result.plot()
    fig.savefig(OUTPUT / "overview.png", dpi=120)
    print("wrote", OUTPUT / "overview.png")

    # JSON is machine-readable: e.g. fail a CI job when a severe pattern appears.
    data = json.loads((OUTPUT / "example_report.json").read_text(encoding="utf-8"))
    severe = [p for p in data["patterns"] if p["lift"] >= 3 and p["p_value_adjusted"] < 0.01]
    print(f"{len(severe)} pattern(s) with lift >= 3x and adjusted p < 0.01")


if __name__ == "__main__":
    main()
