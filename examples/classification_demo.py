"""ErrorLens classification demo: a 96%-accurate model that fails a hidden subgroup.

The training data under-represents young, low-income customers (age < 25 AND
income < 30,000), whose behaviour follows a different rule. Overall metrics look excellent;
ErrorLens finds the subgroup automatically — nobody tells it where to look.

Run:  python examples/classification_demo.py
"""

from __future__ import annotations

from pathlib import Path

from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from _data import credit_dataset, hidden_subgroup
from errorlens import ErrorLens

OUTPUT = Path(__file__).parent / "output"


def main() -> None:
    X_train, y_train, X_test, y_test = credit_dataset()
    model = make_pipeline(
        make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["region", "student"]),
                                remainder="passthrough"),
        HistGradientBoostingClassifier(random_state=0),
    ).fit(X_train, y_train)

    lens = ErrorLens(model, X_test, y_test, max_patterns=10, min_samples=30,
                     significance_level=0.05, title="Credit default model — error analysis")
    result = lens.analyze()
    result.summary(max_patterns=3)

    # Did ErrorLens recover the planted subgroup?
    truth = hidden_subgroup(X_test)
    top = result.patterns[0]
    overlap = (top.mask(X_test) & truth).sum() / (top.mask(X_test) | truth).sum()
    print(f"\nPlanted subgroup: age < 25 AND income < 30,000  ({truth.sum()} test rows)")
    print(f"Top discovered:   {top.description}")
    print(f"Row overlap (Jaccard) with the planted subgroup: {overlap:.2f}")

    OUTPUT.mkdir(exist_ok=True)
    print("\nFalse negatives:", len(result.false_negatives()), "rows; FN patterns:",
          [p.description for p in result.false_negative_patterns()[:3]])
    path = result.export(OUTPUT / "classification_report.html")
    print(f"HTML report: {path}")


if __name__ == "__main__":
    main()
