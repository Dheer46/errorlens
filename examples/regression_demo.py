"""ErrorLens regression demo: where does a house-price model make its largest errors?

Two hidden problems are planted: prices of large homes (> 2,000 sqft) in the east district
are far noisier, and homes with an unknown construction year carry a price premium that the
model only partly captures.

Run:  python examples/regression_demo.py
"""

from __future__ import annotations

from pathlib import Path

from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from _data import housing_dataset
from errorlens import ErrorLens

OUTPUT = Path(__file__).parent / "output"


def main() -> None:
    X_train, y_train, X_test, y_test = housing_dataset()
    model = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore"), ["district", "has_garage"]),
            remainder="passthrough"),
        HistGradientBoostingRegressor(random_state=0),
    ).fit(X_train, y_train)

    result = ErrorLens(model, X_test, y_test, title="House price model — error analysis",
                       max_patterns=8).analyze()
    result.summary(max_patterns=3)

    print()
    print(result.residual_analysis())
    print("\nHigh-error regions:")
    for p in result.high_error_regions():
        print(f"  #{p.rank} {p.description:<45} MAE {p.rate:6.1f} vs {p.baseline:5.1f} "
              f"({p.lift:.2f}x), mean residual {p.mean_residual:+.1f}")
    print("Severe underprediction patterns:",
          [p.description for p in result.underprediction_patterns()])

    OUTPUT.mkdir(exist_ok=True)
    print(f"\nHTML report: {result.export(OUTPUT / 'regression_report.html')}")
    print(f"PDF report:  {result.export(OUTPUT / 'regression_report.pdf')}")


if __name__ == "__main__":
    main()
