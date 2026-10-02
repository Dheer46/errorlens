# ErrorLens

**Automatic, statistically rigorous failure-pattern discovery for tabular ML models.**

ErrorLens is a debugger for model errors. Give it a fitted model and labelled evaluation data;
it finds the interpretable subgroups — combinations of feature conditions — where the model
fails unusually often, tests whether each finding is statistically real, removes redundant
variants, and writes the result as a console summary, JSON, Markdown, HTML or PDF report.

```text
Pattern #1
  age <= 24
  AND 18,400 < income <= 30,000

  Samples:            350  (coverage 3.50%)
  Errors:             188  (43.2% of all errors)
  Error rate:         53.71%
  Baseline:           4.35%
  Lift:               12.35x
  95% CI (held-out):  [44.5%, 59.7%]
  p-value (held-out): < 0.001   adjusted (fdr_bh): < 0.001
```

## Why ErrorLens?

Aggregate metrics hide failures. In the bundled demo a gradient-boosting credit model reaches
**95.7% accuracy** — and is wrong on **54% of young, low-income customers**, a group that was
under-represented in its training data. Nothing in the accuracy, ROC AUC or confusion matrix
reveals it. ErrorLens finds that group without being told where to look.

What makes it different from "slice the metrics by a few features":

* **Automatic discovery** of multi-condition patterns (`age <= 24 AND income <= 30,000`,
  `transactions is missing`) on raw mixed-type data — numeric, categorical, boolean and
  missing values.
* **Honest statistics.** Patterns are found on one half of the data and tested on the other,
  so p-values and confidence intervals are not inflated by the search. Multiple testing is
  controlled (Benjamini–Hochberg by default) over an explicitly defined family of hypotheses.
  On pure-noise data ErrorLens reports nothing.
* **Error-type aware.** Separate patterns for false positives and false negatives, per-class
  misses (multiclass), and absolute error plus severe under- / over-prediction (regression).
* **A few distinct findings, not hundreds.** Conditions are simplified, thresholds sharpened,
  and near-duplicate or "diluted" variants of a better pattern are removed.
* **Headless and lightweight.** Core dependencies: numpy, pandas, scipy, scikit-learn, click.
  Plots and PDF are optional extras. Every output is generated from one structured result.

ErrorLens is **not** an explainer (use SHAP/LIME), a drift monitor (NannyML, Evidently) or a
fairness-mitigation toolkit (Fairlearn). See [RESEARCH.md](RESEARCH.md) for an honest
comparison with Microsoft Error Analysis, Deepchecks, SliceLine, pysubgroup, DivExplorer and
others — the underlying ideas are not new; the combination is the point.

## Installation

```bash
pip install errorlens            # analysis engine + CLI
pip install "errorlens[viz]"     # + matplotlib plots
pip install "errorlens[pdf]"     # + PDF reports (reportlab, matplotlib)
pip install "errorlens[all]"     # everything
```

Python 3.10+.

## Quick start

```python
from errorlens import ErrorLens

lens = ErrorLens(model=model, X=X_test, y=y_test)
result = lens.analyze()

result.summary()                 # console report
result.patterns[0]               # best FailurePattern
result.export("report.html")     # or .pdf / .md / .json
```

Any object with `predict(X)` works (scikit-learn pipelines, XGBoost, LightGBM, CatBoost, …);
`predict_proba` is used for confidence analysis when available. If predictions come from
elsewhere:

```python
result = ErrorLens.from_predictions(X_test, y_test, y_pred).analyze()
```

Configuration:

```python
lens = ErrorLens(
    model, X_test, y_test,
    task="classification",        # "auto" | "classification" | "regression"
    max_patterns=20,              # per discovery target
    min_samples=30,               # minimum rows per reported pattern
    significance_level=0.05,      # after multiple-testing correction
    correction="fdr_bh",          # "fdr_bh" | "fdr_by" | "holm" | "bonferroni" | "none"
    max_depth=3,                  # max conditions per pattern
    positive_label=1,             # binary classification
)
```

See `AnalysisConfig` for all options (beam width, bins, honest splitting, ranking, …).

## Failure-pattern discovery

1. **Conditions.** Numeric features become readable thresholds (`x <= t`, `x > t`) at
   quantiles; categorical/boolean features become `x == level`; missingness becomes
   `x is missing`. Identifier-like, constant and datetime columns are skipped with a warning.
2. **Search.** A vectorized beam search over conjunctions of up to `max_depth` conditions,
   scored by the standardized excess `sqrt(n)·(error − baseline)/sd`, plus the paths of a
   shallow surrogate decision tree fit to the error indicator. Search cost is linear in rows.
3. **Simplify and sharpen.** Conditions that don't raise the error metric by ≥ 10 % are
   dropped; numeric thresholds of shortlisted candidates are re-optimized on a finer grid.
4. **Validate.** Candidates are tested on a held-out split they were not selected on.
5. **Correct.** p-values are adjusted for every tested candidate.
6. **Rank and de-duplicate.** Significant patterns are ranked by a conservative
   excess-error score `n · (CI_low − baseline)` with a complexity penalty; overlapping,
   diluted or non-improving refinements of a better pattern are removed.

Each `FailurePattern` carries its conditions, rows, errors, error rate, baseline, lift,
relative risk (with CI), Wilson confidence interval, p-value, adjusted p-value, effect size
(Cohen's h / d), coverage, error coverage and complexity — and can be applied to new data:

```python
p = result.patterns[0]
p.description            # 'age <= 24 AND 18,400 < income <= 30,000'
p.lift, p.p_value_adjusted, p.confidence_interval
mask = p.mask(X_new)     # rows of new data that fall into this pattern
result.patterns_frame()  # all patterns as a DataFrame
```

## Classification

```python
result.error_rate()                  # 0.0435
result.confusion_matrix()            # labelled DataFrame
result.errors(); result.correct()    # rows
result.false_positives()             # rows (binary)
result.false_negatives()
result.false_positive_patterns()     # subgroups with elevated false positive rate
result.false_negative_patterns()     # subgroups with elevated false negative rate
result.class_patterns("bird")        # multiclass: where class 'bird' is missed
result.feature_associations()        # errors vs correct, per feature
```

The summary includes per-class precision, recall and error counts, the most frequent
confusions, and — when probabilities are available — confidence on errors vs correct
predictions.

## Regression

```python
result = ErrorLens(regressor, X_test, y_test).analyze()
result.residual_analysis()           # bias, skew, heteroscedasticity, decile table, ...
result.high_error_regions()          # subgroups with significantly higher MAE
result.underprediction_patterns()    # subgroups with many severe underpredictions
result.overprediction_patterns()
```

```text
High-error regions:
  #1 district == east AND sqft > 2,010   MAE 53.1 vs 12.7 (4.18x)
  #2 year_built is missing               MAE 16.1 vs 12.7 (1.27x)
```

## Visualizations

Requires `errorlens[viz]`. Every method returns a matplotlib `Figure`.

```python
result.plot()                          # 2×2 overview
result.plot_patterns(kind="false_negative")
result.plot_feature_errors("age")
result.plot_error_distribution("income")
result.plot_error_heatmap("age", "income")
result.plot_confusion_matrix(normalize=True)
result.plot_residuals()                # regression
```

## Reports

All formats are rendered from the same structured result, so they always agree.

```python
result.to_html("report.html")      # self-contained single file, charts embedded
result.to_pdf("report.pdf")        # requires errorlens[pdf]
result.to_markdown("report.md")    # charts written to report_assets/
result.to_json("report.json")      # machine-readable
result.export("report.pdf")        # format inferred from the extension
result.export("out.txt", format="json")
```

Reports contain: executive summary, dataset summary, model information, performance metrics,
error breakdown, failure patterns (table, chart, detailed cards), feature error analysis,
false-positive / false-negative or class-specific analysis, residual analysis (regression),
feature-interaction heatmap, a statistical-analysis table documenting every discovery run
(candidates evaluated, tested, family size), warnings, methodology and limitations.

The PDF is a paginated A4 document with repeated table headers, charts, page numbers and
Unicode-safe fonts. If reportlab is missing, `to_pdf()` raises `MissingDependencyError`
with the exact `pip install` command.

The JSON output is suitable for CI gates:

```python
import json
data = json.load(open("report.json"))
assert not [p for p in data["patterns"] if p["lift"] > 3 and p["p_value_adjusted"] < 0.01]
```

## CLI

```bash
errorlens analyze --model model.pkl --data test.csv --target target --output report.pdf
errorlens analyze --data scored.csv --target y --prediction-column y_hat -o report.html \
    --min-samples 50 --max-patterns 10 --significance-level 0.01 --correction holm
errorlens version
```

Options: `--task`, `--format`, `--min-samples`, `--max-patterns`, `--significance-level`,
`--correction`, `--max-depth`, `--positive-label`, `--drop COL` (repeatable), `--seed`,
`--title`, `--no-figures`, `--quiet`. Data may be CSV, TSV or Parquet. Models are loaded with
joblib/pickle — **only load model files you trust**, unpickling can execute code.

## Statistical methodology

| Step | Method |
|---|---|
| Subgroup discovery | Beam search (subgroup discovery with the binomial/z quality function) + surrogate decision tree |
| Honest inference | Stratified 50/50 discovery/validation split when the validation part holds ≥ 5 × `min_samples` rows and ≥ 10 errors; otherwise in-sample, flagged as exploratory |
| Test (error rates) | One-sided exact Fisher test (hypergeometric tail), subgroup vs rest |
| Test (absolute error) | One-sided Welch t-test, subgroup vs rest |
| Confidence intervals | Wilson score (rates), Student-t (means), Katz log interval (relative risk) |
| Effect sizes | Cohen's h (rates), Cohen's d (means); rank-biserial r, Cramér's V, Spearman ρ, ε² for features |
| Multiple testing | Benjamini–Hochberg FDR (default); Benjamini–Yekutieli, Holm, Bonferroni available |
| Family size | Honest mode: the candidates tested on validation data. In-sample mode: every candidate evaluated during search (conservative) |

**Why Benjamini–Hochberg?** Error analysis is exploratory: a few false leads are acceptable,
missing real failures is costly, and overlapping subgroups produce positively dependent
tests, under which BH controls the false discovery rate. Use `correction="holm"` or
`"bonferroni"` when any false positive is costly, or `"fdr_by"` for arbitrary dependence.

**Why sample splitting?** Testing a subgroup on the same data used to find it makes p-values
far too small (the "winner's curse"). Splitting costs power but makes the reported
p-values and intervals valid. Descriptive numbers (rows, error rate, lift) are reported on
the full data; inferential numbers come from the held-out split and are labelled as such.

**Associations, not causes.** A pattern says *where* the model errs, not *why*. ErrorLens
phrases every finding as an association ("error rate is 53.7% for rows matching …").

## Limitations

* Patterns are conjunctions of thresholds, equality and missingness conditions up to
  `max_depth`; failure modes of other shapes (feature ratios, smooth interactions, text or
  image content) are missed or approximated.
* Power depends on sample size; small evaluation sets reveal only large, frequent failures.
  No reported pattern ≠ uniform performance.
* Honest validation halves the data for each step. In-sample results are exploratory.
* Findings hold for this evaluation set; under distribution shift they may not.
* Hard-label errors only: calibration and cost-sensitive errors are not analysed (yet).
* Some "model errors" are label errors — consider a label-quality tool such as cleanlab.

## Roadmap

* NLP and computer-vision support via metadata / embedding-derived concepts
* Time-series error analysis (time-window conditions)
* Comparing two models' failure patterns; drift of patterns between datasets
* Fairness-oriented reporting (patterns over protected attributes, Fairlearn interop)
* Calibration and cost-sensitive error targets
* Active-learning integration: sample new labels from discovered failure regions
* LLM-evaluation targets (judge failures over prompt metadata)

## Project documents

[ARCHITECTURE.md](ARCHITECTURE.md) · [RESEARCH.md](RESEARCH.md) ·
[CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) · [LICENSE](LICENSE)

Examples: [classification_demo.py](examples/classification_demo.py) ·
[regression_demo.py](examples/regression_demo.py) ·
[pdf_report_demo.py](examples/pdf_report_demo.py)
