# ErrorLens Architecture

## 1. Design principles

1. **Statistical correctness first.** Every number that implies "this is real" (p-value,
   confidence interval) must be computed in a way that survives the adaptive search that
   produced it.
2. **One source of truth.** The analysis produces a single immutable-ish `AnalysisResult`.
   Every output (console summary, plots, JSON, Markdown, HTML, PDF) is derived from it.
3. **Lightweight core.** `numpy`, `pandas`, `scipy`, `scikit-learn`, `click`. Plotting
   (`matplotlib`) and PDF (`reportlab`) are optional extras with explicit, actionable errors
   when missing.
4. **Task-agnostic core, task-specific adapters.** Discovery and statistics operate on a
   *target vector* (binary event or continuous loss) and a *population mask*. Classification,
   regression and future tasks only decide how to build those vectors.
5. **Associational language everywhere.** No output says a feature *causes* errors.

## 2. Package layout

```
errorlens/
├── __init__.py              public API re-exports
├── _version.py
├── exceptions.py            ErrorLensError hierarchy
├── core/
│   ├── config.py            AnalysisConfig (validated dataclass)
│   ├── data.py              input validation, feature typing (FeatureInfo)
│   ├── models.py            ModelAdapter: predict / predict_proba / task inference
│   ├── types.py             Condition, GroupStats, FailurePattern, metric dataclasses
│   ├── results.py           AnalysisResult (accessors, summary, plots, export)
│   └── analyzer.py          ErrorLens — orchestrates the pipeline
├── analysis/
│   ├── errors.py            error identification → DiscoveryTarget objects
│   ├── statistics.py        tests, intervals, effect sizes, p-value adjustment
│   ├── classification.py    classification metrics & per-class analysis
│   ├── regression.py        regression metrics & residual analysis
│   └── distributions.py     errors-vs-correct feature association
├── discovery/
│   ├── items.py             raw features → candidate conditions (binary item matrix)
│   ├── search.py            beam search + surrogate error tree candidate generators
│   ├── subgroup.py          SubgroupDiscovery: split → search → test → correct → rank
│   ├── ranking.py           scoring
│   ├── deduplication.py     simplification, refinement pruning, Jaccard dedup
│   └── interactions.py      2-D error grids, co-occurring feature pairs
├── visualization/
│   ├── _mpl.py              optional-import guard
│   ├── plots.py             user-facing plot functions (return matplotlib Figures)
│   └── report_plots.py      figure set rendered to PNG for reports
├── reporting/
│   ├── document.py          Report document model (sections + typed blocks)
│   ├── report.py            AnalysisResult → Report (the only place content is decided)
│   ├── html.py              Report → self-contained HTML
│   ├── pdf.py               Report → PDF (reportlab)
│   ├── markdown.py          Report → Markdown (+ PNG assets)
│   ├── json_export.py       AnalysisResult → JSON
│   ├── export.py            format inference / dispatch
│   └── templates/report.css
├── cli/main.py              `errorlens analyze`, `errorlens version`
└── utils/                   formatting, optional dependency helper
```

## 3. Public API

```python
from errorlens import ErrorLens

lens = ErrorLens(model, X_test, y_test,           # or y_pred=... instead of model
                 task="auto", max_patterns=20, min_samples=30,
                 significance_level=0.05, correction="fdr_bh")
result = lens.analyze()

result.summary()                 # prints & returns text
result.patterns                  # list[FailurePattern] (ranked, significant)
result.patterns_frame()          # DataFrame view
result.error_rate(); result.confusion_matrix(); result.errors()
result.false_positives(); result.false_negatives()                # rows (binary)
result.false_positive_patterns(); result.false_negative_patterns()
result.class_patterns(label)                                      # multiclass
result.residual_analysis(); result.high_error_regions()           # regression
result.feature_associations()                                      # DataFrame
result.plot(); result.plot_patterns(); ...                        # matplotlib
result.export("report.pdf")      # .html .pdf .md .json, or format=...
result.to_html(...); result.to_pdf(...); result.to_markdown(...); result.to_json(...)

pattern = result.patterns[0]
pattern.mask(X_new)              # apply the discovered rule to new data
str(pattern)                     # "age < 24.5 AND income <= 30000"
```

`ErrorLens.from_predictions(X, y, y_pred, y_proba=None)` supports analysis without a model
object (predictions computed elsewhere, e.g. in Spark or a remote service).

## 4. Data flow

```
             ┌────────────┐     ┌──────────────┐
X, y, model ─► validate &  ├────►│ ModelAdapter │── y_pred, y_proba
             │ type data  │     └──────────────┘
             └─────┬──────┘
                   ▼
        ┌──────────────────────┐
        │ analysis/errors.py   │  builds DiscoveryTargets:
        │                      │   classification: error, FP, FN, class-c misses
        │                      │   regression: |residual|, severe under/over
        └─────┬────────────────┘
              ▼
  ┌──────────────────────────┐   ┌──────────────────────┐   ┌───────────────────┐
  │ metrics (sklearn)         │   │ SubgroupDiscovery     │   │ feature association│
  │ classification/regression │   │ (per target)          │   │ errors vs correct  │
  └──────────┬───────────────┘   └──────────┬───────────┘   └─────────┬─────────┘
             └──────────────┬───────────────┴─────────────────────────┘
                            ▼
                     AnalysisResult  ──► summary() / plots
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
      json_export (direct)        reporting/report.py → Report (document model)
                                    │        │         │
                                  html.py  pdf.py  markdown.py
```

## 5. Failure-pattern discovery

### 5.1 Discovery target abstraction

```python
@dataclass
class DiscoveryTarget:
    kind: str          # "error", "false_positive", "false_negative", "class_error",
                       # "high_error", "underprediction", "overprediction"
    values: ndarray    # binary event (0/1) or continuous loss per row
    population: ndarray[bool]   # rows where the target is defined
    is_binary: bool
    metric_name: str   # "error rate", "false negative rate", "MAE", ...
```

* **Error:** `values = y_pred != y`, population = all rows.
* **False negatives (binary):** population = actual positives, event = predicted negative
  → the subgroup metric is the *false negative rate* (miss rate).
* **False positives (binary):** population = actual negatives, event = predicted positive
  → *false positive rate*.
* **Class-c misses (multiclass):** population = `y == c`, event = `y_pred != c`.
* **Regression magnitude:** continuous `|y - ŷ|` (MAE as subgroup metric).
* **Severe under/over-prediction:** `τ = q90(|residual|)`; event = `residual > τ`
  (underprediction) or `residual < −τ` (overprediction).

### 5.2 Candidate conditions (`discovery/items.py`)

Raw features are converted to a boolean *item matrix* (rows × candidate conditions):

* numeric: thresholds at up to `n_bins − 1` quantiles, rounded to 3 significant digits
  (readable, and membership is computed with the *rounded* threshold so the description is
  exact); both `x <= t` and `x > t`.
* categorical / boolean: `x == level` for the `max_categories` most frequent levels; IDs
  (cardinality > 50 % of rows and > `max_categories`) are skipped with a warning.
* missingness: `x is missing` whenever at least `min_samples` rows are missing.
* constant and datetime features are skipped with a warning.

### 5.3 Candidate generators (`discovery/search.py`)

1. **Beam search** (classic subgroup discovery). Level 1 evaluates every item; each later
   level extends the `beam_width` best subgroups with one more item on a *new* feature, up
   to `max_depth` conditions. Evaluation is vectorized: for a beam member with mask `m`, the
   support and target sums of all extensions are one matrix product
   `items[m].T @ [1, t[m]]`. Quality function (standardized excess, the "binomial"/z quality
   of subgroup discovery):

   `q = sqrt(n_s) · (mean_s − mean_0) / sd_0`

   For binary targets this is the one-sample z statistic; it balances effect size and
   support so neither tiny extreme groups nor huge mild groups dominate. Subgroups smaller
   than the minimum support are pruned (support is anti-monotone, so their refinements are
   pruned too).
2. **Surrogate error tree.** A shallow `DecisionTree{Classifier,Regressor}` with
   `min_samples_leaf = min_samples` is fit to the target; root-to-leaf paths of leaves with
   above-baseline error become candidates (thresholds per feature merged into intervals).

All candidates are re-evaluated by applying their `Condition`s to the *raw* data, so a
pattern's description and its membership can never disagree (e.g. missing values never
satisfy a numeric comparison).

3. **Screening (discovery split only).** Each candidate is *simplified* (a condition is
   dropped if removing it lowers the subgroup metric by < 10 %), filtered (support ≥ scaled
   `min_samples`, discovery lift ≥ `min_lift`), ordered by complexity-penalized quality and
   de-duplicated; the top `max_candidates` survive. Their numeric thresholds are then
   **refined** on a 40-quantile grid of the feature *within the rows selected by the other
   conditions* (vectorized with a cumulative-sum scan), so boundaries are not limited to the
   coarse search grid (`age <= 23` becomes the true `age <= 24`).

### 5.4 Honest inference (`discovery/subgroup.py`)

```
population rows ──split (stratified)──► discovery part (50%)      validation part (50%)
                                          │                         │
                          items, beam search, tree                  │
                          simplify, dedup, select top K ────────────► one-sided test per candidate
                                                                    BH correction over exactly K
                                                                    keep adj. p < α
  full data ◄───────────────── descriptive statistics for survivors ┘
  rank, final dedup, truncate to max_patterns
```

* `honest="auto"` splits when the validation part would hold at least `5 × min_samples`
  rows (and ≥ 10 events for binary targets); otherwise the analysis runs in-sample, the
  correction uses **every candidate evaluated during search** as the family size, and
  patterns are flagged `inference="in-sample (exploratory)"` with a result-level warning.
* Each pattern carries two `GroupStats` blocks: `stats` (descriptive, full data) and
  `validation` (inferential: p-value, CI, effect size on the validation part).

### 5.5 Statistics (`analysis/statistics.py`)

| Quantity | Binary target | Continuous target |
|---|---|---|
| Test (subgroup vs complement, one-sided "greater") | Exact Fisher via hypergeometric survival function | Welch's t-test |
| Confidence interval for subgroup metric | Wilson score | Student-t interval for the mean |
| Relative measure | Relative risk vs complement with Katz log interval (Haldane 0.5 correction on zero cells) | Ratio of means |
| Effect size | Cohen's h | Cohen's d (pooled SD) |
| Lift | subgroup rate / population baseline | subgroup MAE / population MAE |
| Coverage | n_s / N | n_s / N |
| Error coverage | errors in subgroup / all errors | share of total absolute error |

Multiple-testing correction (`correction=`): `fdr_bh` (default), `fdr_by`, `holm`,
`bonferroni`, `none`. BH controls the FDR under independence or positive regression
dependence (PRDS); overlapping subgroups produce positively correlated statistics in most
practical cases, which is why BH is the default. `fdr_by` is valid under arbitrary
dependence at the cost of power; Holm/Bonferroni control the family-wise error rate.

### 5.6 Ranking and de-duplication

* **Simplification:** drop any condition whose removal lowers the subgroup metric by less
  than 10 % relative (keeps descriptions minimal).
* **Greedy de-duplication** in rank order. A pattern is dropped when, relative to an already
  kept pattern K:
  1. its rows overlap K with Jaccard similarity ≥ `dedup_threshold` (0.7); or
  2. it *refines* K (strict superset of conditions, or < `min_samples` rows outside K) and is
     not a meaningful refinement — its metric must be ≥ 10 % higher than K's **and** a
     one-sided test of the sub-region vs the rest of K must reach p < 0.05 *on the
     validation split* (which took no part in selecting either pattern, avoiding
     selection bias); or
  3. it is a **diluted** version of K: the rows it does not share with K retain less than half
     of its excess error over the baseline (its elevation is explained by the overlap).
* **Score** (default `rank_by="score"`):
  `excess_lcb / (1 + 0.25 · (complexity − 1))`, where
  `excess_lcb = n_s · (CI_low − baseline)` — a conservative count of errors in the subgroup
  beyond what the baseline rate predicts (absolute-error units for regression). This rewards
  patterns that are simultaneously large, strongly elevated, precisely estimated and simple.
  Alternatives: `lift`, `p_value`, `coverage`, `error_coverage`.

### 5.7 Complexity

Let `F` = features, `I` ≈ `F · n_bins` items, `w` = beam width, `d` = depth, `N` rows.
Beam search costs `O(N · I)` for level 1 plus `O(d · w · N_m · I)` for deeper levels (`N_m`
= rows of the parent subgroup) — linear in `N`, never exponential. Defaults
(`n_bins=10`, `beam_width=25`, `max_depth=3`) analyze 50 k rows × 20 features in seconds.
`max_discovery_rows` subsamples the discovery part for very large data.

## 6. Feature error association (`analysis/distributions.py`)

Errors vs correct predictions (classification) or |residual| (regression):

| Feature type | Classification | Regression |
|---|---|---|
| numeric | Mann–Whitney U; effect = rank-biserial r; also means/medians | Spearman ρ with |residual| |
| categorical | χ² test of independence (rare levels pooled); effect = Cramér's V; per-level error rate | Kruskal–Wallis; effect = ε² |
| missingness | reported as a level / indicator | same |

p-values are BH-adjusted across features. Magnitude labels (negligible/small/medium/large)
use conventional thresholds and are explicitly heuristic.

## 7. Result representation (`core/types.py`, `core/results.py`)

```
AnalysisResult
├── metadata        (title, timestamp, version, task, model info, config)
├── dataset         DatasetSummary (rows, features, types, missingness, skipped features)
├── performance     ClassificationPerformance | RegressionPerformance
├── error_summary   ErrorSummary (counts, rates, FP/FN, confusion pairs)
├── patterns        list[FailurePattern]               ← headline
├── directional     {"false_positive": [...], "false_negative": [...]}  or
│                   {"underprediction": [...], "overprediction": [...]}
├── class_patterns  {label: [...]}  (multiclass)
├── feature_analysis list[FeatureAssociation]
├── residuals       ResidualAnalysis | None
├── statistics      DiscoveryStatistics per target (candidates, tested, correction, split)
└── warnings        list[str]
```

All dataclasses implement `to_dict()`; JSON export is `json.dumps(result.to_dict())` with
NaN/inf mapped to `null`/strings. The private raw arrays (`X`, `y`, `y_pred`) stay on the
result for plotting and row-level accessors but are never serialized.

## 8. Reporting system

`reporting/report.py` converts an `AnalysisResult` into a format-neutral document:

```python
Report(title, subtitle, sections=[Section(title, blocks=[
    Paragraph(text, style="normal"|"muted"|"lead"),
    KeyValues([(label, value), ...]),
    Table(columns, rows, caption, numeric_columns),
    PatternCard(rank, title, rows, validation_note, emphasis),
    Figure(png_bytes, caption, width_fraction),
    Callout(text, level="info"|"warning"),
    BulletList(items),
])])
```

Renderers only *lay out* blocks; they never compute statistics or decide content. That
guarantees HTML, PDF and Markdown contain the same analysis. Figures are rendered once to
PNG (`visualization/report_plots.py`, Agg canvas, no pyplot global state) and embedded as
base64 (HTML), `Image` flowables (PDF) or sidecar files in `<name>_assets/` (Markdown). If
matplotlib is missing, figures are omitted with an explicit notice in the report.

**PDF (`reporting/pdf.py`)** uses reportlab Platypus: A4, title block, `LongTable` with
`repeatRows=1` for automatic pagination, `Paragraph` cells for wrapping long pattern
descriptions, `KeepTogether` for pattern cards, a two-pass canvas for "Page X of N" footers,
and PDF metadata (title, author, subject).

## 9. CLI

`click`-based. Click is the mature foundation that Typer itself builds on; it adds no
further transitive dependencies (Typer would pull in rich and shellingham), which keeps the
base install lightweight while still giving typed options, validation, help text and a
test runner:

```
errorlens analyze --model model.pkl --data test.csv --target label --output report.pdf
                  [--task auto|classification|regression] [--format html|pdf|md|json]
                  [--min-samples 30] [--max-patterns 20] [--significance-level 0.05]
                  [--correction fdr_bh] [--max-depth 3] [--positive-label 1]
                  [--drop col] [--prediction-column col] [--quiet]
errorlens version
```

Models are loaded with joblib/pickle (documented as unsafe for untrusted files). Data:
CSV or Parquet. `--prediction-column` analyzes precomputed predictions without a model.
Exit codes: 0 success, 1 analysis error, 2 usage error.

## 10. Error handling

`exceptions.py`: `ErrorLensError` → `InvalidInputError` (also `ValueError`),
`InsufficientDataError`, `ModelError`, `PredictionError`, `UnsupportedTaskError`,
`MissingDependencyError` (also `ImportError`, message contains the exact `pip install`).
Prediction failures are re-raised with the original exception chained.

## 11. Extensibility

New tasks plug in by (1) a `ModelAdapter`/prediction step, (2) a function producing
`DiscoveryTarget`s, (3) a performance dataclass, (4) a report section builder. Discovery,
statistics, ranking, export and CLI are reused unchanged. For NLP/CV the feature table
would be metadata or embedding-derived concepts; for time series, time-window conditions;
for LLM evaluation, a binary "judge failed" target over prompt metadata.
