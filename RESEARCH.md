# ErrorLens — Prior-Art Research

This document surveys existing tools for ML model error analysis and debugging, identifies
what they do well, and narrows down the design space that ErrorLens targets. It was written
**before** implementation and updated once the MVP was built.

> **Bottom line.** Automatic discovery of high-error subgroups ("slice finding",
> "subgroup discovery", "weak segment detection") is **not new**. It exists in research
> (SliceFinder 2019, SliceLine 2021, DivExplorer 2021, FreaAI 2021, Divisi 2025), in a general
> subgroup-discovery library (pysubgroup), and inside larger products (Microsoft Error
> Analysis, Deepchecks, Giskard). ErrorLens does not claim a new algorithm. Its value is a
> specific combination that, to the best of our research, is not available in a single
> lightweight, headless, permissively licensed Python package: error-type-aware targets,
> **statistically defensible inference over adaptively discovered patterns** (sample splitting
> + exact tests + an explicitly defined multiple-testing family), redundancy control, and a
> single structured result that feeds Python, JSON, Markdown, HTML and PDF outputs.

---

## 1. Tools surveyed

### 1.1 Microsoft Error Analysis (`erroranalysis`, part of Responsible AI Toolbox / `raiwidgets`)

* **What it does.** Identifies cohorts with high error rates. Two views: an *error tree*
  (a surrogate decision tree trained to predict whether the model is wrong) and an *error
  heatmap* over one or two user-selected features. Reports per-cohort error rate, error
  coverage (share of all errors in the cohort) and data representation.
* **API.** `ResponsibleAIDashboard(RAIInsights(...))` / `ErrorAnalysisDashboard(...)`
  widgets; core APIs in `erroranalysis` (`ModelAnalyzer`, `compute_error_tree`,
  `compute_matrix`). Integrated in Azure ML.
* **Strengths.** The closest conceptual neighbour. Mature, well-documented, coined
  practical metrics (error coverage), good interactive UX, cohort workflow.
* **Limitations (relative to ErrorLens goals).**
  * Dashboard-first; the primary output is an interactive widget rather than a static,
    archivable artifact (no PDF report from the core API).
  * A single surrogate tree partitions the data: every sample sits in exactly one leaf, so
    overlapping explanations (e.g. "age < 25" and "student = True") compete rather than
    being reported side by side; greedy splits can hide strong conjunctions.
  * We found no hypothesis testing, confidence intervals or multiple-testing correction for
    the discovered cohorts — a leaf with 31 samples and 12 errors looks as "real" as one
    with 3,100 samples and 1,200 errors.
  * Heavier dependency footprint (LightGBM, widget stack).
* **Overlap.** Error-rate/coverage metrics and tree-based cohort discovery. ErrorLens reuses
  the *idea* of a surrogate error tree as one candidate generator, not as the final answer.

### 1.2 What-If Tool (Google PAIR)

* **What it does.** Visual, interactive probing of a model in TensorBoard / Jupyter /
  Colab: datapoint editing, counterfactuals, partial dependence, slicing performance by up
  to two user-chosen features, fairness threshold exploration.
* **Strengths.** Excellent for interactive exploration and communication.
* **Limitations.** Slices are chosen manually; no automatic discovery, no statistics; heavy
  front-end; development activity is low.
* **Overlap.** "Performance by slice" views. ErrorLens automates finding the slices.

### 1.3 SHAP

* **What it does.** Shapley-value feature attributions (TreeExplainer, KernelExplainer,
  DeepExplainer, …); global summaries via aggregation. TreeExplainer can attribute the
  model's *loss* rather than its output.
* **Strengths.** Theoretically grounded attribution, rich plots, huge adoption.
* **Limitations for error analysis.** Explains *how features push a prediction*, not *which
  interpretable regions of the data the model gets wrong*. Converting attributions into
  subgroups with error-rate guarantees is left to the user. Cost can be high for
  model-agnostic explainers.
* **Decision.** ErrorLens does **not** compute attributions and does not depend on SHAP.
  Users who want "why did the model predict X for this sample" should use SHAP on samples
  inside an ErrorLens pattern.

### 1.4 LIME

* Local surrogate models per instance; tabular/text/image. Instance-level, known
  instability across runs. **Not duplicated.**

### 1.5 NannyML

* **What it does.** Post-deployment monitoring: performance *estimation without labels*
  (CBPE for classification, DLE for regression), univariate and multivariate drift
  detection (e.g. PCA reconstruction error), chunked time-based analysis.
* **Overlap.** Very little: NannyML asks "is performance changing over time?"; ErrorLens asks
  "where, inside one labelled evaluation set, does the model fail?" **Not duplicated.**

### 1.6 Evidently

* **What it does.** Reports and test suites (presets) for data drift, data quality,
  classification and regression quality, and increasingly LLM evaluation. The regression
  preset includes an *Error Bias* table: top-5% over-predictions vs top-5% under-predictions
  vs the remaining 90%, compared feature by feature.
* **Strengths.** Broad, polished HTML reports, monitoring-oriented, popular.
* **Limitations.** Segment analysis is feature-by-feature; no automatic multi-condition
  subgroup search with significance testing or FDR control.
* **Overlap.** Metric summaries, residual plots, the over/under split idea. ErrorLens keeps
  its performance section small and does not try to be a monitoring/drift platform.

### 1.7 Alibi (Explain) / Alibi Detect (Seldon)

* **Alibi Explain:** anchors (rule-based local explanations), counterfactuals, CEM, ALE,
  integrated gradients. **Alibi Detect:** drift, outlier and adversarial detection.
* **Note.** Since January 2024 both moved from Apache-2.0 to the Business Source License
  1.1 (free production use restricted) — a practical reason to keep ErrorLens permissively
  licensed and independent.
* **Overlap.** Anchors produce rules, but they explain *individual predictions*, not
  high-error *populations*. **Not duplicated.**

### 1.8 Fairlearn

* **What it does.** `MetricFrame` disaggregates any metric by **user-specified** sensitive
  features; mitigation algorithms (ExponentiatedGradient, GridSearch, ThresholdOptimizer).
* **Overlap.** Disaggregated metrics. Fairlearn requires the user to name the groups;
  ErrorLens discovers them. They are complementary: discovered patterns can be passed to
  `MetricFrame` as sensitive features. ErrorLens does **not** do fairness mitigation.

### 1.9 PyOD

* 40+ outlier-detection algorithms with a uniform API. Outliers are sometimes where models
  fail, but outlier-ness is a property of inputs, not of errors. **Not duplicated**; out of
  scope for v0.1 (possible future: "is this pattern an outlier region?").

### 1.10 scikit-learn inspection & metrics

* `confusion_matrix`, `classification_report`, `ConfusionMatrixDisplay`,
  `PredictionErrorDisplay`, `permutation_importance`, `partial_dependence`,
  `DecisionBoundaryDisplay`, `DecisionTreeClassifier`.
* These are *building blocks*. ErrorLens uses scikit-learn metrics and trees internally
  rather than reimplementing them.

### 1.11 MLflow

* Experiment tracking, model registry, `mlflow.models.evaluate()` producing metrics,
  plots and (optionally) SHAP artifacts. **Not duplicated**; ErrorLens' JSON/HTML/PDF
  outputs can be logged as MLflow artifacts.

### 1.12 Directly related slice-finding / subgroup-discovery projects

| Project | Kind | Approach | Statistics | Notes |
|---|---|---|---|---|
| **SliceFinder** (Chung, Kraska, Polyzotis, Whang — ICDE 2019) | Research paper + research code | Lattice search and decision-tree search for problematic, large, interpretable slices | Effect size + significance; controls false discoveries with **α-investing** (marginal FDR) | The methodological ancestor of ErrorLens' "statistics first" stance. Not a maintained package. |
| **SliceLine** (Sagadeeva & Boehm, SIGMOD 2021) / **`sliceline`** (DataDome, PyPI) | Library | Linear-algebra enumeration of the slice lattice with a scoring function trading off size and error; sklearn-style `Slicefinder().fit(X, errors)` | Score-based; no p-values / CIs | Fast and elegant; expects pre-discretized/categorical inputs; returns slices, not a report. |
| **pysubgroup** | General library | Classic subgroup discovery (Apriori, beam, best-first, DFS), many quality functions, binary and numeric targets | Quality functions incl. χ²-based; correction is up to the user | General-purpose data mining, not ML-error aware. A user could build part of ErrorLens' discovery on it. |
| **DivExplorer** (Pastor, de Alfaro, Baralis — SIGMOD 2021) | Library | Frequent-itemset mining (apriori / FP-growth) of subgroups whose FPR/FNR/error *diverges*; Shapley-value attribution of items | Welch t-test per itemset | Requires discretized inputs; heavier deps (mlxtend, plotly, igraph). |
| **Deepchecks — Weak Segments Performance** | Check inside a test-suite framework | Trains many simple trees, each on **exactly two features**, to predict per-sample error; reports weakest leaves | No multiple-testing control documented | Segment depth limited to 2 features; part of a larger framework. |
| **Giskard — scan / performance bias detector** | Detector inside a testing framework | Finds data slices where a metric is markedly worse than overall | Thresholds on relative metric drop | Project focus has shifted toward LLM testing. |
| **FreaAI** (IBM, 2021), **Divisi** (2025) | Research | Feature-model / interactive subgroup analysis | Varies | Not general-purpose packages. |
| **Domino, Spotlight, sliceguard** | Research / libraries | Embedding-based slice discovery for images/text | — | Unstructured data; out of scope for v0.1. |
| **cleanlab** | Library | Detects likely label errors | — | Complementary: some "model errors" are label errors. ErrorLens' limitations section points users to it. |

---

## 2. What overlaps with ErrorLens

* Performance metrics, confusion matrices, residual plots — **commodity**; every tool has
  them. ErrorLens includes them only as context for failure patterns and delegates the
  arithmetic to scikit-learn.
* Tree-based error cohorts (Microsoft EA, Deepchecks) and lattice/itemset slice finding
  (SliceLine, DivExplorer, pysubgroup) — ErrorLens uses the same family of techniques.
* Per-feature error breakdowns (Evidently, WIT, Fairlearn) — ErrorLens includes a
  statistically tested version as its "feature error association" table.

## 3. What ErrorLens should NOT attempt to duplicate

| Not in scope | Use instead |
|---|---|
| Per-prediction attributions | SHAP, LIME, Alibi |
| Drift detection / production monitoring / label-free performance estimation | NannyML, Evidently, Alibi Detect |
| Fairness mitigation and constraint-based training | Fairlearn |
| Generic outlier detection | PyOD |
| Label-noise detection | cleanlab |
| Interactive dashboards / notebook widgets | Responsible AI Toolbox, What-If Tool |
| Experiment tracking | MLflow |
| Unstructured-data slice discovery (images, text) | Domino, Spotlight, sliceguard |

## 4. The underserved design space

Across the surveyed tools we repeatedly found the same gaps:

1. **Inference after search is rarely handled.** Most tools search adaptively and then
   report raw subgroup metrics. Even when a p-value is shown, it is computed on the same
   data that was used to *find* the subgroup, which is optimistic (winner's curse / selective
   inference). SliceFinder's α-investing is the notable exception, and it never shipped as a
   maintained package.
2. **The multiple-testing family is rarely defined.** "We corrected for multiple tests" is
   meaningless unless the set of tests is known. Adaptive search obscures it.
3. **Error-type awareness is shallow.** False positives and false negatives usually have
   different causes; regression under- vs over-prediction too. Most tools discover slices
   for one scalar error signal only.
4. **Redundancy.** Lattice/itemset methods return many near-duplicate slices
   (`A`, `A ∧ B`, `A ∧ B ∧ C` with almost identical rows).
5. **Mixed raw data.** Many slice finders require users to discretize and one-hot encode
   first and do not treat *missingness* as a first-class condition.
6. **Headless, archivable output.** Dashboards are great for exploration but hard to put in
   CI, attach to a model card, or send to a reviewer. Few tools produce the same analysis as
   JSON + Markdown + HTML + PDF from one model.

## 5. ErrorLens' specific differentiation

ErrorLens is a **headless, statistics-first failure-pattern discovery library for tabular
models** with a small dependency footprint (numpy, pandas, scipy, scikit-learn; matplotlib
and reportlab optional) and a permissive (MIT) license.

1. **Error-type-aware discovery targets** — overall errors; false positives and false
   negatives (binary); per-class missed predictions (multiclass); absolute-error magnitude
   plus severe under- and over-prediction (regression).
2. **Two complementary, interpretable candidate generators on raw mixed-type data** —
   beam-search subgroup discovery over automatically built conditions (quantile thresholds,
   frequent categories, *missingness*) and a surrogate error tree. Candidates may overlap.
3. **Honest inference** — by default (when data allow) candidates are found on a discovery
   split and **tested on an independent validation split**, so the multiple-testing family
   is exactly the set of tested candidates and p-values / confidence intervals are not
   inflated by the search. Exact one-sided Fisher (hypergeometric) tests for rates, Welch
   t-tests for error magnitude, Wilson intervals, relative-risk intervals, Cohen's h / d,
   Benjamini–Hochberg FDR by default (Benjamini–Yekutieli, Holm, Bonferroni available).
   When data are too small to split, ErrorLens falls back to in-sample testing, corrects
   over *every* candidate evaluated during the search, and labels the results exploratory.
4. **Redundancy control** — condition simplification, refinement pruning and Jaccard
   de-duplication, then ranking by a conservative "excess errors" score with a complexity
   penalty, so users see a handful of distinct patterns, not hundreds.
5. **One structured result, many outputs** — `AnalysisResult` → document model →
   HTML / PDF / Markdown renderers; JSON is a direct serialization of the same result. A CLI
   makes it usable in CI pipelines.
6. **Careful language** — every output frames findings as associations
   ("higher error rate observed in"), with methodology and limitations embedded in reports.

What ErrorLens deliberately is **not**: a new slice-finding algorithm, an explainer, a
monitoring system, or a fairness toolkit.

## 6. Key references

* Chung, Kraska, Polyzotis, Whang. *Slice Finder: Automated Data Slicing for Model
  Validation.* ICDE 2019.
* Sagadeeva, Boehm. *SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model
  Debugging.* SIGMOD 2021.
* Pastor, de Alfaro, Baralis. *Looking for Trouble: Analyzing Classifier Behavior via
  Pattern Divergence.* SIGMOD 2021 (DivExplorer).
* Nushi, Kamar, Horvitz. *Towards Accountable AI: Hybrid Human-Machine Analyses for
  Characterizing System Failure.* HCOMP 2018 (foundation of Microsoft Error Analysis).
* Wrobel. *An Algorithm for Multi-relational Discovery of Subgroups.* PKDD 1997;
  Klösgen. *Explora.* 1996; Lavrač et al. *Subgroup Discovery with CN2-SD.* JMLR 2004.
* Benjamini, Hochberg. *Controlling the False Discovery Rate.* JRSS-B 1995.
  Benjamini, Yekutieli. *The control of the FDR under dependency.* Ann. Stat. 2001.
* Athey, Imbens. *Recursive partitioning for heterogeneous causal effects.* PNAS 2016
  ("honest" estimation via sample splitting).
* Wilson. *Probable inference, the law of succession, and statistical inference.* JASA 1927.

### Sources consulted (2026-10)

* erroranalysis on PyPI — https://pypi.org/project/erroranalysis
* Deepchecks Weak Segments Performance docs — https://docs.deepchecks.com/dev/tabular/auto_checks/model_evaluation/plot_weak_segments_performance.html
* sliceline — https://github.com/DataDome/sliceline , https://sliceline.readthedocs.io/
* pysubgroup — https://pysubgroup.readthedocs.io/
* DivExplorer — https://pypistats.org/packages/divexplorer
* Giskard — https://legacy-docs.giskard.ai/en/stable/
* Evidently regression preset — https://docs.evidentlyai.com/metrics/preset_regression
* Seldon licensing FAQ — https://seldon.io/resources/licensing-faqs
* Slice Finder — https://research.google/pubs/slice-finder-automated-data-slicing-for-model-validation/
* What-If Tool — https://github.com/PAIR-code/what-if-tool
