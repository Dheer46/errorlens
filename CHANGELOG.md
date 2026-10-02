# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-10-02

### Added
- `ErrorLens` analyzer for binary, multiclass and regression models (`model.predict`) or
  precomputed predictions (`ErrorLens.from_predictions`).
- Automatic failure-pattern discovery: beam search + surrogate error tree over threshold,
  equality and missingness conditions; condition simplification and threshold refinement.
- Honest inference via a stratified discovery/validation split; one-sided Fisher and Welch
  tests, Wilson / t / Katz intervals, Cohen's h / d; Benjamini–Hochberg, Benjamini–Yekutieli,
  Holm and Bonferroni corrections with an explicit hypothesis family.
- Overlap-aware de-duplication and conservative excess-error ranking.
- Error-type-specific targets: false positives, false negatives, per-class misses, absolute
  error, severe under- and over-prediction.
- Feature error association (Mann–Whitney, chi-square, Spearman, Kruskal–Wallis with
  effect sizes and FDR adjustment) and regression residual diagnostics.
- `AnalysisResult` with accessors, DataFrame views, console summary and matplotlib plots.
- Structured report model rendered to self-contained HTML, paginated PDF (reportlab),
  Markdown (with PNG assets) and JSON.
- `errorlens analyze` / `errorlens version` CLI.
