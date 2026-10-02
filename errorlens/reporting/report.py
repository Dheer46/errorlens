"""Build the format-neutral :class:`Report` from an :class:`AnalysisResult`.

This module is the *only* place where report content is decided. HTML, PDF and Markdown
renderers consume the resulting document unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from errorlens.core.types import (
    ClassificationPerformance,
    FailurePattern,
    RegressionPerformance,
)
from errorlens.reporting.document import (
    BulletList,
    Callout,
    Figure,
    Heading,
    KeyValues,
    Paragraph,
    PatternCard,
    Report,
    Section,
    Table,
    Tiles,
)
from errorlens.utils.formatting import fmt_int, fmt_lift, fmt_num, fmt_p, fmt_pct

if TYPE_CHECKING:
    from errorlens.core.results import AnalysisResult
    from errorlens.visualization.report_plots import ReportFigure

CORRECTION_NAMES = {
    "fdr_bh": "Benjamini–Hochberg false discovery rate (FDR)",
    "fdr_by": "Benjamini–Yekutieli false discovery rate (FDR, arbitrary dependence)",
    "holm": "Holm–Bonferroni family-wise error rate (FWER)",
    "bonferroni": "Bonferroni family-wise error rate (FWER)",
    "none": "no correction (not recommended)",
}

KIND_LABELS = {
    "error": "Failure patterns (all errors)",
    "high_error": "High-error regions (absolute error)",
    "false_positive": "False positives",
    "false_negative": "False negatives",
    "class_error": "Missed predictions of one class",
    "underprediction": "Severe underprediction",
    "overprediction": "Severe overprediction",
}

ASSOCIATION_NOTE = (
    "All findings describe statistical associations observed in this evaluation data. A "
    "pattern with a high error rate does not mean its features cause the errors; it may "
    "reflect label noise, missing information, under-representation in training data, or "
    "correlated factors not present in the data."
)


def methodology_text(config: dict[str, Any]) -> list[tuple[str, str]]:
    alpha = config.get("significance_level", 0.05)
    correction = CORRECTION_NAMES.get(config.get("correction", "fdr_bh"), "")
    return [
        ("Error identification",
         "Classification: a row is an error when the predicted label differs from the true "
         "label; for binary tasks false positives and false negatives are analysed separately "
         "within actual negatives and actual positives (false positive / false negative rate). "
         "Multiclass tasks additionally analyse missed predictions per class. Regression: the "
         "error metric is the absolute residual |actual − predicted|; severe under- and "
         "over-prediction are residuals beyond the 90th percentile of |residual|."),
        ("Candidate conditions",
         f"Numeric features are split at up to {config.get('n_bins', 10) - 1} readable quantile "
         "thresholds (x ≤ t, x > t); categorical and boolean features contribute equality "
         f"conditions for their {config.get('max_categories', 30)} most frequent levels; "
         "missingness is a condition of its own. Constant, datetime and identifier-like "
         "columns are excluded."),
        ("Subgroup discovery",
         "Two interpretable candidate generators are combined: (1) a vectorized beam search "
         f"(width {config.get('beam_width', 25)}) over conjunctions of up to "
         f"{config.get('max_depth', 3)} conditions on distinct features, scored by the "
         "standardized excess sqrt(n)·(mean − baseline)/sd; and (2) the node paths of a "
         "shallow surrogate decision tree fit to the error indicator. Conditions that do not "
         "raise the subgroup's error metric by at least 10% are removed so descriptions stay "
         "minimal, and the numeric thresholds of shortlisted candidates are re-optimized on a "
         "finer 40-quantile grid. All of this uses the discovery data only."),
        ("Honest validation",
         "When the data allow it, rows are split (stratified by the error indicator) into a "
         f"discovery part and a validation part ({int(config.get('holdout_fraction', 0.5) * 100)}%"
         " of rows). Candidates are generated on the discovery part only and tested on the "
         "validation part, so the tested hypotheses are fixed before the validation data are "
         "seen. Descriptive figures (rows, error rate, lift) are reported on the full data; "
         "p-values, confidence intervals and effect sizes come from the validation part. Small "
         "datasets fall back to in-sample testing, which is flagged as exploratory."),
        ("Statistical tests",
         "Each candidate is compared with the rest of its population using a one-sided test "
         "(alternative: the subgroup is worse). Error rates use the exact Fisher test "
         "(hypergeometric tail); absolute errors use Welch's t-test. Error-rate confidence "
         "intervals are Wilson score intervals; mean-error intervals are Student-t intervals; "
         "relative risks use Katz log intervals. Effect sizes: Cohen's h (rates) and Cohen's d "
         "(means)."),
        ("Multiple-testing correction",
         f"p-values are adjusted with the {correction} procedure at α = {alpha}. With honest "
         "validation the family is exactly the set of candidates tested on the validation "
         "part. With in-sample testing the family is every candidate evaluated during the "
         "search (untested candidates count with p = 1), which is conservative for the "
         "adjustment but cannot remove the optimism of adaptive search. Benjamini–Hochberg is "
         "the default because overlapping subgroups yield positively dependent tests, under "
         "which it controls the FDR; Holm/Bonferroni (FWER) and Benjamini–Yekutieli are "
         "available for stricter guarantees."),
        ("Ranking and de-duplication",
         "Only patterns with adjusted p < α, a validated lift ≥ "
         f"{config.get('min_lift', 1.1)} and at least {config.get('min_samples', 30)} rows are "
         "reported. They are ranked by a conservative excess-error score, "
         "n · (CI lower bound − baseline), divided by 1 + 0.25·(conditions − 1). A pattern is "
         "dropped as redundant if it overlaps a better-ranked one with Jaccard similarity ≥ "
         f"{config.get('dedup_threshold', 0.7)}, if it is a sub-region that is not detectably "
         "worse than its parent, or if its rows outside the better pattern retain less than "
         "half of its excess error."),
        ("Feature association",
         "Classification: numeric features are compared between misclassified and correct rows "
         "with the Mann–Whitney U test (effect: rank-biserial correlation); categorical "
         "features with a chi-square test of independence (effect: Cramér's V). Regression: "
         "Spearman correlation with |residual| and Kruskal–Wallis (effect: ε²). p-values are "
         "Benjamini–Hochberg adjusted across features. Magnitude labels use conventional "
         "thresholds and are heuristics."),
    ]


LIMITATIONS = [
    "Associations are not causes. A subgroup's elevated error rate can be driven by label "
    "noise, unobserved variables, or correlated features; changing a feature value will not "
    "necessarily change the model's accuracy.",
    "Only conjunctions of simple threshold, equality and missingness conditions are searched, "
    "up to the configured depth. Failure modes that require other shapes (e.g. ratios of "
    "features, smooth interactions, text or image content) may be missed or only approximated.",
    "Statistical power depends on sample size. Small evaluation sets can only reveal large, "
    "frequent failure patterns; absence of a reported pattern is not evidence that the model "
    "performs uniformly.",
    "Honest validation halves the data used for each step, trading power for valid inference. "
    "In-sample (exploratory) results must be confirmed on independent data.",
    "Patterns are found on this evaluation set. Under distribution shift (new time periods, "
    "populations or data pipelines) they may not hold.",
    "Thresholds are rounded for readability, and quantile-based thresholds approximate the "
    "best split point; the exact boundary of a failure region is uncertain.",
    "The error metric is the model's hard prediction (classification) or absolute residual "
    "(regression); calibration and cost-sensitive errors are not analysed in this version.",
    "Some model errors are label errors. Consider a label-quality tool (e.g. cleanlab) before "
    "acting on patterns concentrated in hard-to-label rows.",
]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _metric(p: FailurePattern, value: float) -> str:
    return fmt_pct(value) if p.is_binary else fmt_num(value)


def _ci(p: FailurePattern) -> str:
    v = p.validation
    return f"{_metric(p, v.ci_low)} – {_metric(p, v.ci_high)}"


def pattern_table(patterns: list[FailurePattern], caption: str = "") -> Table:
    if not patterns:
        return Table(["Result"], [["No statistically significant pattern"]], caption)
    binary = patterns[0].is_binary
    if binary:
        columns = ["#", "Pattern", "Rows", "Errors", "Rate", "Baseline", "Lift", "95% CI",
                   "Adj. p"]
        rows = [[str(p.rank), p.description, fmt_int(p.n_samples), fmt_int(p.n_errors),
                 fmt_pct(p.rate), fmt_pct(p.baseline), fmt_lift(p.lift), _ci(p),
                 fmt_p(p.p_value_adjusted)] for p in patterns]
        align = ["r", "l", "r", "r", "r", "r", "r", "r", "r"]
        widths = [0.35, 3.3, 0.75, 0.75, 0.75, 0.95, 0.65, 1.6, 0.9]
    else:
        columns = ["#", "Pattern", "Rows", "MAE", "Baseline", "Lift", "95% CI",
                   "Mean resid.", "Adj. p"]
        rows = [[str(p.rank), p.description, fmt_int(p.n_samples), fmt_num(p.rate),
                 fmt_num(p.baseline), fmt_lift(p.lift), _ci(p),
                 fmt_num(p.mean_residual) if p.mean_residual is not None else "n/a",
                 fmt_p(p.p_value_adjusted)] for p in patterns]
        align = ["r", "l", "r", "r", "r", "r", "r", "r", "r"]
        widths = [0.35, 3.1, 0.75, 0.75, 0.9, 0.65, 1.6, 1.0, 0.9]
    return Table(columns, rows, caption, align, widths)


def pattern_card(p: FailurePattern) -> PatternCard:
    v = p.validation
    label = "held-out" if v.inference == "holdout" else "in-sample, exploratory"
    stats: list[tuple[str, str]] = [
        ("Rows", f"{fmt_int(p.n_samples)} ({fmt_pct(p.coverage, 2)} of population)"),
    ]
    if p.is_binary:
        stats += [
            (p.event_noun.capitalize(), f"{fmt_int(p.n_errors)} "
             f"({fmt_pct(p.stats.event_coverage)} of all {p.event_noun})"),
            (p.metric_name.capitalize(), fmt_pct(p.rate, 2)),
            ("Baseline", fmt_pct(p.baseline, 2)),
            ("Lift", fmt_lift(p.lift)),
            ("Relative risk vs rest", f"{fmt_num(v.relative_risk)} "
             f"[{fmt_num(v.rr_ci_low)}, {fmt_num(v.rr_ci_high)}]"),
        ]
    else:
        stats += [
            (p.metric_name, fmt_num(p.rate)),
            ("Baseline", fmt_num(p.baseline)),
            ("Lift", fmt_lift(p.lift)),
        ]
        if p.mean_residual is not None:
            direction = "underpredicts" if p.mean_residual > 0 else "overpredicts"
            stats.append(("Mean residual", f"{fmt_num(p.mean_residual)} (model {direction})"))
    stats += [
        (f"95% CI ({label})", _ci(p)),
        (f"p-value ({label})", fmt_p(v.p_value)),
        (f"Adjusted p ({p.correction})", fmt_p(p.p_value_adjusted)),
        ("Effect size", f"{fmt_num(v.effect_size)} ({v.effect_size_name})"),
        ("Complexity", f"{p.complexity} condition{'s' if p.complexity != 1 else ''}"),
    ]
    note = ""
    if v.inference == "holdout":
        note = (f"Validated on {fmt_int(v.population_size)} held-out rows: {_metric(p, v.value)} "
                f"vs {_metric(p, v.baseline)} baseline ({fmt_lift(v.lift)}).")
    return PatternCard(p.rank, [str(c) for c in p.conditions], p.headline(), stats, note)


def _figure(figs: dict[str, ReportFigure], key: str) -> list[Figure]:
    f = figs.get(key)
    if f is None:
        return []
    return [Figure(f.key, f.png, f.caption, f.width_in, f.height_in)]


def _patterns_blocks(patterns: list[FailurePattern], figs: dict[str, ReportFigure],
                     fig_key: str, cards: int, empty_text: str) -> list[Any]:
    if not patterns:
        return [Callout(empty_text, "info")]
    blocks: list[Any] = [pattern_table(patterns)]
    blocks += _figure(figs, fig_key)
    for p in patterns[:cards]:
        blocks.append(pattern_card(p))
    return blocks


# --------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------


def build_report(result: AnalysisResult, include_figures: bool = True,
                 max_cards: int = 10) -> Report:
    """Create the format-neutral report document for ``result``."""
    figs: dict[str, ReportFigure] = {}
    figures_included = False
    warnings = list(result.warnings)
    if include_figures:
        from errorlens.visualization.report_plots import (
            matplotlib_available,
            render_report_figures,
        )

        if matplotlib_available():
            # Figures are rendered once per result and reused by every export format.
            if result._figure_cache is None:
                result._figure_cache = render_report_figures(result)
            figs = result._figure_cache
            figures_included = True
        else:
            warnings.append("Charts were omitted because matplotlib is not installed "
                            "(pip install \"errorlens[viz]\").")

    md = result.metadata
    perf = result.performance
    model = md.model_info.get("type", "unknown")
    meta = [
        ("Generated", md.created_at.replace("T", " ")),
        ("ErrorLens version", md.errorlens_version),
        ("Task", md.task),
        ("Model", str(model)),
        ("Rows analysed", fmt_int(result.dataset.n_samples)),
        ("Runtime", f"{md.runtime_seconds:.1f} s"),
    ]
    sections: list[Section] = []

    # 1. Executive summary ---------------------------------------------------------------
    s = Section("Executive summary", "summary")
    tiles: list[tuple[str, str, str]] = []
    if isinstance(perf, ClassificationPerformance):
        tiles = [("Accuracy", fmt_pct(perf.accuracy),
                  f"balanced {fmt_pct(perf.balanced_accuracy)}"),
                 ("Error rate", fmt_pct(perf.error_rate), f"{fmt_int(perf.n_errors)} errors"),
                 ("Samples", fmt_int(perf.n_samples), f"{result.dataset.n_features} features"),
                 ("Failure patterns", str(len(result.patterns)), "statistically significant")]
    elif isinstance(perf, RegressionPerformance):
        tiles = [("MAE", fmt_num(perf.mae), f"median {fmt_num(perf.median_ae)}"),
                 ("RMSE", fmt_num(perf.rmse), f"R² {fmt_num(perf.r2)}"),
                 ("Samples", fmt_int(perf.n_samples), f"{result.dataset.n_features} features"),
                 ("High-error regions", str(len(result.patterns)), "statistically significant")]
    s.add(Tiles(tiles))
    findings = [p.headline() for p in result.patterns[:3]]
    for kind, pats in result.directional_patterns.items():
        if pats:
            findings.append(f"{KIND_LABELS[kind]}: {pats[0].headline()}")
    if findings:
        s.add(Paragraph("Key findings", "lead"), BulletList(findings))
    else:
        s.add(Paragraph("No subgroup with a statistically significant elevated error rate was "
                        "found at the configured significance level.", "lead"))
    s.add(Callout(ASSOCIATION_NOTE, "info"))
    sections.append(s)

    # 2. Dataset -------------------------------------------------------------------------
    ds = result.dataset
    s = Section("Dataset", "dataset")
    s.add(KeyValues([
        ("Rows", fmt_int(ds.n_samples)), ("Features", str(ds.n_features)),
        ("Numeric features", str(ds.n_numeric)),
        ("Categorical / boolean features", str(ds.n_categorical)),
        ("Missing cells", f"{fmt_int(ds.n_missing_cells)} ({fmt_pct(ds.pct_missing, 2)})"),
        ("Target", ds.target_name),
    ]))
    if ds.target_distribution:
        total = sum(ds.target_distribution.values())
        s.add(Table(["Class", "Rows", "Share"],
                    [[k, fmt_int(v), fmt_pct(v / total)] for k, v in
                     list(ds.target_distribution.items())[:30]],
                    "Target distribution", ["l", "r", "r"]))
    s.add(Table(["Feature", "Type", "Unique", "Missing", "Note"],
                [[f.name, f.kind, fmt_int(f.n_unique), fmt_int(f.n_missing),
                  f.skipped_reason or ""] for f in ds.features],
                "Feature overview", ["l", "l", "r", "r", "l"], [2.2, 1.2, 0.9, 0.9, 3.5]))
    sections.append(s)

    # 3. Model performance --------------------------------------------------------------
    s = Section("Model performance", "performance")
    model_rows = [(k.replace("_", " "), str(v)) for k, v in md.model_info.items()
                  if k in ("type", "module", "n_features_in")]
    if "pipeline_steps" in md.model_info:
        model_rows.append(("pipeline", " → ".join(md.model_info["pipeline_steps"])))
    if model_rows:
        s.add(Heading("Model"), KeyValues(model_rows))
    s.add(Heading("Metrics"))
    if isinstance(perf, ClassificationPerformance):
        kv = [("Accuracy", fmt_pct(perf.accuracy)), ("Error rate", fmt_pct(perf.error_rate)),
              ("Balanced accuracy", fmt_pct(perf.balanced_accuracy)),
              ("Macro precision", fmt_pct(perf.macro_precision)),
              ("Macro recall", fmt_pct(perf.macro_recall)), ("Macro F1", fmt_pct(perf.macro_f1))]
        if perf.roc_auc is not None:
            kv.append(("ROC AUC", fmt_num(perf.roc_auc)))
        if perf.log_loss is not None:
            kv.append(("Log loss", fmt_num(perf.log_loss)))
        s.add(KeyValues(kv))
        s.add(Table(["Class", "Precision", "Recall", "F1", "Support", "Missed", "False alarms"],
                    [[str(c.label), fmt_pct(c.precision), fmt_pct(c.recall), fmt_pct(c.f1),
                      fmt_int(c.support), fmt_int(c.n_missed), fmt_int(c.n_false_alarms)]
                     for c in perf.per_class],
                    "Per-class metrics. Missed = rows of the class predicted as another class; "
                    "false alarms = rows of other classes predicted as this class.",
                    ["l", "r", "r", "r", "r", "r", "r"]))
        s.add(*_figure(figs, "confusion"))
    elif isinstance(perf, RegressionPerformance):
        kv = [("MAE", fmt_num(perf.mae)), ("RMSE", fmt_num(perf.rmse)),
              ("Median absolute error", fmt_num(perf.median_ae)), ("R²", fmt_num(perf.r2)),
              ("Max error", fmt_num(perf.max_error)),
              ("Mean residual (actual − predicted)", fmt_num(perf.mean_residual))]
        if perf.mape is not None:
            kv.append(("MAPE", fmt_pct(perf.mape)))
        s.add(KeyValues(kv))
    sections.append(s)

    # 4. Error breakdown ----------------------------------------------------------------
    es = result.error_summary
    s = Section("Error breakdown", "errors")
    if isinstance(perf, ClassificationPerformance):
        kv = [("Errors", f"{fmt_int(es.n_errors)} of {fmt_int(perf.n_samples)} "
                         f"({fmt_pct(es.error_rate)})")]
        if perf.is_binary:
            kv += [("Positive class", str(perf.positive_label)),
                   ("True positives", fmt_int(perf.tp)), ("True negatives", fmt_int(perf.tn)),
                   ("False positives", f"{fmt_int(perf.fp)} (FPR "
                                       f"{fmt_pct(perf.false_positive_rate)})"),
                   ("False negatives", f"{fmt_int(perf.fn)} (FNR "
                                       f"{fmt_pct(perf.false_negative_rate)})")]
        if es.mean_confidence_errors is not None:
            kv += [("Mean confidence on errors", fmt_pct(es.mean_confidence_errors)),
                   ("Mean confidence on correct", fmt_pct(es.mean_confidence_correct))]
            kv.append((f"High-confidence errors (≥ {fmt_pct(es.high_confidence_threshold, 0)})",
                       fmt_int(es.n_high_confidence_errors)))
        s.add(KeyValues(kv))
        if es.top_confusions:
            s.add(Table(["Actual", "Predicted", "Count", "Share of errors"],
                        [[str(c.true_label), str(c.predicted_label), fmt_int(c.count),
                          fmt_pct(c.share_of_errors)] for c in es.top_confusions],
                        "Most frequent confusions", ["l", "l", "r", "r"]))
    else:
        s.add(KeyValues([
            ("Underpredicted rows (actual > predicted)", fmt_int(es.n_underpredicted)),
            ("Overpredicted rows (actual < predicted)", fmt_int(es.n_overpredicted)),
            ("Severe-error threshold (90th pct of |residual|)", fmt_num(es.severe_threshold)),
            ("Severe underpredictions", fmt_int(es.n_severe_under)),
            ("Severe overpredictions", fmt_int(es.n_severe_over)),
        ]))
    sections.append(s)

    # 5. Failure patterns ---------------------------------------------------------------
    s = Section("Failure patterns" if result.is_classification else "High-error regions",
                "patterns")
    stats_main = next((st for st in result.discovery_statistics
                       if st.kind == result.primary_kind), None)
    if stats_main is not None:
        s.add(Paragraph(
            f"ErrorLens evaluated {fmt_int(stats_main.n_candidates_evaluated)} candidate "
            f"subgroups built from {fmt_int(stats_main.n_conditions)} conditions, tested "
            f"{fmt_int(stats_main.n_candidates_tested)} on "
            + ("a held-out validation split" if stats_main.honest
               else "the same data (exploratory)") +
            f", and found {stats_main.n_significant} significant after correction; "
            f"{stats_main.n_reported} distinct patterns are reported after de-duplication.",
            "muted"))
    s.blocks += _patterns_blocks(result.patterns, figs, f"patterns_{result.primary_kind}",
                                 max_cards, "No subgroup had a statistically significant "
                                 "elevated error after multiple-testing correction.")
    sections.append(s)

    # 6. Feature error analysis ---------------------------------------------------------
    s = Section("Feature error analysis", "features")
    if result.feature_analysis:
        s.add(Paragraph(
            ("Each feature is compared between misclassified and correctly classified rows."
             if result.is_classification else
             "Each feature is related to the absolute error of the predictions.") +
            " p-values are Benjamini–Hochberg adjusted across features.", "muted"))
        s.add(Table(["Feature", "Type", "Test", "Effect size", "Magnitude", "Adj. p",
                     "Observation"],
                    [[a.feature, a.feature_type, a.test,
                      f"{fmt_num(a.effect_size)} ({a.effect_size_name.replace('_', ' ')})",
                      a.magnitude, fmt_p(a.p_value_adjusted), a.direction]
                     for a in result.feature_analysis],
                    "Feature error association", ["l", "l", "l", "r", "l", "r", "l"],
                    [1.6, 0.9, 1.9, 1.7, 0.9, 0.7, 2.3]))
        for i in range(4):
            s.add(*_figure(figs, f"feature_{i}"))
        s.add(*_figure(figs, "distribution"))
    else:
        s.add(Callout("Feature analysis requires both errors and correct predictions.", "info"))
    sections.append(s)

    # 7./8. Directional analyses --------------------------------------------------------
    if isinstance(perf, ClassificationPerformance) and perf.is_binary:
        fp = result.directional_patterns.get("false_positive", [])
        s = Section("False-positive analysis", "false-positives")
        s.add(Paragraph(
            f"Among {fmt_int((perf.tn or 0) + (perf.fp or 0))} actual negatives, "
            f"{fmt_int(perf.fp)} were predicted as '{perf.positive_label}' (false positive "
            f"rate {fmt_pct(perf.false_positive_rate)}). Patterns below have a higher false "
            "positive rate than that baseline.", "muted"))
        s.blocks += _patterns_blocks(fp, figs, "patterns_false_positive", 5,
                                     "No significant false-positive pattern was found.")
        sections.append(s)
        fn = result.directional_patterns.get("false_negative", [])
        s = Section("False-negative analysis", "false-negatives")
        s.add(Paragraph(
            f"Among {fmt_int((perf.tp or 0) + (perf.fn or 0))} actual positives, "
            f"{fmt_int(perf.fn)} were missed (false negative rate "
            f"{fmt_pct(perf.false_negative_rate)}). Patterns below have a higher false "
            "negative rate than that baseline.", "muted"))
        s.blocks += _patterns_blocks(fn, figs, "patterns_false_negative", 5,
                                     "No significant false-negative pattern was found.")
        sections.append(s)
    elif isinstance(perf, ClassificationPerformance) and result.class_patterns_map:
        s = Section("Class-specific analysis", "classes")
        s.add(Paragraph("For each class, subgroups of that class's rows where the model misses "
                        "the class unusually often (miss rate = 1 − recall).", "muted"))
        for label, pats in result.class_patterns_map.items():
            s.add(Heading(f"Class {label}", 3))
            s.blocks += _patterns_blocks(pats, figs, "", 2,
                                         f"No significant pattern for class {label}.")
        sections.append(s)

    # 9. Residual analysis --------------------------------------------------------------
    if result.residuals is not None:
        ra = result.residuals
        s = Section("Residual analysis", "residuals")
        s.add(KeyValues([
            ("Mean residual", f"{fmt_num(ra.mean_residual)} (one-sample t-test p "
                              f"{fmt_p(ra.bias_test_p)})"),
            ("Median residual", fmt_num(ra.median_residual)),
            ("Std of residuals", fmt_num(ra.std_residual)),
            ("Underpredicted", f"{fmt_pct(ra.pct_underpredicted)} of rows, mean "
                               f"{fmt_num(ra.mean_underprediction)}"),
            ("Overpredicted", f"{fmt_pct(ra.pct_overpredicted)} of rows, mean "
                              f"{fmt_num(ra.mean_overprediction)}"),
            ("Skewness / excess kurtosis", f"{fmt_num(ra.skewness)} / "
                                           f"{fmt_num(ra.excess_kurtosis)}"),
            ("|error| vs prediction (Spearman)", f"{fmt_num(ra.heteroscedasticity_rho)} "
                                                 f"(p {fmt_p(ra.heteroscedasticity_p)})"),
        ] + ([("Median relative error", fmt_pct(ra.median_relative_error))]
             if ra.median_relative_error is not None else [])))
        s.add(BulletList(ra.interpretation()))
        s.add(*_figure(figs, "residuals"))
        s.add(*_figure(figs, "residual_hist"))
        if ra.residual_by_prediction_decile:
            s.add(Table(["Decile", "Predicted range", "Rows", "Mean actual", "Mean predicted",
                         "Mean residual", "MAE"],
                        [[str(b["bin"]), f"{fmt_num(b['pred_min'])} – {fmt_num(b['pred_max'])}",
                          fmt_int(b["n"]), fmt_num(b["mean_actual"]),
                          fmt_num(b["mean_predicted"]), fmt_num(b["mean_residual"]),
                          fmt_num(b["mae"])] for b in ra.residual_by_prediction_decile],
                        "Errors by prediction decile", ["r", "l", "r", "r", "r", "r", "r"]))
        if ra.feature_residual_correlations:
            s.add(Table(["Feature", "Measure", "Value", "Adj. p"],
                        [[row["feature"], row["measure"].replace("_", " "),
                          fmt_num(row["value"]), fmt_p(row.get("p_value_adjusted"))]
                         for row in ra.feature_residual_correlations],
                        "Association between features and the signed residual (systematic "
                        "over/under-prediction)", ["l", "l", "r", "r"]))
        for kind in ("underprediction", "overprediction"):
            pats = result.directional_patterns.get(kind, [])
            s.add(Heading(KIND_LABELS[kind] + " patterns", 3))
            s.blocks += _patterns_blocks(pats, figs, f"patterns_{kind}", 3,
                                         f"No significant {kind} pattern was found.")
        sections.append(s)

    # 10. Interactions -----------------------------------------------------------------
    if result.interaction_grid is not None and "heatmap" in figs:
        g = result.interaction_grid
        s = Section("Feature interactions", "interactions")
        s.add(Paragraph(f"Error metric across combinations of '{g.feature_x}' and "
                        f"'{g.feature_y}', the feature pair most involved in the discovered "
                        "patterns.", "muted"))
        s.add(*_figure(figs, "heatmap"))
        sections.append(s)

    # 11. Statistical analysis ---------------------------------------------------------
    s = Section("Statistical analysis", "statistics")
    s.add(Paragraph(f"Significance level α = {md.config.get('significance_level')}; "
                    f"correction: {CORRECTION_NAMES.get(md.config.get('correction', ''), '')}.",
                    "muted"))
    s.add(Table(
        ["Target", "Rows", "Baseline", "Testing", "Conditions", "Evaluated", "Tested",
         "Family", "Signif.", "Reported"],
        [[KIND_LABELS.get(st.kind, st.kind) + (f" ({st.target_label})" if st.kind ==
                                               "class_error" else ""),
          fmt_int(st.population_size),
          fmt_pct(st.baseline) if st.n_events is not None else fmt_num(st.baseline),
          (f"held-out, {fmt_int(st.n_validation)}" if st.honest else "in-sample"),
          fmt_int(st.n_conditions), fmt_int(st.n_candidates_evaluated),
          fmt_int(st.n_candidates_tested), fmt_int(st.family_size), fmt_int(st.n_significant),
          fmt_int(st.n_reported)] for st in result.discovery_statistics],
        "Discovery runs. Rows: population of the target; Testing: where candidates were tested "
        "(rows in the held-out split); Evaluated: candidate subgroups scored during search; "
        "Family: number of hypotheses the multiple-testing correction accounts for.",
        ["l", "r", "r", "l", "r", "r", "r", "r", "r", "r"],
        [2.0, 0.75, 0.8, 1.2, 0.95, 0.95, 0.75, 0.75, 0.75, 0.9]))
    notes = [f"{KIND_LABELS.get(st.kind, st.kind)}: {st.note}"
             for st in result.discovery_statistics if st.note]
    if notes:
        s.add(BulletList(notes))
    sections.append(s)

    # 12. Warnings ----------------------------------------------------------------------
    s = Section("Warnings", "warnings")
    if warnings:
        for w in warnings:
            s.add(Callout(w, "warning"))
    else:
        s.add(Paragraph("No warnings.", "muted"))
    sections.append(s)

    # 13. Methodology / 14. Limitations --------------------------------------------------
    s = Section("Methodology", "methodology")
    for title, text in methodology_text(md.config):
        s.add(Heading(title, 3), Paragraph(text))
    sections.append(s)
    sections.append(Section("Limitations", "limitations", [BulletList(LIMITATIONS),
                                                            Callout(ASSOCIATION_NOTE, "info")]))

    return Report(
        title=md.title,
        subtitle="ML model error analysis",
        created_at=md.created_at,
        meta=meta,
        sections=sections,
        figures_included=figures_included,
        footer=f"Generated by ErrorLens {md.errorlens_version}",
    )
