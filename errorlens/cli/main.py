"""Command-line interface: ``errorlens analyze`` and ``errorlens version``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd

from errorlens._version import __version__
from errorlens.core.config import VALID_CORRECTIONS
from errorlens.exceptions import ErrorLensError
from errorlens.reporting.export import infer_format


def _load_model(path: Path) -> Any:
    """Load a pickle / joblib model file. Only load files you trust: unpickling runs code."""
    import joblib

    try:
        return joblib.load(path)
    except Exception as exc:
        raise click.BadParameter(f"Could not load a model from {path.name}: {exc}",
                                 param_hint="--model") from exc


def _load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise click.BadParameter(f"Unsupported data format {suffix!r}; use .csv, .tsv or .parquet.",
                             param_hint="--data")


def _parse_label(value: str | None, y: pd.Series) -> Any:
    """Convert --positive-label to the dtype of the target column."""
    if value is None:
        return None
    for candidate in pd.unique(y):
        if str(candidate) == value:
            return candidate
    raise click.BadParameter(f"{value!r} is not a value of the target column.",
                             param_hint="--positive-label")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="errorlens")
def cli() -> None:
    """ErrorLens — automatic failure-pattern discovery for ML models."""


@cli.command()
def version() -> None:
    """Print the ErrorLens version."""
    click.echo(f"errorlens {__version__}")


@cli.command()
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False,
                                                       path_type=Path),
              help="Pickled/joblib model with a predict() method (load trusted files only).")
@click.option("--data", "data_path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Evaluation data (.csv, .tsv, .parquet) including the target column.")
@click.option("--target", required=True, help="Name of the target column in --data.")
@click.option("--prediction-column", default=None,
              help="Column with precomputed predictions (instead of --model).")
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Report path (.html, .pdf, .md, .json).")
@click.option("--format", "fmt", type=click.Choice(["html", "pdf", "md", "json"]),
              default=None, help="Report format (default: inferred from --output).")
@click.option("--task", type=click.Choice(["auto", "classification", "regression"]),
              default="auto", show_default=True)
@click.option("--min-samples", type=click.IntRange(min=2), default=30, show_default=True,
              help="Minimum rows per reported pattern.")
@click.option("--max-patterns", type=click.IntRange(min=1), default=20, show_default=True,
              help="Maximum patterns reported per target.")
@click.option("--significance-level", type=click.FloatRange(0, 1, min_open=True,
                                                            max_open=True),
              default=0.05, show_default=True, help="Alpha after multiple-testing correction.")
@click.option("--correction", type=click.Choice(list(VALID_CORRECTIONS)), default="fdr_bh",
              show_default=True, help="Multiple-testing correction.")
@click.option("--max-depth", type=click.IntRange(min=1, max=6), default=3, show_default=True,
              help="Maximum conditions per pattern.")
@click.option("--positive-label", default=None, help="Positive class (binary classification).")
@click.option("--drop", multiple=True, help="Column to exclude from the analysis (repeatable).")
@click.option("--seed", type=int, default=0, show_default=True,
              help="Random seed for the discovery/validation split.")
@click.option("--title", default="ErrorLens Error Analysis", show_default=True)
@click.option("--no-figures", is_flag=True, help="Do not embed charts in the report.")
@click.option("--quiet", "-q", is_flag=True, help="Do not print the summary.")
def analyze(model_path: Path | None, data_path: Path, target: str,
            prediction_column: str | None, output: Path | None, fmt: str | None, task: str,
            min_samples: int, max_patterns: int, significance_level: float, correction: str,
            max_depth: int, positive_label: str | None, drop: tuple[str, ...], seed: int,
            title: str, no_figures: bool, quiet: bool) -> None:
    """Analyze a model's errors on labelled data and write a report.

    \b
    Examples:
      errorlens analyze --model model.pkl --data test.csv --target label -o report.pdf
      errorlens analyze --data scored.csv --target y --prediction-column y_hat -o report.html
    """
    from errorlens import ErrorLens

    if (model_path is None) == (prediction_column is None):
        raise click.UsageError("Provide exactly one of --model or --prediction-column.")
    if output is None and fmt is not None:
        raise click.UsageError("--format requires --output.")
    if output is not None:
        try:
            infer_format(output, fmt)
        except ErrorLensError as exc:
            raise click.BadParameter(str(exc), param_hint="--output") from exc

    df = _load_data(data_path)
    missing = [c for c in [target, *([prediction_column] if prediction_column else []), *drop]
               if c not in df.columns]
    if missing:
        raise click.BadParameter(f"Columns not found in {data_path.name}: {missing}. "
                                 f"Available: {list(df.columns)[:20]}", param_hint="--target")
    y = df[target]
    X = df.drop(columns=[target, *drop, *([prediction_column] if prediction_column else [])])
    options: dict[str, Any] = {
        "task": task, "min_samples": min_samples, "max_patterns": max_patterns,
        "significance_level": significance_level, "correction": correction,
        "max_depth": max_depth, "positive_label": _parse_label(positive_label, y),
        "random_state": seed, "title": title,
    }

    try:
        if prediction_column is not None:
            lens = ErrorLens.from_predictions(X, y, df[prediction_column], **options)
        else:
            assert model_path is not None
            model = _load_model(model_path)
            lens = ErrorLens(model, X, y, **options)
        result = lens.analyze()
        if not quiet:
            result.summary()
        if output is not None:
            path = result.export(output, format=fmt, include_figures=not no_figures) \
                if (fmt or infer_format(output)) != "json" else result.export(output, format=fmt)
            click.echo(f"\nReport written to {path}", err=quiet)
    except ErrorLensError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


def main() -> None:
    """Console-script entry point."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
