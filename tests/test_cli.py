import json

import joblib
import pandas as pd
import pytest
from click.testing import CliRunner
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from errorlens import __version__
from errorlens.cli.main import cli
from tests.conftest import make_binary_case


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    X, y, y_pred = make_binary_case(3000, seed=3)
    model = make_pipeline(
        make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["region", "student"]),
                                remainder="passthrough"),
        HistGradientBoostingClassifier(max_iter=30, random_state=0))
    model.fit(X, y)
    joblib.dump(model, d / "model.pkl")
    df = X.assign(target=y, prediction=y_pred)
    df.to_csv(d / "test.csv", index=False)
    return d


def test_version():
    result = CliRunner().invoke(cli, ["version"])
    assert result.exit_code == 0 and __version__ in result.output
    result = CliRunner().invoke(cli, ["--version"])
    assert __version__ in result.output


def test_analyze_with_model_to_html(files):
    out = files / "report.html"
    result = CliRunner().invoke(cli, [
        "analyze", "--model", str(files / "model.pkl"), "--data", str(files / "test.csv"),
        "--target", "target", "--drop", "prediction", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "CLASSIFICATION ERROR ANALYSIS" in result.output
    assert out.exists() and "<html" in out.read_text(encoding="utf-8")


def test_analyze_predictions_to_json(files):
    out = files / "report.json"
    result = CliRunner().invoke(cli, [
        "analyze", "--data", str(files / "test.csv"), "--target", "target",
        "--prediction-column", "prediction", "-o", str(out), "--quiet",
        "--min-samples", "40", "--max-patterns", "5", "--significance-level", "0.01",
        "--correction", "holm"])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["metadata"]["config"]["min_samples"] == 40
    assert data["metadata"]["config"]["correction"] == "holm"
    assert len(data["patterns"]) <= 5
    assert data["patterns"], "planted pattern should be found"


def test_analyze_pdf_and_format_flag(files):
    pytest.importorskip("reportlab")
    out = files / "report.bin"
    result = CliRunner().invoke(cli, [
        "analyze", "--data", str(files / "test.csv"), "--target", "target",
        "--prediction-column", "prediction", "--output", str(out), "--format", "pdf", "-q",
        "--no-figures"])
    assert result.exit_code == 0, result.output
    assert out.read_bytes().startswith(b"%PDF")


def test_cli_errors(files):
    runner = CliRunner()
    r = runner.invoke(cli, ["analyze", "--data", str(files / "test.csv"), "--target", "nope",
                            "--prediction-column", "prediction"])
    assert r.exit_code == 2 and "nope" in r.output
    r = runner.invoke(cli, ["analyze", "--data", str(files / "test.csv"), "--target", "target"])
    assert r.exit_code == 2 and "exactly one of" in r.output
    r = runner.invoke(cli, ["analyze", "--data", str(files / "test.csv"), "--target", "target",
                            "--prediction-column", "prediction", "-o", "report.xlsx"])
    assert r.exit_code == 2
    bad = files / "bad.csv"
    pd.DataFrame({"a": range(5), "target": [0, 1, 0, 1, 0], "p": [0] * 5}).to_csv(bad,
                                                                                    index=False)
    r = runner.invoke(cli, ["analyze", "--data", str(bad), "--target", "target",
                            "--prediction-column", "p"])
    assert r.exit_code == 1 and "Error:" in r.output
