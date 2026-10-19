import json

import pytest
from typer.testing import CliRunner

from medwatch.cli import app
from medwatch.config import get_settings
from medwatch.datasets import MIMIC_IV_DEMO
from tests.unit.anomaly.helpers import make_raw, make_stay, write_fake_chartevents

runner = CliRunner()


@pytest.fixture
def fake_dataset():
    dataset_dir = get_settings().data_dir / "raw" / MIMIC_IV_DEMO.dirname
    raw = make_raw(hours=72, seed=3)
    raw.loc[raw.index[40:42], "spo2"] = [88, 87]
    stays = [make_stay(raw, subject_id=1, stay_id=10)] + [
        make_stay(make_raw(hours=72, seed=s), subject_id=s, stay_id=s * 10) for s in (2, 3, 4)
    ]
    write_fake_chartevents(dataset_dir, stays)
    return dataset_dir


def test_vitals_requires_dataset():
    result = runner.invoke(app, ["vitals", "list"])
    assert result.exit_code == 1
    assert "medwatch data download" in result.output


def test_vitals_list(fake_dataset):
    result = runner.invoke(app, ["vitals", "list"])
    assert result.exit_code == 0
    assert "1            1      72" in result.stdout


def test_vitals_analyze_writes_reports_and_alerts(fake_dataset):
    result = runner.invoke(app, ["vitals", "analyze", "--patient", "1", "--alert"])
    assert result.exit_code == 0, result.output
    assert "spo2_drop" in result.stdout
    assert "Alertas enviados:" in result.stdout
    reports = get_settings().outputs_dir / "reports"
    assert (reports / "vitals_1_10.png").exists()
    payload = json.loads((reports / "vitals_1.json").read_text(encoding="utf-8"))
    assert payload["stays"][0]["events_by_type"]["spo2_drop"] == 1
    assert get_settings().alerts_file.exists()


def test_vitals_analyze_unknown_patient(fake_dataset):
    result = runner.invoke(app, ["vitals", "analyze", "--patient", "999"])
    assert result.exit_code == 1


def test_vitals_evaluate(fake_dataset):
    result = runner.invoke(app, ["vitals", "evaluate", "--per-stay", "2"])
    assert result.exit_code == 0, result.output
    assert "combined" in result.stdout
    reports = get_settings().outputs_dir / "reports"
    metrics = json.loads((reports / "vitals_evaluation.json").read_text(encoding="utf-8"))
    assert metrics["metrics"]["combined"]["injected"] == 8
    assert (reports / "vitals_evaluation_example.png").exists()
