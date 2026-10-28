import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from medwatch.cli import app
from medwatch.config import get_settings
from medwatch.datasets import MIMIC_IV_DEMO

runner = CliRunner()
T0 = pd.Timestamp("2180-01-01 08:00")


@pytest.fixture
def fake_prescriptions():
    """MIMIC falso: paciente 1 com salto de dose de furosemida; 2 e 3 sem anomalias."""
    rows = []
    for subject in (1, 2, 3):
        for h in range(0, 120, 8):
            start, stop = T0 + pd.Timedelta(hours=h), T0 + pd.Timedelta(hours=h + 30)
            rows.append(
                (subject, subject * 10, start, stop, "MAIN", "Acetaminophen", "650", "mg", "PO")
            )
    rows.append(
        (
            1,
            10,
            T0 + pd.Timedelta(hours=2),
            T0 + pd.Timedelta(hours=20),
            "MAIN",
            "Furosemide",
            "20",
            "mg",
            "IV",
        )
    )
    rows.append(
        (
            1,
            10,
            T0 + pd.Timedelta(hours=10),
            T0 + pd.Timedelta(hours=40),
            "MAIN",
            "Furosemide",
            "80",
            "mg",
            "IV",
        )
    )
    columns = [
        "subject_id", "hadm_id", "starttime", "stoptime", "drug_type",
        "drug", "dose_val_rx", "dose_unit_rx", "route",
    ]  # fmt: skip
    path = get_settings().data_dir / "raw" / MIMIC_IV_DEMO.dirname / "hosp" / "prescriptions.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_prescriptions_requires_dataset():
    result = runner.invoke(app, ["prescriptions", "list"])
    assert result.exit_code == 1
    assert "medwatch data download" in result.output


def test_prescriptions_list(fake_prescriptions):
    result = runner.invoke(app, ["prescriptions", "list"])
    assert result.exit_code == 0, result.output
    assert "1            1           17" in result.stdout


def test_prescriptions_analyze_writes_reports_and_alerts(fake_prescriptions):
    result = runner.invoke(app, ["prescriptions", "analyze", "--patient", "1", "--alert"])
    assert result.exit_code == 0, result.output
    assert "dose_increase" in result.stdout
    assert "Furosemide 20.0 → 80.0 (4.0×)" in result.stdout
    reports = get_settings().outputs_dir / "reports"
    payload = json.loads((reports / "prescriptions_1.json").read_text(encoding="utf-8"))
    assert payload["stays"][0]["events_by_type"]["dose_increase"] == 1
    assert (reports / "prescriptions_1_10.png").exists()
    alerts = get_settings().alerts_file.read_text(encoding="utf-8").splitlines()
    assert any('"prescription.dose_increase"' in line for line in alerts)


def test_prescriptions_analyze_unknown_patient(fake_prescriptions):
    result = runner.invoke(app, ["prescriptions", "analyze", "--patient", "999"])
    assert result.exit_code == 1


def test_prescriptions_evaluate(fake_prescriptions):
    result = runner.invoke(app, ["prescriptions", "evaluate", "--per-stay", "1"])
    assert result.exit_code == 0, result.output
    assert "Internações avaliadas: 3 de 3" in result.stdout
    metrics = json.loads(
        (get_settings().outputs_dir / "reports" / "prescriptions_evaluation.json").read_text()
    )
    assert metrics["stays_evaluated"] == 3


def test_movement_analyze_simulated_demo_with_alerts():
    result = runner.invoke(app, ["movement", "analyze", "--patient", "SIM-001", "--alert"])
    assert result.exit_code == 0, result.output
    for kind in ("prolonged_immobility", "night_bed_exit", "fall_suspected", "activity_drop"):
        assert kind in result.stdout
    reports = get_settings().outputs_dir / "reports"
    assert (reports / "movement_SIM-001.png").exists()
    alerts = get_settings().alerts_file.read_text(encoding="utf-8").splitlines()
    assert any('"movement.fall_suspected"' in line for line in alerts)


def test_movement_analyze_from_file(tmp_path):
    index = pd.date_range("2026-01-05", periods=6 * 60, freq="1min")
    series = pd.DataFrame({"activity": 0.0, "in_bed": 1.0}, index=index)
    path = tmp_path / "serie.csv"
    series.to_csv(path)
    result = runner.invoke(app, ["movement", "analyze", "--input", str(path), "--patient", "PAC-9"])
    assert result.exit_code == 0, result.output
    assert "prolonged_immobility" in result.stdout


def test_movement_analyze_rejects_file_without_columns(tmp_path):
    path = tmp_path / "ruim.csv"
    pd.DataFrame({"x": [1]}, index=pd.date_range("2026-01-05", periods=1)).to_csv(path)
    result = runner.invoke(app, ["movement", "analyze", "--input", str(path)])
    assert result.exit_code == 1
    assert "Colunas ausentes" in result.output


def test_movement_evaluate():
    result = runner.invoke(app, ["movement", "evaluate"])
    assert result.exit_code == 0, result.output
    assert "Alarmes falsos em internações normais" in result.stdout
    assert (get_settings().outputs_dir / "reports" / "movement_evaluation.json").exists()


def test_movement_from_pose(tmp_path):
    times = np.arange(0, 4, 0.5)
    pose = pd.DataFrame(
        {
            "time": times,
            "bbox_y1": 0.0,
            "bbox_y2": 100.0,
            "kp_nose_x": 50 + times * 5,
            "kp_nose_y": 30.0,
            "hip_x": 40.0,
            "hip_y": 50.0,
        }
    )
    source, output = tmp_path / "clip_pose.parquet", tmp_path / "mov.parquet"
    pose.to_parquet(source)
    result = runner.invoke(
        app,
        [
            "movement",
            "from-pose",
            str(source),
            "--start-time",
            "2026-01-05T10:00:00-03:00",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    series = pd.read_parquet(output)
    assert {"activity", "in_bed"} <= set(series.columns)
    assert series.index[0].tz is not None


@pytest.mark.parametrize("value", ["ontem", "2026-01-05T10:00:00"])
def test_movement_from_pose_rejects_bad_start_time(tmp_path, value):
    result = runner.invoke(
        app,
        [
            "movement",
            "from-pose",
            str(tmp_path / "x.parquet"),
            "--start-time",
            value,
            "--output",
            str(tmp_path / "o.parquet"),
        ],
    )
    assert result.exit_code == 1
    assert "--start-time" in result.output
