import pandas as pd
import pytest

from medwatch.anomaly.mimic import build_stay_grid, clean_values, iter_stays, load_chartevents
from tests.unit.anomaly.helpers import load_config, make_stay, write_fake_chartevents

CONFIG = load_config()
T0 = pd.Timestamp("2180-01-01 10:00")


def events(*rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["charttime", "vital", "valuenum", "value"])
    frame["charttime"] = pd.to_datetime(frame["charttime"])
    return frame


def test_clean_values_converts_fahrenheit_and_drops_artifacts():
    raw = events(
        (T0, "temp_f", 98.6, "98.6"),
        (T0, "temp", 99.0, "99"),  # °F digitado no campo de °C
        (T0, "hr", 0.0, "0"),  # sensor desconectado
        (T0, "spo2", 29.0, "29"),  # artefato de oximetria
        (T0, "hr", 85.0, "85"),
    )
    cleaned, removed = clean_values(raw, CONFIG)
    temps = cleaned[cleaned["vital"] == "temp"]["valuenum"].round(1).tolist()
    assert temps == [37.0, 37.2]
    assert removed == 2
    assert cleaned[cleaned["vital"] == "hr"]["valuenum"].tolist() == [85.0]


def test_build_stay_grid_resamples_and_derives_oxygen_and_alertness():
    raw = events(
        (T0, "hr", 80.0, "80"),
        (T0 + pd.Timedelta("20min"), "hr", 90.0, "90"),
        (T0, "o2_device", None, "Nasal cannula"),
        (T0 + pd.Timedelta("1h"), "hr", 100.0, "100"),
        (T0 + pd.Timedelta("1h"), "o2_device", None, "Room air"),
        (T0 + pd.Timedelta("1h"), "gcs_eye", 3.0, "To Speech"),
    )
    grid = build_stay_grid(raw, CONFIG)
    assert grid["hr"].tolist() == [85.0, 100.0]  # mediana dentro da hora
    assert grid["on_oxygen"].tolist() == [1.0, 0.0]
    assert pd.isna(grid["alert"].iloc[0]) and grid["alert"].iloc[1] == 0.0


def test_iter_stays_reads_chartevents(tmp_path):
    write_fake_chartevents(
        tmp_path, [make_stay(subject_id=1, stay_id=10), make_stay(subject_id=2, stay_id=20)]
    )
    stays = iter_stays(tmp_path, CONFIG)
    assert [(s.patient_id, s.stay_id) for s in stays] == [("MIMIC-1", 10), ("MIMIC-2", 20)]
    assert len(stays[0].raw) == 48
    assert iter_stays(tmp_path, CONFIG, subject_ids=[2])[0].subject_id == 2


def test_load_chartevents_keeps_only_vital_items(tmp_path):
    write_fake_chartevents(tmp_path, [make_stay()])
    loaded = load_chartevents(tmp_path)
    assert set(loaded["vital"]) == {"hr", "sbp", "dbp", "rr", "spo2", "temp"}


@pytest.mark.slow
def test_real_mimic_demo_if_downloaded():
    from medwatch.config import PROJECT_ROOT

    dataset = PROJECT_ROOT / "data" / "raw" / "mimic-iv-demo-2.2"
    if not (dataset / "icu" / "chartevents.csv.gz").exists():
        pytest.skip("MIMIC-IV demo não baixado (medwatch data download)")
    stays = iter_stays(dataset, CONFIG)
    assert len(stays) == 140
    assert sum(s.artifacts for s in stays) > 0
