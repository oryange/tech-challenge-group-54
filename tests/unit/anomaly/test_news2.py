import pandas as pd
import pytest

from medwatch.anomaly.news2 import compute_news2, news2_severity, subscore_severity
from medwatch.core.events import Severity
from tests.unit.anomaly.helpers import load_config

CONFIG = load_config()


def news2_of(**vitals) -> pd.Series:
    frame = pd.DataFrame([vitals], index=pd.date_range("2180-01-01", periods=1))
    return compute_news2(frame, CONFIG).iloc[0]


@pytest.mark.parametrize(
    ("vitals", "total", "severity"),
    [
        # paciente estável em ar ambiente
        ({"rr": 16, "spo2": 97, "sbp": 120, "hr": 80, "temp": 37.0}, 0, Severity.INFO),
        # taquicardia leve isolada
        ({"rr": 16, "spo2": 97, "sbp": 120, "hr": 95, "temp": 37.0}, 1, Severity.LOW),
        # um único parâmetro vermelho (FR 26) → resposta urgente
        ({"rr": 26, "spo2": 97, "sbp": 120, "hr": 80, "temp": 37.0}, 3, Severity.MEDIUM),
        # FR 22 (2) + SpO2 94 (1) + FC 105 (1) + febre 38.5 (1) = 5
        ({"rr": 22, "spo2": 94, "sbp": 120, "hr": 105, "temp": 38.5}, 5, Severity.HIGH),
        # sepse provável: FR 26 (3) + SpO2 91 (3) + PAS 95 (2) + FC 125 (2) + 39.5 (2) = 12
        ({"rr": 26, "spo2": 91, "sbp": 95, "hr": 125, "temp": 39.5}, 12, Severity.CRITICAL),
    ],
)
def test_news2_known_cases(vitals, total, severity):
    row = news2_of(**vitals)
    assert row["news2_total"] == total
    assert row["news2_severity"] == severity


def test_oxygen_and_consciousness_add_points():
    base = {"rr": 16, "spo2": 97, "sbp": 120, "hr": 80, "temp": 37.0}
    assert news2_of(**base, on_oxygen=1.0)["news2_total"] == 2
    assert news2_of(**base, alert=0.0)["news2_total"] == 3
    # consciência alterada sozinha já é um parâmetro vermelho
    assert news2_of(**base, alert=0.0)["news2_severity"] == Severity.MEDIUM


def test_missing_oxygen_and_alert_assume_room_air_and_alert():
    row = news2_of(rr=16, spo2=97, sbp=120, hr=80, temp=37.0, on_oxygen=float("nan"))
    assert row["news2_total"] == 0


def test_row_without_vitals_has_no_score():
    frame = pd.DataFrame({"hr": [float("nan")]}, index=pd.date_range("2180-01-01", periods=1))
    row = compute_news2(frame, CONFIG).iloc[0]
    assert pd.isna(row["news2_total"])
    assert row["news2_severity"] is None


@pytest.mark.parametrize(
    ("total", "single", "expected"),
    [
        (0, 0, Severity.INFO),
        (4, 2, Severity.LOW),
        (3, 3, Severity.MEDIUM),
        (6, 3, Severity.HIGH),
        (7, 2, Severity.CRITICAL),
    ],
)
def test_news2_severity_bands(total, single, expected):
    assert news2_severity(total, single, CONFIG) == expected


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (0, Severity.LOW),
        (float("nan"), Severity.LOW),
        (1, Severity.MEDIUM),
        (2, Severity.MEDIUM),
        (3, Severity.HIGH),
    ],
)
def test_subscore_severity(points, expected):
    assert subscore_severity(points) == expected


@pytest.mark.parametrize(
    ("vital", "value", "points"),
    [("temp", 38.04, 0), ("temp", 38.06, 1), ("hr", 90.4, 0), ("hr", 90.6, 1), ("rr", 20.4, 0)],
)
def test_values_are_rounded_to_table_resolution_before_scoring(vital, value, points):
    base = {"rr": 16, "spo2": 97, "sbp": 120, "hr": 80, "temp": 37.0}
    assert news2_of(**(base | {vital: value}))[f"news2_{vital}"] == points
