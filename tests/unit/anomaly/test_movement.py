import numpy as np
import pandas as pd
import pytest

from medwatch.anomaly.movement import (
    MovementAnalyzer,
    MovementConfig,
    activity_from_pose,
    simulate_stay,
)
from medwatch.anomaly.movement_evaluation import evaluate_movement, inject_movement_anomalies
from medwatch.config import PROJECT_ROOT
from medwatch.core.events import Severity, Source

CONFIG = MovementConfig.load(PROJECT_ROOT / "configs" / "movement.yaml")
ANALYZER = MovementAnalyzer(CONFIG)
DAY = pd.Timestamp("2026-01-05")


def calm_series(hours: int = 24, start: pd.Timestamp = DAY) -> pd.DataFrame:
    """No leito, com um movimento a cada 30 min (nada deve disparar)."""
    index = pd.date_range(start, periods=hours * 60, freq="1min")
    activity = np.full(len(index), 0.01)
    activity[::30] = 0.2
    return pd.DataFrame({"activity": activity, "in_bed": 1.0}, index=index)


def events(series: pd.DataFrame, type_: str):
    return [e for e in ANALYZER.analyze("PAC-001", series).events if e.type == type_]


def test_calm_series_has_no_events():
    assert ANALYZER.analyze("PAC-001", calm_series()).events == []


def test_still_in_bed_beyond_two_hours_is_immobility():
    series = calm_series()
    series.loc["2026-01-05 09:00":"2026-01-05 11:30", "activity"] = 0.0  # 151 min
    [event] = events(series, "prolonged_immobility")
    assert event.source is Source.MOVEMENT
    assert event.severity is Severity.MEDIUM
    # o último movimento foi às 08:30: parado desde 08:31, o limiar de 120 min cai às 10:30
    assert event.timestamp.strftime("%H:%M") == "10:30"


def test_immobility_escalates_after_four_hours():
    series = calm_series()
    series.loc["2026-01-05 09:00":"2026-01-05 13:30", "activity"] = 0.0
    found = events(series, "prolonged_immobility")
    assert [e.severity for e in found] == [Severity.MEDIUM, Severity.HIGH]


def test_still_out_of_bed_is_not_immobility():
    series = calm_series()
    series.loc["2026-01-05 09:00":"2026-01-05 12:00", ["activity", "in_bed"]] = [0.0, 0.0]
    assert events(series, "prolonged_immobility") == []


def test_night_bed_exit():
    series = calm_series()
    series.loc["2026-01-05 03:00":"2026-01-05 03:09", ["activity", "in_bed"]] = [0.3, 0.0]
    [event] = events(series, "night_bed_exit")
    assert event.evidence["minutes_out"] == 10
    assert event.severity is Severity.MEDIUM


def test_daytime_walk_is_not_night_exit():
    series = calm_series()
    series.loc["2026-01-05 15:00":"2026-01-05 15:20", ["activity", "in_bed"]] = [0.4, 0.0]
    assert events(series, "night_bed_exit") == []


def test_one_minute_sensor_dropout_at_night_is_ignored():
    series = calm_series()
    series.loc["2026-01-05 02:00", "in_bed"] = 0.0
    assert events(series, "night_bed_exit") == []


def test_spike_then_still_out_of_bed_is_fall():
    series = calm_series()
    series.loc["2026-01-05 14:00", ["activity", "in_bed"]] = [0.95, 0.0]
    series.loc["2026-01-05 14:01":"2026-01-05 14:08", ["activity", "in_bed"]] = [0.0, 0.0]
    [event] = events(series, "fall_suspected")
    assert event.severity is Severity.CRITICAL
    assert event.evidence["spike_activity"] == 0.95


def test_spike_back_into_bed_is_not_fall():
    series = calm_series()
    series.loc["2026-01-05 14:00", "activity"] = 0.95  # movimento brusco no leito
    assert events(series, "fall_suspected") == []


def test_activity_drop_against_previous_days():
    series = pd.concat([calm_series(24, DAY + pd.Timedelta(days=d)) for d in range(3)])
    series.loc["2026-01-07", "activity"] *= 0.3
    [event] = events(series, "activity_drop")
    assert event.evidence["ratio"] == pytest.approx(0.3, abs=0.01)


def test_simulated_normal_stay_has_no_false_immobility_or_night_exit():
    rng = np.random.default_rng(3)
    found = {e.type for e in ANALYZER.analyze("SIM", simulate_stay(4, rng)).events}
    assert not found & {"prolonged_immobility", "night_bed_exit", "activity_drop"}


def test_injection_is_detected():
    rng = np.random.default_rng(5)
    series, labels = inject_movement_anomalies(simulate_stay(7, rng), rng, 4, fall_intensity=0.95)
    assert [label.kind for label in labels] == [
        "prolonged_immobility", "night_bed_exit", "fall", "activity_drop",
    ]  # fmt: skip
    found = {e.type for e in ANALYZER.analyze("SIM", series).events}
    assert {"prolonged_immobility", "night_bed_exit", "fall_suspected", "activity_drop"} <= found


def test_evaluation_reports_every_kind():
    metrics = evaluate_movement(ANALYZER, patients=4, days=7, per_stay=4, seed=0)
    assert metrics["combined"].injected == 16
    assert all(metrics[kind].injected == 4 for kind in ("fall", "activity_drop"))
    assert metrics["combined"].recall >= 0.7


def test_activity_from_pose_measures_motion_and_bed():
    times = np.arange(0, 6, 0.5)
    frame = pd.DataFrame({"time": times, "bbox_y1": 0.0, "bbox_y2": 100.0})
    still, moving = times < 3, times >= 3
    frame["kp_nose_x"] = np.where(still, 50.0, 50.0 + 20 * (times - 3))
    frame["kp_nose_y"] = 30.0
    frame["hip_x"] = np.where(still, 20.0, 80.0)  # no leito à esquerda, depois fora
    frame["hip_y"] = 50.0
    series = activity_from_pose(
        frame,
        pd.Timestamp("2026-01-05 10:00", tz="UTC"),
        freq="1s",
        bed_polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        frame_size=(100, 100),
    )
    assert series["activity"].iloc[:2].max() == 0
    assert series["activity"].iloc[-1] > 0.5
    assert series["in_bed"].iloc[0] == 1 and series["in_bed"].iloc[-1] == 0
    assert moving.any()
