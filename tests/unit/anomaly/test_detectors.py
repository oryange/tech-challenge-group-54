import numpy as np
import pandas as pd
import pytest

from medwatch.anomaly.detectors import (
    IsolationForestDetector,
    RobustZScore,
    ThresholdBands,
    find_episodes,
)

NEWS2_HR = [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (None, 3)]


@pytest.mark.parametrize(
    ("value", "points"),
    [(35, 3), (40, 3), (41, 1), (50, 1), (51, 0), (90, 0), (91, 1), (111, 2), (130, 2), (131, 3)],
)
def test_threshold_bands_inclusive_upper_limits(value, points):
    assert ThresholdBands(NEWS2_HR).score(pd.Series([value])).iloc[0] == points


def test_threshold_bands_keep_missing_as_nan():
    assert ThresholdBands(NEWS2_HR).score(pd.Series([np.nan])).isna().all()


def hourly(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2180-01-01", periods=len(values), freq="1h"))


def test_zscore_flags_spike_and_not_stable_series():
    rng = np.random.default_rng(0)
    values = rng.normal(80, 2, 30)
    detector = RobustZScore(window="12h", min_periods=4, threshold=3.5, min_mad=1.0)
    assert not detector.flags(detector.score(hourly(values))).any()

    values[20] = 120
    flags = detector.flags(detector.score(hourly(values)))
    assert flags.iloc[20]
    assert flags.sum() == 1


def test_zscore_is_causal():
    """Mudar o futuro não pode alterar o score do passado (requisito de tempo real)."""
    rng = np.random.default_rng(1)
    values = rng.normal(80, 2, 30)
    altered = values.copy()
    altered[25:] = 200
    detector = RobustZScore()
    original = detector.score(hourly(values)).iloc[:25]
    changed = detector.score(hourly(altered)).iloc[:25]
    pd.testing.assert_series_equal(original, changed)


def test_zscore_min_mad_avoids_explosion_on_constant_series():
    values = [97.0] * 20 + [96.0]
    detector = RobustZScore(min_mad=1.0, threshold=3.5)
    z = detector.score(hourly(values))
    assert abs(z.iloc[-1]) < 3.5  # 1 ponto de SpO2 numa série constante não é anomalia


def test_zscore_ignores_missing_measurements():
    values = [80.0, np.nan] * 15
    z = RobustZScore(min_periods=4).score(hourly(values))
    assert z.iloc[1::2].isna().all()


def test_isolation_forest_scores_outlier_higher():
    rng = np.random.default_rng(2)
    normal = pd.DataFrame(rng.normal(0, 1, (500, 3)), columns=["a", "b", "c"])
    detector = IsolationForestDetector(contamination=0.02).fit(normal)
    probe = pd.DataFrame([[0, 0, 0], [8, -8, 8]], columns=["a", "b", "c"])
    scores = detector.score(probe)
    assert scores.iloc[1] > 0.5 > scores.iloc[0]
    assert scores.between(0, 1).all()


def test_isolation_forest_requires_fit_and_handles_missing():
    detector = IsolationForestDetector()
    with pytest.raises(RuntimeError):
        detector.score(pd.DataFrame({"a": [1.0]}))
    detector.fit(pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0)}))
    scores = detector.score(pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]}))
    assert pd.notna(scores.iloc[0]) and pd.isna(scores.iloc[1])


def test_find_episodes_groups_nearby_flags_and_finds_peak():
    flags = hourly([False, True, True, False, True, False, False, False, True])
    magnitude = hourly([0, 4, 6, 0, 5, 0, 0, 0, -9])
    episodes = find_episodes(flags, magnitude, max_gap="2h")
    assert len(episodes) == 2
    assert episodes[0].start == flags.index[1]
    assert episodes[0].end == flags.index[4]
    assert episodes[0].peak == flags.index[2]
    assert episodes[1].peak == flags.index[8]


def test_find_episodes_empty():
    assert find_episodes(hourly([False, False]), hourly([0, 0])) == []


def test_zscore_sees_spike_in_first_hours_of_stay():
    """O NaN do shift(1) não pode cegar o detector enquanto estiver na janela (12 h)."""
    rng = np.random.default_rng(0)
    values = rng.normal(80, 2, 24)
    values[6] = 140
    detector = RobustZScore(window="12h", min_periods=4, threshold=3.5, min_mad=1.0)
    z = detector.score(hourly(values))
    assert z.iloc[4:12].notna().all()
    assert detector.flags(z).iloc[6]
