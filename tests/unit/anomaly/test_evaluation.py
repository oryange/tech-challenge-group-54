import numpy as np
import pandas as pd

from medwatch.anomaly.evaluation import ANOMALY_KINDS, MethodMetrics, evaluate, inject_anomalies
from medwatch.anomaly.vitals import VitalsAnalyzer
from tests.unit.anomaly.helpers import load_config, make_raw, make_stay

CONFIG = load_config()


def test_inject_applies_deltas_without_overlap():
    raw = make_raw(hours=96)
    modified, labels = inject_anomalies(raw, np.random.default_rng(0), 4, CONFIG.validity)
    assert len(labels) == 4
    starts = sorted(label.start for label in labels)
    assert all(b - a >= pd.Timedelta("12h") for a, b in zip(starts, starts[1:], strict=False))
    for label in labels:
        window = slice(label.start, label.end)
        diff = (modified.loc[window, label.vital] - raw.loc[window, label.vital]).dropna()
        low, high = CONFIG.validity[label.vital]
        assert ((diff == label.delta) | (modified.loc[diff.index, label.vital] == high)).all()
        assert label.kind in ANOMALY_KINDS
    untouched = modified.drop(columns=[label.vital for label in labels])
    pd.testing.assert_frame_equal(untouched, raw.drop(columns=[label.vital for label in labels]))


def test_inject_respects_history_requirement():
    raw = make_raw(hours=96)
    _, labels = inject_anomalies(raw, np.random.default_rng(1), 3, CONFIG.validity)
    assert all(label.start >= raw.index[12] for label in labels)


def test_metrics_math():
    metric = MethodMetrics(detected=8, injected=10, true_detections=9, new_detections=10)
    assert metric.recall == 0.8
    assert metric.precision == 0.9
    assert round(metric.f1, 4) == round(2 * 0.9 * 0.8 / 1.7, 4)
    assert MethodMetrics().f1 == 0.0


def test_evaluate_on_synthetic_stays_detects_most_injections():
    stays = [make_stay(make_raw(hours=96, seed=s), subject_id=s, stay_id=s) for s in range(6)]
    analyzer = VitalsAnalyzer(CONFIG).fit(stays)
    metrics = evaluate(analyzer, stays, per_stay=3, seed=0)
    combined = metrics["combined"]
    assert combined.injected == 18
    assert combined.recall >= 0.8
    assert combined.precision >= 0.8
    assert set(metrics) == {"robust_zscore", "isolation_forest", "news2", "combined"}
