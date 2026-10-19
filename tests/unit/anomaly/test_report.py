import json

import numpy as np

from medwatch.anomaly.evaluation import inject_anomalies
from medwatch.anomaly.report import plot_stay, write_json
from medwatch.anomaly.vitals import VitalsAnalyzer
from tests.unit.anomaly.helpers import load_config, make_raw, make_stay


def test_json_report_and_plot(tmp_path):
    config = load_config()
    raw = make_raw()
    raw.loc[raw.index[30], "spo2"] = 85
    analysis = VitalsAnalyzer(config).analyze(make_stay(raw))

    report = write_json([analysis], tmp_path / "vitals.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    stay = payload["stays"][0]
    assert stay["patient_id"] == "MIMIC-1"
    assert stay["events_by_type"]["spo2_drop"] == 1
    assert stay["events"][0]["source"] == "vitals"

    _, labels = inject_anomalies(raw, np.random.default_rng(0), 1, config.validity)
    png = plot_stay(analysis, tmp_path / "vitals.png", labels)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
