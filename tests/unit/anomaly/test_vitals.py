from datetime import timedelta

import numpy as np
import pandas as pd

from medwatch.anomaly.vitals import VitalsAnalyzer
from medwatch.core.events import Severity, Source
from tests.unit.anomaly.helpers import load_config, make_raw, make_stay

CONFIG = load_config()


def analyzer(fit_on=None) -> VitalsAnalyzer:
    an = VitalsAnalyzer(CONFIG)
    if fit_on is not None:
        an.fit(fit_on)
    return an


def zscore_events(analysis):
    return [e for e in analysis.events if e.evidence.get("detector") == "robust_zscore"]


def test_stable_patient_has_no_deviation_events():
    analysis = analyzer().analyze(make_stay())
    assert zscore_events(analysis) == []
    # o ruído pode levar a FC a 91 (1 ponto no NEWS2 → LOW), mas nada que exija resposta
    assert all(e.severity < Severity.MEDIUM for e in analysis.events)


def test_spo2_drop_is_detected_with_clinical_severity():
    raw = make_raw()
    raw.loc[raw.index[30:33], "spo2"] = [89, 88, 90]
    analysis = analyzer().analyze(make_stay(raw))
    drops = [e for e in zscore_events(analysis) if e.type == "spo2_drop"]
    assert len(drops) == 1  # três horas seguidas viram um único episódio
    event = drops[0]
    assert event.source is Source.VITALS
    assert event.severity is Severity.HIGH  # SpO2 ≤ 91 vale 3 pontos no NEWS2
    assert event.evidence["value"] == 88
    assert event.timestamp.utcoffset() == timedelta(0)
    assert 0 < event.score <= 1


def test_statistical_deviation_within_normal_range_is_low_severity():
    raw = make_raw()
    raw.loc[raw.index[30], "dbp"] = 95  # fora do padrão do paciente, mas não pontua no NEWS2
    events = zscore_events(analyzer().analyze(make_stay(raw)))
    assert [(e.type, e.severity) for e in events] == [("dbp_rise", Severity.LOW)]


def test_spo2_rise_is_not_an_alert():
    raw = make_raw()
    raw["spo2"] = 93.0
    raw.loc[raw.index[30], "spo2"] = 100
    assert not [e for e in analyzer().analyze(make_stay(raw)).events if e.type == "spo2_rise"]


def test_news2_rise_with_hysteresis():
    raw = make_raw(hours=40)
    # FR 22 (2) + FC 105 (1) + SpO2 94 (1) + PAS 105 (1) = 5 → HIGH
    # sobe, oscila e só alerta de novo depois de 6 h abaixo da faixa
    worse = {"rr": 22, "hr": 105, "spo2": 94, "sbp": 105}
    for start, end in [(14, 16), (17, 18), (27, 29)]:
        for vital, value in worse.items():
            raw.loc[raw.index[start:end], vital] = value
    events = [e for e in analyzer().analyze(make_stay(raw)).events if e.type == "news2_rise"]
    rises = [(e.timestamp.hour, e.severity) for e in events if e.severity >= Severity.HIGH]
    # 14h: primeira subida · 17h: oscilação após 1 h abaixo (ignorada) · 27h: após 9 h abaixo
    assert [hour for hour, _ in rises] == [14, 3]
    assert all(severity is Severity.HIGH for _, severity in rises)


def test_iforest_events_only_after_fit():
    stays = [make_stay(make_raw(seed=seed), subject_id=seed) for seed in range(5)]
    raw = make_raw(seed=99)
    raw.loc[raw.index[30], ["hr", "rr", "spo2", "sbp"]] = [150, 40, 80, 70]
    probe = make_stay(raw, subject_id=99)

    assert "iforest_score" not in analyzer().analyze(probe).frame
    analysis = analyzer(fit_on=stays).analyze(probe)
    multivariate = [e for e in analysis.events if e.type == "multivariate_anomaly"]
    # contaminação de 2% sempre marca alguns pontos limítrofes; o pico tem de ser a anomalia
    strongest = max(multivariate, key=lambda e: e.score)
    assert strongest.evidence["vitals"]["hr"] == 150
    assert strongest.severity >= Severity.MEDIUM


def test_analyze_accepts_modified_raw_without_touching_stay():
    stay = make_stay()
    modified = stay.raw.copy()
    modified.loc[modified.index[30], "hr"] = 160
    analysis = analyzer().analyze(stay, raw=modified)
    assert analysis.frame.loc[modified.index[30], "hr"] == 160
    assert not np.isclose(stay.raw.loc[stay.raw.index[30], "hr"], 160)


def test_news2_rearm_uses_lowest_band_seen_below():
    """CRITICAL → 6 h em LOW → HIGH: a piora LOW → HIGH precisa gerar evento."""
    raw = make_raw(hours=30)
    critical = {"rr": 26, "spo2": 91, "sbp": 95, "hr": 125}  # 3 + 3 + 2 + 2 = 10
    low = {"rr": 16, "spo2": 97, "sbp": 120, "hr": 95}  # só a FC pontua (1)
    high = {"rr": 22, "hr": 105, "spo2": 94, "sbp": 105}  # 2 + 1 + 1 + 1 = 5
    for (start, end), values in [((10, 12), critical), ((12, 18), low), ((18, 22), high)]:
        for vital, value in values.items():
            raw.loc[raw.index[start:end], vital] = value
    events = [e for e in analyzer().analyze(make_stay(raw)).events if e.type == "news2_rise"]
    serious = [(e.timestamp.hour, e.severity) for e in events if e.severity >= Severity.HIGH]
    assert serious == [(10, Severity.CRITICAL), (18, Severity.HIGH)]


def test_iforest_severity_uses_news2_as_floor(monkeypatch):
    """Anomalia multivariada pouco rara, mas com NEWS2 alto, não pode sair LOW."""
    raw = make_raw()
    worse = {"rr": 26, "spo2": 91, "sbp": 95, "hr": 125}  # NEWS2 10 → CRITICAL
    for vital, value in worse.items():
        raw.loc[raw.index[30], vital] = value
    an = analyzer()
    scores = pd.Series(0.1, index=raw.index)
    scores.iloc[30] = 0.51  # acima de 0.5 (anômalo), abaixo do corte de MEDIUM
    monkeypatch.setattr(an.iforest, "score", lambda features: scores)
    an._fitted = True

    events = [e for e in an.analyze(make_stay(raw)).events if e.type == "multivariate_anomaly"]
    assert len(events) == 1
    assert events[0].severity is Severity.CRITICAL
    assert events[0].evidence["iforest_severity"] == "low"
