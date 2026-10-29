from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from medwatch.config import PROJECT_ROOT
from medwatch.core.events import ClinicalEvent, Severity, Source
from medwatch.fusion.config import FusionConfig, RiskBands
from medwatch.fusion.risk_engine import RISK_ALERT_TYPE, RiskEngine, noisy_or

T0 = datetime(2026, 10, 19, 14, 0, tzinfo=UTC)


@pytest.fixture
def config() -> FusionConfig:
    return FusionConfig.load(PROJECT_ROOT / "configs" / "fusion.yaml")


def event(source, type_, minute, score, severity, patient="PAC-001") -> ClinicalEvent:
    return ClinicalEvent(
        patient_id=patient,
        source=source,
        type=type_,
        timestamp=T0 + timedelta(minutes=minute),
        score=score,
        severity=severity,
    )


def types(alerts) -> list[tuple[str, Severity]]:
    return [(a.alert_type, a.severity) for a in alerts]


def test_isolated_event_does_not_escalate(config):
    engine = RiskEngine(config)
    alerts = engine.alerts_for(event(Source.VITALS, "spo2_drop", 0, 0.9, Severity.HIGH))
    # só o alerta direto, com a severidade do próprio evento
    assert types(alerts) == [("vitals.spo2_drop", Severity.HIGH)]
    risk = engine.risk("PAC-001")
    assert risk.score == pytest.approx(0.9)
    assert risk.sources == (Source.VITALS,)


def test_same_modality_does_not_combine(config):
    engine = RiskEngine(config)
    engine.alerts_for(event(Source.MOVEMENT, "night_bed_exit", 0, 0.9, Severity.MEDIUM))
    alerts = engine.alerts_for(event(Source.MOVEMENT, "low_activity", 1, 0.9, Severity.LOW))
    assert types(alerts) == [("movement.low_activity", Severity.LOW)]


def test_cross_modal_rule_escalates_to_critical(config):
    engine = RiskEngine(config)
    engine.alerts_for(event(Source.AUDIO, "dyspnea_terms", 0, 0.64, Severity.MEDIUM))
    alerts = engine.alerts_for(event(Source.VITALS, "spo2_drop", 10, 0.82, Severity.HIGH))
    assert ("fusion.respiratory_deterioration", Severity.CRITICAL) in types(alerts)
    rule_alert = next(a for a in alerts if a.alert_type == "fusion.respiratory_deterioration")
    # explicável: os dois eventos da combinação estão em reasons, em ordem cronológica
    assert [(e.source, e.type) for e in rule_alert.reasons] == [
        (Source.AUDIO, "dyspnea_terms"),
        (Source.VITALS, "spo2_drop"),
    ]
    assert rule_alert.recommended_action.startswith("Acionar a equipe de resposta rápida")


def test_rule_respects_its_window(config):
    engine = RiskEngine(config)
    engine.alerts_for(event(Source.VIDEO, "fall_detected", 0, 0.9, Severity.CRITICAL))
    # fall_confirmed tem janela de 5 min: o sensor 6 min depois não confirma a mesma queda
    alerts = engine.alerts_for(event(Source.MOVEMENT, "fall_suspected", 6, 0.9, Severity.CRITICAL))
    assert "fusion.fall_confirmed" not in [a.alert_type for a in alerts]


def test_rule_works_with_events_out_of_order(config):
    engine = RiskEngine(config)
    # vitais chegam antes; o áudio da consulta (10 min antes) só é processado depois
    engine.alerts_for(event(Source.VITALS, "spo2_drop", 10, 0.82, Severity.HIGH))
    alerts = engine.alerts_for(event(Source.AUDIO, "dyspnea_terms", 0, 0.64, Severity.MEDIUM))
    assert "fusion.respiratory_deterioration" in [a.alert_type for a in alerts]


def test_two_weak_signals_from_different_modalities_raise_combined_risk(config):
    engine = RiskEngine(config)
    first = engine.alerts_for(event(Source.VITALS, "hr_rise", 0, 0.45, Severity.LOW))
    assert types(first) == [("vitals.hr_rise", Severity.LOW)]
    alerts = engine.alerts_for(event(Source.AUDIO, "fatigue_terms", 3, 0.40, Severity.LOW))
    # 1 − (1 − 0,45)·(1 − 0,7·0,40) = 0,604 → faixa MEDIUM, acima do pior evento (LOW)
    assert (RISK_ALERT_TYPE, Severity.MEDIUM) in types(alerts)
    assert engine.risk("PAC-001").score == pytest.approx(0.604)


def test_repeated_finding_counts_once_by_its_strongest_score(config):
    engine = RiskEngine(config)
    for minute, score in ((0, 0.5), (1, 0.7), (2, 0.6)):
        engine.alerts_for(event(Source.VITALS, "spo2_drop", minute, score, Severity.MEDIUM))
    risk = engine.risk("PAC-001")
    assert risk.score == pytest.approx(0.7)
    assert len(risk.contributions) == 1


def test_risk_alert_skipped_when_rule_already_explains_it(config):
    engine = RiskEngine(config)
    engine.alerts_for(event(Source.AUDIO, "dyspnea_terms", 0, 0.9, Severity.MEDIUM))
    alerts = engine.alerts_for(event(Source.VITALS, "spo2_drop", 1, 0.95, Severity.HIGH))
    # risco combinado seria CRITICAL, mas a regra respiratória CRITICAL já cobre
    assert RISK_ALERT_TYPE not in [a.alert_type for a in alerts]


def test_events_leave_the_window(config):
    engine = RiskEngine(config)
    engine.alerts_for(event(Source.AUDIO, "fatigue_terms", 0, 0.9, Severity.LOW))
    engine.alerts_for(event(Source.VITALS, "hr_rise", 31, 0.4, Severity.LOW))
    risk = engine.risk("PAC-001")
    assert risk.sources == (Source.VITALS,)
    assert risk.score == pytest.approx(0.4)


def test_clock_decays_risk_without_new_events(config):
    now = [T0]
    engine = RiskEngine(config, clock=lambda: now[0])
    engine.alerts_for(event(Source.VITALS, "spo2_drop", 0, 0.9, Severity.HIGH))
    assert engine.risk("PAC-001").score == pytest.approx(0.9)
    now[0] = T0 + timedelta(minutes=45)
    assert engine.risk("PAC-001").score == 0.0


def test_patients_are_sorted_by_risk(config):
    engine = RiskEngine(config)
    engine.alerts_for(event(Source.VITALS, "hr_rise", 0, 0.3, Severity.LOW, patient="A"))
    engine.alerts_for(event(Source.VITALS, "spo2_drop", 0, 0.9, Severity.HIGH, patient="B"))
    assert [p.patient_id for p in engine.patients()] == ["B", "A"]


def test_noisy_or_of_nothing_is_zero():
    assert noisy_or([]) == 0.0


def test_risk_bands_must_increase():
    with pytest.raises(ValidationError):
        RiskBands(low=0.5, medium=0.4, high=0.7, critical=0.9)
    bands = RiskBands(low=0.3, medium=0.5, high=0.7, critical=0.9)
    assert bands.severity(0.1) is Severity.INFO
    assert bands.severity(0.95) is Severity.CRITICAL


def test_config_rejects_invalid_weights_and_duplicated_rules(config):
    data = config.model_dump(mode="json")
    with pytest.raises(ValidationError, match="pesos"):
        FusionConfig.model_validate(data | {"source_weights": {"vitals": 1.5}})
    with pytest.raises(ValidationError, match="repetido"):
        FusionConfig.model_validate(data | {"rules": [data["rules"][0], data["rules"][0]]})


def test_config_history_covers_the_longest_rule(config):
    assert config.history == timedelta(minutes=360)


def test_historical_replay_fuses_even_with_a_live_clock(config):
    # API ao vivo recebendo um relatório antigo: as regras usam o horário dos eventos
    engine = RiskEngine(config, clock=lambda: T0 + timedelta(days=30))
    engine.alerts_for(event(Source.AUDIO, "dyspnea_terms", 0, 0.64, Severity.MEDIUM))
    alerts = engine.alerts_for(event(Source.VITALS, "spo2_drop", 10, 0.82, Severity.HIGH))
    assert "fusion.respiratory_deterioration" in [a.alert_type for a in alerts]
    # o retrato de "agora" já não conta esses eventos
    assert engine.risk("PAC-001").score == 0.0
