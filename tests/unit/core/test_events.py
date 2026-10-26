import copy
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from medwatch.core.events import RECOMMENDED_ACTIONS, Alert, ClinicalEvent, Severity, Source


def make_event(**overrides) -> ClinicalEvent:
    data = {
        "patient_id": "PAC-001",
        "source": Source.VITALS,
        "type": "spo2_drop",
        "score": 0.8,
        "severity": Severity.HIGH,
        "evidence": {"spo2": 88, "nested": {"window": [1, 2, 3]}},
    }
    return ClinicalEvent(**(data | overrides))


def test_severity_uses_clinical_order_not_alphabetical():
    assert Severity.INFO < Severity.LOW < Severity.MEDIUM < Severity.HIGH < Severity.CRITICAL
    # em ordem alfabética "critical" < "high"; a ordem clínica precisa prevalecer
    assert Severity.CRITICAL > Severity.HIGH
    assert max([Severity.HIGH, Severity.CRITICAL, Severity.LOW]) is Severity.CRITICAL
    assert Severity.MEDIUM >= Severity.MEDIUM


def test_event_json_round_trip():
    event = make_event()
    assert ClinicalEvent.model_validate_json(event.model_dump_json()) == event


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_event_rejects_score_out_of_range(score):
    with pytest.raises(ValidationError):
        make_event(score=score)


def test_event_rejects_naive_timestamp():
    with pytest.raises(ValidationError):
        make_event(timestamp=datetime(2026, 1, 1, 12, 0))


def test_event_normalizes_timestamp_to_utc():
    brt = timezone(timedelta(hours=-3))
    event = make_event(timestamp=datetime(2026, 1, 1, 9, 0, tzinfo=brt))
    assert event.timestamp == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert event.timestamp.utcoffset() == timedelta(0)


def test_event_rejects_non_serializable_evidence():
    with pytest.raises(ValidationError):
        make_event(evidence={"obj": object()})


def test_event_is_immutable():
    event = make_event()
    with pytest.raises(ValidationError):
        event.score = 0.1


def test_alert_from_event():
    event = make_event(severity=Severity.CRITICAL)
    alert = Alert.from_event(event)
    assert alert.patient_id == "PAC-001"
    assert alert.alert_type == "vitals.spo2_drop"
    assert alert.dedup_key == "PAC-001:vitals.spo2_drop"
    assert alert.severity is Severity.CRITICAL
    assert alert.reasons == (event,)
    assert alert.recommended_action == RECOMMENDED_ACTIONS[Severity.CRITICAL]
    assert not alert.acked


def test_alert_json_round_trip():
    alert = Alert.from_event(make_event())
    assert Alert.model_validate_json(alert.model_dump_json()) == alert


def test_alert_requires_at_least_one_reason():
    with pytest.raises(ValidationError):
        Alert(
            patient_id="PAC-001",
            alert_type="x",
            severity=Severity.LOW,
            title="t",
            reasons=[],
            recommended_action="a",
        )


def test_every_severity_has_recommended_action():
    assert set(RECOMMENDED_ACTIONS) == set(Severity)


@pytest.mark.parametrize("text", ["high", "HIGH"])
def test_severity_compares_with_text_in_clinical_order(text):
    # YAML e .env carregam a severidade como texto; a ordem alfabética daria o resultado errado.
    # O AlertManager compara `event.severity < min_severity`, com a Severity à esquerda.
    assert Severity.CRITICAL >= text  # noqa: SIM300
    assert Severity.CRITICAL > text  # noqa: SIM300
    assert not Severity.LOW >= text  # noqa: SIM300
    assert Severity.LOW < text  # noqa: SIM300
    assert Severity.HIGH <= text  # noqa: SIM300
    # e com o texto à esquerda (operador refletido)
    assert text <= Severity.CRITICAL
    assert text > Severity.LOW


def test_severity_rejects_unknown_text():
    with pytest.raises(ValueError, match="urgente"):
        _ = Severity.HIGH < "urgente"


def test_alert_is_immutable():
    alert = Alert.from_event(make_event())
    with pytest.raises(ValidationError):
        alert.acked = True


def test_event_evidence_is_deeply_immutable():
    event = make_event()
    with pytest.raises(TypeError, match="imutável"):
        event.evidence["spo2"] = 50
    with pytest.raises(TypeError, match="imutável"):
        event.evidence["nested"]["window"].append(4)  # type: ignore[index, union-attr]
    assert event.evidence == {"spo2": 88, "nested": {"window": [1, 2, 3]}}


def test_frozen_evidence_still_behaves_like_json():
    """Quem consome o evento pode serializar e comparar o evidence como dict/list comuns."""
    event = make_event()
    assert json.loads(json.dumps(event.evidence)) == {"spo2": 88, "nested": {"window": [1, 2, 3]}}
    assert event.evidence["nested"]["window"] == [1, 2, 3]  # type: ignore[index]
    copied = copy.deepcopy(event)
    assert copied == event
    assert event.model_copy(update={"score": 0.5}).evidence == event.evidence
