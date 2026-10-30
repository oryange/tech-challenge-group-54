import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from medwatch.api.app import create_app
from medwatch.api.ingest import IngestResult
from medwatch.config import get_settings
from medwatch.core.events import ClinicalEvent, Severity, Source
from tests.unit.core.test_alert_manager import RecordingChannel

# perto de agora: o retrato do risco usa o relógio e decai após a janela de 30 min
T0 = datetime.now(UTC) - timedelta(minutes=12)


def event_json(source, type_, minute, score, severity, patient="PAC-001") -> dict:
    return ClinicalEvent(
        patient_id=patient,
        source=source,
        type=type_,
        timestamp=T0 + timedelta(minutes=minute),
        score=score,
        severity=severity,
        evidence={"note": "teste"},
    ).model_dump(mode="json")


DYSPNEA = (Source.AUDIO, "dyspnea_terms", 0, 0.64, Severity.MEDIUM)
SPO2 = (Source.VITALS, "spo2_drop", 10, 0.82, Severity.HIGH)


class FakeIngestors:
    def __init__(self, tmp_path) -> None:
        self.tmp_path = tmp_path
        self.calls: list[tuple] = []

    def audio(self, path, patient_id, start_time):
        self.calls.append(("audio", path.suffix, path.exists(), patient_id))
        event = ClinicalEvent.model_validate(event_json(*DYSPNEA, patient=patient_id))
        return IngestResult([event], self.tmp_path / "audio.json")

    def video(self, path, patient_id, start_time, exercise, scene, backend):
        self.calls.append(("video", path.suffix, patient_id, exercise, scene, backend))
        if exercise == "desconhecido":
            raise ValueError("Exercício desconhecido: desconhecido")
        fall = (Source.VIDEO, "fall_detected", 0, 0.95, Severity.CRITICAL)
        return IngestResult(
            [ClinicalEvent.model_validate(event_json(*fall, patient=patient_id))],
            self.tmp_path / "video.html",
        )


@pytest.fixture
def ingestors(tmp_path) -> FakeIngestors:
    return FakeIngestors(tmp_path)


@pytest.fixture
def channel() -> RecordingChannel:
    return RecordingChannel()


@pytest.fixture
def client(ingestors, channel):
    app = create_app(audio=ingestors.audio, video=ingestors.video, channels=[channel])
    with TestClient(app) as test_client:
        yield test_client


def wait_alerts(client, count: int, **params) -> list[dict]:
    """O consumidor roda em paralelo à requisição: espera os alertas aparecerem."""
    for _ in range(200):
        alerts = client.get("/alerts", params=params).json()
        if len(alerts) >= count:
            return alerts
        time.sleep(0.01)
    raise AssertionError(f"esperava {count} alertas, vieram {len(alerts)}")


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["consumer_running"] is True
    assert body["min_severity"] == "medium"


def test_ingest_events_runs_fusion_and_lists_alerts(client, channel):
    response = client.post("/ingest/events", json=[event_json(*SPO2), event_json(*DYSPNEA)])
    assert response.status_code == 202
    assert response.json() == {"accepted": 2}

    alerts = wait_alerts(client, 3)
    types = {(a["alert_type"], a["severity"]) for a in alerts}
    assert ("fusion.respiratory_deterioration", "critical") in types
    assert ("audio.dyspnea_terms", "medium") in types
    # os canais configurados recebem o mesmo que a API expõe
    assert len(channel.alerts) == len(alerts)


def test_patient_risk_and_ranking(client):
    client.post("/ingest/events", json=[event_json(*DYSPNEA), event_json(*SPO2)])
    client.post(
        "/ingest/events",
        json=[event_json(Source.MOVEMENT, "low_activity", 5, 0.3, Severity.LOW, patient="PAC-2")],
    )
    wait_alerts(client, 3)

    ranking = client.get("/patients").json()
    assert [p["patient_id"] for p in ranking] == ["PAC-001", "PAC-2"]
    assert ranking[0]["severity"] == "high"  # R = 0,90
    assert set(ranking[0]["sources"]) == {"audio", "vitals"}
    assert ranking[0]["open_alerts"] == 3

    detail = client.get("/patients/PAC-001/risk").json()
    assert [e["type"] for e in detail["timeline"]] == ["dyspnea_terms", "spo2_drop"]
    assert detail["score"] == pytest.approx(1 - (1 - 0.7 * 0.64) * (1 - 0.82), abs=1e-3)
    assert client.get("/patients/NINGUEM/risk").status_code == 404


def test_ack_alert(client, tmp_path):
    client.post("/ingest/events", json=[event_json(*SPO2)])
    [alert] = wait_alerts(client, 1)

    response = client.post(f"/alerts/{alert['alert_id']}/ack")
    assert response.status_code == 200
    acked = response.json()
    assert acked["acked"] is True
    assert acked["acked_at"] is not None
    # ACK repetido é idempotente
    assert client.post(f"/alerts/{alert['alert_id']}/ack").json()["acked_at"] == acked["acked_at"]

    assert client.get("/alerts", params={"acked": False}).json() == []
    assert client.get("/patients").json()[0]["open_alerts"] == 0
    audit = get_settings().outputs_dir / "alerts" / "acks.jsonl"
    assert json.loads(audit.read_text().splitlines()[0])["alert_id"] == alert["alert_id"]

    assert client.post("/alerts/00000000-0000-0000-0000-000000000000/ack").status_code == 404


def test_websocket_streams_alerts_and_acks(client):
    with client.websocket_connect("/ws/alerts") as ws:
        client.post("/ingest/events", json=[event_json(*SPO2)])
        message = ws.receive_json()
        assert message["type"] == "alert"
        assert message["alert"]["alert_type"] == "vitals.spo2_drop"

        client.post(f"/alerts/{message['alert']['alert_id']}/ack")
        ack = ws.receive_json()
        assert ack["type"] == "ack"
        assert ack["alert"]["acked"] is True


def test_ingest_audio_upload(client, ingestors):
    response = client.post(
        "/ingest/audio",
        files={"file": ("consulta.WAV", b"RIFF....", "audio/wav")},
        data={"patient_id": "PAC-007", "start_time": "2026-10-19T14:30:00-03:00"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["patient_id"] == "PAC-007"
    assert [e["type"] for e in body["events"]] == ["dyspnea_terms"]
    # o pipeline recebe um arquivo temporário com a extensão original (sem o nome)
    assert ingestors.calls == [("audio", ".wav", True, "PAC-007")]
    assert wait_alerts(client, 1, patient_id="PAC-007")[0]["alert_type"] == "audio.dyspnea_terms"


def test_ingest_audio_rejects_bad_input(client):
    bad_ext = client.post(
        "/ingest/audio",
        files={"file": ("consulta.exe", b"x", "application/octet-stream")},
        data={"patient_id": "PAC-007"},
    )
    assert bad_ext.status_code == 415
    # patient_id vira nome de arquivo do relatório: "../" é recusado
    traversal = client.post(
        "/ingest/audio",
        files={"file": ("consulta.wav", b"x", "audio/wav")},
        data={"patient_id": "../../etc"},
    )
    assert traversal.status_code == 422
    naive = client.post(
        "/ingest/audio",
        files={"file": ("consulta.wav", b"x", "audio/wav")},
        data={"patient_id": "PAC-007", "start_time": "2026-10-19T14:30:00"},
    )
    assert naive.status_code == 422


def test_ingest_audio_maps_pipeline_errors(tmp_path, channel):
    def invalid(path, patient_id, start_time):
        raise ValueError("Áudio curto demais")

    app = create_app(audio=invalid, video=FakeIngestors(tmp_path).video, channels=[channel])
    with TestClient(app) as client:
        response = client.post(
            "/ingest/audio",
            files={"file": ("c.wav", b"x", "audio/wav")},
            data={"patient_id": "PAC-1"},
        )
    assert response.status_code == 422
    assert "curto" in response.json()["detail"]


def test_ingest_video_upload(client, ingestors):
    response = client.post(
        "/ingest/video",
        files={"file": ("quarto.mp4", b"\x00\x00", "video/mp4")},
        data={"patient_id": "PAC-008", "scene": "quarto", "backend": "openpose"},
    )
    assert response.status_code == 201, response.text
    assert ingestors.calls == [("video", ".mp4", "PAC-008", None, "quarto", "openpose")]
    assert wait_alerts(client, 1)[0]["severity"] == "critical"


def test_ingest_video_validation(client):
    bad_backend = client.post(
        "/ingest/video",
        files={"file": ("v.mp4", b"x", "video/mp4")},
        data={"patient_id": "PAC-1", "backend": "resnet"},
    )
    assert bad_backend.status_code == 422
    bad_exercise = client.post(
        "/ingest/video",
        files={"file": ("v.mp4", b"x", "video/mp4")},
        data={"patient_id": "PAC-1", "exercise": "desconhecido"},
    )
    assert bad_exercise.status_code == 422


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr("medwatch.api.app.MAX_VIDEO_MB", 0.000001)
    response = client.post(
        "/ingest/video",
        files={"file": ("v.mp4", b"x" * 100, "video/mp4")},
        data={"patient_id": "PAC-1"},
    )
    assert response.status_code == 413
