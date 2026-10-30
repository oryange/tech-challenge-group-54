from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from medwatch.dashboard import client as client_module
from medwatch.dashboard.client import (
    ApiError,
    MedWatchClient,
    contribution_rows,
    patient_rows,
    timeline_rows,
)

APP = Path(client_module.__file__).with_name("app.py")

EVENT = {
    "event_id": "e1",
    "patient_id": "PAC-001",
    "source": "vitals",
    "type": "spo2_drop",
    "timestamp": "2026-10-19T17:10:00Z",
    "score": 0.82,
    "severity": "high",
}
PATIENT = {
    "patient_id": "PAC-001",
    "score": 0.9,
    "severity": "high",
    "sources": ["audio", "vitals"],
    "contributions": [
        {"source": "vitals", "type": "spo2_drop", "score": 0.82, "weight": 1.0, "event_id": "e1"},
        {
            "source": "audio",
            "type": "dyspnea_terms",
            "score": 0.64,
            "weight": 0.7,
            "event_id": "e0",
        },
    ],
    "open_alerts": 2,
    "last_event_at": "2026-10-19T17:10:00Z",
    "as_of": "2026-10-19T17:10:00Z",
}
ALERT = {
    "alert_id": "a1",
    "patient_id": "PAC-001",
    "alert_type": "fusion.respiratory_deterioration",
    "severity": "critical",
    "title": "Piora respiratória",
    "reasons": [EVENT],
    "recommended_action": "Acionar a equipe de resposta rápida imediatamente.",
    "created_at": "2026-10-19T17:10:01Z",
    "acked": False,
    "acked_at": None,
}
HEALTH = {"status": "ok", "consumer_running": True, "events": 2}


def api_handler(request: httpx.Request) -> httpx.Response:
    routes = {
        "/health": HEALTH,
        "/patients": [PATIENT],
        "/patients/PAC-001/risk": PATIENT | {"timeline": [EVENT]},
        "/alerts": [ALERT],
    }
    if request.method == "POST" and request.url.path == "/alerts/a1/ack":
        return httpx.Response(200, json=ALERT | {"acked": True})
    if request.url.path in routes:
        return httpx.Response(200, json=routes[request.url.path])
    return httpx.Response(404, json={"detail": "não encontrado"})


def make_client() -> MedWatchClient:
    return MedWatchClient("http://api", transport=httpx.MockTransport(api_handler))


def test_client_reads_api():
    api = make_client()
    assert api.health()["status"] == "ok"
    assert api.patients()[0]["patient_id"] == "PAC-001"
    assert api.alerts(acked=False)[0]["alert_id"] == "a1"
    assert api.ack("a1")["acked"] is True
    assert api.patient("PAC-001")["timeline"][0]["type"] == "spo2_drop"
    assert api.patient("NINGUEM") is None


def test_client_wraps_network_errors():
    def down(request):
        raise httpx.ConnectError("recusado")

    api = MedWatchClient("http://api", transport=httpx.MockTransport(down))
    with pytest.raises(ApiError):
        api.health()
    with pytest.raises(ApiError):
        api.ack("a1")


def test_view_rows():
    [row] = patient_rows([PATIENT])
    assert row["Severidade"] == "🟠 HIGH"
    assert row["Modalidades"] == "audio, vitals"
    assert timeline_rows(PATIENT | {"timeline": [EVENT]})[0]["Modalidade"] == "vitals"
    assert [r["Contribuição"] for r in contribution_rows(PATIENT)] == [0.82, 0.448]


def test_app_renders_patients_alerts_and_detail(monkeypatch):
    original = MedWatchClient.__init__

    def with_mock_transport(self, base_url, timeout=5.0, transport=None):
        original(self, base_url, timeout, httpx.MockTransport(api_handler))

    monkeypatch.setattr(MedWatchClient, "__init__", with_mock_transport)
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["1", "2", "1", "2"]
    assert any("PAC-001 · risco 0.90" in md.value for md in at.markdown)
    assert any("Piora respiratória" in md.value for md in at.markdown)

    at.button(key="ack-a1").click().run()
    assert not at.exception


def test_app_shows_error_when_api_is_down(monkeypatch):
    monkeypatch.setenv("MEDWATCH_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert "API indisponível" in at.error[0].value
