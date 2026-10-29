import json

import httpx
import pytest
from typer.testing import CliRunner

from medwatch.cli import app
from medwatch.commands import realtime
from medwatch.config import get_settings
from tests.unit.core.test_events import make_event

runner = CliRunner()


def test_simulate_demo_matches_documented_sequence():
    result = runner.invoke(app, ["simulate", "--speed", "0"])
    assert result.exit_code == 0, result.output
    assert "Cenário demo: 12 eventos em 40 min" in result.stdout
    assert "fusion.respiratory_deterioration" in result.stdout
    assert "Alertas enviados: 14 · suprimidos (cooldown): 2" in result.stdout
    assert "✅ Sequência de alertas igual à documentada" in result.stdout
    lines = get_settings().alerts_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 14


def test_simulate_reports_divergence_from_documented_sequence(tmp_path):
    scenario = tmp_path / "s.yaml"
    scenario.write_text(
        """
name: divergente
events:
  - {at_minutes: 0, patient_id: P1, source: vitals, type: spo2_drop, score: 0.9, severity: high}
expected_alerts:
  - {at_minutes: 0, patient_id: P1, alert_type: vitals.spo2_drop, severity: critical}
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["simulate", "--scenario", str(scenario), "--speed", "0"])
    assert result.exit_code == 1
    assert "faltou: min 0 · P1 · vitals.spo2_drop · critical" in result.output


def test_simulate_replay_of_a_module_report(tmp_path):
    report = tmp_path / "video.json"
    events = [make_event(type="fall_detected", source="video", severity="critical")]
    report.write_text(json.dumps({"events": [e.model_dump(mode="json") for e in events]}))
    result = runner.invoke(
        app,
        [
            "simulate",
            "--replay",
            str(report),
            "--replay-patient",
            "PAC-010",
            "--only-replay",
            "--speed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PAC-010  video.fall_detected" in result.stdout
    # sem eventos roteirizados, não há sequência documentada para conferir
    assert "Sequência" not in result.stdout


def test_simulate_unknown_scenario():
    result = runner.invoke(app, ["simulate", "--scenario", "nao_existe"])
    assert result.exit_code == 1
    assert "Disponíveis: demo" in result.output


def test_simulate_missing_replay_file():
    result = runner.invoke(app, ["simulate", "--replay", "nao_existe.json", "--speed", "0"])
    assert result.exit_code == 1
    assert "Erro ao montar o cenário" in result.output


def test_simulate_to_api(monkeypatch):
    received = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.extend(json.loads(request.content))
        return httpx.Response(202, json={"accepted": 1})

    original = realtime.HttpPublisher.__init__

    def mocked(self, base_url, timeout=10.0, transport=None):
        original(self, base_url, timeout, httpx.MockTransport(handler))

    monkeypatch.setattr(realtime.HttpPublisher, "__init__", mocked)
    result = runner.invoke(app, ["simulate", "--target", "api", "--speed", "0"])
    assert result.exit_code == 0, result.output
    assert len(received) == 12
    assert "12 eventos enviados para http://localhost:8000" in result.stdout


def test_simulate_to_api_offline(monkeypatch):
    def down(request):
        raise httpx.ConnectError("recusado")

    original = realtime.HttpPublisher.__init__
    monkeypatch.setattr(
        realtime.HttpPublisher,
        "__init__",
        lambda self, url, timeout=10.0, transport=None: original(
            self, url, timeout, httpx.MockTransport(down)
        ),
    )
    result = runner.invoke(app, ["simulate", "--target", "http://api:9", "--speed", "0"])
    assert result.exit_code == 1
    assert "A API está no ar?" in result.output


def test_api_command_starts_uvicorn(monkeypatch):
    calls = []
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = runner.invoke(app, ["api", "--port", "9000"])
    assert result.exit_code == 0, result.output
    assert calls == [
        (
            ("medwatch.api.app:create_app",),
            {"factory": True, "host": "127.0.0.1", "port": 9000},
        )
    ]


@pytest.mark.parametrize("api_url", [None, "http://api:8000"])
def test_dashboard_command_runs_streamlit(monkeypatch, api_url):
    calls = []
    monkeypatch.setattr(
        realtime.subprocess, "call", lambda command, env: calls.append((command, env)) or 0
    )
    args = ["dashboard"] + (["--api-url", api_url] if api_url else [])
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    command, env = calls[0]
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert command[4].endswith("dashboard/app.py")
    assert env["MEDWATCH_API_URL"] == (api_url or "http://localhost:8000")
