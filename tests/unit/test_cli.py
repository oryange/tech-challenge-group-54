import asyncio

import httpx
import pytest
from typer.testing import CliRunner

from medwatch import cli
from medwatch.cli import app
from medwatch.config import get_settings
from medwatch.core.alerts import ChannelError, WebhookChannel
from medwatch.core.events import Alert, Severity

runner = CliRunner()


@pytest.fixture
def azure_env(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "speech-key")
    monkeypatch.setenv("AZURE_LANGUAGE_KEY", "language-key")
    monkeypatch.setenv("AZURE_LANGUAGE_ENDPOINT", "https://fake.cognitiveservices.azure.com/")
    get_settings.cache_clear()


def test_healthcheck_passes_without_azure():
    result = runner.invoke(app, ["healthcheck"])
    assert result.exit_code == 0
    assert "azure.speech.config" in result.stdout
    assert "Chaves Azure ausentes" in result.stdout


def test_healthcheck_strict_fails_without_azure():
    result = runner.invoke(app, ["healthcheck", "--strict"])
    assert result.exit_code == 1


def test_healthcheck_online_fails_without_azure():
    result = runner.invoke(app, ["healthcheck", "--online"])
    assert result.exit_code == 1
    assert "❌ azure.speech.online" in result.stdout


def test_healthcheck_online_passes_with_valid_credentials(azure_env, monkeypatch):
    monkeypatch.setattr(cli, "_ping_speech", lambda settings: True)
    monkeypatch.setattr(cli, "_ping_language", lambda settings: True)
    monkeypatch.setattr(cli, "_has_module", lambda name: True)
    result = runner.invoke(app, ["healthcheck", "--online"])
    assert result.exit_code == 0
    assert "✅ azure.speech.online" in result.stdout
    assert "✅ azure.language.online" in result.stdout
    assert "Chaves Azure ausentes" not in result.stdout


def test_healthcheck_online_reports_network_error(azure_env, monkeypatch):
    def boom(settings):
        raise ConnectionError("sem rede")

    monkeypatch.setattr(cli, "_ping_speech", boom)
    monkeypatch.setattr(cli, "_ping_language", lambda settings: True)
    monkeypatch.setattr(cli, "_has_module", lambda name: True)
    result = runner.invoke(app, ["healthcheck", "--online"])
    assert result.exit_code == 1
    assert "❌ azure.speech.online" in result.stdout


@pytest.mark.parametrize(("status", "expected"), [(200, True), (401, False)])
def test_ping_speech_checks_token_endpoint(azure_env, status, expected):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers["Ocp-Apim-Subscription-Key"]
        return httpx.Response(status, text="token")

    assert cli._ping_speech(get_settings(), transport=httpx.MockTransport(handler)) is expected
    assert seen["url"] == "https://brazilsouth.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    assert seen["key"] == "speech-key"


def test_alerts_demo_writes_history():
    result = runner.invoke(app, ["alerts", "demo"])
    assert result.exit_code == 0
    assert (
        "Alertas enviados: 4 · suprimidos (cooldown): 1 · ignorados (< medium): 1" in result.stdout
    )
    history = get_settings().alerts_file.read_text(encoding="utf-8").splitlines()
    assert len(history) == 4


def test_alerts_demo_lists_channels_without_webhook():
    result = runner.invoke(app, ["alerts", "demo"])
    assert "Canais: console, jsonl\n" in result.stdout


@pytest.fixture
def webhook_env(monkeypatch):
    monkeypatch.setenv("MEDWATCH_ALERT_WEBHOOK_URL", "https://hooks.example.com/segredo")
    monkeypatch.setenv("MEDWATCH_ALERT_WEBHOOK_FORMAT", "discord")
    get_settings.cache_clear()


def recording_webhook(monkeypatch, delays: list[float] | None = None) -> list[Alert]:
    """Troca o envio HTTP do webhook por um registro dos alertas (com atraso opcional)."""
    sent: list[Alert] = []
    pending = list(delays or [])

    async def send(self, alert: Alert) -> None:
        await asyncio.sleep(pending.pop(0) if pending else 0)
        sent.append(alert)

    monkeypatch.setattr(WebhookChannel, "send", send)
    return sent


def test_alerts_demo_delivers_webhook_in_scenario_order(webhook_env, monkeypatch):
    # o primeiro alerta é o mais lento: em paralelo ele chegaria por último no chat
    sent = recording_webhook(monkeypatch, delays=[0.05, 0.0, 0.0, 0.0])
    result = runner.invoke(app, ["alerts", "demo"])
    assert result.exit_code == 0
    assert "Canais: console, jsonl, webhook (discord)" in result.stdout
    assert "segredo" not in result.stdout
    assert [(a.patient_id, a.alert_type, a.severity) for a in sent] == [
        ("PAC-001", "vitals.spo2_drop", Severity.HIGH),
        ("PAC-001", "audio.dyspnea_terms", Severity.MEDIUM),
        ("PAC-001", "vitals.spo2_drop", Severity.CRITICAL),
        ("PAC-002", "video.fall_detected", Severity.CRITICAL),
    ]


def test_alerts_test_webhook_requires_url():
    result = runner.invoke(app, ["alerts", "test-webhook"])
    assert result.exit_code == 1
    assert "MEDWATCH_ALERT_WEBHOOK_URL" in result.stderr


def test_alerts_test_webhook_sends_one_message(webhook_env, monkeypatch):
    sent = recording_webhook(monkeypatch)
    result = runner.invoke(app, ["alerts", "test-webhook"])
    assert result.exit_code == 0
    assert "✅" in result.stdout
    assert len(sent) == 1
    assert sent[0].patient_id == "PAC-TESTE"
    assert not get_settings().alerts_file.exists()  # teste não entra no histórico


def test_alerts_test_webhook_reports_failure(webhook_env, monkeypatch):
    async def send(self, alert: Alert) -> None:
        raise ChannelError("Webhook recusou o alerta (HTTP 404)")

    monkeypatch.setattr(WebhookChannel, "send", send)
    result = runner.invoke(app, ["alerts", "test-webhook"])
    assert result.exit_code == 1
    assert "HTTP 404" in result.stderr
    assert "segredo" not in result.stderr
