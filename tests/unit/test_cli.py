import httpx
import pytest
from typer.testing import CliRunner

from medwatch import cli
from medwatch.cli import app
from medwatch.config import get_settings

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
