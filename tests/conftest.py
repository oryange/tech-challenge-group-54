import pytest

from medwatch.config import get_settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    # isola os testes de um .env local: sem chaves Azure e sem webhook real
    for var in (
        "AZURE_SPEECH_KEY",
        "AZURE_LANGUAGE_KEY",
        "AZURE_LANGUAGE_ENDPOINT",
        "MEDWATCH_ALERT_WEBHOOK_URL",
    ):
        monkeypatch.setenv(var, "")
    # URL da API do dashboard/simulador: os testes usam o padrão (localhost:8000)
    monkeypatch.delenv("MEDWATCH_API_URL", raising=False)
    monkeypatch.setenv("MEDWATCH_ALERT_MIN_SEVERITY", "medium")
    monkeypatch.setenv("MEDWATCH_ALERT_COOLDOWN_MINUTES", "15")
    monkeypatch.setenv("MEDWATCH_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("MEDWATCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEDWATCH_MODELS_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
