import pytest

from medwatch.config import get_settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    for var in ("AZURE_SPEECH_KEY", "AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"):
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("MEDWATCH_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("MEDWATCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEDWATCH_MODELS_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
