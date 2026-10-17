from medwatch.config import get_settings


def test_env_override(monkeypatch):
    monkeypatch.setenv("MEDWATCH_POSE_BACKEND", "openpose")
    get_settings.cache_clear()
    assert get_settings().pose_backend == "openpose"


def test_azure_not_configured_without_keys():
    settings = get_settings()
    assert not settings.azure.speech_configured
    assert not settings.azure.language_configured


def test_azure_key_is_secret(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "super-secret")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.azure.speech_configured
    assert "super-secret" not in repr(settings)


def test_ensure_dirs_creates_tree():
    settings = get_settings()
    settings.ensure_dirs()
    assert (settings.outputs_dir / "alerts").is_dir()
    assert (settings.data_dir / "raw").is_dir()
    assert settings.logs_dir.is_dir()
