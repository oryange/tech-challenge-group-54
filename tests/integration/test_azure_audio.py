"""Chamadas reais ao Azure (não rodam no CI): `pytest -m azure` com o .env preenchido."""

import os

import pytest

from medwatch.audio.azure_language import AzureLanguageAnalyzer
from medwatch.audio.azure_stt import AzureSpeechTranscriber
from medwatch.audio.preprocessing import load_audio
from medwatch.config import PROJECT_ROOT, AzureSettings
from tests.unit.audio.helpers import audio_config

pytestmark = pytest.mark.azure

SAMPLE = PROJECT_ROOT / "data" / "samples" / "consulta_01_normal.wav"


@pytest.fixture
def azure(monkeypatch):
    # o conftest zera as chaves para isolar os testes; aqui lemos o .env de propósito
    for var in ("AZURE_SPEECH_KEY", "AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    settings = AzureSettings()
    if not (settings.speech_configured and settings.language_configured):
        pytest.skip("Azure não configurado no .env")
    return settings


@pytest.mark.skipif(not SAMPLE.exists(), reason="rode antes: medwatch audio synth-samples")
def test_transcricao_real_da_consulta_01(azure):
    config = audio_config()
    clip = load_audio(SAMPLE, config.preprocessing)
    transcript = AzureSpeechTranscriber(azure, config.speech).transcribe(clip)

    assert transcript.word_count > 20
    assert any(segment.words for segment in transcript.segments)
    assert "falta de ar" in transcript.text.lower()


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="nunca no CI")
def test_language_real_em_pt(azure):
    analysis = AzureLanguageAnalyzer(azure, audio_config().language).analyze(
        "Não tenho falta de ar. Estou com muita tontura desde ontem."
    )
    assert analysis.sentiment is not None
    if analysis.health_available:
        assert any(e.category == "SymptomOrSign" for e in analysis.entities)
