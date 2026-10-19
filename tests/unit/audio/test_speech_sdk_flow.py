"""Fluxo de eventos do Azure Speech com um dublê do SDK (sem rede, sem chave real)."""

import copy
import json
import sys

import azure.cognitiveservices
import pytest

from medwatch.audio.azure_stt import AzureSpeechTranscriber, SpeechServiceError
from medwatch.audio.config import SpeechConfig
from medwatch.audio.synthesis import SampleScript, Turn, build_ssml, synthesize
from medwatch.config import AzureSettings
from tests.unit.audio import fake_speechsdk as sdk
from tests.unit.audio.helpers import FIXTURES, clip, voiced

DETAILED = json.loads((FIXTURES / "speech_detailed_result.json").read_text("utf-8"))


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "azure.cognitiveservices.speech", sdk)
    monkeypatch.setattr(azure.cognitiveservices, "speech", sdk, raising=False)
    sdk.SpeechConfig.instances.clear()
    sdk.SCRIPT.update(results=[], error=None, stop_session=True)
    sdk.SYNTHESIS.update(error=None, ssml=None)


def azure_settings():
    return AzureSettings(speech_key="chave-fake", speech_region="brazilsouth")


def segment(text: str, offset_s: float) -> dict:
    payload = copy.deepcopy(DETAILED)
    payload["Offset"] = int(offset_s * 10_000_000)
    payload["NBest"][0]["Display"] = text
    return payload


def test_reconhecimento_continuo_configura_e_ordena_segmentos():
    sdk.SCRIPT["results"] = [
        (segment("Segunda frase.", 5.0), None),
        (segment("Primeira frase.", 1.0), None),
    ]
    audio = clip(voiced(2.0))
    transcript = AzureSpeechTranscriber(azure_settings(), SpeechConfig()).transcribe(audio)

    assert [s.text for s in transcript.segments] == ["Primeira frase.", "Segunda frase."]
    assert transcript.language == "pt-BR"
    config = sdk.SpeechConfig.instances[-1]
    assert config.subscription == "chave-fake"
    assert config.region == "brazilsouth"
    assert config.speech_recognition_language == "pt-BR"
    assert config.output_format == sdk.OutputFormat.Detailed
    assert config.word_level_timestamps is True
    recognizer = sdk.SpeechRecognizer.last
    assert recognizer.started and recognizer.stopped
    assert recognizer.bytes_received == len(audio.to_pcm16())
    stream_format = recognizer.audio_config.stream.stream_format
    assert (stream_format.samples_per_second, stream_format.bits_per_sample) == (16000, 16)


def test_diarizacao_registra_falantes():
    sdk.SCRIPT["results"] = [
        (segment("Como está?", 0.5), "Guest-1"),
        (segment("Com falta de ar.", 2.0), "Guest-2"),
        (segment("Hum.", 3.0), "Unknown"),
    ]
    transcriber = AzureSpeechTranscriber(azure_settings(), SpeechConfig(), diarize=True)
    transcript = transcriber.transcribe(clip(voiced(1.0)))

    assert [s.speaker for s in transcript.segments] == ["Guest-1", "Guest-2", None]
    assert transcript.speakers == ["Guest-1", "Guest-2"]


def test_cancelamento_por_erro_vira_speech_service_error():
    sdk.SCRIPT["error"] = sdk.CancellationErrorCode.AuthenticationFailure
    transcriber = AzureSpeechTranscriber(azure_settings(), SpeechConfig(max_attempts=3))
    with pytest.raises(SpeechServiceError, match="AuthenticationFailure"):
        transcriber.transcribe(clip(voiced(1.0)))
    assert len(sdk.SpeechConfig.instances) == 1  # erro permanente: sem nova tentativa


def test_sessao_que_nao_termina_estoura_timeout(monkeypatch):
    class Silent(sdk.SpeechRecognizer):
        def _play(self):  # nem fim de stream, nem session_stopped
            self.bytes_received = len(self.audio_config.stream.data)

    monkeypatch.setattr(sdk, "SpeechRecognizer", Silent)
    config = SpeechConfig(timeout_seconds=0.05, max_attempts=1)
    with pytest.raises(SpeechServiceError, match="ServiceTimeout"):
        AzureSpeechTranscriber(azure_settings(), config).transcribe(clip(voiced(1.0)))


def script():
    return SampleScript(description="t", turns=[Turn(speaker="paciente", phrases=["Oi."])])


VOICES = {"medico": "pt-BR-AntonioNeural", "paciente": "pt-BR-FranciscaNeural"}


def test_sintese_grava_wav_16k(tmp_path):
    output = tmp_path / "samples" / "consulta.wav"
    ssml = build_ssml(script(), VOICES)
    synthesize(ssml, output, azure_settings())

    assert output.exists()
    assert sdk.SYNTHESIS["ssml"] == ssml
    format_ = sdk.SpeechConfig.instances[-1].synthesis_format
    assert format_ == sdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm


def test_sintese_com_erro_remove_arquivo_parcial(tmp_path):
    sdk.SYNTHESIS["error"] = sdk.CancellationErrorCode.TooManyRequests
    output = tmp_path / "consulta.wav"
    with pytest.raises(SpeechServiceError, match="TooManyRequests"):
        synthesize(build_ssml(script(), VOICES), output, azure_settings())
    assert not output.exists()


def test_sintese_sem_chave():
    with pytest.raises(ValueError, match="AZURE_SPEECH_KEY"):
        synthesize("<speak/>", None, AzureSettings(speech_key=None))  # type: ignore[arg-type]
