import json

import pytest

from medwatch.audio import azure_stt
from medwatch.audio.azure_stt import (
    AzureSpeechTranscriber,
    RecordedTranscriber,
    RecordingTranscriber,
    SpeechServiceError,
    parse_detailed_result,
)
from medwatch.audio.config import SpeechConfig
from medwatch.audio.transcript import Segment, Transcript
from medwatch.config import AzureSettings
from tests.unit.audio.helpers import FIXTURES, FakeTranscriber, clip, voiced


def test_parse_resultado_detalhado():
    payload = (FIXTURES / "speech_detailed_result.json").read_text("utf-8")
    segment = parse_detailed_result(payload)

    assert segment is not None
    assert segment.text == "Estou com falta de ar."
    assert segment.start == pytest.approx(0.3)
    assert segment.end == pytest.approx(2.4)
    assert segment.confidence == pytest.approx(0.91)
    assert [w.text for w in segment.words] == ["estou", "com", "falta", "de", "ar"]
    assert segment.words[1].start == pytest.approx(0.7)
    assert segment.words[1].end == pytest.approx(1.05)


def test_parse_com_falante_da_diarizacao():
    payload = json.loads((FIXTURES / "speech_detailed_result.json").read_text("utf-8"))
    payload["SpeakerId"] = "Guest-2"
    assert parse_detailed_result(payload).speaker == "Guest-2"
    assert parse_detailed_result(payload, speaker="Guest-1").speaker == "Guest-1"


@pytest.mark.parametrize(
    "payload",
    [
        {"RecognitionStatus": "NoMatch", "Offset": 0, "Duration": 0},
        {"RecognitionStatus": "Success", "DisplayText": "  ", "Offset": 0, "Duration": 0},
    ],
)
def test_parse_sem_fala(payload):
    assert parse_detailed_result(payload) is None


def test_parse_sem_nbest_usa_display_text():
    segment = parse_detailed_result(
        {"DisplayText": "Bom dia.", "Offset": 0, "Duration": 10_000_000}
    )
    assert segment.text == "Bom dia."
    assert segment.words == []
    assert segment.word_count == 2


def test_erros_transitorios():
    assert SpeechServiceError("TooManyRequests").transient
    assert SpeechServiceError("ConnectionFailure").transient
    assert not SpeechServiceError("AuthenticationFailure").transient
    assert "AuthenticationFailure" in str(SpeechServiceError("AuthenticationFailure"))


def _transcriber(**overrides):
    azure = AzureSettings(speech_key="chave-fake", speech_region="brazilsouth")
    config = SpeechConfig(max_attempts=3, backoff_seconds=1.0).model_copy(update=overrides)
    return AzureSpeechTranscriber(azure, config)


def test_sem_chave_recusa_criar_cliente():
    with pytest.raises(ValueError, match="AZURE_SPEECH_KEY"):
        AzureSpeechTranscriber(AzureSettings(speech_key=None), SpeechConfig())


def test_retry_em_erro_transitorio_com_backoff(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(azure_stt.time, "sleep", sleeps.append)
    expected = Transcript(language="pt-BR", segments=[Segment(text="Oi.", start=0, end=1)])
    outcomes = [SpeechServiceError("ServiceTimeout"), SpeechServiceError("TooManyRequests")]

    def flaky(self, audio):
        if outcomes:
            raise outcomes.pop(0)
        return expected

    monkeypatch.setattr(AzureSpeechTranscriber, "_transcribe_once", flaky)
    assert _transcriber().transcribe(clip(voiced(1.0))) == expected
    assert sleeps == [1.0, 2.0]


def test_erro_permanente_nao_repete(monkeypatch):
    calls = []

    def denied(self, audio):
        calls.append(1)
        raise SpeechServiceError("AuthenticationFailure")

    monkeypatch.setattr(AzureSpeechTranscriber, "_transcribe_once", denied)
    with pytest.raises(SpeechServiceError, match="AuthenticationFailure"):
        _transcriber().transcribe(clip(voiced(1.0)))
    assert len(calls) == 1


def test_desiste_apos_max_attempts(monkeypatch):
    monkeypatch.setattr(azure_stt.time, "sleep", lambda _s: None)

    def always_busy(self, audio):
        raise SpeechServiceError("ServiceUnavailable")

    monkeypatch.setattr(AzureSpeechTranscriber, "_transcribe_once", always_busy)
    with pytest.raises(SpeechServiceError, match="ServiceUnavailable"):
        _transcriber(max_attempts=2).transcribe(clip(voiced(1.0)))


def test_resposta_gravada_ida_e_volta(tmp_path):
    transcript = Transcript(language="pt-BR", segments=[Segment(text="Bom dia.", start=0, end=1)])
    path = tmp_path / "azure" / "abc.stt.json"

    RecordingTranscriber(FakeTranscriber(transcript), path).transcribe(clip(voiced(1.0)))

    assert RecordedTranscriber(path).transcribe(clip(voiced(1.0))) == transcript


def test_resposta_gravada_ausente(tmp_path):
    with pytest.raises(FileNotFoundError, match="--record"):
        RecordedTranscriber(tmp_path / "x.stt.json").transcribe(clip(voiced(1.0)))


def test_time_at_interpola_dentro_do_segmento():
    transcript = Transcript(
        language="pt-BR",
        segments=[
            Segment(text="Bom dia.", start=0.0, end=1.0),
            Segment(text="Estou com falta de ar.", start=5.0, end=7.2),
        ],
    )
    text = transcript.text
    assert transcript.time_at(0) == 0.0
    assert transcript.time_at(text.index("Estou")) == pytest.approx(5.0)
    assert 5.0 < transcript.time_at(text.index("falta")) < 7.2
    assert Transcript(language="pt-BR").time_at(10) == 0.0
