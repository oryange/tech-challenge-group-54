"""Transcrição com Azure Speech to Text: reconhecimento contínuo, pt-BR, tempo por palavra.

O áudio vai por stream (PCM 16 kHz mono) direto da memória, sem arquivo temporário.
Com `diarize=True` usa o ConversationTranscriber, que separa os falantes (médico × paciente).
"""

import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

from medwatch.audio.config import SpeechConfig
from medwatch.audio.preprocessing import AudioClip
from medwatch.audio.transcript import Segment, Transcript, Word

if TYPE_CHECKING:
    from medwatch.config import AzureSettings

TICKS_PER_SECOND = 10_000_000  # o Speech informa offsets em unidades de 100 ns

# CancellationErrorCode que valem nova tentativa; os demais (chave, cota, request) não mudam
TRANSIENT_ERRORS = frozenset(
    {"TooManyRequests", "ServiceTimeout", "ServiceUnavailable", "ConnectionFailure", "ServiceError"}
)


class SpeechServiceError(RuntimeError):
    """Falha do Azure Speech. A mensagem traz só o código do erro, nunca o áudio ou o texto."""

    def __init__(self, code: str) -> None:
        super().__init__(f"Azure Speech cancelou a transcrição (código {code})")
        self.code = code

    @property
    def transient(self) -> bool:
        return self.code in TRANSIENT_ERRORS


class Transcriber(Protocol):
    def transcribe(self, clip: AudioClip) -> Transcript: ...


def parse_detailed_result(
    payload: str | dict[str, Any], speaker: str | None = None
) -> Segment | None:
    """Converte o JSON do formato `Detailed` do Speech em um `Segment`."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    if data.get("RecognitionStatus", "Success") != "Success":
        return None
    best = (data.get("NBest") or [{}])[0]
    text = (best.get("Display") or data.get("DisplayText") or "").strip()
    if not text:
        return None
    start = data.get("Offset", 0) / TICKS_PER_SECOND
    words = [
        Word(
            text=w["Word"],
            start=w["Offset"] / TICKS_PER_SECOND,
            end=(w["Offset"] + w["Duration"]) / TICKS_PER_SECOND,
        )
        for w in best.get("Words", [])
    ]
    confidence = best.get("Confidence")
    return Segment(
        text=text,
        start=start,
        end=start + data.get("Duration", 0) / TICKS_PER_SECOND,
        speaker=speaker or data.get("SpeakerId") or None,
        confidence=float(confidence) if confidence is not None else None,
        words=words,
    )


class AzureSpeechTranscriber:
    """Cliente do Azure Speech com retry em erro transitório e timeout por transcrição."""

    def __init__(self, azure: "AzureSettings", config: SpeechConfig, diarize: bool = False) -> None:
        if not (azure.speech_key and azure.speech_key.get_secret_value()):
            raise ValueError("AZURE_SPEECH_KEY não configurada")
        self._azure = azure
        self._config = config
        self._diarize = diarize

    def transcribe(self, clip: AudioClip) -> Transcript:
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                return self._transcribe_once(clip)
            except SpeechServiceError as exc:
                if not exc.transient or attempt == self._config.max_attempts:
                    raise
                delay = self._config.backoff_seconds * 2 ** (attempt - 1)
                logger.warning(
                    "Azure Speech: erro {} (tentativa {}/{}), nova tentativa em {:.0f} s",
                    exc.code,
                    attempt,
                    self._config.max_attempts,
                    delay,
                )
                time.sleep(delay)
        raise AssertionError("inalcançável")  # pragma: no cover

    def _speech_config(self) -> Any:
        import azure.cognitiveservices.speech as speechsdk

        assert self._azure.speech_key is not None  # noqa: S101 - validado no __init__
        config = speechsdk.SpeechConfig(
            subscription=self._azure.speech_key.get_secret_value(), region=self._azure.speech_region
        )
        config.speech_recognition_language = self._azure.speech_language
        config.output_format = speechsdk.OutputFormat.Detailed
        config.request_word_level_timestamps()
        return config

    def _transcribe_once(self, clip: AudioClip) -> Transcript:
        import azure.cognitiveservices.speech as speechsdk

        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=clip.sample_rate, bits_per_sample=16, channels=1
        )
        stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
        audio_config = speechsdk.audio.AudioConfig(stream=stream)
        speech_config = self._speech_config()

        segments: list[Segment] = []
        errors: list[str] = []
        done = threading.Event()

        def on_result(evt: Any) -> None:
            reason = evt.result.reason
            if reason in (
                speechsdk.ResultReason.RecognizedSpeech,
                speechsdk.ResultReason.TranslatedSpeech,
            ):
                speaker = getattr(evt.result, "speaker_id", None)
                if speaker in ("", "Unknown"):
                    speaker = None
                segment = parse_detailed_result(evt.result.json, speaker=speaker)
                if segment is not None:
                    segments.append(segment)

        def on_canceled(evt: Any) -> None:
            details = evt.cancellation_details
            if details.reason == speechsdk.CancellationReason.Error:
                errors.append(details.code.name)
            done.set()

        if self._diarize:
            engine = speechsdk.transcription.ConversationTranscriber(
                speech_config=speech_config, audio_config=audio_config
            )
            engine.transcribed.connect(on_result)
            start, stop = engine.start_transcribing_async, engine.stop_transcribing_async
        else:
            engine = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )
            engine.recognized.connect(on_result)
            start = engine.start_continuous_recognition_async
            stop = engine.stop_continuous_recognition_async
        engine.canceled.connect(on_canceled)
        engine.session_stopped.connect(lambda _evt: done.set())

        start().get()
        pcm = clip.to_pcm16()
        chunk = clip.sample_rate * 2  # 1 s de áudio por escrita
        for offset in range(0, len(pcm), chunk):
            stream.write(pcm[offset : offset + chunk])
        stream.close()
        finished = done.wait(timeout=self._config.timeout_seconds)
        stop().get()

        if errors:
            raise SpeechServiceError(errors[0])
        if not finished:
            raise SpeechServiceError("ServiceTimeout")
        segments.sort(key=lambda s: s.start)
        logger.info("Azure Speech: {} segmentos, {:.0f} s de áudio", len(segments), clip.duration)
        return Transcript(language=self._azure.speech_language, segments=segments)


class RecordedTranscriber:
    """Devolve uma transcrição gravada (modo mock: CI e demonstração offline)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def transcribe(self, clip: AudioClip) -> Transcript:
        if not self._path.is_file():
            raise FileNotFoundError(
                f"Resposta gravada não encontrada: {self._path.name}. "
                "Rode uma vez com o Azure real e --record para gravá-la."
            )
        return Transcript.model_validate_json(self._path.read_text(encoding="utf-8"))


class RecordingTranscriber:
    """Chama o provedor real e grava a resposta para o modo mock."""

    def __init__(self, inner: Transcriber, path: Path) -> None:
        self._inner = inner
        self._path = path

    def transcribe(self, clip: AudioClip) -> Transcript:
        transcript = self._inner.transcribe(clip)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        return transcript
