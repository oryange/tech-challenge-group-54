"""Dublê mínimo do `azure.cognitiveservices.speech` para testar o fluxo de eventos sem rede.

Imita o contrato usado pelo código: o stream de áudio é fechado → o reconhecedor dispara
`recognized`/`transcribed` com o JSON detalhado → `canceled` (EndOfStream ou erro) →
`session_stopped`.
"""

import enum
import json
from types import SimpleNamespace
from typing import Any


class ResultReason(enum.Enum):
    RecognizedSpeech = 1
    TranslatedSpeech = 2
    NoMatch = 3
    SynthesizingAudioCompleted = 4
    Canceled = 5


class CancellationReason(enum.Enum):
    EndOfStream = 1
    Error = 2


class CancellationErrorCode(enum.Enum):
    NoError = 0
    AuthenticationFailure = 1
    TooManyRequests = 2
    ServiceTimeout = 3


class OutputFormat(enum.Enum):
    Simple = 0
    Detailed = 1


class SpeechSynthesisOutputFormat(enum.Enum):
    Riff16Khz16BitMonoPcm = 1


class Signal:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def connect(self, handler: Any) -> None:
        self.handlers.append(handler)

    def fire(self, evt: Any) -> None:
        for handler in self.handlers:
            handler(evt)


class _Future:
    def __init__(self, value: Any = None) -> None:
        self.value = value

    def get(self) -> Any:
        return self.value


class SpeechConfig:
    instances: list["SpeechConfig"] = []

    def __init__(self, subscription: str, region: str) -> None:
        self.subscription = subscription
        self.region = region
        self.speech_recognition_language = None
        self.output_format = OutputFormat.Simple
        self.word_level_timestamps = False
        self.synthesis_format = None
        SpeechConfig.instances.append(self)

    def request_word_level_timestamps(self) -> None:
        self.word_level_timestamps = True

    def set_speech_synthesis_output_format(self, fmt: Any) -> None:
        self.synthesis_format = fmt


class _AudioModule:
    class AudioStreamFormat:
        def __init__(self, samples_per_second: int, bits_per_sample: int, channels: int) -> None:
            self.samples_per_second = samples_per_second
            self.bits_per_sample = bits_per_sample
            self.channels = channels

    class PushAudioInputStream:
        def __init__(self, stream_format: Any) -> None:
            self.stream_format = stream_format
            self.data = bytearray()
            self.on_close: Any = None

        def write(self, chunk: bytes) -> None:
            self.data.extend(chunk)

        def close(self) -> None:
            if self.on_close:
                self.on_close()

    class AudioConfig:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

    class AudioOutputConfig:
        def __init__(self, filename: str) -> None:
            self.filename = filename


audio = _AudioModule()

# o que o próximo reconhecedor vai "ouvir": lista de (json, speaker_id) e erro opcional
SCRIPT: dict[str, Any] = {"results": [], "error": None, "stop_session": True}


def _result_event(payload: dict[str, Any], speaker: str | None) -> Any:
    result = SimpleNamespace(
        reason=ResultReason.RecognizedSpeech, json=json.dumps(payload), speaker_id=speaker
    )
    return SimpleNamespace(result=result)


def _canceled_event(error: CancellationErrorCode | None) -> Any:
    reason = CancellationReason.Error if error else CancellationReason.EndOfStream
    details = SimpleNamespace(reason=reason, code=error or CancellationErrorCode.NoError)
    return SimpleNamespace(cancellation_details=details)


class _Engine:
    result_signal_name = "recognized"

    def __init__(self, speech_config: Any, audio_config: Any) -> None:
        self.speech_config = speech_config
        self.audio_config = audio_config
        self.canceled = Signal()
        self.session_stopped = Signal()
        setattr(self, self.result_signal_name, Signal())
        audio_config.stream.on_close = self._play
        self.started = self.stopped = False
        self.bytes_received = 0

    def _play(self) -> None:
        self.bytes_received = len(self.audio_config.stream.data)
        results: Signal = getattr(self, self.result_signal_name)
        no_match = SimpleNamespace(result=SimpleNamespace(reason=ResultReason.NoMatch))
        results.fire(no_match)
        for payload, speaker in SCRIPT["results"]:
            results.fire(_result_event(payload, speaker))
        self.canceled.fire(_canceled_event(SCRIPT["error"]))
        if SCRIPT["stop_session"]:
            self.session_stopped.fire(SimpleNamespace())

    def _start(self) -> _Future:
        self.started = True
        return _Future()

    def _stop(self) -> _Future:
        self.stopped = True
        return _Future()


class SpeechRecognizer(_Engine):
    last: "SpeechRecognizer | None" = None

    def __init__(self, speech_config: Any, audio_config: Any) -> None:
        super().__init__(speech_config, audio_config)
        SpeechRecognizer.last = self

    def start_continuous_recognition_async(self) -> _Future:
        return self._start()

    def stop_continuous_recognition_async(self) -> _Future:
        return self._stop()


class _TranscriptionModule:
    class ConversationTranscriber(_Engine):
        result_signal_name = "transcribed"

        def start_transcribing_async(self) -> _Future:
            return self._start()

        def stop_transcribing_async(self) -> _Future:
            return self._stop()


transcription = _TranscriptionModule()

SYNTHESIS: dict[str, Any] = {"error": None, "ssml": None}


class SpeechSynthesizer:
    def __init__(self, speech_config: Any, audio_config: Any) -> None:
        self.speech_config = speech_config
        self.audio_config = audio_config

    def speak_ssml_async(self, ssml: str) -> _Future:
        SYNTHESIS["ssml"] = ssml
        with open(self.audio_config.filename, "wb") as fh:
            fh.write(b"RIFF")
        error = SYNTHESIS["error"]
        if error:
            details = SimpleNamespace(error_code=error)
            return _Future(
                SimpleNamespace(reason=ResultReason.Canceled, cancellation_details=details)
            )
        return _Future(SimpleNamespace(reason=ResultReason.SynthesizingAudioCompleted))
