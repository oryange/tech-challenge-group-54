"""Consultas fictícias com Azure Text to Speech (configs/audio_samples.yaml).

Gera WAV PCM 16 kHz mono em data/samples/ para testes reprodutíveis e para a demonstração.
O texto do roteiro é escapado antes de entrar no SSML.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal
from xml.sax.saxutils import escape, quoteattr

import yaml
from pydantic import BaseModel, Field

from medwatch.audio.azure_stt import SpeechServiceError

if TYPE_CHECKING:
    from medwatch.config import AzureSettings

PROSODY_PATTERN = r"^[+-]?\d{1,2}%$"
VOICE_PATTERN = r"^[a-z]{2}-[A-Z]{2}-[A-Za-z]+Neural$"
SAMPLE_NAME_PATTERN = r"^[a-z0-9_]{1,64}$"

Speaker = Literal["medico", "paciente"]
VoiceName = Annotated[str, Field(pattern=VOICE_PATTERN)]
SampleName = Annotated[str, Field(pattern=SAMPLE_NAME_PATTERN)]


class Prosody(BaseModel):
    rate: str = Field(default="0%", pattern=PROSODY_PATTERN)
    pitch: str = Field(default="0%", pattern=PROSODY_PATTERN)


class Turn(BaseModel):
    speaker: Speaker
    phrases: list[str] = Field(min_length=1)


class SampleScript(BaseModel):
    description: str
    patient_prosody: Prosody = Field(default_factory=Prosody)
    phrase_break_ms: int = Field(default=250, ge=0, le=5000)
    turn_break_ms: int = Field(default=600, ge=0, le=5000)
    turns: list[Turn] = Field(min_length=1)


class SamplesConfig(BaseModel):
    """Vozes e roteiros das consultas fictícias (configs/audio_samples.yaml)."""

    voices: dict[Speaker, VoiceName]
    # o nome vira o arquivo data/samples/<nome>.wav
    samples: dict[SampleName, SampleScript]

    @classmethod
    def load(cls, path: Path) -> "SamplesConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


def build_ssml(script: SampleScript, voices: dict[Speaker, str]) -> str:
    """SSML com uma voz por falante, prosódia do paciente e pausas entre frases."""
    parts = ['<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR">']
    for turn in script.turns:
        prosody = script.patient_prosody if turn.speaker == "paciente" else Prosody()
        phrases = f'<break time="{script.phrase_break_ms}ms"/>'.join(
            escape(phrase) for phrase in turn.phrases
        )
        parts.append(
            f"<voice name={quoteattr(voices[turn.speaker])}>"
            f"<prosody rate={quoteattr(prosody.rate)} pitch={quoteattr(prosody.pitch)}>"
            f"{phrases}</prosody>"
            f'<break time="{script.turn_break_ms}ms"/></voice>'
        )
    parts.append("</speak>")
    return "".join(parts)


def synthesize(ssml: str, output: Path, azure: "AzureSettings") -> None:
    """Sintetiza o SSML em `output` (WAV PCM 16 kHz mono, 16 bits)."""
    import azure.cognitiveservices.speech as speechsdk

    if not (azure.speech_key and azure.speech_key.get_secret_value()):
        raise ValueError("AZURE_SPEECH_KEY não configurada")
    config = speechsdk.SpeechConfig(
        subscription=azure.speech_key.get_secret_value(), region=azure.speech_region
    )
    config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output))
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=audio_config)
    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        details = result.cancellation_details
        code = details.error_code.name if details is not None else "Unknown"
        output.unlink(missing_ok=True)
        raise SpeechServiceError(code)
