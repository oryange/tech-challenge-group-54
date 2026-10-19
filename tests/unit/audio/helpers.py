"""Sinais sintéticos e dublês dos clientes Azure para os testes de áudio."""

from pathlib import Path

import numpy as np
import soundfile as sf

from medwatch.audio.azure_language import LanguageAnalysis
from medwatch.audio.config import AudioConfig, CriticalTermsConfig
from medwatch.audio.preprocessing import AudioClip
from medwatch.audio.transcript import Transcript
from medwatch.config import PROJECT_ROOT

CONFIGS = PROJECT_ROOT / "configs"
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "azure"


def audio_config() -> AudioConfig:
    return AudioConfig.load(CONFIGS / "audio.yaml")


def terms_config() -> CriticalTermsConfig:
    return CriticalTermsConfig.load(CONFIGS / "critical_terms.yaml")


def voiced(seconds: float, f0: float = 180.0, sr: int = 16000, vibrato_semitones: float = 0.0):
    """Tom com harmônicos (parecido com voz) e vibrato opcional para variar o F0."""
    t = np.arange(int(seconds * sr)) / sr
    semitones = vibrato_semitones * np.sin(2 * np.pi * 0.5 * t)
    freq = f0 * 2 ** (semitones / 12)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    wave = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (0.3 * wave / np.max(np.abs(wave))).astype(np.float32)


def silence(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def phrases(
    count: int, phrase_seconds: float, pause_seconds: float, sr: int = 16000, **kwargs: float
) -> np.ndarray:
    """`count` frases sonoras separadas por pausas, com 0,3 s de silêncio nas pontas."""
    parts = [silence(0.3, sr)]
    for index in range(count):
        parts.append(voiced(phrase_seconds, sr=sr, **kwargs))
        parts.append(silence(pause_seconds if index < count - 1 else 0.3, sr))
    return np.concatenate(parts)


def write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> Path:
    sf.write(path, samples, sr, subtype="PCM_16")
    return path


def clip(samples: np.ndarray, sr: int = 16000) -> AudioClip:
    return AudioClip(samples=samples, sample_rate=sr)


def load_transcript(name: str) -> Transcript:
    return Transcript.model_validate_json((FIXTURES / f"{name}.stt.json").read_text("utf-8"))


def load_language(name: str) -> LanguageAnalysis:
    path = FIXTURES / f"{name}.language.json"
    return LanguageAnalysis.model_validate_json(path.read_text("utf-8"))


class FakeTranscriber:
    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.calls = 0

    def transcribe(self, clip: AudioClip) -> Transcript:
        self.calls += 1
        return self.transcript


class FakeLanguage:
    def __init__(self, analysis: LanguageAnalysis) -> None:
        self.analysis = analysis
        self.texts: list[str] = []

    def analyze(self, text: str) -> LanguageAnalysis:
        self.texts.append(text)
        return self.analysis
