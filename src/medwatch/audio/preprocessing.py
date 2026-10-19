"""Leitura e normalização do áudio: qualquer formato aceito → PCM 16 kHz mono.

A decodificação usa o libsndfile (soundfile) dentro do processo, sem chamar ffmpeg por
subprocess: o caminho do arquivo nunca vira argumento de linha de comando.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from medwatch.audio.config import PreprocessingConfig


class AudioValidationError(ValueError):
    """Arquivo de áudio recusado (formato, tamanho, duração ou conteúdo)."""


@dataclass(frozen=True)
class AudioClip:
    """Áudio mono em float32 no intervalo [-1, 1]."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def to_pcm16(self) -> bytes:
        """PCM 16 bits little-endian, o formato que o Azure Speech recebe por stream."""
        clipped = np.clip(self.samples, -1.0, 1.0)
        return (clipped * 32767).astype("<i2").tobytes()


def validate_file(path: Path, config: PreprocessingConfig) -> Path:
    """Confere extensão, tipo e tamanho antes de abrir o arquivo."""
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() not in config.allowed_extensions:
        allowed = ", ".join(config.allowed_extensions)
        suffix = resolved.suffix or "(sem extensão)"
        raise AudioValidationError(f"Formato {suffix} não aceito. Use: {allowed}")
    if not resolved.is_file():
        raise AudioValidationError("Arquivo de áudio não encontrado")
    size_mb = resolved.stat().st_size / 1_000_000
    if size_mb > config.max_file_mb:
        raise AudioValidationError(
            f"Arquivo com {size_mb:.0f} MB excede o limite de {config.max_file_mb:.0f} MB"
        )
    return resolved


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Média dos canais; `samples` no formato (quadros, canais)."""
    return samples.mean(axis=1) if samples.ndim == 2 else samples


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Reamostra para `target_sr` (librosa); devolve o próprio sinal se a taxa já bate."""
    if orig_sr == target_sr:
        return samples
    import librosa

    resampled: np.ndarray = librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
    return resampled


def load_audio(path: Path, config: PreprocessingConfig) -> AudioClip:
    """Lê o arquivo, converte para mono na taxa configurada e valida a duração."""
    resolved = validate_file(path, config)
    try:
        samples, sample_rate = sf.read(resolved, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioValidationError("Não foi possível decodificar o áudio") from exc

    mono = resample(to_mono(samples), int(sample_rate), config.sample_rate).astype(np.float32)
    clip = AudioClip(samples=mono, sample_rate=config.sample_rate)
    if not np.isfinite(clip.samples).all():
        raise AudioValidationError("Áudio com amostras inválidas (NaN/inf)")
    if clip.duration < config.min_seconds:
        raise AudioValidationError(
            f"Áudio com {clip.duration:.1f} s; o mínimo é {config.min_seconds:.0f} s"
        )
    if clip.duration > config.max_seconds:
        limit_min = config.max_seconds / 60
        raise AudioValidationError(
            f"Áudio com {clip.duration / 60:.0f} min; o máximo é {limit_min:.0f} min"
        )
    if float(np.max(np.abs(clip.samples))) < 1e-4:
        raise AudioValidationError("Áudio em silêncio")
    return clip
