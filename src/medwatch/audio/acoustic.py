"""Features acústicas locais: F0, jitter, shimmer e HNR (Praat via parselmouth), pausas e
taxas de fala. Nada aqui sai da máquina.

Pausas e frases vêm da segmentação por energia (librosa); taxa de fala e de articulação usam
o número de palavras da transcrição, quando existe.
"""

import math

import numpy as np
from pydantic import BaseModel

from medwatch.audio.config import FEATURES, AcousticConfig
from medwatch.audio.preprocessing import AudioClip

Interval = tuple[float, float]

VOICE_FEATURES = (
    "f0_mean_hz",
    "f0_std_hz",
    "f0_std_semitones",
    "jitter_local_pct",
    "shimmer_local_pct",
    "hnr_db",
)


class AcousticFeatures(BaseModel):
    """Features de uma consulta; `None` quando não puderam ser calculadas."""

    duration_seconds: float
    speech_seconds: float
    pause_count: int
    pause_ratio: float
    pauses_per_minute: float | None = None
    mean_pause_seconds: float | None = None
    mean_phrase_seconds: float | None = None
    f0_mean_hz: float | None = None
    f0_std_hz: float | None = None
    f0_std_semitones: float | None = None
    jitter_local_pct: float | None = None
    shimmer_local_pct: float | None = None
    hnr_db: float | None = None
    word_count: int | None = None
    speech_rate_wpm: float | None = None
    articulation_rate_wpm: float | None = None

    def indicator_values(self) -> dict[str, float]:
        """Só as features usadas nos indicadores e que puderam ser calculadas."""
        return {name: v for name in FEATURES if (v := getattr(self, name)) is not None}


def _finite(value: float) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def speech_intervals(clip: AudioClip, config: AcousticConfig) -> list[Interval]:
    """Trechos com fala (s), unindo silêncios mais curtos que `min_pause_seconds`."""
    import librosa

    hop = 256
    frames = librosa.effects.split(
        clip.samples, top_db=config.silence_top_db, frame_length=1024, hop_length=hop
    )
    merged: list[Interval] = []
    for start, end in frames / clip.sample_rate:
        if merged and start - merged[-1][1] < config.min_pause_seconds:
            merged[-1] = (merged[-1][0], float(end))
        else:
            merged.append((float(start), float(end)))
    return merged


def pause_features(intervals: list[Interval]) -> dict[str, float | int | None]:
    """Pausas internas (entre o primeiro e o último trecho de fala) e duração das frases."""
    speech = sum(end - start for start, end in intervals)
    pauses = [nxt[0] - cur[1] for cur, nxt in zip(intervals, intervals[1:], strict=False)]
    span = intervals[-1][1] - intervals[0][0] if intervals else 0.0
    return {
        "speech_seconds": speech,
        "pause_count": len(pauses),
        "pause_ratio": sum(pauses) / span if span else 0.0,
        "pauses_per_minute": len(pauses) / (span / 60) if span else None,
        "mean_pause_seconds": float(np.mean(pauses)) if pauses else None,
        "mean_phrase_seconds": speech / len(intervals) if intervals else None,
    }


def voice_features(clip: AudioClip, config: AcousticConfig) -> dict[str, float | None]:
    """F0 (média, desvio em Hz e em semitons), jitter e shimmer locais (%) e HNR (dB)."""
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(clip.samples.astype(np.float64), sampling_frequency=clip.sample_rate)
    floor, ceiling = config.pitch_floor_hz, config.pitch_ceiling_hz
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    if len(voiced) < 10:  # menos de 0,1 s de voz: F0 e perturbações não são confiáveis
        return dict.fromkeys(VOICE_FEATURES, None)
    semitones = 12 * np.log2(voiced / np.median(voiced))

    points = call(sound, "To PointProcess (periodic, cc)", floor, ceiling)
    jitter = call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([sound, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, floor, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return {
        "f0_mean_hz": _finite(float(np.mean(voiced))),
        "f0_std_hz": _finite(float(np.std(voiced))),
        "f0_std_semitones": _finite(float(np.std(semitones))),
        "jitter_local_pct": _finite(jitter * 100),
        "shimmer_local_pct": _finite(shimmer * 100),
        "hnr_db": _finite(hnr),
    }


def extract_features(
    clip: AudioClip, config: AcousticConfig, word_count: int | None = None
) -> AcousticFeatures:
    """Calcula todas as features; `word_count` (da transcrição) habilita as taxas de fala."""
    intervals = speech_intervals(clip, config)
    pauses = pause_features(intervals)
    speech_seconds = float(pauses["speech_seconds"] or 0.0)
    rates: dict[str, float | None] = {"speech_rate_wpm": None, "articulation_rate_wpm": None}
    if word_count:
        span = intervals[-1][1] - intervals[0][0] if intervals else clip.duration
        rates["speech_rate_wpm"] = word_count / (span / 60) if span else None
        rates["articulation_rate_wpm"] = (
            word_count / (speech_seconds / 60) if speech_seconds else None
        )
    return AcousticFeatures.model_validate(
        {
            "duration_seconds": clip.duration,
            **pauses,
            **voice_features(clip, config),
            "word_count": word_count,
            **rates,
        }
    )
