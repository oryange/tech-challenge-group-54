"""Configuração tipada do módulo de áudio (configs/audio.yaml e configs/critical_terms.yaml)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from medwatch.core.events import Severity

FEATURES = (
    "speech_rate_wpm",
    "articulation_rate_wpm",
    "f0_std_semitones",
    "mean_pause_seconds",
    "pauses_per_minute",
    "mean_phrase_seconds",
    "jitter_local_pct",
    "shimmer_local_pct",
    "hnr_db",
)


class PreprocessingConfig(BaseModel):
    sample_rate: int = Field(default=16000, gt=0)
    min_seconds: float = Field(default=3.0, ge=0)
    max_seconds: float = Field(default=1800.0, gt=0)
    max_file_mb: float = Field(default=200.0, gt=0)
    allowed_extensions: list[str] = Field(default_factory=lambda: [".wav", ".flac", ".ogg"])

    @field_validator("allowed_extensions")
    @classmethod
    def _lowercase(cls, value: list[str]) -> list[str]:
        return [ext.lower() for ext in value]


class SpeechConfig(BaseModel):
    timeout_seconds: float = Field(default=900.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    backoff_seconds: float = Field(default=2.0, ge=0)


class LanguageConfig(BaseModel):
    sentiment_language: str = "pt-BR"
    health_language: str = "pt"
    key_phrase_language: str = "pt-BR"
    summary_language: str = "pt"
    summary_mode: Literal["extractive", "abstractive", "off"] = "extractive"
    summary_sentences: int = Field(default=3, ge=1, le=20)
    max_key_phrases: int = Field(default=15, ge=1)
    max_chars_per_document: int = Field(default=5000, gt=0, le=5120)
    sentiment_batch_size: int = Field(default=10, ge=1, le=10)
    key_phrase_batch_size: int = Field(default=10, ge=1, le=10)
    health_batch_size: int = Field(default=25, ge=1, le=25)
    retry_total: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)


class AcousticConfig(BaseModel):
    pitch_floor_hz: float = Field(default=75.0, gt=0)
    pitch_ceiling_hz: float = Field(default=500.0, gt=0)
    silence_top_db: float = Field(default=30.0, gt=0)
    min_pause_seconds: float = Field(default=0.25, ge=0)


class FeatureStats(BaseModel):
    mean: float
    std: float = Field(gt=0)


class IndicatorFeature(BaseModel):
    direction: Literal[-1, 1]
    weight: float = Field(default=1.0, gt=0)


class IndicatorConfig(BaseModel):
    label: str
    related_terms: str | None = None
    features: dict[str, IndicatorFeature]

    @field_validator("features")
    @classmethod
    def _known_features(cls, value: dict[str, IndicatorFeature]) -> dict[str, IndicatorFeature]:
        unknown = set(value) - set(FEATURES)
        if unknown:
            raise ValueError(f"features desconhecidas: {sorted(unknown)}")
        return value


class IndicatorSeverityConfig(BaseModel):
    low: float = 1.5
    medium: float = 2.0
    high: float = 3.0

    @model_validator(mode="after")
    def _ordered(self) -> "IndicatorSeverityConfig":
        if not self.low < self.medium < self.high:
            raise ValueError("limiares de severidade devem ser crescentes: low < medium < high")
        return self

    def classify(self, zscore: float) -> Severity | None:
        if zscore >= self.high:
            return Severity.HIGH
        if zscore >= self.medium:
            return Severity.MEDIUM
        if zscore >= self.low:
            return Severity.LOW
        return None


class AudioConfig(BaseModel):
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    acoustic: AcousticConfig = Field(default_factory=AcousticConfig)
    baseline: dict[str, FeatureStats]
    indicators: dict[str, IndicatorConfig]
    indicator_severity: IndicatorSeverityConfig = Field(default_factory=IndicatorSeverityConfig)

    @model_validator(mode="after")
    def _baseline_covers_indicators(self) -> "AudioConfig":
        used = {name for ind in self.indicators.values() for name in ind.features}
        missing = used - set(self.baseline)
        if missing:
            raise ValueError(f"baseline sem features usadas nos indicadores: {sorted(missing)}")
        return self

    @classmethod
    def load(cls, path: Path) -> "AudioConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


class NegationConfig(BaseModel):
    cues: list[str]
    window_tokens: int = Field(default=4, ge=1)
    scope_breakers: list[str] = Field(default_factory=list)


class TermCategory(BaseModel):
    label: str
    severity: Severity
    terms: list[str] = Field(min_length=1)


class CriticalTermsConfig(BaseModel):
    negation: NegationConfig
    categories: dict[str, TermCategory]

    @classmethod
    def load(cls, path: Path) -> "CriticalTermsConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))
