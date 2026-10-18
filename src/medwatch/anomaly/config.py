"""Configuração tipada dos detectores de sinais vitais (configs/thresholds.yaml)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

VITALS = ("hr", "sbp", "dbp", "rr", "spo2", "temp")
NEWS2_VITALS = ("rr", "spo2", "sbp", "hr", "temp")

Band = tuple[float | None, int]


class News2Config(BaseModel):
    rr: list[Band]
    spo2: list[Band]
    sbp: list[Band]
    hr: list[Band]
    temp: list[Band]
    oxygen_points: int = 2
    consciousness_points: int = 3

    def bands(self, vital: str) -> list[Band]:
        bands: list[Band] = getattr(self, vital)
        return bands


class News2SeverityConfig(BaseModel):
    low: int = 1
    red_single_parameter: int = 3
    high: int = 5
    critical: int = 7


class RobustZScoreConfig(BaseModel):
    window: str = "12h"
    min_periods: int = Field(default=4, ge=1)
    window_by_vital: dict[str, str] = Field(default_factory=dict)
    threshold: float = Field(default=3.5, gt=0)
    min_mad: dict[str, float] = Field(default_factory=dict)


class IsolationForestConfig(BaseModel):
    contamination: float = Field(default=0.02, gt=0, lt=0.5)
    n_estimators: int = 200
    random_state: int = 42
    severity_medium: float = 0.54
    severity_high: float = 0.58


class VitalsConfig(BaseModel):
    grid: str = "1h"
    max_ffill_hours: int = 4
    validity: dict[str, tuple[float, float]]
    news2: News2Config
    news2_severity: News2SeverityConfig = Field(default_factory=News2SeverityConfig)
    news2_rearm: str = "6h"
    robust_zscore: RobustZScoreConfig = Field(default_factory=RobustZScoreConfig)
    isolation_forest: IsolationForestConfig = Field(default_factory=IsolationForestConfig)
    episode_max_gap: str = "2h"

    @classmethod
    def load(cls, path: Path) -> "VitalsConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))
