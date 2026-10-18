"""Regras clínicas sobre as séries de métricas: desvios, repetições, quedas e áreas críticas."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from medwatch.core.events import Severity


class Rule(BaseModel):
    name: str
    metric: str
    kind: Literal["frame", "repetition"] = "frame"
    min: float | None = None
    max: float | None = None
    target_min: float | None = None
    target_max: float | None = None
    severity: Severity = Severity.LOW
    message: str
    recommendation: str = ""


class RepetitionConfig(BaseModel):
    metric: str
    enter_below: float | None = None
    exit_above: float | None = None
    enter_above: float | None = None
    exit_below: float | None = None

    @model_validator(mode="after")
    def _one_direction(self) -> "RepetitionConfig":
        down = self.enter_below is not None and self.exit_above is not None
        up = self.enter_above is not None and self.exit_below is not None
        if down == up:
            raise ValueError("use enter_below/exit_above (descida) ou enter_above/exit_below")
        return self


class Exercise(BaseModel):
    description: str
    repetition: RepetitionConfig | None = None
    rules: list[Rule] = Field(default_factory=list)


class FallConfig(BaseModel):
    drop_fraction: float = 0.25
    descent_seconds: float = 1.0
    lying_aspect_ratio: float = 1.6
    posture_frames: int = 3
    window_seconds: float = 3.0


class Roi(BaseModel):
    name: str
    polygon: list[tuple[float, float]]
    forbidden_classes: list[str]
    severity: Severity = Severity.HIGH
    message: str


class Scene(BaseModel):
    rois: list[Roi] = Field(default_factory=list)
    count_classes: list[str] = Field(default_factory=list)


class VideoConfig(BaseModel):
    sample_fps: float = 10
    smoothing_frames: int = 5
    min_keypoint_conf: float = 0.3
    min_deviation_seconds: float = 0.5
    exercises: dict[str, Exercise] = Field(default_factory=dict)
    fall: FallConfig = Field(default_factory=FallConfig)
    scenes: dict[str, Scene] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "VideoConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


@dataclass
class Finding:
    """Achado de uma regra, com intervalo em segundos desde o início do vídeo."""

    type: str
    severity: Severity
    message: str
    start: float
    end: float
    peak: float
    score: float
    evidence: dict[str, float | int | str | None] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class Repetition:
    start: float
    end: float
    extreme_time: float
    extreme_value: float


def _intervals(mask: pd.Series, times: pd.Series, min_seconds: float) -> list[tuple[int, int]]:
    """Trechos contínuos onde `mask` é verdadeiro e duram ao menos `min_seconds`."""
    intervals = []
    start = None
    values = mask.fillna(False).to_numpy(dtype=bool)
    for i, flagged in enumerate([*values, False]):
        if flagged and start is None:
            start = i
        elif not flagged and start is not None:
            end = i - 1
            if times.iloc[end] - times.iloc[start] >= min_seconds - 1e-9:
                intervals.append((start, end))
            start = None
    return intervals


def frame_findings(
    rule: Rule, metrics: pd.DataFrame, times: pd.Series, min_seconds: float
) -> list[Finding]:
    values = metrics[rule.metric]
    low = values < rule.min if rule.min is not None else pd.Series(False, index=values.index)
    high = values > rule.max if rule.max is not None else pd.Series(False, index=values.index)
    findings = []
    for start, end in _intervals(low | high, times, min_seconds):
        window = values.iloc[start : end + 1]
        if low.iloc[start : end + 1].any():
            peak_idx, limit = window.idxmin(), rule.min
        else:
            peak_idx, limit = window.idxmax(), rule.max
        peak_value = float(values.loc[peak_idx])
        assert limit is not None  # noqa: S101 - garantido pelo ramo acima
        excess = abs(peak_value - limit)
        findings.append(
            Finding(
                type=rule.name,
                severity=rule.severity,
                message=rule.message,
                start=float(times.iloc[start]),
                end=float(times.iloc[end]),
                peak=float(times.loc[peak_idx]),
                score=min(1.0, 0.5 + excess / 60),
                evidence={
                    "metric": rule.metric,
                    "value": round(peak_value, 1),
                    "limit": limit,
                    "duration_s": round(float(times.iloc[end] - times.iloc[start]), 2),
                },
                recommendation=rule.recommendation,
            )
        )
    return findings


def count_repetitions(
    config: RepetitionConfig, metrics: pd.DataFrame, times: pd.Series
) -> list[Repetition]:
    """Repetições por histerese: entra ao cruzar `enter` e conclui ao voltar além de `exit`."""
    values = metrics[config.metric]
    going_down = config.enter_below is not None
    reps: list[Repetition] = []
    start = None
    extreme_i = None
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        if start is None:
            entered = value < config.enter_below if going_down else value > config.enter_above  # type: ignore[operator]
            if entered:
                start, extreme_i = i, i
            continue
        assert extreme_i is not None  # noqa: S101
        if (going_down and value < values.iloc[extreme_i]) or (
            not going_down and value > values.iloc[extreme_i]
        ):
            extreme_i = i
        exited = value > config.exit_above if going_down else value < config.exit_below  # type: ignore[operator]
        if exited:
            reps.append(
                Repetition(
                    start=float(times.iloc[start]),
                    end=float(times.iloc[i]),
                    extreme_time=float(times.iloc[extreme_i]),
                    extreme_value=float(values.iloc[extreme_i]),
                )
            )
            start = extreme_i = None
    return reps


def repetition_findings(rule: Rule, reps: list[Repetition]) -> list[Finding]:
    findings = []
    for number, rep in enumerate(reps, start=1):
        too_shallow = rule.target_max is not None and rep.extreme_value > rule.target_max
        too_low = rule.target_min is not None and rep.extreme_value < rule.target_min
        if not (too_shallow or too_low):
            continue
        target = rule.target_max if too_shallow else rule.target_min
        findings.append(
            Finding(
                type=rule.name,
                severity=rule.severity,
                message=f"{rule.message} (repetição {number})",
                start=rep.start,
                end=rep.end,
                peak=rep.extreme_time,
                score=min(1.0, 0.5 + abs(rep.extreme_value - float(target or 0)) / 60),
                evidence={
                    "metric": rule.metric,
                    "repetition": number,
                    "value": round(rep.extreme_value, 1),
                    "target": target,
                },
                recommendation=rule.recommendation,
            )
        )
    return findings


def descent(times: pd.Series, y: pd.Series, seconds: float) -> pd.Series:
    """Quanto `y` desce (cresce, no eixo da imagem) nos próximos `seconds` a partir de cada t."""
    series = pd.Series(y.to_numpy(), index=pd.to_timedelta(times.to_numpy(), unit="s"))
    future_max = series[::-1].rolling(pd.Timedelta(seconds=seconds)).max()[::-1]
    return pd.Series((future_max - series).to_numpy(), index=times.index)


def detect_falls(
    config: FallConfig,
    times: pd.Series,
    hip_y: pd.Series,
    person_height: float,
    aspect_ratio: pd.Series,
) -> list[Finding]:
    """Queda = descida rápida do quadril seguida de postura horizontal sustentada.

    Sentar, agachar ou deitar devagar não dispara: ou a descida é lenta, ou a pessoa não
    termina na horizontal (largura/altura da caixa abaixo do limiar).
    """
    if person_height <= 0 or np.isnan(person_height) or len(times) < 3:
        return []
    hip_y = hip_y.interpolate(limit=5)
    drop = descent(times, hip_y, config.descent_seconds) / person_height
    # mediana só sobre quadros com pessoa detectada: no chão a pose falha em alguns quadros
    valid = aspect_ratio.dropna()
    posture = valid.rolling(config.posture_frames, min_periods=2).median().reindex(times.index)
    lying = posture > config.lying_aspect_ratio
    findings: list[Finding] = []
    last_end = -np.inf
    for i in np.flatnonzero(drop.to_numpy() >= config.drop_fraction):
        t_drop = float(times.iloc[i])
        if t_drop <= last_end:
            continue
        window = (times > t_drop) & (times <= t_drop + config.window_seconds)
        hits = np.flatnonzero((lying & window).to_numpy())
        if len(hits) == 0:
            continue
        start, end = int(hits[0]), int(hits[-1])
        max_drop = float(drop.iloc[i : start + 1].max())
        # a janela olha 1 s à frente: o início real é quando o quadril começa a descer
        moved = (hip_y.iloc[i : start + 1] - hip_y.iloc[i]) > 0.05 * person_height
        onset = float(times.iloc[i + int(np.argmax(moved.to_numpy()))]) if moved.any() else t_drop
        findings.append(
            Finding(
                type="fall_detected",
                severity=Severity.CRITICAL,
                message="Queda detectada: descida rápida seguida de postura horizontal",
                start=onset,
                end=float(times.iloc[end]),
                peak=float(times.iloc[start]),
                score=min(1.0, 0.6 + max_drop / 2),
                evidence={
                    "hip_drop_fraction_1s": round(max_drop, 2),
                    "posture_aspect_ratio": round(float(posture.iloc[start]), 2),
                    "lying_s": round(float(times.iloc[end] - times.iloc[start]), 2),
                },
                recommendation="Acionar a equipe imediatamente para avaliar o paciente.",
            )
        )
        last_end = float(times.iloc[end])
    return findings


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside
