"""Padrões de movimentação durante a internação: imobilidade, saída noturna do leito, queda
e queda de atividade em relação aos dias anteriores.

A série de entrada tem um ponto por minuto com `activity` (intensidade de movimento) e
`in_bed` (1 no leito, 0 fora). Ela pode vir de sensor de leito/acelerômetro ou da série de
pose exportada pelo vídeo (`activity_from_pose`).
"""

from dataclasses import dataclass, field
from datetime import UTC, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from medwatch.core.events import ClinicalEvent, Severity, Source


class ImmobilityRule(BaseModel):
    minutes: int = Field(default=120, gt=0)
    high_minutes: int = Field(default=240, gt=0)
    severity: Severity = Severity.MEDIUM
    severity_high: Severity = Severity.HIGH


class NightExitRule(BaseModel):
    night_start: time = time(22, 0)
    night_end: time = time(6, 0)
    min_minutes: int = Field(default=3, gt=0)
    severity: Severity = Severity.MEDIUM


class FallRule(BaseModel):
    spike: float = 0.8
    still_minutes: int = 2
    within_minutes: int = 2
    severity: Severity = Severity.CRITICAL


class ActivityDropRule(BaseModel):
    min_history_days: int = 2
    ratio: float = Field(default=0.5, gt=0, lt=1)
    severity: Severity = Severity.LOW


class MovementRules(BaseModel):
    prolonged_immobility: ImmobilityRule = Field(default_factory=ImmobilityRule)
    night_bed_exit: NightExitRule = Field(default_factory=NightExitRule)
    fall: FallRule = Field(default_factory=FallRule)
    activity_drop: ActivityDropRule = Field(default_factory=ActivityDropRule)


class MovementEvaluationConfig(BaseModel):
    patients: int = 30
    days: int = 5
    per_stay: int = 4


class MovementConfig(BaseModel):
    moving_threshold: float = 0.05
    rules: MovementRules = Field(default_factory=MovementRules)
    evaluation: MovementEvaluationConfig = Field(default_factory=MovementEvaluationConfig)

    @classmethod
    def load(cls, path: Path) -> "MovementConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


@dataclass
class MovementAnalysis:
    patient_id: str
    series: pd.DataFrame
    events: list[ClinicalEvent] = field(default_factory=list)


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)


def _runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Trechos contínuos (início, fim inclusivo) em que `mask` é verdadeiro."""
    values = mask.fillna(False).to_numpy(dtype=bool)
    if not values.any():
        return []
    edges = np.diff(np.concatenate([[0], values.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    index = mask.index
    return [(index[s], index[e]) for s, e in zip(starts, ends, strict=True)]


def _in_night(ts: pd.DatetimeIndex, start: time, end: time) -> np.ndarray:
    clock = ts.time
    if start <= end:
        return np.array([start <= t < end for t in clock])
    return np.array([t >= start or t < end for t in clock])


class MovementAnalyzer:
    def __init__(self, config: MovementConfig) -> None:
        self.config = config

    def _event(
        self,
        patient_id: str,
        type_: str,
        at: pd.Timestamp,
        severity: Severity,
        score: float,
        evidence: dict[str, object],
    ) -> ClinicalEvent:
        return ClinicalEvent(
            patient_id=patient_id,
            source=Source.MOVEMENT,
            type=type_,
            timestamp=_utc(at),
            score=float(np.clip(score, 0.0, 1.0)),
            severity=severity,
            evidence=evidence,  # type: ignore[arg-type]
        )

    def immobility(self, patient_id: str, series: pd.DataFrame) -> list[ClinicalEvent]:
        rule = self.config.rules.prolonged_immobility
        still = (series["activity"] < self.config.moving_threshold) & (series["in_bed"] == 1)
        events = []
        for start, end in _runs(still):
            minutes = int((end - start) / pd.Timedelta("1min")) + 1
            # alerta ao cruzar cada limiar (tempo real): MEDIUM na hora de reposicionar e
            # HIGH se continuar parado; o AlertManager deixa a escalada passar o cooldown
            levels = [(rule.minutes, rule.severity), (rule.high_minutes, rule.severity_high)]
            for threshold, severity in levels:
                if minutes < threshold:
                    break
                events.append(
                    self._event(
                        patient_id,
                        "prolonged_immobility",
                        start + pd.Timedelta(minutes=threshold - 1),
                        severity,
                        0.5 + min(threshold, 2 * rule.high_minutes) / (4 * rule.high_minutes),
                        {
                            "rule": "prolonged_immobility",
                            "minutes_still": threshold,
                            "episode_minutes": minutes,
                            "threshold_minutes": threshold,
                            "episode_start": _utc(start).isoformat(),
                            "episode_end": _utc(end).isoformat(),
                        },
                    )
                )
        return events

    def night_exits(self, patient_id: str, series: pd.DataFrame) -> list[ClinicalEvent]:
        rule = self.config.rules.night_bed_exit
        night = _in_night(pd.DatetimeIndex(series.index), rule.night_start, rule.night_end)
        out = (series["in_bed"] == 0) & pd.Series(night, index=series.index)
        events = []
        for start, end in _runs(out):
            minutes = int((end - start) / pd.Timedelta("1min")) + 1
            if minutes < rule.min_minutes:
                continue
            events.append(
                self._event(
                    patient_id,
                    "night_bed_exit",
                    start + pd.Timedelta(minutes=rule.min_minutes - 1),
                    rule.severity,
                    0.5 + min(minutes, 30) / 60,
                    {
                        "rule": "night_bed_exit",
                        "minutes_out": minutes,
                        "episode_start": _utc(start).isoformat(),
                        "episode_end": _utc(end).isoformat(),
                    },
                )
            )
        return events

    def falls(self, patient_id: str, series: pd.DataFrame) -> list[ClinicalEvent]:
        rule = self.config.rules.fall
        activity, in_bed = series["activity"], series["in_bed"]
        still_out = (activity < self.config.moving_threshold) & (in_bed == 0)
        still_run = still_out.astype(int).groupby((~still_out).cumsum()).cumsum()
        events: list[ClinicalEvent] = []
        last_fall: pd.Timestamp | None = None
        for at in series.index[activity >= rule.spike]:
            if last_fall is not None and at - last_fall < pd.Timedelta("30min"):
                continue
            window = still_run.loc[
                at : at + pd.Timedelta(minutes=rule.within_minutes + rule.still_minutes)
            ]
            confirmed = window[window >= rule.still_minutes]
            if confirmed.empty:
                continue
            events.append(
                self._event(
                    patient_id,
                    "fall_suspected",
                    confirmed.index[0],
                    rule.severity,
                    0.6 + float(activity.loc[at]) / 4,
                    {
                        "rule": "fall",
                        "spike_activity": round(float(activity.loc[at]), 2),
                        "spike_at": _utc(at).isoformat(),
                        "minutes_still_out_of_bed": int(rule.still_minutes),
                        "episode_start": _utc(at).isoformat(),
                        "episode_end": _utc(confirmed.index[0]).isoformat(),
                    },
                )
            )
            last_fall = at
        return events

    def activity_drops(self, patient_id: str, series: pd.DataFrame) -> list[ClinicalEvent]:
        rule = self.config.rules.activity_drop
        daily = series["activity"].resample("1D").sum()
        complete = series["activity"].resample("1D").count() >= 23 * 60
        events = []
        for i, (day, total) in enumerate(daily.items()):
            history = daily.iloc[max(0, i - 3) : i][complete.iloc[max(0, i - 3) : i]]
            if not complete.loc[day] or len(history) < rule.min_history_days:
                continue
            baseline = float(history.median())
            if baseline <= 0 or total > rule.ratio * baseline:
                continue
            end = day + pd.Timedelta("1D") - pd.Timedelta("1min")
            events.append(
                self._event(
                    patient_id,
                    "activity_drop",
                    end,
                    rule.severity,
                    0.5 + (1 - total / baseline) / 2,
                    {
                        "rule": "activity_drop",
                        "day_activity": round(float(total), 1),
                        "baseline_activity": round(baseline, 1),
                        "ratio": round(float(total / baseline), 2),
                        "episode_start": _utc(day).isoformat(),
                        "episode_end": _utc(end).isoformat(),
                    },
                )
            )
        return events

    def analyze(self, patient_id: str, series: pd.DataFrame) -> MovementAnalysis:
        series = series.sort_index()
        events = [
            *self.immobility(patient_id, series),
            *self.night_exits(patient_id, series),
            *self.falls(patient_id, series),
            *self.activity_drops(patient_id, series),
        ]
        events.sort(key=lambda e: e.timestamp)
        return MovementAnalysis(patient_id=patient_id, series=series, events=events)


def activity_from_pose(
    timeline: pd.DataFrame,
    start: pd.Timestamp,
    freq: str = "1min",
    bed_polygon: list[tuple[float, float]] | None = None,
    frame_size: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Converte a série de pose do vídeo (`*_pose.parquet`) na série de movimentação.

    `activity` é o deslocamento médio dos keypoints entre quadros, em fração da altura da
    pessoa (escala ~0–1, robusta à distância da câmera). `in_bed` vem do quadril dentro de
    `bed_polygon` (coordenadas normalizadas 0–1 do quadro); sem polígono, fica NaN.
    """
    from medwatch.video.rules import point_in_polygon

    xs = timeline.filter(regex=r"^kp_.*_x$").to_numpy(dtype=float)
    ys = timeline.filter(regex=r"^kp_.*_y$").to_numpy(dtype=float)
    height = (timeline["bbox_y2"] - timeline["bbox_y1"]).median()
    step = np.sqrt(np.diff(xs, axis=0) ** 2 + np.diff(ys, axis=0) ** 2)
    with np.errstate(invalid="ignore"):
        motion = np.nanmean(step, axis=1) / height if height and height > 0 else step[:, 0] * np.nan
    seconds = timeline["time"].to_numpy(dtype=float)
    index = start + pd.to_timedelta(seconds[1:], unit="s")
    frame = pd.DataFrame({"activity": np.clip(motion * 10, 0, 1)}, index=index)
    if bed_polygon is not None and frame_size is not None:
        width, height_px = frame_size
        inside = [
            float(point_in_polygon(x / width, y / height_px, bed_polygon))
            if not (np.isnan(x) or np.isnan(y))
            else np.nan
            for x, y in zip(timeline["hip_x"].iloc[1:], timeline["hip_y"].iloc[1:], strict=True)
        ]
        frame["in_bed"] = inside
    else:
        frame["in_bed"] = np.nan
    return frame.resample(freq).agg({"activity": "max", "in_bed": "median"})


def simulate_stay(
    days: int, rng: np.random.Generator, start: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Internação sintética por minuto, sem anomalias (o "normal" usado na avaliação).

    - Noite (22h–6h): no leito, com movimentos breves de sono a cada 20–60 min.
    - Dia: reposicionamento a cada 30–100 min e 2–3 caminhadas de 10–25 min fora do leito.
    - Confusores realistas: 1–2 idas à poltrona por dia (transferência com movimento brusco e
      depois parado fora do leito por 20–60 min, parecido com uma queda) e falhas de 1 min no
      sensor de leito.
    """
    start = start or pd.Timestamp("2026-01-05 00:00")
    index = pd.date_range(start, periods=days * 24 * 60, freq="1min")
    activity = rng.uniform(0.0, 0.02, len(index))
    in_bed = np.ones(len(index))
    minute = 0
    while minute < len(index):
        hour = index[minute].hour
        night = hour >= 22 or hour < 6
        gap = int(rng.integers(20, 61)) if night else int(rng.integers(30, 101))
        burst = min(int(rng.integers(1, 4)), len(index) - minute)
        activity[minute : minute + burst] = rng.uniform(0.12, 0.35 if night else 0.5, burst)
        minute += burst + gap
    for day in range(days):
        for _ in range(int(rng.integers(2, 4))):
            begin = day * 1440 + int(rng.integers(8 * 60, 20 * 60))
            length = int(rng.integers(10, 26))
            activity[begin : begin + length] = rng.uniform(0.25, 0.6, length)
            in_bed[begin : begin + length] = 0
        for _ in range(int(rng.integers(1, 3))):
            begin = day * 1440 + int(rng.integers(9 * 60, 19 * 60))
            length = int(rng.integers(20, 61))
            activity[begin] = rng.uniform(0.3, 0.85)  # transferência para a poltrona
            activity[begin + 1 : begin + length] = rng.uniform(0.0, 0.04, length - 1)
            in_bed[begin : begin + length] = 0
    dropouts = rng.choice(len(index), size=days, replace=False)
    in_bed[dropouts] = 0
    return pd.DataFrame({"activity": activity, "in_bed": in_bed}, index=index)
