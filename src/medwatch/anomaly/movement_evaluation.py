"""Avaliação das regras de movimentação com anomalias injetadas em internações sintéticas.

Não há dataset público de movimentação contínua durante a internação, então a série
"normal" vem de `simulate_stay` e as anomalias são injetadas com rótulo, no mesmo método
dos sinais vitais e das prescrições (recall e precision de injeção).
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from medwatch.anomaly.evaluation import MethodMetrics
from medwatch.anomaly.movement import MovementAnalyzer, _utc, simulate_stay
from medwatch.core.events import ClinicalEvent

KINDS = ("prolonged_immobility", "night_bed_exit", "fall", "activity_drop")
EVENT_FOR_KIND = {
    "prolonged_immobility": "prolonged_immobility",
    "night_bed_exit": "night_bed_exit",
    "fall": "fall_suspected",
    "activity_drop": "activity_drop",
}
TOLERANCE = pd.Timedelta("10min")


@dataclass(frozen=True)
class InjectedMovement:
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp


def inject_movement_anomalies(
    series: pd.DataFrame,
    rng: np.random.Generator,
    count: int,
    fall_intensity: float | None = None,
) -> tuple[pd.DataFrame, list[InjectedMovement]]:
    """Até `count` anomalias em dias diferentes (os dois primeiros ficam como histórico).

    `fall_intensity` fixa o pico da queda (demonstração); sem ele, é sorteado em 0,7–1,0.
    """
    modified = series.copy()
    days = pd.DatetimeIndex(series.index.normalize().unique())
    available = list(days[2:])  # activity_drop precisa de 2 dias de histórico
    rng.shuffle(available)
    labels: list[InjectedMovement] = []
    for day in available[:count]:
        kind = KINDS[len(labels) % len(KINDS)]
        if kind == "prolonged_immobility":
            start = day + pd.Timedelta(minutes=int(rng.integers(9 * 60, 13 * 60)))
            end = start + pd.Timedelta(minutes=int(rng.integers(150, 300)))
            modified.loc[start:end, "activity"] = 0.01
            modified.loc[start:end, "in_bed"] = 1
        elif kind == "night_bed_exit":
            start = day + pd.Timedelta(minutes=int(rng.integers(60, 4 * 60)))
            end = start + pd.Timedelta(minutes=int(rng.integers(5, 16)))
            modified.loc[start:end, "activity"] = rng.uniform(0.2, 0.5)
            modified.loc[start:end, "in_bed"] = 0
        elif kind == "fall":
            start = day + pd.Timedelta(minutes=int(rng.integers(8 * 60, 20 * 60)))
            end = start + pd.Timedelta(minutes=6)
            # intensidade variável: quedas mais leves podem ficar abaixo do limiar
            peak = fall_intensity if fall_intensity is not None else float(rng.uniform(0.7, 1.0))
            modified.loc[start, ["activity", "in_bed"]] = [peak, 0]
            modified.loc[start + pd.Timedelta("1min") : end, ["activity", "in_bed"]] = [0.0, 0]
        else:
            start, end = day, day + pd.Timedelta("1D") - pd.Timedelta("1min")
            # dia quase todo no leito, com menos da metade do movimento habitual
            modified.loc[start:end, "activity"] *= 0.45
            modified.loc[start:end, "in_bed"] = 1
        labels.append(InjectedMovement(kind, start, end))
    return modified, labels


def _window(event: ClinicalEvent) -> tuple[datetime, datetime]:
    start, end = event.evidence.get("episode_start"), event.evidence.get("episode_end")
    if isinstance(start, str) and isinstance(end, str):
        return datetime.fromisoformat(start), datetime.fromisoformat(end)
    return event.timestamp, event.timestamp


def _overlaps(event: ClinicalEvent, start: datetime, end: datetime) -> bool:
    ev_start, ev_end = _window(event)
    return ev_start <= end + TOLERANCE and ev_end >= start - TOLERANCE


def evaluate_movement(
    analyzer: MovementAnalyzer, patients: int, days: int, per_stay: int, seed: int = 42
) -> dict[str, MethodMetrics]:
    rng = np.random.default_rng(seed)
    metrics = {kind: MethodMetrics() for kind in [*KINDS, "combined"]}
    for number in range(patients):
        patient_id = f"SIM-{number:03d}"
        series = simulate_stay(days, rng)
        clean = analyzer.analyze(patient_id, series).events
        modified, labels = inject_movement_anomalies(series, rng, per_stay)
        injected = analyzer.analyze(patient_id, modified).events
        new = [
            e
            for e in injected
            if not any(c.type == e.type and _overlaps(c, *_window(e)) for c in clean)
        ]
        windows = [(label, _utc(label.start), _utc(label.end)) for label in labels]
        for name, metric in metrics.items():
            metric.stays += 1
            mine = [e for e in new if name == "combined" or e.type == EVENT_FOR_KIND[name]]
            relevant = [w for w in windows if name == "combined" or w[0].kind == name]
            for label, start, end in relevant:
                hit = any(
                    e.type == EVENT_FOR_KIND[label.kind] and _overlaps(e, start, end) for e in mine
                )
                metric.injected += 1
                metric.detected += int(hit)
                counts = metric.by_kind.setdefault(label.kind, [0, 0])
                counts[0] += int(hit)
                counts[1] += 1
            metric.new_detections += len(mine)
            metric.true_detections += sum(
                any(
                    e.type == EVENT_FOR_KIND[label.kind] and _overlaps(e, start, end)
                    for label, start, end in windows
                )
                for e in mine
            )
    return metrics
