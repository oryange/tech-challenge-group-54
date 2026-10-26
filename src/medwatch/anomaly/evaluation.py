"""Avaliação dos detectores com anomalias sintéticas injetadas em internações reais.

Os dados reais não têm rótulo de anomalia, então medimos assim:
1. injetamos desvios conhecidos (ex.: SpO2 −10 pontos por 3 h) em pontos aleatórios;
2. recall = fração das injeções detectadas por cada método;
3. precision = entre as detecções que *surgiram* com a injeção (não existiam na análise
   da série original), a fração que coincide com uma injeção. Detecções que já existiam na
   série original são anomalias reais sem rótulo e não entram na conta.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from medwatch.anomaly.mimic import StayVitals
from medwatch.anomaly.vitals import VitalsAnalyzer, _utc
from medwatch.core.events import ClinicalEvent

# tipo → (sinal, deslocamento aplicado aos valores medidos)
ANOMALY_KINDS: dict[str, tuple[str, float]] = {
    "spo2_drop": ("spo2", -10.0),
    "hr_spike": ("hr", 40.0),
    "sbp_drop": ("sbp", -40.0),
    "rr_rise": ("rr", 12.0),
    "temp_spike": ("temp", 2.0),
}

MIN_HISTORY_HOURS = 12
TOLERANCE = pd.Timedelta("1h")


@dataclass(frozen=True)
class InjectedAnomaly:
    kind: str
    vital: str
    start: pd.Timestamp
    end: pd.Timestamp
    delta: float


@dataclass
class MethodMetrics:
    detected: int = 0
    injected: int = 0
    true_detections: int = 0
    new_detections: int = 0
    stays: int = 0  # internações que de fato entraram na avaliação (≥ min_hours e com injeção)
    by_kind: dict[str, list[int]] = field(default_factory=dict)  # kind → [detectadas, total]

    @property
    def recall(self) -> float:
        return self.detected / self.injected if self.injected else 0.0

    @property
    def precision(self) -> float:
        return self.true_detections / self.new_detections if self.new_detections else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "injected": self.injected,
            "detected": self.detected,
            "new_detections": self.new_detections,
            "true_detections": self.true_detections,
            "stays": self.stays,
            "recall_by_kind": {
                kind: round(hit / total, 3) for kind, (hit, total) in sorted(self.by_kind.items())
            },
        }


def _event_window(event: ClinicalEvent) -> tuple[datetime, datetime]:
    start = event.evidence.get("episode_start")
    end = event.evidence.get("episode_end")
    if isinstance(start, str) and isinstance(end, str):
        return datetime.fromisoformat(start), datetime.fromisoformat(end)
    return event.timestamp, event.timestamp


def _overlaps(event: ClinicalEvent, start: datetime, end: datetime) -> bool:
    ev_start, ev_end = _event_window(event)
    return ev_start <= end + TOLERANCE and ev_end >= start - TOLERANCE


# Cada método decide quais eventos são "dele" e se um evento pode explicar a injeção.
METHODS: dict[str, Callable[[ClinicalEvent, str], bool]] = {
    "robust_zscore": lambda e, vital: (
        e.evidence.get("detector") == "robust_zscore" and e.type.startswith(f"{vital}_")
    ),
    "isolation_forest": lambda e, vital: e.type == "multivariate_anomaly",
    "news2": lambda e, vital: e.type == "news2_rise",
    "combined": lambda e, vital: (
        e.type == "news2_rise" or e.type == "multivariate_anomaly" or e.type.startswith(f"{vital}_")
    ),
}
METHOD_OF_EVENT: dict[str, Callable[[ClinicalEvent], bool]] = {
    "robust_zscore": lambda e: e.evidence.get("detector") == "robust_zscore",
    "isolation_forest": lambda e: e.type == "multivariate_anomaly",
    "news2": lambda e: e.type == "news2_rise",
    "combined": lambda e: True,
}


def inject_anomalies(
    raw: pd.DataFrame,
    rng: np.random.Generator,
    count: int,
    validity: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, list[InjectedAnomaly]]:
    """Aplica `count` anomalias sem sobreposição em posições com histórico suficiente."""
    modified = raw.copy()
    labels: list[InjectedAnomaly] = []
    kinds = list(ANOMALY_KINDS)
    candidates = list(range(MIN_HISTORY_HOURS, len(raw) - 4))
    rng.shuffle(candidates)
    for position in candidates:
        if len(labels) == count:
            break
        kind = str(rng.choice(kinds))
        vital, delta = ANOMALY_KINDS[kind]
        duration = int(rng.integers(2, 5))
        window = raw.index[position : position + duration]
        start, end = window[0], window[-1]
        busy = any(
            abs((start - label.start).total_seconds()) < MIN_HISTORY_HOURS * 3600
            for label in labels
        )
        measured = raw.loc[window, vital].notna()
        # temperatura é medida a cada ~4 h: uma medição na janela já basta
        if busy or measured.sum() < 1:
            continue
        low, high = validity[vital]
        rows = window[measured.to_numpy()]
        modified.loc[rows, vital] = (raw.loc[rows, vital] + delta).clip(low, high)
        labels.append(InjectedAnomaly(kind, vital, start, end, delta))
    return modified, labels


def evaluate(
    analyzer: VitalsAnalyzer,
    stays: list[StayVitals],
    per_stay: int = 2,
    seed: int = 42,
    min_hours: int = 24,
) -> dict[str, MethodMetrics]:
    rng = np.random.default_rng(seed)
    metrics = {name: MethodMetrics() for name in METHODS}
    for stay in stays:
        if len(stay.raw) < min_hours:
            continue
        clean_events = analyzer.analyze(stay).events
        modified, labels = inject_anomalies(stay.raw, rng, per_stay, analyzer.config.validity)
        if not labels:
            continue
        injected_events = analyzer.analyze(stay, raw=modified).events
        for name, metric in metrics.items():
            metric.stays += 1
            _score_method(name, metric, labels, clean_events, injected_events)
    return metrics


def _score_method(
    name: str,
    metric: MethodMetrics,
    labels: list[InjectedAnomaly],
    clean_events: list[ClinicalEvent],
    injected_events: list[ClinicalEvent],
) -> None:
    belongs = METHOD_OF_EVENT[name]
    explains = METHODS[name]
    clean = [e for e in clean_events if belongs(e)]
    new = [
        e
        for e in injected_events
        if belongs(e)
        and not any(c.type == e.type and _overlaps(c, *_event_window(e)) for c in clean)
    ]
    windows = [(label, _utc(label.start), _utc(label.end)) for label in labels]

    for label, start, end in windows:
        hit = any(explains(e, label.vital) and _overlaps(e, start, end) for e in new)
        metric.injected += 1
        metric.detected += int(hit)
        counts = metric.by_kind.setdefault(label.kind, [0, 0])
        counts[0] += int(hit)
        counts[1] += 1

    metric.new_detections += len(new)
    metric.true_detections += sum(
        any(explains(e, label.vital) and _overlaps(e, start, end) for label, start, end in windows)
        for e in new
    )
