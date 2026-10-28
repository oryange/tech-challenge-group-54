"""Avaliação das regras de prescrição com anomalias injetadas nas prescrições reais.

Mesmo método dos sinais vitais (`anomaly/evaluation.py`): os dados reais não têm rótulo,
então injetamos anomalias conhecidas e medimos
- recall: fração das injeções detectadas pelo método;
- precision: entre as detecções que surgiram com a injeção (ausentes na análise original),
  a fração que coincide com uma injeção.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from medwatch.anomaly.evaluation import MethodMetrics
from medwatch.anomaly.prescriptions import PrescriptionAnalyzer, PrescriptionStay, _utc
from medwatch.core.events import ClinicalEvent

KINDS = ("dose_spike", "duplicate_therapy", "early_stop")
TOLERANCE = pd.Timedelta("6h")
# par de drogas da mesma classe usado para injetar duplicidade (opioides, alto risco)
DUPLICATE_PAIR = ("morphine sulfate", "hydromorphone")
DUPLICATE_CLASS = "opioides"


@dataclass(frozen=True)
class InjectedPrescription:
    kind: str
    drug: str
    at: pd.Timestamp  # quando a anomalia passa a ser detectável


def _new_order(orders: pd.DataFrame, template: pd.Series, **changes: object) -> pd.DataFrame:
    """Prescrição nova a partir de uma existente, com os mesmos tipos de coluna da tabela."""
    row = template.to_dict() | changes
    return pd.DataFrame([row], columns=orders.columns).astype(orders.dtypes.to_dict())


def _inject_dose_spike(
    orders: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, InjectedPrescription] | None:
    long_enough = (orders["stop"] - orders["start"]) >= pd.Timedelta("12h")
    candidates = orders[~orders["titratable"] & orders["dose"].gt(0) & long_enough]
    if candidates.empty:
        return None
    base = candidates.iloc[int(rng.integers(len(candidates)))]
    start = base["start"] + pd.Timedelta("8h")
    spike = _new_order(orders, base, start=start, dose=base["dose"] * 4)
    return pd.concat([orders, spike], ignore_index=True), InjectedPrescription(
        "dose_spike", str(base["drug"]), start
    )


def _inject_duplicate(
    orders: pd.DataFrame, rng: np.random.Generator, analyzer: PrescriptionAnalyzer
) -> tuple[pd.DataFrame, InjectedPrescription] | None:
    first, last = orders["start"].min(), orders["start"].max()
    span = (last - first) - pd.Timedelta("36h")
    if span <= pd.Timedelta(0) or orders.empty:
        return None
    start = (first + span * float(rng.random())).floor("h")
    stop = start + pd.Timedelta("36h")
    same_class = orders[orders["drug_class"] == DUPLICATE_CLASS]
    already = ((same_class["start"] < stop) & (same_class["stop"].fillna(stop) > start)).any()
    if already:
        return None
    template = orders.iloc[0]
    config = analyzer.config
    injected = [
        _new_order(
            orders,
            template,
            drug=name.title(),
            drug_key=name,
            route_key="iv",
            start=start + pd.Timedelta(hours=2 * i),
            stop=stop,
            dose=2.0,
            unit="mass_mg",
            high_alert=True,
            titratable=False,
            drug_class=config.drug_class(name),
        )
        for i, name in enumerate(DUPLICATE_PAIR)
    ]
    return pd.concat([orders, *injected], ignore_index=True), InjectedPrescription(
        "duplicate_therapy", " + ".join(DUPLICATE_PAIR), start + pd.Timedelta("2h")
    )


def _inject_early_stop(
    orders: pd.DataFrame, rng: np.random.Generator, analyzer: PrescriptionAnalyzer
) -> tuple[pd.DataFrame, InjectedPrescription] | None:
    window = pd.Timedelta(analyzer.config.rules.early_stop.no_restart_within)
    last = orders["start"].max()
    long_enough = (orders["stop"] - orders["start"]) >= pd.Timedelta("24h")
    room = orders["start"] + pd.Timedelta("2h") + window <= last
    candidates = orders[~orders["titratable"] & long_enough & room]
    if candidates.empty:
        return None
    index = candidates.index[int(rng.integers(len(candidates)))]
    base = orders.loc[index]
    stop = base["start"] + pd.Timedelta("2h")
    modified = orders.copy()
    modified.loc[index, "stop"] = stop
    # a droga não volta a ser prescrita na janela: é o que caracteriza a suspensão
    restarts = (
        (modified["drug_key"] == base["drug_key"])
        & (modified["start"] > base["start"])
        & (modified["start"] <= stop + window)
    )
    modified = modified[~restarts].reset_index(drop=True)
    return modified, InjectedPrescription("early_stop", str(base["drug"]), stop + window)


def inject_prescription_anomalies(
    orders: pd.DataFrame,
    rng: np.random.Generator,
    count: int,
    analyzer: PrescriptionAnalyzer,
) -> tuple[pd.DataFrame, list[InjectedPrescription]]:
    """Aplica até `count` anomalias de tipos sorteados, separadas por pelo menos 24 h."""
    modified = orders.copy()
    labels: list[InjectedPrescription] = []
    for _ in range(count * 5):
        if len(labels) == count:
            break
        kind = str(rng.choice(KINDS))
        if kind == "dose_spike":
            result = _inject_dose_spike(modified, rng)
        elif kind == "duplicate_therapy":
            result = _inject_duplicate(modified, rng, analyzer)
        else:
            result = _inject_early_stop(modified, rng, analyzer)
        if result is None:
            continue
        candidate, label = result
        if any(abs(label.at - other.at) < pd.Timedelta("24h") for other in labels):
            continue
        modified, labels = candidate, [*labels, label]
    return modified.sort_values("start").reset_index(drop=True), labels


def _window(event: ClinicalEvent) -> tuple[datetime, datetime]:
    start, end = event.evidence.get("episode_start"), event.evidence.get("episode_end")
    if isinstance(start, str) and isinstance(end, str):
        return datetime.fromisoformat(start), datetime.fromisoformat(end)
    return event.timestamp, event.timestamp


def _overlaps(event: ClinicalEvent, at: datetime) -> bool:
    start, end = _window(event)
    return start - TOLERANCE <= at <= end + TOLERANCE


RULE_FOR_KIND = {
    "dose_spike": ("dose_increase", "dose_decrease"),
    "duplicate_therapy": ("duplicate_therapy",),
    "early_stop": ("early_stop",),
}
METHODS: dict[str, Callable[[ClinicalEvent, str], bool]] = {
    "rules": lambda e, kind: e.type in RULE_FOR_KIND[kind],
    "isolation_forest": lambda e, kind: e.type == "prescription_pattern_anomaly",
    "combined": lambda e, kind: (
        e.type in RULE_FOR_KIND[kind] or e.type == "prescription_pattern_anomaly"
    ),
}
METHOD_OF_EVENT: dict[str, Callable[[ClinicalEvent], bool]] = {
    "rules": lambda e: e.type != "prescription_pattern_anomaly",
    "isolation_forest": lambda e: e.type == "prescription_pattern_anomaly",
    "combined": lambda e: True,
}


def evaluate_prescriptions(
    analyzer: PrescriptionAnalyzer,
    stays: list[PrescriptionStay],
    per_stay: int = 2,
    seed: int = 42,
    min_days: int = 3,
) -> dict[str, MethodMetrics]:
    rng = np.random.default_rng(seed)
    metrics = {name: MethodMetrics() for name in METHODS}
    for stay in stays:
        span = stay.orders["start"].max() - stay.orders["start"].min()
        if span < pd.Timedelta(days=min_days):
            continue
        clean = analyzer.analyze(stay).events
        modified, labels = inject_prescription_anomalies(stay.orders, rng, per_stay, analyzer)
        if not labels:
            continue
        injected = analyzer.analyze(stay, orders=modified).events
        for name, metric in metrics.items():
            metric.stays += 1
            _score(name, metric, labels, clean, injected)
    return metrics


def _score(
    name: str,
    metric: MethodMetrics,
    labels: list[InjectedPrescription],
    clean: list[ClinicalEvent],
    injected: list[ClinicalEvent],
) -> None:
    belongs, explains = METHOD_OF_EVENT[name], METHODS[name]
    original = [e for e in clean if belongs(e)]
    new = [
        e
        for e in injected
        if belongs(e) and not any(c.type == e.type and _overlaps(c, e.timestamp) for c in original)
    ]
    for label in labels:
        at = _utc(label.at).to_pydatetime()
        hit = any(explains(e, label.kind) and _overlaps(e, at) for e in new)
        metric.injected += 1
        metric.detected += int(hit)
        counts = metric.by_kind.setdefault(label.kind, [0, 0])
        counts[0] += int(hit)
        counts[1] += 1
    metric.new_detections += len(new)
    metric.true_detections += sum(
        any(
            explains(e, label.kind) and _overlaps(e, _utc(label.at).to_pydatetime())
            for label in labels
        )
        for e in new
    )
