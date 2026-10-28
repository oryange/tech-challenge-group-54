"""Anomalias na evolução de prescrições (MIMIC-IV demo, hosp/prescriptions).

Duas técnicas complementares, como nos sinais vitais:
- regras explicáveis por prescrição: salto de dose, duplicidade terapêutica e suspensão
  precoce, com severidade maior para medicamentos potencialmente perigosos (ISMP);
- Isolation Forest por paciente-dia: o volume de mudanças no tratamento naquele dia é raro
  na população?
"""

import re
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel, Field, JsonValue

from medwatch.anomaly.detectors import Episode, IsolationForestDetector, find_episodes
from medwatch.core.events import ClinicalEvent, Severity, Source

PRESCRIPTIONS_FILE = Path("hosp") / "prescriptions.csv.gz"
DAY_FEATURES = ("active", "new", "stopped", "high_alert_active", "dose_changes")


class DoseChangeRule(BaseModel):
    window: str = "24h"
    increase_ratio: float = Field(default=2.0, gt=1)
    decrease_ratio: float = Field(default=0.5, gt=0, lt=1)
    severity: Severity = Severity.MEDIUM


class DuplicateTherapyRule(BaseModel):
    min_overlap: str = "24h"
    severity: Severity = Severity.MEDIUM
    classes: dict[str, list[str]] = Field(default_factory=dict)


class EarlyStopRule(BaseModel):
    max_duration: str = "4h"
    no_restart_within: str = "24h"
    severity: Severity = Severity.LOW


class PrescriptionRules(BaseModel):
    dose_change: DoseChangeRule = Field(default_factory=DoseChangeRule)
    duplicate_therapy: DuplicateTherapyRule = Field(default_factory=DuplicateTherapyRule)
    early_stop: EarlyStopRule = Field(default_factory=EarlyStopRule)


class PrescriptionForestConfig(BaseModel):
    contamination: float = Field(default=0.03, gt=0, lt=0.5)
    n_estimators: int = 200
    random_state: int = 42
    severity_medium: float = 0.58
    severity_high: float = 0.65


class PrescriptionEvaluationConfig(BaseModel):
    min_days: int = 3
    per_stay: int = 2


class PrescriptionsConfig(BaseModel):
    unit_factors: dict[str, dict[str, float]] = Field(default_factory=dict)
    high_alert: list[str] = Field(default_factory=list)
    titratable: list[str] = Field(default_factory=list)
    titratable_routes: list[str] = Field(default_factory=list)
    rules: PrescriptionRules = Field(default_factory=PrescriptionRules)
    isolation_forest: PrescriptionForestConfig = Field(default_factory=PrescriptionForestConfig)
    evaluation: PrescriptionEvaluationConfig = Field(default_factory=PrescriptionEvaluationConfig)

    @classmethod
    def load(cls, path: Path) -> "PrescriptionsConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def drug_class(self, name: str) -> str | None:
        for klass, members in self.rules.duplicate_therapy.classes.items():
            if _matches(name, members):
                return klass
        return None


@dataclass(frozen=True)
class PrescriptionStay:
    """Prescrições de uma internação (hadm_id), já normalizadas por `normalize_orders`."""

    patient_id: str
    subject_id: int
    hadm_id: int
    orders: pd.DataFrame


@dataclass
class PrescriptionAnalysis:
    stay: PrescriptionStay
    days: pd.DataFrame  # features e score do Isolation Forest por dia
    events: list[ClinicalEvent] = field(default_factory=list)


def _matches(name: str, terms: list[str]) -> bool:
    return any(term in name for term in terms)


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    # o MIMIC desloca as datas (ano ~2100–2200) e não informa fuso; tratamos como UTC
    return ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)


def parse_dose(value: object) -> float:
    """Dose prescrita em número: "1-2" vira 2 (o máximo da faixa) e "25,000" vira 25000."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = str(value).strip().replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return max(float(n) for n in numbers) if numbers else float("nan")


def normalize_unit(
    dose: float, unit: object, factors: dict[str, dict[str, float]]
) -> tuple[float, str]:
    """Converte para a unidade base do grupo (mg, mL). Outras unidades ficam como estão."""
    name = str(unit).strip().lower() if isinstance(unit, str) else ""
    for group, table in factors.items():
        if name in table:
            return dose * table[name], group
    return dose, name or "unknown"


def normalize_orders(raw: pd.DataFrame, config: PrescriptionsConfig) -> pd.DataFrame:
    """Colunas padronizadas, dose numérica na unidade base e marcações de risco/titulação."""
    orders = raw[raw["drug_type"].fillna("MAIN") == "MAIN"].copy()
    orders["start"] = pd.to_datetime(orders["starttime"])
    orders["stop"] = pd.to_datetime(orders["stoptime"])
    orders = orders.dropna(subset=["start"])
    orders["drug_key"] = orders["drug"].astype(str).str.lower().str.strip()
    orders["route_key"] = orders["route"].astype("string").str.lower().str.strip().fillna("")
    doses = orders["dose_val_rx"].map(parse_dose)
    converted = [
        normalize_unit(dose, unit, config.unit_factors)
        for dose, unit in zip(doses, orders["dose_unit_rx"], strict=True)
    ]
    orders["dose"] = [dose for dose, _ in converted]
    orders["unit"] = [unit for _, unit in converted]
    orders["high_alert"] = orders["drug_key"].map(lambda d: _matches(d, config.high_alert))
    orders["titratable"] = orders["drug_key"].map(
        lambda d: _matches(d, config.titratable)
    ) | orders["route_key"].isin(config.titratable_routes)
    orders["drug_class"] = orders["drug_key"].map(config.drug_class)
    columns = [
        "subject_id", "hadm_id", "drug", "drug_key", "route_key", "start", "stop",
        "dose", "unit", "high_alert", "titratable", "drug_class",
    ]  # fmt: skip
    return orders[columns].sort_values(["start", "drug_key"]).reset_index(drop=True)


def patient_id_for(subject_id: int) -> str:
    return f"MIMIC-{subject_id}"


def load_prescriptions(dataset_dir: Path, subject_ids: list[int] | None = None) -> pd.DataFrame:
    raw = pd.read_csv(
        dataset_dir / PRESCRIPTIONS_FILE,
        usecols=[
            "subject_id",
            "hadm_id",
            "starttime",
            "stoptime",
            "drug_type",
            "drug",
            "dose_val_rx",
            "dose_unit_rx",
            "route",
        ],  # fmt: skip
    )
    if subject_ids is not None:
        raw = raw[raw["subject_id"].isin(subject_ids)]
    return raw


def iter_prescription_stays(
    dataset_dir: Path, config: PrescriptionsConfig, subject_ids: list[int] | None = None
) -> list[PrescriptionStay]:
    orders = normalize_orders(load_prescriptions(dataset_dir, subject_ids), config)
    stays = [
        PrescriptionStay(
            patient_id=patient_id_for(int(subject_id)),
            subject_id=int(subject_id),
            hadm_id=int(hadm_id),
            orders=group.reset_index(drop=True),
        )
        for (subject_id, hadm_id), group in orders.groupby(["subject_id", "hadm_id"])
    ]
    logger.debug("{} internações com prescrições", len(stays))
    return stays


def _bump(severity: Severity, high_alert: bool) -> Severity:
    """Medicamento potencialmente perigoso sobe um nível (até HIGH, nunca CRITICAL sozinho)."""
    if not high_alert:
        return severity
    order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]
    return order[min(order.index(severity) + 1, len(order) - 1)] if severity in order else severity


def _num(value: object) -> float | None:
    return None if pd.isna(value) else round(float(value), 3)  # type: ignore[arg-type]


def _iso(ts: object) -> str | None:
    return None if pd.isna(ts) else _utc(pd.Timestamp(ts)).isoformat()  # type: ignore[arg-type]


class PrescriptionAnalyzer:
    def __init__(self, config: PrescriptionsConfig) -> None:
        self.config = config
        cfg = config.isolation_forest
        self.iforest = IsolationForestDetector(
            contamination=cfg.contamination,
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
        )
        self._fitted = False

    # --- regras ---------------------------------------------------------------------------

    def dose_changes(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Pares (anterior, nova) da mesma droga e via, com a razão entre as doses."""
        rule = self.config.rules.dose_change
        ordered = orders.sort_values("start")
        grouped = ordered.groupby(["drug_key", "route_key"], sort=False)
        pairs = ordered.assign(
            prev_dose=grouped["dose"].shift(),
            prev_unit=grouped["unit"].shift(),
            prev_start=grouped["start"].shift(),
        )
        comparable = (
            pairs["prev_dose"].gt(0)
            & pairs["dose"].gt(0)
            & (pairs["unit"] == pairs["prev_unit"])
            & ((pairs["start"] - pairs["prev_start"]) <= pd.Timedelta(rule.window))
        )
        pairs = pairs[comparable].copy()
        pairs["ratio"] = pairs["dose"] / pairs["prev_dose"]
        return pairs

    def _dose_change_events(
        self, stay: PrescriptionStay, orders: pd.DataFrame
    ) -> list[ClinicalEvent]:
        rule = self.config.rules.dose_change
        pairs = self.dose_changes(orders)
        pairs = pairs[~pairs["titratable"]]
        flagged = pairs[
            (pairs["ratio"] >= rule.increase_ratio) | (pairs["ratio"] <= rule.decrease_ratio)
        ]
        events = []
        for row in flagged.itertuples(index=False):
            increase = row.ratio > 1
            magnitude = row.ratio if increase else 1 / row.ratio
            events.append(
                ClinicalEvent(
                    patient_id=stay.patient_id,
                    source=Source.PRESCRIPTION,
                    type="dose_increase" if increase else "dose_decrease",
                    timestamp=_utc(row.start),
                    score=min(1.0, 0.5 + (magnitude - 1) / 10),
                    severity=_bump(rule.severity, bool(row.high_alert)),
                    evidence={
                        "hadm_id": stay.hadm_id,
                        "rule": "dose_change",
                        "drug": row.drug,
                        "route": row.route_key,
                        "previous_dose": _num(row.prev_dose),
                        "new_dose": _num(row.dose),
                        "unit": row.unit,
                        "ratio": round(float(row.ratio), 2),
                        "previous_start": _iso(row.prev_start),
                        "high_alert": bool(row.high_alert),
                    },
                )
            )
        return events

    def _duplicate_events(
        self, stay: PrescriptionStay, orders: pd.DataFrame
    ) -> list[ClinicalEvent]:
        rule = self.config.rules.duplicate_therapy
        min_overlap = pd.Timedelta(rule.min_overlap)
        events: list[ClinicalEvent] = []
        classed = orders.dropna(subset=["drug_class", "stop"])
        for klass, group in classed.groupby("drug_class"):
            rows = list(group.sort_values("start").itertuples(index=False))
            reported: set[tuple[str, str]] = set()
            for i, first in enumerate(rows):
                for second in rows[i + 1 :]:
                    if second.drug_key == first.drug_key:
                        continue
                    pair = tuple(sorted((first.drug_key, second.drug_key)))
                    overlap = min(first.stop, second.stop) - max(first.start, second.start)
                    if overlap < min_overlap or pair in reported:
                        continue
                    reported.add(pair)  # type: ignore[arg-type]
                    starts_at = max(first.start, second.start)
                    events.append(
                        ClinicalEvent(
                            patient_id=stay.patient_id,
                            source=Source.PRESCRIPTION,
                            type="duplicate_therapy",
                            timestamp=_utc(starts_at),
                            score=min(1.0, 0.5 + overlap / pd.Timedelta("96h") / 2),
                            severity=_bump(
                                rule.severity, bool(first.high_alert or second.high_alert)
                            ),
                            evidence={
                                "hadm_id": stay.hadm_id,
                                "rule": "duplicate_therapy",
                                "class": str(klass),
                                "drugs": [first.drug, second.drug],
                                "overlap_hours": round(overlap / pd.Timedelta("1h"), 1),
                                "high_alert": bool(first.high_alert or second.high_alert),
                            },
                        )
                    )
        return events

    def _early_stop_events(
        self, stay: PrescriptionStay, orders: pd.DataFrame
    ) -> list[ClinicalEvent]:
        rule = self.config.rules.early_stop
        max_duration = pd.Timedelta(rule.max_duration)
        restart_window = pd.Timedelta(rule.no_restart_within)
        last_known = orders["start"].max()
        events = []
        stoppable = orders.dropna(subset=["stop"])
        stoppable = stoppable[
            ~stoppable["titratable"]
            & (stoppable["stop"] > stoppable["start"])
            & ((stoppable["stop"] - stoppable["start"]) <= max_duration)
        ]
        for row in stoppable.itertuples(index=False):
            # só dá para afirmar que não houve nova prescrição se a internação continuou
            if last_known < row.stop + restart_window:
                continue
            same = orders[(orders["drug_key"] == row.drug_key) & (orders["start"] > row.start)]
            restarted = (same["start"] <= row.stop + restart_window).any()
            if restarted:
                continue
            duration_h = (row.stop - row.start) / pd.Timedelta("1h")
            events.append(
                ClinicalEvent(
                    patient_id=stay.patient_id,
                    source=Source.PRESCRIPTION,
                    type="early_stop",
                    # só se confirma depois da janela sem nova prescrição
                    timestamp=_utc(row.stop + restart_window),
                    score=min(
                        1.0, 0.5 + (1 - duration_h / (max_duration / pd.Timedelta("1h"))) / 2
                    ),
                    severity=_bump(rule.severity, bool(row.high_alert)),
                    evidence={
                        "hadm_id": stay.hadm_id,
                        "rule": "early_stop",
                        "drug": row.drug,
                        "route": row.route_key,
                        "duration_hours": round(float(duration_h), 1),
                        "started": _iso(row.start),
                        "stopped": _iso(row.stop),
                        "high_alert": bool(row.high_alert),
                    },
                )
            )
        return events

    # --- Isolation Forest por paciente-dia ---------------------------------------------------

    def day_features(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Uma linha por dia da internação; o rótulo do dia é o fim dele (00:00 seguinte)."""
        if orders.empty:
            return pd.DataFrame(columns=list(DAY_FEATURES))
        first = orders["start"].min().normalize()
        last = orders[["start", "stop"]].max().max().normalize()
        days = pd.date_range(first, last, freq="1D")
        changes = self.dose_changes(orders)
        changed = changes[
            (changes["ratio"] >= self.config.rules.dose_change.increase_ratio)
            | (changes["ratio"] <= self.config.rules.dose_change.decrease_ratio)
        ]
        stop = orders["stop"].fillna(orders["start"])
        rows = []
        for day in days:
            end = day + pd.Timedelta("1D")
            active = (orders["start"] < end) & (stop >= day)
            rows.append(
                {
                    "active": int(active.sum()),
                    "new": int(((orders["start"] >= day) & (orders["start"] < end)).sum()),
                    "stopped": int(((stop >= day) & (stop < end)).sum()),
                    "high_alert_active": int((active & orders["high_alert"]).sum()),
                    "dose_changes": int(
                        ((changed["start"] >= day) & (changed["start"] < end)).sum()
                    ),
                }
            )
        return pd.DataFrame(rows, index=days + pd.Timedelta("1D"))

    def fit(self, stays: list[PrescriptionStay]) -> "PrescriptionAnalyzer":
        population = pd.concat([self.day_features(s.orders) for s in stays])
        self.iforest.fit(population[list(DAY_FEATURES)].astype(float))
        self._fitted = True
        return self

    def iforest_severity(self, score: float) -> Severity:
        cfg = self.config.isolation_forest
        if score >= cfg.severity_high:
            return Severity.HIGH
        if score >= cfg.severity_medium:
            return Severity.MEDIUM
        return Severity.LOW

    def _iforest_events(self, stay: PrescriptionStay, days: pd.DataFrame) -> list[ClinicalEvent]:
        scores = days["iforest_score"]
        events = []
        for episode in find_episodes(self.iforest.flags(scores), scores, max_gap="1D"):
            score = float(scores[episode.peak])
            snapshot: dict[str, JsonValue] = {
                f: int(days.at[episode.peak, f]) for f in DAY_FEATURES
            }
            events.append(
                ClinicalEvent(
                    patient_id=stay.patient_id,
                    source=Source.PRESCRIPTION,
                    type="prescription_pattern_anomaly",
                    timestamp=_utc(episode.peak),
                    score=score,
                    severity=self.iforest_severity(score),
                    evidence={
                        "hadm_id": stay.hadm_id,
                        "detector": "isolation_forest",
                        "day": snapshot,
                        **_episode_evidence(episode),
                    },
                )
            )
        return events

    def analyze(
        self, stay: PrescriptionStay, orders: pd.DataFrame | None = None
    ) -> PrescriptionAnalysis:
        """Analisa uma internação. `orders` permite analisar uma versão modificada (avaliação)."""
        orders = stay.orders if orders is None else orders
        events = [
            *self._dose_change_events(stay, orders),
            *self._duplicate_events(stay, orders),
            *self._early_stop_events(stay, orders),
        ]
        days = self.day_features(orders)
        if self._fitted and not days.empty:
            days["iforest_score"] = self.iforest.score(days[list(DAY_FEATURES)].astype(float))
            events += self._iforest_events(stay, days)
        events.sort(key=lambda e: e.timestamp)
        return PrescriptionAnalysis(stay=stay, days=days, events=events)


def _episode_evidence(episode: Episode) -> dict[str, str]:
    # cada ponto é o fim de um dia: o episódio começa 24 h antes do primeiro ponto
    return {
        "episode_start": _utc(episode.start - pd.Timedelta("1D")).isoformat(),
        "episode_end": _utc(episode.end).isoformat(),
    }
