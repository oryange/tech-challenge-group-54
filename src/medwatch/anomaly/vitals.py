"""Detecção de anomalias em sinais vitais: NEWS2 + z-score robusto + Isolation Forest.

As três técnicas são complementares:
- NEWS2: o valor é clinicamente perigoso? (limiares absolutos, iguais para todos)
- z-score robusto: o valor mudou muito em relação ao histórico do próprio paciente?
- Isolation Forest: a combinação de sinais é rara na população?
"""

from dataclasses import dataclass, field
from datetime import UTC

import pandas as pd
from pydantic import JsonValue

from medwatch.anomaly.config import NEWS2_VITALS, VITALS, VitalsConfig
from medwatch.anomaly.detectors import Episode, IsolationForestDetector, RobustZScore, find_episodes
from medwatch.anomaly.mimic import StayVitals
from medwatch.anomaly.news2 import compute_news2, subscore_severity
from medwatch.core.events import ClinicalEvent, Severity, Source

# direções sem relevância clínica de alerta (ex.: SpO2 subindo é melhora)
BENIGN_DIRECTIONS = {("spo2", "rise")}
FEATURE_DELTAS = ("hr", "sbp", "rr", "spo2")


@dataclass
class VitalsAnalysis:
    stay: StayVitals
    frame: pd.DataFrame
    events: list[ClinicalEvent] = field(default_factory=list)


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    # o MIMIC desloca as datas (ano ~2180) e não informa fuso; tratamos como UTC
    return ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)


def _num(value: object) -> float | None:
    return None if pd.isna(value) else round(float(value), 2)  # type: ignore[arg-type]


class VitalsAnalyzer:
    def __init__(self, config: VitalsConfig) -> None:
        self.config = config
        cfg = config.isolation_forest
        self.iforest = IsolationForestDetector(
            contamination=cfg.contamination,
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
        )
        self._fitted = False

    def fill(self, raw: pd.DataFrame) -> pd.DataFrame:
        return raw.ffill(limit=self.config.max_ffill_hours)

    @staticmethod
    def features(filled: pd.DataFrame) -> pd.DataFrame:
        feats = filled[list(VITALS)].copy()
        for vital in FEATURE_DELTAS:
            feats[f"d_{vital}"] = filled[vital].diff()
        return feats

    def fit(self, stays: list[StayVitals]) -> "VitalsAnalyzer":
        """Treina o Isolation Forest na população de internações."""
        population = pd.concat([self.features(self.fill(s.raw)) for s in stays])
        self.iforest.fit(population)
        self._fitted = True
        return self

    def zscore_detector(self, vital: str) -> RobustZScore:
        cfg = self.config.robust_zscore
        return RobustZScore(
            window=cfg.window_by_vital.get(vital, cfg.window),
            min_periods=cfg.min_periods,
            threshold=cfg.threshold,
            min_mad=cfg.min_mad.get(vital, 1.0),
        )

    def analyze(self, stay: StayVitals, raw: pd.DataFrame | None = None) -> VitalsAnalysis:
        """Analisa uma internação. `raw` permite analisar uma versão modificada (avaliação)."""
        raw = stay.raw if raw is None else raw
        filled = self.fill(raw)
        frame = filled.join(compute_news2(filled, self.config))

        events: list[ClinicalEvent] = []
        events += self._news2_events(stay, frame)
        for vital in VITALS:
            detector = self.zscore_detector(vital)
            z = detector.score(raw[vital])
            frame[f"z_{vital}"] = z
            frame[f"baseline_{vital}"] = detector.baseline(raw[vital])
            events += self._zscore_events(stay, frame, vital, detector)
        if self._fitted:
            frame["iforest_score"] = self.iforest.score(self.features(filled))
            events += self._iforest_events(stay, frame)

        events.sort(key=lambda e: e.timestamp)
        return VitalsAnalysis(stay=stay, frame=frame, events=events)

    def _vitals_snapshot(self, frame: pd.DataFrame, ts: pd.Timestamp) -> dict[str, JsonValue]:
        return {v: _num(frame.at[ts, v]) for v in VITALS}

    def _news2_events(self, stay: StayVitals, frame: pd.DataFrame) -> list[ClinicalEvent]:
        """Um evento quando a faixa de risco do NEWS2 sobe acima da última faixa alertada.

        Histerese: a faixa alertada só "rearma" (baixa) depois de o paciente ficar abaixo
        dela por `news2_rearm`, para oscilações entre faixas vizinhas não gerarem alertas.
        Ao rearmar, a faixa alertada passa a ser a mais baixa vista nesse período, e não a do
        momento: se o paciente caiu para LOW e já voltou a HIGH, essa piora gera evento.
        """
        events = []
        rearm = pd.Timedelta(self.config.news2_rearm)
        alerted = Severity.INFO
        below_since: pd.Timestamp | None = None
        below_floor = Severity.CRITICAL
        # o pandas 3 guarda a coluna como string comum: reconverter garante a ordem clínica
        for ts, value in frame["news2_severity"].dropna().items():
            severity = Severity(value)
            if severity < alerted:
                below_since = below_since if below_since is not None else ts
                below_floor = min(below_floor, severity)
                if ts - below_since < rearm:
                    continue
                alerted, below_since, below_floor = below_floor, None, Severity.CRITICAL
            else:
                below_since, below_floor = None, Severity.CRITICAL
            if severity > alerted and severity >= Severity.LOW:
                total = float(frame.at[ts, "news2_total"])
                events.append(
                    ClinicalEvent(
                        patient_id=stay.patient_id,
                        source=Source.VITALS,
                        type="news2_rise",
                        timestamp=_utc(ts),
                        score=min(1.0, total / 10),
                        severity=severity,
                        evidence={
                            "stay_id": stay.stay_id,
                            "news2_total": total,
                            "previous_band": alerted.value,
                            "subscores": {
                                v: _num(frame.at[ts, f"news2_{v}"])
                                for v in (*NEWS2_VITALS, "oxygen", "consciousness")
                            },
                            "vitals": self._vitals_snapshot(frame, ts),
                        },
                    )
                )
                alerted = severity
        return events

    def _zscore_events(
        self, stay: StayVitals, frame: pd.DataFrame, vital: str, detector: RobustZScore
    ) -> list[ClinicalEvent]:
        z = frame[f"z_{vital}"]
        events = []
        for episode in find_episodes(detector.flags(z), z, self.config.episode_max_gap):
            peak_z = float(z[episode.peak])
            direction = "rise" if peak_z > 0 else "drop"
            if (vital, direction) in BENIGN_DIRECTIONS:
                continue
            points = frame.at[episode.peak, f"news2_{vital}"] if vital in NEWS2_VITALS else None
            events.append(
                ClinicalEvent(
                    patient_id=stay.patient_id,
                    source=Source.VITALS,
                    type=f"{vital}_{direction}",
                    timestamp=_utc(episode.peak),
                    score=min(1.0, abs(peak_z) / (2 * detector.threshold)),
                    severity=subscore_severity(points),
                    evidence={
                        "stay_id": stay.stay_id,
                        "detector": "robust_zscore",
                        "value": _num(stay_value(frame, vital, episode)),
                        "baseline_median": _num(frame.at[episode.peak, f"baseline_{vital}"]),
                        "zscore": round(peak_z, 2),
                        "news2_points": _num(points),
                        **_episode_evidence(episode),
                    },
                )
            )
        return events

    def iforest_severity(self, score: float) -> Severity:
        cfg = self.config.isolation_forest
        if score >= cfg.severity_high:
            return Severity.HIGH
        if score >= cfg.severity_medium:
            return Severity.MEDIUM
        return Severity.LOW

    def _iforest_events(self, stay: StayVitals, frame: pd.DataFrame) -> list[ClinicalEvent]:
        scores = frame["iforest_score"]
        events = []
        for episode in find_episodes(
            self.iforest.flags(scores), scores, self.config.episode_max_gap
        ):
            score = float(scores[episode.peak])
            news2_band = frame.at[episode.peak, "news2_severity"]
            severity = self.iforest_severity(score)
            if isinstance(news2_band, str):
                # o IF mede quão rara é a combinação, não quão grave ela é: o NEWS2 no pico
                # serve de piso, senão uma piora gradual com NEWS2 9–12 sairia LOW e seria
                # descartada pelo min_severity, justo o caso que só o IF enxerga
                severity = max(severity, Severity(news2_band))
            events.append(
                ClinicalEvent(
                    patient_id=stay.patient_id,
                    source=Source.VITALS,
                    type="multivariate_anomaly",
                    timestamp=_utc(episode.peak),
                    score=score,
                    severity=severity,
                    evidence={
                        "stay_id": stay.stay_id,
                        "detector": "isolation_forest",
                        "news2_total": _num(frame.at[episode.peak, "news2_total"]),
                        "iforest_severity": self.iforest_severity(score).value,
                        "vitals": self._vitals_snapshot(frame, episode.peak),
                        **_episode_evidence(episode),
                    },
                )
            )
        return events


def stay_value(frame: pd.DataFrame, vital: str, episode: Episode) -> object:
    return frame.at[episode.peak, vital]


def _episode_evidence(episode: Episode) -> dict[str, str]:
    return {
        "episode_start": _utc(episode.start).isoformat(),
        "episode_end": _utc(episode.end).isoformat(),
    }
