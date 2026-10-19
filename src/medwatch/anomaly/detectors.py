"""Detectores genéricos de anomalia para séries temporais com DatetimeIndex."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from medwatch.anomaly.config import Band


@dataclass(frozen=True)
class ThresholdBands:
    """Converte um valor em pontos por faixas [(limite_superior_inclusivo, pontos), ...].

    A última faixa usa limite None ("acima de todos os anteriores"), como nas tabelas do NEWS2.
    """

    bands: list[Band]

    def score(self, series: pd.Series) -> pd.Series:
        points = pd.Series(np.nan, index=series.index)
        remaining = series.notna()
        for upper, value in self.bands:
            hit = remaining if upper is None else remaining & (series <= upper)
            points[hit] = value
            remaining &= ~hit
        return points


def _mad(window: np.ndarray) -> float:
    return float(np.median(np.abs(window - np.median(window))))


@dataclass(frozen=True)
class RobustZScore:
    """z-score robusto em relação ao histórico recente do próprio paciente.

    Causal: o baseline (mediana e MAD) usa só pontos anteriores, como seria em tempo real.
    """

    window: str = "12h"
    min_periods: int = 4
    threshold: float = 3.5
    min_mad: float = 1.0

    def score(self, series: pd.Series) -> pd.Series:
        history = series.dropna().shift(1)
        rolling = history.rolling(self.window, min_periods=self.min_periods)
        median = rolling.median()
        mad = rolling.apply(_mad, raw=True).clip(lower=self.min_mad)
        z = 0.6745 * (series.dropna() - median) / mad
        return z.reindex(series.index)

    def flags(self, z: pd.Series) -> pd.Series:
        return z.abs() >= self.threshold

    def baseline(self, series: pd.Series) -> pd.Series:
        history = series.dropna().shift(1)
        return (
            history.rolling(self.window, min_periods=self.min_periods)
            .median()
            .reindex(series.index)
        )


@dataclass
class IsolationForestDetector:
    """Isolation Forest multivariado. Score em [0, 1]; acima de 0.5 é anômalo."""

    contamination: float = 0.02
    n_estimators: int = 200
    random_state: int = 42
    _model: IsolationForest | None = field(default=None, init=False, repr=False)
    _columns: list[str] = field(default_factory=list, init=False, repr=False)

    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        complete = features.dropna()
        if complete.empty:
            raise ValueError("Sem linhas completas para treinar o Isolation Forest")
        self._columns = list(features.columns)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        ).fit(complete.to_numpy())
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Chame fit() antes de score()")
        scores = pd.Series(np.nan, index=features.index)
        complete = features[self._columns].dropna()
        if not complete.empty:
            # decision_function: > 0 normal, < 0 anômalo (≈ [-0.5, 0.5])
            decision = self._model.decision_function(complete.to_numpy())
            scores[complete.index] = np.clip(0.5 - decision, 0.0, 1.0)
        return scores

    @staticmethod
    def flags(scores: pd.Series) -> pd.Series:
        return scores > 0.5


@dataclass(frozen=True)
class Episode:
    start: pd.Timestamp
    end: pd.Timestamp
    peak: pd.Timestamp


def find_episodes(flags: pd.Series, magnitude: pd.Series, max_gap: str = "2h") -> list[Episode]:
    """Agrupa pontos sinalizados próximos em episódios; o pico é o ponto de maior |magnitude|."""
    flagged = flags[flags.fillna(False).astype(bool)].index
    if len(flagged) == 0:
        return []
    gap = pd.Timedelta(max_gap)
    episodes: list[Episode] = []
    group = [flagged[0]]
    for ts in flagged[1:]:
        if ts - group[-1] <= gap:
            group.append(ts)
            continue
        episodes.append(_episode(group, magnitude))
        group = [ts]
    episodes.append(_episode(group, magnitude))
    return episodes


def _episode(group: list[pd.Timestamp], magnitude: pd.Series) -> Episode:
    peak = magnitude.loc[group].abs().idxmax()
    return Episode(start=group[0], end=group[-1], peak=peak)
