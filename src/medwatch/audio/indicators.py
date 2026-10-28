"""Indicadores de fadiga, disartria e dificuldade respiratória por z-score contra um baseline.

São indicadores de TRIAGEM: apontam uma consulta para revisão humana, não fazem diagnóstico.
O baseline é a referência populacional de configs/audio.yaml ou, de preferência, uma consulta
anterior do mesmo paciente (`baseline_from_features`).
"""

from collections.abc import Collection

from pydantic import BaseModel

from medwatch.audio.acoustic import AcousticFeatures
from medwatch.audio.config import AudioConfig, FeatureStats
from medwatch.core.events import Severity

# um indicador precisa de pelo menos esta fração das suas features para ser calculado
MIN_FEATURE_COVERAGE = 0.5

_ESCALATION = {Severity.LOW: Severity.MEDIUM, Severity.MEDIUM: Severity.HIGH}


class FeatureScore(BaseModel):
    value: float
    baseline_mean: float
    baseline_std: float
    zscore: float  # orientado: positivo = na direção do achado; negativo = melhor que o baseline


class IndicatorResult(BaseModel):
    """Indicador calculado; `zscore` é None sem features suficientes."""

    name: str
    label: str
    zscore: float | None
    severity: Severity | None
    features: dict[str, FeatureScore]
    missing_features: list[str]
    related_terms_present: bool = False


def baseline_from_features(
    features: AcousticFeatures, population: dict[str, FeatureStats]
) -> dict[str, FeatureStats]:
    """Baseline do próprio paciente: média = valor da consulta de referência.

    Com uma única consulta de referência não há variância própria, então o desvio-padrão
    continua sendo o populacional.
    """
    values = features.indicator_values()
    return {
        name: FeatureStats(mean=values.get(name, stats.mean), std=stats.std)
        for name, stats in population.items()
    }


def score_indicators(
    features: AcousticFeatures,
    config: AudioConfig,
    baseline: dict[str, FeatureStats] | None = None,
    term_categories: Collection[str] = (),
) -> list[IndicatorResult]:
    """Calcula cada indicador; `term_categories` são as categorias de termo não negadas."""
    reference = baseline or config.baseline
    values = features.indicator_values()
    results = []
    for name, indicator in config.indicators.items():
        scores: dict[str, FeatureScore] = {}
        weighted = 0.0
        total_weight = 0.0
        for feature, spec in indicator.features.items():
            if feature not in values:
                continue
            stats = reference[feature]
            zscore = spec.direction * (values[feature] - stats.mean) / stats.std
            scores[feature] = FeatureScore(
                value=round(values[feature], 3),
                baseline_mean=stats.mean,
                baseline_std=stats.std,
                zscore=round(zscore, 2),
            )
            # só o desvio na direção do achado conta: estar melhor que o baseline numa feature
            # (ex.: HNR alto) não pode compensar o desvio de outra (ex.: pausas frequentes)
            weighted += spec.weight * max(zscore, 0.0)
            total_weight += spec.weight

        missing = sorted(set(indicator.features) - set(scores))
        coverage = len(scores) / len(indicator.features)
        zscore_total: float | None = None
        severity: Severity | None = None
        if scores and coverage >= MIN_FEATURE_COVERAGE:
            zscore_total = weighted / total_weight
            severity = config.indicator_severity.classify(zscore_total)
        related = indicator.related_terms in term_categories if indicator.related_terms else False
        if severity is not None and related:
            severity = _ESCALATION.get(severity, severity)
        results.append(
            IndicatorResult(
                name=name,
                label=indicator.label,
                zscore=round(zscore_total, 2) if zscore_total is not None else None,
                severity=severity,
                features=scores,
                missing_features=missing,
                related_terms_present=related,
            )
        )
    return results
