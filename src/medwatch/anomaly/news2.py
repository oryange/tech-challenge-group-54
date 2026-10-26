"""National Early Warning Score 2 (Royal College of Physicians, 2017)."""

import pandas as pd

from medwatch.anomaly.config import NEWS2_VITALS, VitalsConfig
from medwatch.anomaly.detectors import ThresholdBands
from medwatch.core.events import Severity

# casas decimais com que a tabela do RCP é escrita: a temperatura com 1 casa, o resto inteiro.
# Arredondar antes de pontuar evita que 38,04 °C (registrado como 38,0) caia na faixa de 1 ponto.
TABLE_DECIMALS: dict[str, int] = {"temp": 1}


def compute_news2(frame: pd.DataFrame, config: VitalsConfig) -> pd.DataFrame:
    """Subscores por parâmetro, total e severidade.

    `frame` precisa das colunas de NEWS2_VITALS (valores já carregados para a frente) e,
    opcionalmente, `on_oxygen` e `alert` (booleanos). Linhas sem nenhum vital ficam com NaN.
    """
    out = pd.DataFrame(index=frame.index)
    for vital in NEWS2_VITALS:
        series = frame[vital] if vital in frame else pd.Series(float("nan"), index=frame.index)
        series = series.round(TABLE_DECIMALS.get(vital, 0))
        out[f"news2_{vital}"] = ThresholdBands(config.news2.bands(vital)).score(series)

    if "on_oxygen" in frame:
        on_oxygen = frame["on_oxygen"].astype("boolean").fillna(False).astype(bool)
    else:
        on_oxygen = pd.Series(False, index=frame.index)
    out["news2_oxygen"] = on_oxygen.astype(int) * config.news2.oxygen_points

    if "alert" in frame:
        not_alert = ~frame["alert"].astype("boolean").fillna(True).astype(bool)
    else:
        not_alert = pd.Series(False, index=frame.index)
    out["news2_consciousness"] = not_alert.astype(int) * config.news2.consciousness_points

    vital_cols = [f"news2_{v}" for v in NEWS2_VITALS]
    has_vitals = out[vital_cols].notna().any(axis=1)
    subscores = out.fillna(0)
    out["news2_total"] = subscores.sum(axis=1).where(has_vitals)
    out["news2_max_single"] = subscores.max(axis=1).where(has_vitals)
    out["news2_severity"] = [
        news2_severity(total, single, config) if pd.notna(total) else None
        for total, single in zip(out["news2_total"], out["news2_max_single"], strict=True)
    ]
    return out


def news2_severity(total: float, max_single: float, config: VitalsConfig) -> Severity:
    thresholds = config.news2_severity
    if total >= thresholds.critical:
        return Severity.CRITICAL
    if total >= thresholds.high:
        return Severity.HIGH
    if max_single >= thresholds.red_single_parameter:
        return Severity.MEDIUM
    if total >= thresholds.low:
        return Severity.LOW
    return Severity.INFO


def subscore_severity(points: float | None) -> Severity:
    """Severidade de um desvio estatístico conforme o quanto o valor é anormal no NEWS2."""
    if points is None or pd.isna(points) or points == 0:
        return Severity.LOW
    if points >= 3:
        return Severity.HIGH
    return Severity.MEDIUM
