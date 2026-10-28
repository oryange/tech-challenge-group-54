"""Relatórios de prescrições e movimentação: JSON com os eventos e gráfico PNG."""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: gera o arquivo mesmo em servidor/CI

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from medwatch.anomaly.movement import MovementAnalysis  # noqa: E402
from medwatch.anomaly.prescriptions import PrescriptionAnalysis  # noqa: E402
from medwatch.core.events import ClinicalEvent  # noqa: E402

SEVERITY_COLORS = {"low": "#9e9e9e", "medium": "#f9a825", "high": "#ef6c00", "critical": "#c62828"}


def _events_summary(events: list[ClinicalEvent]) -> dict[str, object]:
    return {
        "events_by_type": dict(Counter(e.type for e in events).most_common()),
        "events_by_severity": dict(Counter(e.severity.value for e in events).most_common()),
        "events": [e.model_dump(mode="json") for e in events],
    }


def write_json(payload: dict[str, object], path: Path) -> Path:
    """Grava o relatório em JSON (UTF-8, indentado)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def prescriptions_payload(analyses: list[PrescriptionAnalysis]) -> dict[str, object]:
    """Resumo por internação com os eventos de prescrição."""
    return {
        "stays": [
            {
                "patient_id": a.stay.patient_id,
                "hadm_id": a.stay.hadm_id,
                "orders": len(a.stay.orders),
                "days": len(a.days),
                **_events_summary(a.events),
            }
            for a in analyses
        ]
    }


def movement_payload(analysis: MovementAnalysis) -> dict[str, object]:
    """Resumo da série de movimentação com os eventos."""
    series = analysis.series
    return {
        "patient_id": analysis.patient_id,
        "minutes": len(series),
        "start": series.index.min().isoformat() if len(series) else None,
        "end": series.index.max().isoformat() if len(series) else None,
        **_events_summary(analysis.events),
    }


def _mark_events(ax: plt.Axes, events: list[ClinicalEvent]) -> None:
    for event in events:
        at = pd.Timestamp(event.timestamp).tz_localize(None)
        ax.axvline(
            at, color=SEVERITY_COLORS.get(event.severity.value, "#607d8b"), lw=1.2, alpha=0.8
        )


def plot_prescriptions(analysis: PrescriptionAnalysis, path: Path, top: int = 25) -> Path:
    """Linha do tempo das prescrições (uma linha por droga) com os eventos marcados."""
    orders = analysis.stay.orders
    drugs = orders["drug_key"].value_counts().index[:top]
    shown = orders[orders["drug_key"].isin(drugs)]
    fig, ax = plt.subplots(figsize=(13, 0.32 * len(drugs) + 2))
    positions = {drug: i for i, drug in enumerate(drugs)}
    for row in shown.itertuples(index=False):
        stop = row.stop if pd.notna(row.stop) else row.start + pd.Timedelta("1h")
        color = "#c62828" if row.high_alert else "#1565c0"
        ax.plot([row.start, stop], [positions[row.drug_key]] * 2, color=color, lw=4, alpha=0.6)
    _mark_events(ax, analysis.events)
    ax.set_yticks(range(len(drugs)), [d[:40] for d in drugs], fontsize=8)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.set_title(
        f"{analysis.stay.patient_id} · internação {analysis.stay.hadm_id} · "
        f"{len(analysis.events)} eventos\n"
        "vermelho = medicamento de alto risco (ISMP) · "
        "linhas verticais = eventos (cor = severidade)",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_movement(analysis: MovementAnalysis, path: Path) -> Path:
    """Atividade por minuto e permanência no leito, com os eventos marcados."""
    series = analysis.series
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 5), sharex=True, height_ratios=[3, 1])
    top.plot(series.index, series["activity"], color="#1565c0", lw=0.5)
    top.set_ylabel("atividade")
    bottom.fill_between(series.index, series["in_bed"].fillna(0), step="post", color="#90a4ae")
    bottom.set_ylabel("no leito")
    bottom.set_yticks([0, 1])
    for ax in (top, bottom):
        _mark_events(ax, analysis.events)
        ax.grid(alpha=0.3)
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    top.set_title(
        f"{analysis.patient_id} · {len(analysis.events)} eventos de movimentação "
        "(linhas verticais; cor = severidade)",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
