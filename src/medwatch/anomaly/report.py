"""Relatórios da análise de sinais vitais: JSON com os eventos e gráfico PNG."""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: gera o arquivo mesmo em servidor/CI

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from medwatch.anomaly.evaluation import InjectedAnomaly  # noqa: E402
from medwatch.anomaly.vitals import VitalsAnalysis  # noqa: E402

PANELS = [
    ("hr", "FC (bpm)"),
    ("sbp", "PAS (mmHg)"),
    ("rr", "FR (irpm)"),
    ("spo2", "SpO2 (%)"),
    ("temp", "Temp (°C)"),
]
SEVERITY_COLORS = {"low": "#9e9e9e", "medium": "#f9a825", "high": "#ef6c00", "critical": "#c62828"}


def summarize(analysis: VitalsAnalysis) -> dict[str, object]:
    events = analysis.events
    return {
        "patient_id": analysis.stay.patient_id,
        "stay_id": analysis.stay.stay_id,
        "hours": len(analysis.frame),
        "artifacts_removed": analysis.stay.artifacts,
        "events_by_type": dict(Counter(e.type for e in events).most_common()),
        "events_by_severity": dict(Counter(e.severity.value for e in events).most_common()),
        "events": [e.model_dump(mode="json") for e in events],
    }


def write_json(analyses: list[VitalsAnalysis], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stays": [summarize(a) for a in analyses]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def plot_stay(
    analysis: VitalsAnalysis, path: Path, injected: list[InjectedAnomaly] | None = None
) -> Path:
    frame = analysis.frame
    fig, axes = plt.subplots(len(PANELS) + 1, 1, figsize=(13, 13), sharex=True)

    for ax, (vital, label) in zip(axes, PANELS, strict=False):
        ax.plot(frame.index, frame[vital], color="#1565c0", lw=1, label="medido")
        if f"baseline_{vital}" in frame:
            ax.plot(
                frame.index,
                frame[f"baseline_{vital}"],
                color="#90a4ae",
                lw=1,
                ls="--",
                label="baseline (mediana móvel)",
            )
        if vital == "sbp":
            ax.plot(frame.index, frame["dbp"], color="#64b5f6", lw=0.8, label="PAD")
        for event in analysis.events:
            if event.type.startswith(f"{vital}_"):
                ts = pd.Timestamp(event.timestamp).tz_localize(None)
                ax.scatter(
                    ts,
                    frame.at[ts, vital],
                    s=60,
                    zorder=3,
                    marker="X",
                    color=SEVERITY_COLORS[event.severity.value],
                )
        for label_ in injected or []:
            if label_.vital == vital:
                ax.axvspan(label_.start, label_.end, color="#66bb6a", alpha=0.25)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    news = axes[-1]
    news.step(frame.index, frame["news2_total"], where="post", color="#6a1b9a", lw=1.2)
    news.axhline(5, color="#ef6c00", ls=":", lw=1)
    news.axhline(7, color="#c62828", ls=":", lw=1)
    news.set_ylabel("NEWS2")
    news.grid(alpha=0.3)

    for event in analysis.events:
        ts = pd.Timestamp(event.timestamp).tz_localize(None)
        if event.type == "multivariate_anomaly":
            for ax in axes:
                ax.axvline(ts, color="#ff8f00", alpha=0.5, lw=1.5)
        elif event.type == "news2_rise":
            news.scatter(
                ts,
                frame.at[ts, "news2_total"],
                s=50,
                zorder=3,
                color=SEVERITY_COLORS[event.severity.value],
            )

    axes[0].legend(loc="upper right", fontsize=8)
    title = f"{analysis.stay.patient_id} · internação {analysis.stay.stay_id}"
    subtitle = (
        "X = desvio do próprio histórico (z-score) · linha laranja = Isolation Forest · "
        "pontos no NEWS2 = subida de faixa · cor = severidade"
    )
    if injected:
        subtitle += " · faixa verde = anomalia injetada"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
