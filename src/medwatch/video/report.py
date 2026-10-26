"""Relatório automático da análise de vídeo: JSON, HTML (autocontido) e série de pose."""

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from medwatch.core.events import ClinicalEvent  # noqa: E402
from medwatch.video.pipeline import VideoAnalysis  # noqa: E402
from medwatch.video.render import METRIC_LABELS  # noqa: E402
from medwatch.video.rules import VideoConfig  # noqa: E402

SEVERITY_HEX = {
    "info": "#9e9e9e",
    "low": "#757575",
    "medium": "#f9a825",
    "high": "#ef6c00",
    "critical": "#c62828",
}


def summary(analysis: VideoAnalysis) -> dict[str, object]:
    return {
        "video": analysis.video.name,
        "backend": analysis.backend,
        "exercise": analysis.exercise,
        "scene": analysis.scene,
        "duration_s": round(analysis.duration_s, 2),
        "frames_analyzed": len(analysis.timeline),
        "processing_fps": round(analysis.processing_fps, 1),
        "repetitions": len(analysis.repetitions),
        "deviation_ratio": round(analysis.deviation_ratio, 3),
        "person_detected_ratio": round(float(analysis.timeline["bbox_x1"].notna().mean()), 3),
    }


def write_json(analysis: VideoAnalysis, events: list[ClinicalEvent], path: Path) -> Path:
    payload = {
        "summary": summary(analysis),
        "repetitions": [
            {
                "number": i,
                "start_s": round(r.start, 2),
                "end_s": round(r.end, 2),
                "extreme_value": round(r.extreme_value, 1),
            }
            for i, r in enumerate(analysis.repetitions, start=1)
        ],
        "events": [e.model_dump(mode="json") for e in events],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_pose_series(analysis: VideoAnalysis, path: Path) -> Path:
    """Série temporal de pose e métricas (entrada da análise de movimentação, PR 6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    analysis.timeline.to_parquet(path, index=False)
    return path


def plot_timeline(analysis: VideoAnalysis, config: VideoConfig, path: Path) -> Path:
    timeline = analysis.timeline
    rules = config.exercises[analysis.exercise].rules if analysis.exercise else []
    metrics = list(dict.fromkeys(r.metric for r in rules))
    panels = metrics or ["hip_y"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.4 * len(panels) + 0.6), sharex=True)
    axes = axes if len(panels) > 1 else [axes]
    for ax, metric in zip(axes, panels, strict=True):
        ax.plot(timeline["time"], timeline[metric], color="#1565c0", lw=1.4)
        for rule in rules:
            if rule.metric != metric:
                continue
            for limit in (rule.min, rule.max, rule.target_min, rule.target_max):
                if limit is not None:
                    ax.axhline(limit, color="#c62828", ls=":", lw=1)
        for finding in analysis.findings:
            if finding.evidence.get("metric") == metric or (
                metric == "hip_y" and finding.type == "fall_detected"
            ):
                ax.axvspan(
                    finding.start,
                    finding.end + 0.05,
                    alpha=0.25,
                    color=SEVERITY_HEX[finding.severity.value],
                )
        for rep in analysis.repetitions:
            ax.axvline(rep.end, color="#9e9e9e", lw=0.8, ls="--")
        label = METRIC_LABELS.get(metric, "Altura do quadril (px, ↓ = mais baixo)")
        ax.set_ylabel(label, fontsize=9)
        if metric == "hip_y":
            ax.invert_yaxis()
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("tempo (s)")
    fig.suptitle(
        f"{analysis.video.name} · faixas = desvios · tracejado = fim de repetição · "
        "pontilhado = limite",
        fontsize=9,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _img(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.name)}">'


def write_html(
    analysis: VideoAnalysis,
    events: list[ClinicalEvent],
    timeline_png: Path,
    path: Path,
) -> Path:
    """Relatório autocontido (imagens embutidas): abre direto no navegador."""
    s = summary(analysis)
    rows = []
    recommendations: dict[str, str] = {}
    for event, finding in zip(events, analysis.findings, strict=True):
        color = SEVERITY_HEX[event.severity.value]
        evidence = ", ".join(
            f"{k}: {v}"
            for k, v in event.evidence.items()
            if k not in {"message", "video", "backend", "exercise"} and v is not None
        )
        rows.append(
            f"<tr><td>{finding.start:.1f}–{finding.end:.1f} s</td>"
            f'<td><span class="sev" style="background:{color}">{event.severity.value}</span></td>'
            f"<td>{html.escape(finding.message)}</td><td>{html.escape(evidence)}</td>"
            f"<td>{_img(event.artifacts[0] if event.artifacts else None)}</td></tr>"
        )
        if finding.recommendation:
            recommendations[finding.type] = finding.recommendation
    reps = "".join(
        f"<tr><td>{i}</td><td>{r.start:.1f}–{r.end:.1f} s</td><td>{r.extreme_value:.0f}°</td></tr>"
        for i, r in enumerate(analysis.repetitions, start=1)
    )
    findings_html = (
        "<table><tr><th>Intervalo</th><th>Severidade</th><th>Achado</th><th>Evidência</th>"
        f"<th>Quadro</th></tr>{''.join(rows)}</table>"
        if rows
        else "<p>Nenhum desvio ou evento detectado.</p>"
    )
    recs = "".join(f"<li>{html.escape(r)}</li>" for r in recommendations.values())
    video = html.escape(analysis.video.name)
    backend = html.escape(analysis.backend)
    exercise = html.escape(analysis.exercise or "—")
    scene = html.escape(analysis.scene or "—")
    document = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório de vídeo · {video}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #212121; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 td, th {{ border-bottom: 1px solid #e0e0e0; padding: .5rem; text-align: left; }}
 td {{ vertical-align: top; }}
 td img {{ max-width: 320px; border-radius: 4px; }} img {{ max-width: 100%; }}
 .sev {{ color: white; padding: .15rem .5rem; border-radius: 10px; font-size: .8rem; }}
 .kpis {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
 .kpi {{ background: #f5f5f5; border-radius: 8px; padding: .8rem 1rem; min-width: 140px; }}
 .kpi b {{ display: block; font-size: 1.3rem; }}
</style></head><body>
<h1>Relatório automático de vídeo clínico</h1>
<p>{video} · modelo de pose: <b>{backend}</b> · exercício: <b>{exercise}</b> · cena: {scene}</p>
<div class="kpis">
 <div class="kpi"><b>{s["duration_s"]} s</b>duração analisada</div>
 <div class="kpi"><b>{s["repetitions"]}</b>repetições</div>
 <div class="kpi"><b>{analysis.deviation_ratio:.0%}</b>do tempo fora do padrão</div>
 <div class="kpi"><b>{len(events)}</b>achados</div>
 <div class="kpi"><b>{s["processing_fps"]}</b>quadros/s processados</div>
</div>
<h2>Achados</h2>{findings_html}
<h2>Recomendações</h2><ul>{recs or "<li>Nenhuma ação necessária.</li>"}</ul>
<h2>Métricas ao longo do tempo</h2>{_img(timeline_png)}
<h2>Repetições</h2>
<table><tr><th>#</th><th>Intervalo</th><th>Amplitude (extremo)</th></tr>{reps}</table>
<p style="color:#757575;font-size:.8rem">Gerado pelo MedWatch. Apoio à decisão: não substitui a
avaliação do profissional de saúde.</p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
