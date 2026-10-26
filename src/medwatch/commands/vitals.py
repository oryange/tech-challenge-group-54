"""medwatch vitals: detecção de anomalias em sinais vitais (MIMIC-IV demo)."""

import asyncio
import json

import numpy as np
import typer

from medwatch.anomaly.config import VitalsConfig
from medwatch.anomaly.evaluation import evaluate as run_evaluation
from medwatch.anomaly.evaluation import inject_anomalies
from medwatch.anomaly.mimic import StayVitals, iter_stays
from medwatch.anomaly.report import plot_stay, write_json
from medwatch.anomaly.vitals import VitalsAnalyzer
from medwatch.config import Settings, get_settings
from medwatch.core.alerts.factory import build_alert_manager
from medwatch.core.alerts.manager import DispatchStats
from medwatch.core.bus import InMemoryEventBus
from medwatch.core.events import ClinicalEvent
from medwatch.datasets import MIMIC_IV_DEMO

app = typer.Typer(help="Sinais vitais: anomalias com NEWS2, z-score e Isolation Forest.")


def _load(settings: Settings) -> tuple[VitalsConfig, list[StayVitals]]:
    dataset_dir = settings.data_dir / "raw" / MIMIC_IV_DEMO.dirname
    if not (dataset_dir / "icu" / "chartevents.csv.gz").exists():
        typer.echo("Dataset não encontrado. Rode antes: medwatch data download", err=True)
        raise typer.Exit(code=1)
    config = VitalsConfig.load(settings.configs_dir / "thresholds.yaml")
    return config, iter_stays(dataset_dir, config)


def _describe(event: ClinicalEvent) -> str:
    ev = event.evidence
    if ev.get("detector") == "robust_zscore":
        return f"valor {ev['value']} · baseline {ev['baseline_median']} · z={ev['zscore']}"
    if event.type == "multivariate_anomaly":
        return f"score {event.score:.2f} · NEWS2 {ev['news2_total']}"
    return f"NEWS2 {ev['news2_total']} (faixa anterior: {ev['previous_band']})"


async def _send_alerts(settings: Settings, events: list[ClinicalEvent]) -> DispatchStats:
    bus = InMemoryEventBus()
    manager = build_alert_manager(settings, event_time=True)
    consumer = asyncio.create_task(
        manager.consume(bus.subscribe(), min_severity=settings.alert_min_severity)
    )
    for event in events:
        await bus.publish(event)
    await bus.close()
    return await consumer


@app.command("list")
def list_patients() -> None:
    """Lista os pacientes disponíveis com o total de horas monitoradas."""
    _, stays = _load(get_settings())
    by_patient: dict[int, list[StayVitals]] = {}
    for stay in stays:
        by_patient.setdefault(stay.subject_id, []).append(stay)
    typer.echo(f"{'subject_id':>10}  {'internações':>11}  {'horas':>6}")
    for subject_id, patient_stays in sorted(by_patient.items()):
        hours = sum(len(s.raw) for s in patient_stays)
        typer.echo(f"{subject_id:>10}  {len(patient_stays):>11}  {hours:>6}")


@app.command()
def analyze(
    patient: int = typer.Option(..., "--patient", help="subject_id do MIMIC (ver: vitals list)"),
    plot: bool = typer.Option(True, help="Gera o gráfico PNG de cada internação."),
    alert: bool = typer.Option(False, help="Envia os eventos ao AlertManager (canais do .env)."),
) -> None:
    """Analisa as internações de um paciente e gera relatório JSON e gráfico."""
    settings = get_settings()
    config, stays = _load(settings)
    selected = [s for s in stays if s.subject_id == patient]
    if not selected:
        typer.echo(f"Paciente {patient} não encontrado. Veja: medwatch vitals list", err=True)
        raise typer.Exit(code=1)

    analyzer = VitalsAnalyzer(config).fit(stays)
    analyses = [analyzer.analyze(stay) for stay in selected]
    reports = settings.outputs_dir / "reports"
    for analysis in analyses:
        stay = analysis.stay
        typer.echo(
            f"\n{stay.patient_id} · internação {stay.stay_id} · {len(analysis.frame)} h · "
            f"{stay.artifacts} artefatos descartados · {len(analysis.events)} eventos"
        )
        for event in analysis.events:
            typer.echo(
                f"  {event.timestamp:%Y-%m-%d %H:%M}  {event.severity.value:<8}  "
                f"{event.type:<22}  {_describe(event)}"
            )
        if plot:
            png = plot_stay(analysis, reports / f"vitals_{patient}_{stay.stay_id}.png")
            typer.echo(f"  gráfico: {png}")

    report = write_json(analyses, reports / f"vitals_{patient}.json")
    typer.echo(f"\nRelatório: {report}")

    if alert:
        events = sorted((e for a in analyses for e in a.events), key=lambda e: e.timestamp)
        stats = asyncio.run(_send_alerts(settings, events))
        typer.echo(
            f"Alertas enviados: {stats.sent} · suprimidos (cooldown): {stats.suppressed} · "
            f"ignorados (< {settings.alert_min_severity}): {stats.ignored}"
        )
        if stats.failed:
            typer.echo(f"Sem entrega em nenhum canal: {stats.failed}", err=True)


@app.command()
def evaluate(
    per_stay: int = typer.Option(3, help="Anomalias injetadas por internação."),
    seed: int = typer.Option(42, help="Semente aleatória (reprodutibilidade)."),
) -> None:
    """Mede precision/recall/F1 com anomalias sintéticas injetadas nos dados reais."""
    settings = get_settings()
    config, stays = _load(settings)
    analyzer = VitalsAnalyzer(config).fit(stays)
    metrics = run_evaluation(analyzer, stays, per_stay=per_stay, seed=seed)

    typer.echo(f"\n{'método':<18} {'precision':>9} {'recall':>7} {'F1':>6}  injetadas")
    for name, metric in metrics.items():
        typer.echo(
            f"{name:<18} {metric.precision:>9.3f} {metric.recall:>7.3f} {metric.f1:>6.3f}  "
            f"{metric.injected}"
        )
    combined = metrics["combined"]
    typer.echo(f"\nInternações avaliadas: {combined.stays} de {len(stays)} (≥ 24 h de dados)")
    typer.echo("\nRecall por tipo de anomalia (combinado):")
    for kind, (hit, total) in sorted(metrics["combined"].by_kind.items()):
        typer.echo(f"  {kind:<12} {hit / total:.3f}")

    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    result = {
        "per_stay": per_stay,
        "seed": seed,
        "stays": len(stays),
        "stays_evaluated": combined.stays,
        "metrics": {name: metric.as_dict() for name, metric in metrics.items()},
    }
    path = reports / "vitals_evaluation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # exemplo visual: a internação mais longa com anomalias injetadas
    example = max(stays, key=lambda s: len(s.raw))
    modified, labels = inject_anomalies(
        example.raw, np.random.default_rng(seed), 4, config.validity
    )
    png = plot_stay(
        analyzer.analyze(example, raw=modified), reports / "vitals_evaluation_example.png", labels
    )
    typer.echo(f"\nMétricas: {path}\nExemplo: {png}")
