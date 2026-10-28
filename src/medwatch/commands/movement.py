"""medwatch movement: padrões de movimentação durante a internação."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from medwatch.anomaly.clinical_report import movement_payload, plot_movement, write_json
from medwatch.anomaly.movement import (
    MovementAnalyzer,
    MovementConfig,
    activity_from_pose,
    simulate_stay,
)
from medwatch.anomaly.movement_evaluation import evaluate_movement, inject_movement_anomalies
from medwatch.commands._alerts import send_and_report
from medwatch.config import Settings, get_settings
from medwatch.core.events import ClinicalEvent

app = typer.Typer(help="Movimentação: imobilidade, saída noturna do leito, queda e atividade.")

START_TIME_EXAMPLE = "2026-10-19T14:30:00-03:00"


def _config(settings: Settings) -> MovementConfig:
    return MovementConfig.load(settings.configs_dir / "movement.yaml")


def _read_series(path: Path) -> pd.DataFrame:
    """Série por minuto (parquet ou CSV) com índice de horário, `activity` e `in_bed`."""
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = {"activity", "in_bed"} - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {', '.join(sorted(missing))}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{path.name} precisa de índice de data e hora")
    return frame


def _describe(event: ClinicalEvent) -> str:
    ev = event.evidence
    if event.type == "prolonged_immobility":
        return f"{ev['minutes_still']} min sem se mover no leito"
    if event.type == "night_bed_exit":
        return f"{ev['minutes_out']} min fora do leito à noite"
    if event.type == "fall_suspected":
        return f"pico de movimento {ev['spike_activity']} e parado fora do leito"
    return f"atividade do dia {ev['ratio']}× a mediana dos dias anteriores"


@app.command()
def analyze(
    input_file: Annotated[
        Path | None,
        typer.Option(
            "--input", help="Série por minuto (.parquet/.csv). Sem ela, simula uma internação."
        ),
    ] = None,
    patient: str = typer.Option("SIM-001", help="Identificador pseudonimizado do paciente."),
    days: int = typer.Option(7, help="Dias da internação simulada."),
    seed: int = typer.Option(42, help="Semente da simulação."),
    alert: bool = typer.Option(False, help="Envia os eventos ao AlertManager (canais do .env)."),
) -> None:
    """Analisa a movimentação de um paciente e gera relatório JSON e gráfico.

    Sem `--input`, simula uma internação com uma anomalia de cada tipo (demonstração).
    """
    settings = get_settings()
    analyzer = MovementAnalyzer(_config(settings))
    if input_file is not None:
        try:
            series = _read_series(input_file)
        except (OSError, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
    else:
        rng = np.random.default_rng(seed)
        series, labels = inject_movement_anomalies(
            simulate_stay(days, rng), rng, 4, fall_intensity=0.95
        )
        typer.echo(
            "Simulação com anomalias injetadas: "
            + ", ".join(f"{label.kind} em {label.start:%d/%m %H:%M}" for label in labels)
        )

    analysis = analyzer.analyze(patient, series)
    typer.echo(f"\n{patient} · {len(series)} min · {len(analysis.events)} eventos")
    for event in analysis.events:
        typer.echo(
            f"  {event.timestamp:%Y-%m-%d %H:%M}  {event.severity.value:<8}  "
            f"{event.type:<22}  {_describe(event)}"
        )
    reports = settings.outputs_dir / "reports"
    png = plot_movement(analysis, reports / f"movement_{patient}.png")
    report = write_json(movement_payload(analysis), reports / f"movement_{patient}.json")
    typer.echo(f"\nRelatório: {report}\nGráfico: {png}")
    if alert:
        send_and_report(settings, analysis.events)


@app.command()
def evaluate(
    seed: int = typer.Option(42, help="Semente aleatória (reprodutibilidade)."),
) -> None:
    """Precision/recall/F1 com anomalias injetadas em internações sintéticas."""
    settings = get_settings()
    config = _config(settings)
    analyzer = MovementAnalyzer(config)
    cfg = config.evaluation
    metrics = evaluate_movement(analyzer, cfg.patients, cfg.days, cfg.per_stay, seed=seed)

    rng = np.random.default_rng(seed)
    normal = [
        e
        for i in range(cfg.patients)
        for e in analyzer.analyze(f"N{i}", simulate_stay(cfg.days, rng)).events
    ]
    patient_days = cfg.patients * cfg.days

    typer.echo(f"\n{'tipo':<22} {'precision':>9} {'recall':>7} {'F1':>6}  injetadas")
    for name, metric in metrics.items():
        typer.echo(
            f"{name:<22} {metric.precision:>9.3f} {metric.recall:>7.3f} {metric.f1:>6.3f}  "
            f"{metric.injected}"
        )
    by_type = pd.Series([e.type for e in normal], dtype="string").value_counts().to_dict()
    typer.echo(
        f"\nAlarmes falsos em internações normais: {len(normal)} em {patient_days} "
        f"paciente-dias {by_type or ''}"
    )
    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": seed,
        "patients": cfg.patients,
        "days": cfg.days,
        "per_stay": cfg.per_stay,
        "metrics": {name: metric.as_dict() for name, metric in metrics.items()},
        "false_alarms_normal": {"total": len(normal), "patient_days": patient_days, **by_type},
    }
    path = reports / "movement_evaluation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Métricas: {path}")


@app.command("from-pose")
def from_pose(
    pose_file: Annotated[Path, typer.Argument(help="Série de pose do vídeo (*_pose.parquet).")],
    start_time: Annotated[
        str, typer.Option("--start-time", help=f"Início do vídeo, ex.: {START_TIME_EXAMPLE}")
    ],
    output: Annotated[Path, typer.Option("--output", help="Arquivo .parquet de saída.")],
    freq: str = typer.Option("1s", help="Resolução da série (vídeos curtos: 1s; longos: 1min)."),
) -> None:
    """Converte a série de pose do vídeo na série de movimentação (`activity`, `in_bed`)."""
    try:
        start = datetime.fromisoformat(start_time)
    except ValueError as exc:
        typer.echo(f"--start-time inválido; use ISO 8601, ex.: {START_TIME_EXAMPLE}", err=True)
        raise typer.Exit(code=1) from exc
    if start.tzinfo is None:
        typer.echo(f"--start-time precisa de fuso horário, ex.: {START_TIME_EXAMPLE}", err=True)
        raise typer.Exit(code=1)
    timeline = pd.read_parquet(pose_file)
    series = activity_from_pose(timeline, pd.Timestamp(start.astimezone(UTC)), freq=freq)
    output.parent.mkdir(parents=True, exist_ok=True)
    series.to_parquet(output)
    typer.echo(f"{len(series)} pontos ({freq}) em {output}")
