"""medwatch video: análise de vídeos clínicos (pose, objetos, quedas, áreas críticas)."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from medwatch.config import Settings, get_settings
from medwatch.core.alerts.factory import build_alert_manager
from medwatch.core.bus import InMemoryEventBus
from medwatch.core.events import ClinicalEvent
from medwatch.datasets import VIDEO_SAMPLES_BY_NAME, urfall_split
from medwatch.video.pipeline import VideoAnalyzer, build_events
from medwatch.video.pose.base import PoseEstimator
from medwatch.video.pose.factory import build_pose_estimator
from medwatch.video.report import plot_timeline, write_html, write_json, write_pose_series
from medwatch.video.rules import VideoConfig

app = typer.Typer(help="Vídeo clínico: OpenPose/YOLOv8, desvios de exercício e quedas.")


def _config(settings: Settings) -> VideoConfig:
    return VideoConfig.load(settings.configs_dir / "video_rules.yaml")


def _pose(settings: Settings, backend: str) -> PoseEstimator:
    try:
        return build_pose_estimator(backend, settings.models_dir, settings.yolo_pose_weights)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None


def _resolve(settings: Settings, video: str) -> tuple[Path, str | None]:
    """Aceita um caminho ou o nome de um vídeo de exemplo (data download --dataset videos)."""
    path = Path(video)
    if path.exists():
        sample = VIDEO_SAMPLES_BY_NAME.get(path.name)
        return path, sample.crop if sample else None
    sample = VIDEO_SAMPLES_BY_NAME.get(video)
    candidate = settings.data_dir / "raw" / "videos" / video
    if sample and candidate.exists():
        return candidate, sample.crop
    typer.echo(
        f"Vídeo não encontrado: {video}. Exemplos: medwatch data download --dataset videos",
        err=True,
    )
    raise typer.Exit(code=1)


START_TIME_EXAMPLE = "2026-10-19T14:30:00-03:00"


def parse_start_time(value: str | None) -> datetime:
    """Início da gravação em ISO 8601 com fuso; agora (UTC) se omitido."""
    if value is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"--start-time inválido; use ISO 8601, ex.: {START_TIME_EXAMPLE}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"--start-time precisa de fuso horário, ex.: {START_TIME_EXAMPLE}")
    return parsed


async def _send_alerts(
    settings: Settings, events: list[ClinicalEvent]
) -> tuple[int, int, int, int]:
    bus = InMemoryEventBus()
    manager = build_alert_manager(settings)
    consumer = asyncio.create_task(
        manager.consume(bus.subscribe(), min_severity=settings.alert_min_severity)
    )
    for event in events:
        await bus.publish(event)
    await bus.close()
    stats = await consumer
    return stats.sent, stats.suppressed, stats.ignored, stats.failed


@app.command()
def analyze(
    video: str = typer.Argument(..., help="Caminho do vídeo ou nome de um vídeo de exemplo."),
    exercise: str | None = typer.Option(None, help="agachamento, agachamento_unipodal, ..."),
    scene: str | None = typer.Option(None, help="quarto ou centro_cirurgico (ativa objetos)."),
    backend: str = typer.Option("yolo", help="yolo (rápido) ou openpose (CMU)."),
    crop: str | None = typer.Option(None, help="right/left: usa metade do quadro."),
    patient: str = typer.Option("PAC-DEMO", help="Identificador pseudonimizado do paciente."),
    start_time: str | None = typer.Option(
        None,
        "--start-time",
        help="Início da gravação, ISO 8601 com fuso (padrão: agora). Alinha com os vitais.",
    ),
    alert: bool = typer.Option(False, help="Envia os achados ao AlertManager."),
    annotated: bool = typer.Option(True, help="Gera o vídeo anotado."),
) -> None:
    """Analisa um vídeo e gera relatório HTML/JSON, vídeo anotado e série de pose."""
    settings = get_settings()
    config = _config(settings)
    try:
        recorded_at = parse_start_time(start_time)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    path, sample_crop = _resolve(settings, video)
    objects = None
    if scene:
        from medwatch.video.objects import ObjectDetector

        objects = ObjectDetector(
            weights=settings.yolo_detection_weights, models_dir=settings.models_dir
        )
    analyzer = VideoAnalyzer(config, _pose(settings, backend), objects)
    try:
        analysis = analyzer.analyze(path, exercise=exercise, scene=scene, crop=crop or sample_crop)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    from medwatch.video.render import render

    reports = settings.outputs_dir / "reports" / "video"
    stem = f"{path.stem}_{analysis.backend}"
    rendered = render(analysis, config, reports, stem, write_video=annotated)
    events = build_events(analysis, patient, start_time=recorded_at, keyframes=rendered.keyframes)
    timeline_png = plot_timeline(analysis, config, reports / f"{stem}_metricas.png")
    report_html = write_html(analysis, events, timeline_png, reports / f"{stem}.html")
    report_json = write_json(analysis, events, reports / f"{stem}.json")
    pose = write_pose_series(analysis, reports / f"{stem}_pose.parquet")

    typer.echo(
        f"\n{path.name} · {analysis.backend} · {analysis.duration_s:.1f} s · "
        f"{len(analysis.timeline)} quadros a {analysis.processing_fps:.1f} quadros/s · "
        f"{len(analysis.repetitions)} repetições · {analysis.deviation_ratio:.0%} fora do padrão"
    )
    for finding in analysis.findings:
        typer.echo(
            f"  {finding.start:5.1f}–{finding.end:5.1f} s  {finding.severity.value:<8}  "
            f"{finding.message}"
        )
    if not analysis.findings:
        typer.echo("  nenhum desvio detectado")
    typer.echo(f"\nRelatório: {report_html}\nJSON: {report_json}\nPose: {pose}")
    if rendered.video:
        typer.echo(f"Vídeo anotado: {rendered.video}")

    if alert:
        sent, suppressed, ignored, failed = asyncio.run(_send_alerts(settings, events))
        typer.echo(
            f"Alertas enviados: {sent} · suprimidos (cooldown): {suppressed} · "
            f"ignorados (< {settings.alert_min_severity}): {ignored}"
        )
        if failed:
            typer.echo(f"Sem entrega em nenhum canal: {failed}", err=True)


@app.command("evaluate-falls")
def evaluate_falls(
    backend: str = typer.Option("yolo", help="yolo ou openpose."),
) -> None:
    """Acurácia da detecção de quedas no UR Fall Dataset (calibração 01–10, validação 11–20)."""
    settings = get_settings()
    folder = settings.data_dir / "raw" / "urfall"
    if not all((folder / name).exists() for name in urfall_split("validation")):
        typer.echo("Dataset não encontrado. Rode antes: medwatch data download --dataset urfall")
        raise typer.Exit(code=1)
    analyzer = VideoAnalyzer(_config(settings), _pose(settings, backend))
    results: dict[str, dict[str, object]] = {}
    typer.echo(f"\n{'conjunto':<12} {'quedas detectadas':>18} {'alarmes falsos':>15}")
    for split in ("calibration", "validation"):
        detected: dict[str, bool] = {}
        for name in urfall_split(split):
            analysis = analyzer.analyze(folder / name, crop="right")
            detected[name] = any(f.type == "fall_detected" for f in analysis.findings)
        falls = [n for n in detected if n.startswith("fall")]
        adls = [n for n in detected if n.startswith("adl")]
        tp = sum(detected[n] for n in falls)
        fp = sum(detected[n] for n in adls)
        results[split] = {
            "recall": tp / len(falls),
            "false_alarm_rate": fp / len(adls),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "missed": [n for n in falls if not detected[n]],
            "false_alarms": [n for n in adls if detected[n]],
        }
        typer.echo(f"{split:<12} {tp:>12}/{len(falls):<5} {fp:>10}/{len(adls)}")
    path = settings.outputs_dir / "reports" / "video" / f"falls_evaluation_{backend}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"\nResultado: {path}")
