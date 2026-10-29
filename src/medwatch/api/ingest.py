"""Análise de arquivos enviados à API (`POST /ingest/audio` e `/ingest/video`).

As funções são síncronas e pesadas (Azure, modelos de pose): a API as roda numa thread.
Reaproveitam os mesmos componentes da CLI, então o resultado é idêntico ao de
`medwatch audio analyze` / `medwatch video analyze`.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from medwatch.config import Settings
from medwatch.core.events import ClinicalEvent

VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mov", ".mkv")
MAX_VIDEO_MB = 500


@dataclass(frozen=True)
class IngestResult:
    events: list[ClinicalEvent]
    report: Path


class AudioIngestor(Protocol):
    def __call__(self, path: Path, patient_id: str, start_time: datetime) -> IngestResult: ...


class VideoIngestor(Protocol):
    def __call__(
        self,
        path: Path,
        patient_id: str,
        start_time: datetime,
        exercise: str | None,
        scene: str | None,
        backend: str,
    ) -> IngestResult: ...


def audio_ingestor(settings: Settings) -> AudioIngestor:
    """Pipeline de áudio: Azure Speech + Azure Language (ou respostas gravadas no mock)."""

    def ingest(path: Path, patient_id: str, start_time: datetime) -> IngestResult:
        from medwatch.audio.config import AudioConfig, CriticalTermsConfig
        from medwatch.audio.pipeline import AudioAnalyzer
        from medwatch.audio.report import write_report
        from medwatch.commands.audio import build_providers

        config = AudioConfig.load(settings.configs_dir / "audio.yaml")
        terms = CriticalTermsConfig.load(settings.configs_dir / "critical_terms.yaml")
        transcriber, language = build_providers(settings, config, path, diarize=False, record=False)
        analyzer = AudioAnalyzer(config, terms, transcriber, language)
        report = analyzer.analyze(path, patient_id, start_time, mock=settings.azure_mock)
        return IngestResult(report.events, write_report(report, settings.outputs_dir / "reports"))

    return ingest


def video_ingestor(settings: Settings) -> VideoIngestor:
    """Pipeline de vídeo: pose (YOLOv8-pose ou OpenPose), objetos por cena e regras."""

    def ingest(
        path: Path,
        patient_id: str,
        start_time: datetime,
        exercise: str | None,
        scene: str | None,
        backend: str,
    ) -> IngestResult:
        from medwatch.video.pipeline import VideoAnalyzer, build_events
        from medwatch.video.pose.factory import build_pose_estimator
        from medwatch.video.render import render
        from medwatch.video.report import plot_timeline, write_html, write_json
        from medwatch.video.rules import VideoConfig

        config = VideoConfig.load(settings.configs_dir / "video_rules.yaml")
        pose = build_pose_estimator(backend, settings.models_dir, settings.yolo_pose_weights)
        objects = None
        if scene:
            from medwatch.video.objects import ObjectDetector

            objects = ObjectDetector(
                weights=settings.yolo_detection_weights, models_dir=settings.models_dir
            )
        analysis = VideoAnalyzer(config, pose, objects).analyze(
            path, exercise=exercise, scene=scene
        )
        reports = settings.outputs_dir / "reports" / "video"
        stem = f"{patient_id}_{start_time:%Y%m%dT%H%M%S}_{analysis.backend}"
        # sem o vídeo anotado: a resposta da API não deve esperar a codificação do MP4
        rendered = render(analysis, config, reports, stem, write_video=False)
        events = build_events(analysis, patient_id, start_time, keyframes=rendered.keyframes)
        timeline_png = plot_timeline(analysis, config, reports / f"{stem}_metricas.png")
        write_json(analysis, events, reports / f"{stem}.json")
        return IngestResult(
            events, write_html(analysis, events, timeline_png, reports / f"{stem}.html")
        )

    return ingest
