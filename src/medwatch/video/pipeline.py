"""Pipeline de vídeo: amostragem de quadros → pose (+ objetos) → métricas → regras → eventos."""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger

from medwatch.core.events import ClinicalEvent, Source
from medwatch.video.keypoints import COCO17, KP, Person
from medwatch.video.metrics import compute_metrics, smooth_keypoints
from medwatch.video.objects import Detection, ObjectDetector, count_findings, roi_findings
from medwatch.video.pose.base import PoseEstimator
from medwatch.video.rules import (
    Finding,
    Repetition,
    VideoConfig,
    count_repetitions,
    detect_falls,
    frame_findings,
    repetition_findings,
)


@dataclass
class VideoAnalysis:
    video: Path
    backend: str
    exercise: str | None
    scene: str | None
    crop: str | None
    duration_s: float
    frame_size: tuple[int, int]
    processing_fps: float
    timeline: pd.DataFrame  # um registro por quadro analisado
    coords: np.ndarray  # (T, 17, 2) suavizado, NaN nos pontos ausentes
    detections: list[list[Detection]]
    repetitions: list[Repetition] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def deviation_ratio(self) -> float:
        """Fração do tempo analisado com algum desvio de exercício ativo."""
        times = self.timeline["time"]
        if times.empty:
            return 0.0
        active = pd.Series(False, index=times.index)
        for f in self.findings:
            if f.type != "fall_detected" and not f.type.startswith("roi_"):
                active |= (times >= f.start) & (times <= f.end)
        return float(active.mean())


def crop_frame(frame: np.ndarray, crop: str | None) -> np.ndarray:
    if crop == "right":  # vídeos com profundidade | RGB lado a lado (UR Fall Dataset)
        return frame[:, frame.shape[1] // 2 :]
    if crop == "left":
        return frame[:, : frame.shape[1] // 2]
    return frame


def iter_sampled_frames(
    video: Path, sample_fps: float, crop: str | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """(índice do quadro, tempo em s, quadro) a cada 1/sample_fps segundos de vídeo."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    next_time = 0.0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = index / fps
            if t + 1e-9 >= next_time:
                yield index, t, crop_frame(frame, crop)
                next_time += 1.0 / sample_fps
            index += 1
    finally:
        capture.release()


def choose_person(
    people: list[Person], previous: tuple[float, float] | None, width: int
) -> Person | None:
    """Mantém a mesma pessoa entre quadros (a mais próxima da anterior) ou pega a maior."""
    if not people:
        return None
    if previous is not None:

        def distance(p: Person) -> float:
            cx, cy = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
            return float(np.hypot(cx - previous[0], cy - previous[1]))

        nearest = min(people, key=distance)
        if distance(nearest) < 0.3 * width:
            return nearest
    return max(people, key=lambda p: p.area)


class VideoAnalyzer:
    def __init__(
        self,
        config: VideoConfig,
        pose: PoseEstimator,
        objects: ObjectDetector | None = None,
    ) -> None:
        self.config = config
        self.pose = pose
        self.objects = objects

    def analyze(
        self,
        video: Path,
        exercise: str | None = None,
        scene: str | None = None,
        crop: str | None = None,
    ) -> VideoAnalysis:
        if exercise is not None and exercise not in self.config.exercises:
            raise ValueError(f"Exercício desconhecido: {exercise}")
        if scene is not None and scene not in self.config.scenes:
            raise ValueError(f"Cena desconhecida: {scene}")

        records: list[dict[str, float | int]] = []
        keypoints: list[np.ndarray] = []
        detections: list[list[Detection]] = []
        previous: tuple[float, float] | None = None
        frame_size = (0, 0)
        started = time.perf_counter()

        for index, t, frame in iter_sampled_frames(video, self.config.sample_fps, crop):
            frame_size = (frame.shape[1], frame.shape[0])
            person = choose_person(self.pose.estimate(frame), previous, frame.shape[1])
            record: dict[str, float | int] = {"frame": index, "time": t}
            if person is None:
                keypoints.append(np.zeros((17, 3)))
                record.update(
                    {"bbox_x1": np.nan, "bbox_y1": np.nan, "bbox_x2": np.nan, "bbox_y2": np.nan}
                )
            else:
                keypoints.append(person.keypoints)
                x1, y1, x2, y2 = person.bbox
                record.update({"bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2})
                previous = ((x1 + x2) / 2, (y1 + y2) / 2)
            detections.append(self.objects.detect(frame) if self.objects else [])
            records.append(record)

        elapsed = time.perf_counter() - started
        if not records:
            raise ValueError(f"Nenhum quadro lido de {video}")
        logger.info(
            "{}: {} quadros em {:.1f} s ({:.1f} quadros/s, {})",
            video.name,
            len(records),
            elapsed,
            len(records) / elapsed,
            self.pose.name,
        )

        raw = np.stack(keypoints)
        coords = smooth_keypoints(raw, self.config.smoothing_frames, self.config.min_keypoint_conf)
        timeline = pd.DataFrame(records).join(compute_metrics(coords))
        for name in COCO17:
            timeline[f"kp_{name}_x"] = coords[:, KP[name], 0]
            timeline[f"kp_{name}_y"] = coords[:, KP[name], 1]
            timeline[f"kp_{name}_conf"] = raw[:, KP[name], 2]
        timeline["n_detections"] = [len(d) for d in detections]

        analysis = VideoAnalysis(
            video=video,
            backend=self.pose.name,
            exercise=exercise,
            scene=scene,
            crop=crop,
            duration_s=float(timeline["time"].iloc[-1]),
            frame_size=frame_size,
            processing_fps=len(records) / elapsed,
            timeline=timeline,
            coords=coords,
            detections=detections,
        )
        analysis.findings = self._findings(analysis)
        analysis.findings.sort(key=lambda f: f.start)
        return analysis

    def _findings(self, analysis: VideoAnalysis) -> list[Finding]:
        cfg, timeline = self.config, analysis.timeline
        times = timeline["time"]
        findings: list[Finding] = []

        if analysis.exercise:
            exercise = cfg.exercises[analysis.exercise]
            if exercise.repetition:
                analysis.repetitions = count_repetitions(exercise.repetition, timeline, times)
            for rule in exercise.rules:
                if rule.kind == "repetition":
                    findings += repetition_findings(rule, analysis.repetitions)
                else:
                    findings += frame_findings(rule, timeline, times, cfg.min_deviation_seconds)

        heights = timeline["bbox_y2"] - timeline["bbox_y1"]
        widths = timeline["bbox_x2"] - timeline["bbox_x1"]
        upright = heights[heights / widths > 1.3]
        person_height = float(upright.median() if len(upright) else heights.quantile(0.9))
        # sem quadril visível (ex.: pessoa caída de lado), usa o centro da caixa da pessoa
        hip_y = timeline["hip_y"].fillna((timeline["bbox_y1"] + timeline["bbox_y2"]) / 2)
        findings += detect_falls(cfg.fall, times, hip_y, person_height, widths / heights)

        if analysis.scene:
            scene = cfg.scenes[analysis.scene]
            for roi in scene.rois:
                findings += roi_findings(
                    roi, analysis.detections, times, analysis.frame_size, cfg.min_deviation_seconds
                )
            findings += count_findings(scene.count_classes, analysis.detections, times)
        return findings


def build_events(
    analysis: VideoAnalysis,
    patient_id: str,
    start_time: datetime | None = None,
    keyframes: dict[int, Path] | None = None,
) -> list[ClinicalEvent]:
    """Converte os achados em ClinicalEvents. `keyframes`: índice do achado → imagem."""
    t0 = start_time or datetime.now(UTC)
    events = []
    for i, finding in enumerate(analysis.findings):
        artifact = (keyframes or {}).get(i)
        events.append(
            ClinicalEvent(
                patient_id=patient_id,
                source=Source.VIDEO,
                type=finding.type,
                timestamp=t0 + timedelta(seconds=finding.peak),
                score=finding.score,
                severity=finding.severity,
                evidence={
                    **finding.evidence,
                    "message": finding.message,
                    "video": analysis.video.name,
                    "backend": analysis.backend,
                    "exercise": analysis.exercise,
                    "start_s": round(finding.start, 2),
                    "end_s": round(finding.end, 2),
                },
                artifacts=[artifact] if artifact else [],
            )
        )
    return events
