"""Detecção e rastreamento de objetos (YOLOv8 + ByteTrack) e regras de áreas críticas."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medwatch.core.events import Severity
from medwatch.video.pose.base import pick_device, yolo_weights
from medwatch.video.rules import Finding, Roi, _intervals, point_in_polygon


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

    def anchor(self, width: int, height: int) -> tuple[float, float]:
        """Ponto de referência normalizado: os pés para pessoas, o centro para objetos."""
        x1, y1, x2, y2 = self.bbox
        y = y2 if self.label == "person" else (y1 + y2) / 2
        return ((x1 + x2) / 2 / width, y / height)


class ObjectDetector:
    """YOLOv8 pré-treinado no COCO (80 classes: pessoa, cama, cadeira, tesoura, celular...)."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        models_dir: Path | None = None,
        conf: float = 0.35,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(yolo_weights(weights, models_dir)))
        self._conf = conf
        self._device = pick_device(device)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results: Any = self._model.track(
            frame_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self._conf,
            device=self._device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None:
            return []
        names = result.names
        ids = result.boxes.id
        track_ids = ids.int().cpu().tolist() if ids is not None else [None] * len(result.boxes)
        return [
            Detection(
                label=names[int(cls)],
                confidence=float(conf),
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                track_id=track_id,
            )
            for cls, conf, box, track_id in zip(
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.xyxy.cpu().numpy(),
                track_ids,
                strict=True,
            )
        ]


def roi_findings(
    roi: Roi,
    detections: list[list[Detection]],
    times: pd.Series,
    frame_size: tuple[int, int],
    min_seconds: float,
) -> list[Finding]:
    width, height = frame_size
    labels_inside: list[set[str]] = []
    for frame_detections in detections:
        inside = {
            d.label
            for d in frame_detections
            if d.label in roi.forbidden_classes
            and point_in_polygon(*d.anchor(width, height), roi.polygon)
        }
        labels_inside.append(inside)
    mask = pd.Series([bool(labels) for labels in labels_inside], index=times.index)
    findings = []
    for start, end in _intervals(mask, times, min_seconds):
        seen = Counter(label for labels in labels_inside[start : end + 1] for label in labels)
        findings.append(
            Finding(
                type=f"roi_{roi.name}",
                severity=roi.severity,
                message=roi.message,
                start=float(times.iloc[start]),
                end=float(times.iloc[end]),
                peak=float(times.iloc[start]),
                score=0.8,
                evidence={
                    "roi": roi.name,
                    "classes": ", ".join(sorted(seen)),
                    "duration_s": round(float(times.iloc[end] - times.iloc[start]), 2),
                },
                recommendation="Verificar a área e remover a pessoa/objeto não autorizado.",
            )
        )
    return findings


def count_findings(
    classes: list[str],
    detections: list[list[Detection]],
    times: pd.Series,
    edge_seconds: float = 2.0,
) -> list[Finding]:
    """Contagem de instrumentos no início e no fim do vídeo (ex.: instrumental cirúrgico)."""
    if len(times) == 0:
        return []
    first = times <= times.iloc[0] + edge_seconds
    last = times >= times.iloc[-1] - edge_seconds
    findings = []
    for label in classes:
        counts = pd.Series(
            [sum(d.label == label for d in f) for f in detections], index=times.index
        )
        at_start, at_end = int(counts[first].median()), int(counts[last].median())
        if at_start != at_end:
            findings.append(
                Finding(
                    type="instrument_count_mismatch",
                    severity=Severity.HIGH,
                    message=f"Contagem de '{label}' diferente entre início e fim do procedimento",
                    start=float(times.iloc[0]),
                    end=float(times.iloc[-1]),
                    peak=float(times.iloc[-1]),
                    score=0.9,
                    evidence={"class": label, "count_start": at_start, "count_end": at_end},
                    recommendation="Conferir o instrumental antes de encerrar o procedimento.",
                )
            )
    return findings
