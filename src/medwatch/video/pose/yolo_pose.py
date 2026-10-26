"""Backend YOLOv8-pose (Ultralytics): rápido em CPU, padrão para desenvolvimento."""

from pathlib import Path
from typing import Any

import numpy as np

from medwatch.video.keypoints import Person
from medwatch.video.pose.base import PoseEstimator, pick_device, yolo_weights


class YoloPoseEstimator(PoseEstimator):
    name = "yolov8-pose"

    def __init__(
        self,
        weights: str = "yolov8n-pose.pt",
        models_dir: Path | None = None,
        conf: float = 0.25,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(yolo_weights(weights, models_dir)))
        self._conf = conf
        self._device = pick_device(device)

    def estimate(self, frame_bgr: np.ndarray) -> list[Person]:
        results: Any = self._model(frame_bgr, conf=self._conf, device=self._device, verbose=False)
        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return []
        keypoints = result.keypoints.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [
            Person(
                keypoints=kp.astype(float),
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                score=float(score),
            )
            for kp, box, score in zip(keypoints, boxes, scores, strict=True)
        ]
