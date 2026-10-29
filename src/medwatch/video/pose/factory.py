"""Escolha do estimador de pose pelo nome do backend (CLI e API)."""

from pathlib import Path

from medwatch.video.pose.base import PoseEstimator

POSE_BACKENDS = ("yolo", "openpose")


def build_pose_estimator(
    backend: str, models_dir: Path, yolo_weights: str = "yolov8n-pose.pt"
) -> PoseEstimator:
    """`yolo` (YOLOv8-pose, rápido em CPU) ou `openpose` (rede original da CMU)."""
    if backend == "openpose":
        from medwatch.video.pose.openpose import OpenPoseEstimator

        return OpenPoseEstimator(models_dir=models_dir)
    if backend == "yolo":
        from medwatch.video.pose.yolo_pose import YoloPoseEstimator

        return YoloPoseEstimator(weights=yolo_weights, models_dir=models_dir)
    raise ValueError(f"Backend desconhecido: {backend} (use {' ou '.join(POSE_BACKENDS)})")
