"""Interface comum dos estimadores de pose."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from medwatch.datasets import download_file
from medwatch.video.keypoints import Person

# Pesos oficiais do Ultralytics com SHA-256 fixado, como já é feito com os do OpenPose:
# um arquivo trocado no caminho (ou no release) é recusado antes do torch.load.
ULTRALYTICS_ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0"
YOLO_WEIGHTS_SHA256: dict[str, str] = {
    "yolov8n-pose.pt": "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212",
    "yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
}


def yolo_weights(weights: str, models_dir: Path | None) -> Path:
    """Caminho dos pesos do YOLO, baixados e conferidos pelo SHA-256 quando conhecidos.

    Pesos fora da tabela (ex.: um modelo próprio) seguem sem verificação, com aviso.
    """
    path = (models_dir / weights) if models_dir else Path(weights)
    expected = YOLO_WEIGHTS_SHA256.get(path.name)
    if expected is None:
        logger.warning("Pesos {} sem SHA-256 fixado: integridade não verificada", path.name)
        return path
    return download_file(f"{ULTRALYTICS_ASSETS_URL}/{path.name}", path, expected)


def pick_device(preferred: str | None = None) -> str:
    """cuda > mps (Apple Silicon) > cpu, salvo se `preferred` for informado."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PoseEstimator(ABC):
    name: str

    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray) -> list[Person]:
        """Pessoas detectadas no quadro, com keypoints no padrão COCO-17."""
