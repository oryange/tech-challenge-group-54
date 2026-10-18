"""Interface comum dos estimadores de pose."""

from abc import ABC, abstractmethod

import numpy as np
import torch

from medwatch.video.keypoints import Person


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
