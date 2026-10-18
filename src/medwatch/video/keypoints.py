"""Esquema comum de keypoints (COCO-17) e conversão dos formatos do OpenPose.

Todos os backends de pose devolvem `Person` com 17 keypoints no padrão COCO, para que as
regras clínicas não dependam do modelo usado.
"""

from dataclasses import dataclass

import numpy as np

COCO17 = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
KP = {name: index for index, name in enumerate(COCO17)}

# Segmentos para desenhar o esqueleto
SKELETON = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
)

# OpenPose COCO-18 (modelo usado pelo backend PyTorch): índice COCO-17 → índice OpenPose.
# Ordem OpenPose-18: nose, neck, r_sho, r_elb, r_wri, l_sho, l_elb, l_wri, r_hip, r_knee,
# r_ank, l_hip, l_knee, l_ank, r_eye, l_eye, r_ear, l_ear. O "neck" não existe no COCO-17.
OPENPOSE18_FOR_COCO17 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)

# OpenPose BODY_25 (saída JSON do binário oficial): índice COCO-17 → índice BODY_25.
# BODY_25 adiciona neck (1), mid_hip (8) e os pés (19–24), descartados aqui.
BODY25_FOR_COCO17 = (0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11)


@dataclass
class Person:
    """Uma pessoa detectada no quadro. `keypoints`: array (17, 3) com x, y e confiança."""

    keypoints: np.ndarray
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels
    score: float
    track_id: int | None = None

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def area(self) -> float:
        return self.height * self.width


def from_openpose(points: np.ndarray, mapping: tuple[int, ...]) -> np.ndarray:
    """Converte keypoints do OpenPose (N, 3) para COCO-17 (17, 3). Ausentes ficam com conf 0."""
    converted = points[list(mapping)].astype(float)
    converted[converted[:, 2] <= 0] = 0.0
    return converted


def bbox_from_keypoints(
    keypoints: np.ndarray, min_conf: float = 0.1
) -> tuple[float, float, float, float]:
    visible = keypoints[keypoints[:, 2] > min_conf]
    if len(visible) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    x1, y1 = visible[:, :2].min(axis=0)
    x2, y2 = visible[:, :2].max(axis=0)
    return (float(x1), float(y1), float(x2), float(y2))
