"""Backend OpenPose (CMU): rede original de corpo (COCO-18) rodando em PyTorch.

Os pesos são os do OpenPose convertidos para PyTorch (distribuídos em
lllyasviel/Annotators no Hugging Face). Licença dos pesos: acadêmica, não comercial (CMU).
Também lê a saída JSON (BODY_25) do binário oficial do OpenPose, para quem rodar em GPU.
"""

import json
from pathlib import Path

import numpy as np
import torch

from medwatch.datasets import download_file
from medwatch.video.keypoints import (
    BODY25_FOR_COCO17,
    OPENPOSE18_FOR_COCO17,
    Person,
    bbox_from_keypoints,
    from_openpose,
)
from medwatch.video.pose.base import PoseEstimator, pick_device
from medwatch.video.pose.openpose_net import (
    find_peaks,
    group_people,
    infer_maps,
    load_body_model,
    people_keypoints,
)

WEIGHTS_URL = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
WEIGHTS_SHA256 = "25a948c16078b0f08e236bda51a385d855ef4c153598947c28c0d47ed94bb746"
WEIGHTS_FILE = "openpose_body_coco18.pth"


class OpenPoseEstimator(PoseEstimator):
    name = "openpose"

    def __init__(
        self,
        models_dir: Path,
        scales: tuple[float, ...] = (0.5,),
        device: str | None = None,
    ) -> None:
        weights = download_file(WEIGHTS_URL, models_dir / WEIGHTS_FILE, WEIGHTS_SHA256)
        self._device = torch.device(pick_device(device))
        self._model = load_body_model(str(weights), self._device)
        self._scales = scales

    def estimate(self, frame_bgr: np.ndarray) -> list[Person]:
        heatmaps, pafs = infer_maps(self._model, frame_bgr, scales=self._scales)
        candidate, subset = group_people(find_peaks(heatmaps), pafs, frame_bgr.shape[0])
        people = []
        for points, score in people_keypoints(candidate, subset):
            keypoints = from_openpose(points, OPENPOSE18_FOR_COCO17)
            people.append(
                Person(keypoints=keypoints, bbox=bbox_from_keypoints(keypoints), score=score)
            )
        return people


def read_openpose_json(path: Path) -> list[Person]:
    """Lê um `*_keypoints.json` do OpenPose oficial (modelo BODY_25) em COCO-17."""
    data = json.loads(path.read_text(encoding="utf-8"))
    people = []
    for person in data.get("people", []):
        points = np.asarray(person["pose_keypoints_2d"], dtype=float).reshape(-1, 3)
        keypoints = from_openpose(points, BODY25_FOR_COCO17)
        visible = keypoints[keypoints[:, 2] > 0, 2]
        people.append(
            Person(
                keypoints=keypoints,
                bbox=bbox_from_keypoints(keypoints),
                score=float(visible.mean()) if len(visible) else 0.0,
            )
        )
    return people
