"""YoloPoseEstimator com o modelo do Ultralytics substituído: sem pesos nem inferência real."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from medwatch.video.pose import yolo_pose
from medwatch.video.pose.yolo_pose import YoloPoseEstimator


class FakeYolo:
    def __init__(self, weights: str, result: SimpleNamespace) -> None:
        self.weights = weights
        self.result = result
        self.calls: list[dict] = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def _result(keypoints, boxes, scores) -> SimpleNamespace:
    return SimpleNamespace(
        keypoints=SimpleNamespace(data=torch.tensor(keypoints)),
        boxes=SimpleNamespace(xyxy=torch.tensor(boxes), conf=torch.tensor(scores)),
    )


@pytest.fixture
def estimator_with(monkeypatch, tmp_path):
    def build(result: SimpleNamespace) -> tuple[YoloPoseEstimator, FakeYolo]:
        created: list[FakeYolo] = []

        def fake_yolo(weights: str) -> FakeYolo:
            created.append(FakeYolo(weights, result))
            return created[-1]

        fake_module = ModuleType("ultralytics")
        fake_module.YOLO = fake_yolo  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
        monkeypatch.setattr(yolo_pose, "yolo_weights", lambda name, models_dir: tmp_path / name)
        estimator = YoloPoseEstimator(models_dir=tmp_path, conf=0.4, device="cpu")
        return estimator, created[0]

    return build


def test_converts_ultralytics_result_to_people(estimator_with):
    keypoints = np.random.default_rng(0).random((2, 17, 3)).astype(np.float32)
    boxes = [[10.0, 20.0, 110.0, 220.0], [200.0, 30.0, 260.0, 190.0]]
    estimator, model = estimator_with(_result(keypoints, boxes, [0.9, 0.6]))

    people = estimator.estimate(np.zeros((240, 320, 3), dtype=np.uint8))

    assert Path(model.weights).name == "yolov8n-pose.pt"
    assert model.calls == [{"conf": 0.4, "device": "cpu", "verbose": False}]
    assert len(people) == 2
    assert people[0].bbox == (10.0, 20.0, 110.0, 220.0)
    assert people[1].score == pytest.approx(0.6)
    np.testing.assert_allclose(people[0].keypoints, keypoints[0])
    assert people[0].keypoints.dtype == np.float64


def test_frame_without_detections_returns_empty_list(estimator_with):
    estimator, _ = estimator_with(SimpleNamespace(keypoints=None, boxes=None))
    assert estimator.estimate(np.zeros((10, 10, 3), dtype=np.uint8)) == []
