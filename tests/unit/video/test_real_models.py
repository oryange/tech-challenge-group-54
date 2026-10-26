"""Modelos reais (YOLOv8-pose e OpenPose CMU) num quadro de vídeo de exemplo.

Marcados como slow: rodam quando os pesos (models/) e os vídeos de exemplo já foram baixados.
"""

import cv2
import numpy as np
import pytest

from medwatch.config import PROJECT_ROOT
from medwatch.video.keypoints import KP

pytestmark = pytest.mark.slow

VIDEO = PROJECT_ROOT / "data" / "raw" / "videos" / "agachamento_unipodal.webm"
MODELS = PROJECT_ROOT / "models"


@pytest.fixture(scope="module")
def frame():
    if not VIDEO.exists():
        pytest.skip("vídeos de exemplo não baixados (medwatch data download --dataset videos)")
    capture = cv2.VideoCapture(str(VIDEO))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 120)
    ok, image = capture.read()
    assert ok
    return image


def main_person(estimator, frame):
    people = estimator.estimate(frame)
    assert people, f"{estimator.name} não detectou ninguém"
    return max(people, key=lambda p: p.area)


def test_yolo_and_openpose_agree_on_keypoints(frame):
    if (
        not (MODELS / "yolov8n-pose.pt").exists()
        or not (MODELS / "openpose_body_coco18.pth").exists()
    ):
        pytest.skip("pesos não baixados (rode medwatch video analyze com cada backend)")
    from medwatch.video.pose.openpose import OpenPoseEstimator
    from medwatch.video.pose.yolo_pose import YoloPoseEstimator

    yolo = main_person(YoloPoseEstimator(models_dir=MODELS), frame)
    openpose = main_person(OpenPoseEstimator(models_dir=MODELS), frame)
    for joint in ("left_shoulder", "left_hip", "left_knee", "left_ankle"):
        a, b = yolo.keypoints[KP[joint]], openpose.keypoints[KP[joint]]
        assert a[2] > 0.3 and b[2] > 0.3
        assert np.hypot(*(a[:2] - b[:2])) < 15, joint  # pixels
