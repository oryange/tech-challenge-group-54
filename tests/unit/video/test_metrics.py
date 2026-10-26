import numpy as np
import pytest

from medwatch.video.metrics import angle, compute_metrics, smooth_keypoints, tilt_from_vertical
from tests.unit.video.helpers import skeleton


def arr(*points):
    return np.array([points], dtype=float)


def test_angle():
    assert angle(arr(0, 0), arr(1, 0), arr(1, 1))[0] == pytest.approx(90)
    assert angle(arr(0, 0), arr(1, 0), arr(2, 0))[0] == pytest.approx(180)
    assert np.isnan(angle(arr(np.nan, 0), arr(1, 0), arr(2, 0))[0])


def test_tilt_from_vertical_uses_image_axis():
    assert tilt_from_vertical(arr(0, 100), arr(0, 0))[0] == pytest.approx(0)  # em pé
    assert tilt_from_vertical(arr(0, 100), arr(100, 100))[0] == pytest.approx(90)  # deitado
    assert tilt_from_vertical(arr(0, 100), arr(-100, 0))[0] == pytest.approx(45)


@pytest.mark.parametrize(("knee", "lean"), [(175, 5), (90, 30), (70, 50)])
def test_metrics_recover_known_angles(knee, lean):
    kp = skeleton(knee_angle=knee, trunk_lean=lean)
    metrics = compute_metrics(kp[None, :, :2]).iloc[0]
    assert metrics["knee_angle_min"] == pytest.approx(knee, abs=0.5)
    assert metrics["support_knee_angle"] == pytest.approx(knee, abs=0.5)
    assert metrics["trunk_lean"] == pytest.approx(lean, abs=0.5)
    assert metrics["knee_asymmetry"] == pytest.approx(0, abs=0.5)


def test_smoothing_removes_single_frame_spike_and_keeps_missing_points():
    frames = np.stack([skeleton() for _ in range(5)])
    frames[2, 0, :2] += 80  # tremor do estimador no nariz
    frames[3, 5, 2] = 0.1  # ombro pouco confiável neste quadro
    smoothed = smooth_keypoints(frames, window=5, min_conf=0.3)
    assert smoothed[2, 0, 0] == pytest.approx(frames[0, 0, 0])
    assert np.isnan(smoothed[3, 5]).all()
