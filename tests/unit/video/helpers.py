"""Esqueletos sintéticos (vista lateral), vídeo falso e estimador de pose roteirizado."""

import math
from pathlib import Path

import cv2
import numpy as np

from medwatch.config import PROJECT_ROOT
from medwatch.video.keypoints import KP, Person, bbox_from_keypoints
from medwatch.video.pose.base import PoseEstimator
from medwatch.video.rules import VideoConfig


def load_config() -> VideoConfig:
    return VideoConfig.load(PROJECT_ROOT / "configs" / "video_rules.yaml")


def _rotate(vec: np.ndarray, degrees: float) -> np.ndarray:
    t = math.radians(degrees)
    return np.array(
        [vec[0] * math.cos(t) - vec[1] * math.sin(t), vec[0] * math.sin(t) + vec[1] * math.cos(t)]
    )


def skeleton(
    knee_angle: float = 175,
    trunk_lean: float = 5,
    hip: tuple[float, float] = (200, 200),
    lying: bool = False,
    conf: float = 0.9,
) -> np.ndarray:
    """Keypoints COCO-17 (17, 3) de uma pessoa de perfil com ângulos conhecidos."""
    hip_pt = np.array(hip, dtype=float)
    thigh, shank, trunk = 45.0, 45.0, 60.0
    phi = math.radians((180 - knee_angle) / 2)
    u = np.array([math.sin(phi), math.cos(phi)])  # coxa: para baixo e para frente
    knee = hip_pt + thigh * u
    ankle = knee + shank * _rotate(u, 180 - knee_angle)
    lean = math.radians(trunk_lean)
    shoulder = hip_pt + trunk * np.array([math.sin(lean), -math.cos(lean)])
    elbow = shoulder + np.array([0, 25])
    wrist = elbow + np.array([0, 25])
    nose = shoulder + trunk * 0.35 * np.array([math.sin(lean), -math.cos(lean)])
    points = {
        "nose": nose, "left_eye": nose + (-3, -3), "right_eye": nose + (3, -3),
        "left_ear": nose + (-6, 0), "right_ear": nose + (6, 0),
        "left_shoulder": shoulder, "right_shoulder": shoulder,
        "left_elbow": elbow, "right_elbow": elbow, "left_wrist": wrist, "right_wrist": wrist,
        "left_hip": hip_pt, "right_hip": hip_pt, "left_knee": knee, "right_knee": knee,
        "left_ankle": ankle, "right_ankle": ankle,
    }  # fmt: skip
    kp = np.zeros((17, 3))
    for name, point in points.items():
        kp[KP[name], :2] = point
        kp[KP[name], 2] = conf
    if lying:  # gira 90° em torno do quadril e apoia no chão
        rel = kp[:, :2] - hip_pt
        kp[:, :2] = hip_pt + np.stack([-rel[:, 1], rel[:, 0]], axis=1)
    return kp


def person(kp: np.ndarray) -> Person:
    return Person(keypoints=kp, bbox=bbox_from_keypoints(kp), score=0.9)


def write_video(
    path: Path, seconds: float, fps: int = 30, size: tuple[int, int] = (400, 320)
) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(int(seconds * fps)):
        frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path


class ScriptedPose(PoseEstimator):
    """Devolve, a cada chamada, a próxima pose de um roteiro (função do índice do quadro)."""

    name = "scripted"

    def __init__(self, script) -> None:
        self.script = script
        self.calls = 0

    def estimate(self, frame_bgr):
        kp = self.script(self.calls)
        self.calls += 1
        return [] if kp is None else [person(kp)]


def squat_script(sample_fps: float = 10, reps: int = 3, bottom: float = 85, lean: float = 10):
    """Agachamentos de 2 s cada: em pé (175°) → fundo (`bottom`) → em pé."""

    def script(i: int) -> np.ndarray:
        t = i / sample_fps
        phase = (t % 2.0) / 2.0 if t < reps * 2.0 else 0.0
        knee = 175 - (175 - bottom) * math.sin(math.pi * phase)
        trunk = 5 + (lean - 5) * math.sin(math.pi * phase)
        hip_y = 200 + (175 - knee) * 0.6
        return skeleton(knee_angle=knee, trunk_lean=trunk, hip=(200, hip_y))

    return script


def fall_script(sample_fps: float = 10, at: float = 1.0, descent: float = 0.4):
    """Em pé até `at`, cai em `descent` segundos e fica deitado no chão."""

    def script(i: int) -> np.ndarray:
        t = i / sample_fps
        if t < at:
            return skeleton(hip=(200, 160))
        progress = min(1.0, (t - at) / descent)
        hip_y = 160 + 110 * progress
        return skeleton(hip=(200, hip_y), lying=progress >= 0.6, knee_angle=175)

    return script


def sit_script(sample_fps: float = 10, at: float = 1.0, duration: float = 2.5, lie: bool = False):
    """Desce devagar (sentar ou deitar com calma): não deve ser queda."""

    def script(i: int) -> np.ndarray:
        t = i / sample_fps
        progress = min(1.0, max(0.0, (t - at) / duration))
        hip_y = 160 + 60 * progress
        knee = 175 - 85 * progress
        return skeleton(hip=(200, hip_y), knee_angle=knee, lying=lie and progress >= 1.0)

    return script
