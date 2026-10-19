"""Métricas biomecânicas por quadro a partir dos keypoints COCO-17 suavizados."""

import numpy as np
import pandas as pd

from medwatch.video.keypoints import KP


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ângulo ABC em graus (vértice em B) para arrays (N, 2). NaN se faltar algum ponto."""
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def tilt_from_vertical(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Inclinação do segmento bottom→top em relação à vertical (0° = em pé), em graus."""
    vec = top - bottom
    # no eixo da imagem o y cresce para baixo: "para cima" é (0, -1)
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arctan2(np.abs(vec[:, 0]), -vec[:, 1]))


def smooth_keypoints(keypoints: np.ndarray, window: int, min_conf: float) -> np.ndarray:
    """Mediana móvel causal das coordenadas (T, 17, 2); pontos pouco confiáveis viram NaN."""
    coords = keypoints[:, :, :2].copy()
    coords[keypoints[:, :, 2] < min_conf] = np.nan
    flat = pd.DataFrame(coords.reshape(len(coords), -1))
    smoothed = flat.rolling(window, min_periods=1).median().to_numpy(copy=True)
    # ponto ausente no quadro atual continua ausente: a mediana não inventa articulações
    smoothed[np.isnan(flat.to_numpy())] = np.nan
    return smoothed.reshape(coords.shape)


def compute_metrics(coords: np.ndarray) -> pd.DataFrame:
    """Métricas por quadro a partir de coordenadas (T, 17, 2) com NaN nos pontos ausentes."""

    def p(name: str) -> np.ndarray:
        return coords[:, KP[name]]

    shoulder_mid = (p("left_shoulder") + p("right_shoulder")) / 2
    hip_mid = (p("left_hip") + p("right_hip")) / 2
    # na vista lateral um dos lados some: usa o lado visível
    shoulder_mid = np.where(
        np.isnan(shoulder_mid), np.fmax(p("left_shoulder"), p("right_shoulder")), shoulder_mid
    )
    hip_mid = np.where(np.isnan(hip_mid), np.fmax(p("left_hip"), p("right_hip")), hip_mid)

    m = pd.DataFrame(index=range(len(coords)))
    m["knee_angle_left"] = angle(p("left_hip"), p("left_knee"), p("left_ankle"))
    m["knee_angle_right"] = angle(p("right_hip"), p("right_knee"), p("right_ankle"))
    m["knee_angle_min"] = m[["knee_angle_left", "knee_angle_right"]].min(axis=1)
    m["knee_asymmetry"] = (m["knee_angle_left"] - m["knee_angle_right"]).abs()
    # perna de apoio: a do tornozelo mais baixo na imagem (maior y)
    left_lower = p("left_ankle")[:, 1] >= p("right_ankle")[:, 1]
    m["support_knee_angle"] = np.where(left_lower, m["knee_angle_left"], m["knee_angle_right"])
    m.loc[np.isnan(p("left_ankle")[:, 1]), "support_knee_angle"] = m["knee_angle_right"]
    m.loc[np.isnan(p("right_ankle")[:, 1]), "support_knee_angle"] = m["knee_angle_left"]

    m["hip_angle_left"] = angle(p("left_shoulder"), p("left_hip"), p("left_knee"))
    m["hip_angle_right"] = angle(p("right_shoulder"), p("right_hip"), p("right_knee"))
    m["shoulder_angle_left"] = angle(p("left_hip"), p("left_shoulder"), p("left_elbow"))
    m["shoulder_angle_right"] = angle(p("right_hip"), p("right_shoulder"), p("right_elbow"))
    m["shoulder_angle_min"] = m[["shoulder_angle_left", "shoulder_angle_right"]].min(axis=1)
    m["shoulder_asymmetry"] = (m["shoulder_angle_left"] - m["shoulder_angle_right"]).abs()
    m["trunk_lean"] = tilt_from_vertical(hip_mid, shoulder_mid)
    m["hip_x"] = hip_mid[:, 0]
    m["hip_y"] = hip_mid[:, 1]
    return m
