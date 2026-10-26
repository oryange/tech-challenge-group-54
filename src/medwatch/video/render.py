"""Vídeo anotado (esqueleto, métricas, desvios, objetos, áreas críticas) e quadros-chave."""

import shutil
import subprocess  # noqa: S404 - só chama o ffmpeg do PATH com argumentos fixos
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from medwatch.core.events import Severity
from medwatch.video.keypoints import KP, SKELETON
from medwatch.video.pipeline import VideoAnalysis, iter_sampled_frames
from medwatch.video.rules import Finding, VideoConfig

# BGR (OpenCV) e RGB (Pillow) por severidade
SEVERITY_BGR = {
    Severity.INFO: (200, 200, 200),
    Severity.LOW: (180, 180, 180),
    Severity.MEDIUM: (37, 168, 249),
    Severity.HIGH: (0, 108, 239),
    Severity.CRITICAL: (40, 40, 198),
}
OK_BGR = (90, 200, 90)
METRIC_LABELS = {
    "knee_angle_min": "Joelho (menor)",
    "support_knee_angle": "Joelho de apoio",
    "trunk_lean": "Tronco",
    "knee_asymmetry": "Assimetria joelhos",
    "shoulder_angle_min": "Ombro (menor)",
    "shoulder_asymmetry": "Assimetria ombros",
}


@dataclass
class RenderResult:
    video: Path | None
    keyframes: dict[int, Path]  # índice do achado → PNG


@cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    # DejaVu Sans vem com o matplotlib e tem acentos (a fonte padrão do Pillow não tem "ç")
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def _fit(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def _rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return bgr[2], bgr[1], bgr[0]


def _active(findings: list[Finding], t: float) -> list[Finding]:
    return [f for f in findings if f.start - 1e-6 <= t <= f.end + 1e-6]


def draw_frame(
    frame: np.ndarray,
    analysis: VideoAnalysis,
    config: VideoConfig,
    k: int,
) -> np.ndarray:
    row = analysis.timeline.iloc[k]
    t = float(row["time"])
    active = _active(analysis.findings, t)
    worst = max((f.severity for f in active), default=None)
    color = SEVERITY_BGR[worst] if worst else OK_BGR
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.5, w / 640)

    if analysis.scene:
        overlay = out.copy()
        for roi in config.scenes[analysis.scene].rois:
            pts = np.array([[x * w, y * h] for x, y in roi.polygon], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (60, 60, 200))
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

    for det in analysis.detections[k]:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), max(1, int(scale)))

    coords = analysis.coords[k]
    for a, b in SKELETON:
        pa, pb = coords[KP[a]], coords[KP[b]]
        if not (np.isnan(pa).any() or np.isnan(pb).any()):
            cv2.line(out, tuple(pa.astype(int)), tuple(pb.astype(int)), color, int(3 * scale))
    for point in coords:
        if not np.isnan(point).any():
            cv2.circle(out, tuple(point.astype(int)), int(4 * scale), (255, 255, 255), -1)

    image = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    font, small = _font(int(16 * scale)), _font(int(13 * scale))
    lines = [f"t = {t:4.1f} s · {analysis.backend}"]
    if analysis.exercise:
        reps = sum(1 for r in analysis.repetitions if r.end <= t)
        lines.append(f"{analysis.exercise} · repetições: {reps}")
        metrics = {r.metric for r in config.exercises[analysis.exercise].rules}
        for metric in sorted(metrics):
            value = row.get(metric)
            if value is not None and not np.isnan(value):
                lines.append(f"{METRIC_LABELS.get(metric, metric)}: {value:5.0f}°")
    box_h = int((len(lines) * 20 + 10) * scale)
    draw.rectangle((0, 0, int(260 * scale), box_h), fill=(0, 0, 0, 150))
    for i, text in enumerate(lines):
        draw.text((int(8 * scale), int((6 + i * 20) * scale)), text, font=small, fill="white")

    for i, finding in enumerate(sorted(active, key=lambda f: f.severity, reverse=True)[:3]):
        y = h - int((i + 1) * 30 * scale)
        draw.rectangle(
            (0, y, w, y + int(28 * scale)), fill=(*_rgb(SEVERITY_BGR[finding.severity]), 210)
        )
        label = _fit(
            draw, f"{finding.severity.value.upper()} · {finding.message}", font, w - int(20 * scale)
        )
        draw.text((int(10 * scale), y + int(5 * scale)), label, font=font, fill="white")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    # avc1 (H.264) abre no navegador; mp4v é o fallback disponível em qualquer build do OpenCV
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
    raise RuntimeError("Nenhum codec de vídeo disponível no OpenCV")


def transcode_to_h264(path: Path) -> bool:
    """Converte o MP4 (mp4v) para H.264 com o ffmpeg, para o vídeo abrir no navegador.

    O OpenCV do pip no Linux/Docker não traz o encoder H.264 e cai no mp4v, que os
    navegadores não reproduzem. O ffmpeg já está no Dockerfile. Retorna False sem o ffmpeg.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning(
            "{} gravado em mp4v (sem H.264 no OpenCV) e ffmpeg não encontrado: o vídeo pode "
            "não abrir no navegador",
            path.name,
        )
        return False
    converted = path.with_name(f"{path.stem}.h264{path.suffix}")
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        converted.unlink(missing_ok=True)
        logger.warning("ffmpeg falhou ao converter {}: {}", path.name, result.stderr.strip())
        return False
    converted.replace(path)
    return True


def render(
    analysis: VideoAnalysis,
    config: VideoConfig,
    output_dir: Path,
    stem: str,
    write_video: bool = True,
) -> RenderResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    times = analysis.timeline["time"].to_numpy()
    wanted = {
        i: int(np.searchsorted(times, finding.peak - 1e-6))
        for i, finding in enumerate(analysis.findings)
    }
    frames_needed = set(wanted.values())
    keyframes: dict[int, Path] = {}
    video_path = output_dir / f"{stem}_anotado.mp4" if write_video else None
    writer = None
    codec = ""
    for k, (_, _, frame) in enumerate(
        iter_sampled_frames(analysis.video, config.sample_fps, analysis.crop)
    ):
        if k >= len(analysis.timeline):
            break
        if not write_video and k not in frames_needed:
            continue
        annotated = draw_frame(frame, analysis, config, k)
        if video_path is not None:
            if writer is None:
                size = (annotated.shape[1], annotated.shape[0])
                writer, codec = _writer(video_path, config.sample_fps, size)
            writer.write(annotated)
        for i, frame_index in wanted.items():
            if frame_index == k:
                path = output_dir / f"{stem}_achado_{i + 1:02d}.png"
                cv2.imwrite(str(path), annotated)
                keyframes[i] = path
    if writer is not None:
        writer.release()
        if codec == "mp4v" and video_path is not None:
            transcode_to_h264(video_path)
    return RenderResult(video=video_path, keyframes=keyframes)
