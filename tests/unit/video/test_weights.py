from pathlib import Path

from medwatch.video.pose import base
from medwatch.video.pose.base import ULTRALYTICS_ASSETS_URL, YOLO_WEIGHTS_SHA256, yolo_weights


def test_known_yolo_weights_are_downloaded_with_pinned_hash(monkeypatch, tmp_path):
    calls = []

    def fake_download(url: str, path: Path, expected: str) -> Path:
        calls.append((url, path, expected))
        return path

    monkeypatch.setattr(base, "download_file", fake_download)
    path = yolo_weights("yolov8n-pose.pt", tmp_path)
    assert path == tmp_path / "yolov8n-pose.pt"
    assert calls == [
        (
            f"{ULTRALYTICS_ASSETS_URL}/yolov8n-pose.pt",
            tmp_path / "yolov8n-pose.pt",
            YOLO_WEIGHTS_SHA256["yolov8n-pose.pt"],
        )
    ]


def test_unknown_weights_are_used_as_is(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "download_file", lambda *args: (_ for _ in ()).throw(AssertionError))
    assert yolo_weights("meu_modelo.pt", tmp_path) == tmp_path / "meu_modelo.pt"
