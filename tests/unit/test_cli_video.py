import json

import pytest
from typer.testing import CliRunner

from medwatch.cli import app
from medwatch.commands import video as video_commands
from medwatch.config import get_settings
from tests.unit.video.helpers import ScriptedPose, fall_script, write_video

runner = CliRunner()


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(
        video_commands, "_pose", lambda settings, backend: ScriptedPose(fall_script())
    )


def test_video_analyze_end_to_end(tmp_path, scripted):
    video = write_video(tmp_path / "queda.mp4", seconds=5)
    result = runner.invoke(app, ["video", "analyze", str(video), "--alert"])
    assert result.exit_code == 0, result.output
    assert "Queda detectada" in result.stdout
    assert "Alertas enviados: 1" in result.stdout
    reports = get_settings().outputs_dir / "reports" / "video"
    payload = json.loads((reports / "queda_scripted.json").read_text(encoding="utf-8"))
    assert payload["events"][0]["type"] == "fall_detected"
    assert (reports / "queda_scripted.html").exists()
    assert (reports / "queda_scripted_anotado.mp4").exists()


def test_video_analyze_missing_file():
    result = runner.invoke(app, ["video", "analyze", "nao_existe.mp4"])
    assert result.exit_code == 1
    assert "medwatch data download --dataset videos" in result.output


def test_video_analyze_unknown_exercise(tmp_path, scripted):
    video = write_video(tmp_path / "clip.mp4", seconds=1)
    result = runner.invoke(app, ["video", "analyze", str(video), "--exercise", "nado"])
    assert result.exit_code == 1


def test_unknown_backend():
    result = runner.invoke(app, ["video", "evaluate-falls"])
    assert result.exit_code == 1  # dataset ausente
    assert "--dataset urfall" in result.output


def test_data_download_unknown_dataset():
    result = runner.invoke(app, ["data", "download", "--dataset", "nada"])
    assert result.exit_code == 1
