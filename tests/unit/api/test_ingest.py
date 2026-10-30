from datetime import UTC, datetime

import pytest

from medwatch.api.ingest import audio_ingestor, video_ingestor
from medwatch.config import get_settings
from medwatch.video.pose import factory
from tests.unit.test_cli_audio import consulta, install_recording, mock_azure  # noqa: F401
from tests.unit.video.helpers import ScriptedPose, fall_script, write_video

START = datetime(2026, 10, 19, 17, 30, tzinfo=UTC)


def test_audio_ingestor_runs_the_audio_pipeline_in_mock_mode(mock_azure, consulta):  # noqa: F811
    install_recording(consulta, "consulta_dispneia")
    result = audio_ingestor(get_settings())(consulta, "PAC-001", START)
    assert "dyspnea_terms" in [e.type for e in result.events]
    assert all(e.patient_id == "PAC-001" for e in result.events)
    assert result.report.exists()


def test_video_ingestor_runs_the_video_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        factory, "build_pose_estimator", lambda *args, **kwargs: ScriptedPose(fall_script())
    )
    video = write_video(tmp_path / "quarto.mp4", seconds=5)
    result = video_ingestor(get_settings())(video, "PAC-002", START, None, None, "yolo")
    assert [e.type for e in result.events] == ["fall_detected"]
    assert result.events[0].timestamp > START
    assert result.report.name == "PAC-002_20261019T173000_scripted.html"
    assert result.report.exists()


def test_unknown_pose_backend():
    with pytest.raises(ValueError, match="Backend desconhecido"):
        factory.build_pose_estimator("resnet", get_settings().models_dir)
