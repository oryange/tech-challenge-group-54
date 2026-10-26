import json
import shutil
from datetime import UTC, datetime, timedelta

import cv2
import pandas as pd
import pytest

from medwatch.core.events import Severity, Source
from medwatch.video.objects import Detection
from medwatch.video.pipeline import VideoAnalyzer, build_events, choose_person, iter_sampled_frames
from medwatch.video.render import render
from medwatch.video.report import plot_timeline, write_html, write_json, write_pose_series
from medwatch.video.rules import Finding
from tests.unit.video.helpers import (
    ScriptedPose,
    fall_script,
    load_config,
    person,
    sit_script,
    skeleton,
    squat_script,
    write_video,
)

CONFIG = load_config()


@pytest.fixture
def video(tmp_path):
    return write_video(tmp_path / "clip.mp4", seconds=7)


def test_sampling_respects_sample_fps_and_crop(video):
    frames = list(iter_sampled_frames(video, sample_fps=10, crop="right"))
    assert len(frames) == 70
    assert frames[1][1] == pytest.approx(0.1)
    assert frames[0][2].shape == (320, 200, 3)


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_sampled_frames(tmp_path / "nao_existe.mp4", 10))


def test_choose_person_keeps_track_of_the_same_person():
    near = person(skeleton(hip=(100, 200)))
    big_far = person(skeleton(hip=(380, 200)))
    big_far.bbox = (300, 0, 400, 320)
    assert choose_person([near, big_far], previous=(100, 180), width=400) is near
    assert choose_person([near, big_far], previous=None, width=400) is big_far
    assert choose_person([], previous=None, width=400) is None


def test_correct_squats_count_reps_without_deviations(video):
    analysis = VideoAnalyzer(CONFIG, ScriptedPose(squat_script(bottom=85, lean=20))).analyze(
        video, exercise="agachamento"
    )
    assert len(analysis.repetitions) == 3
    assert analysis.findings == []
    assert analysis.deviation_ratio == 0


def test_incorrect_squats_generate_deviations(video):
    script = squat_script(bottom=50, lean=60)  # fundo demais e tronco muito inclinado
    analysis = VideoAnalyzer(CONFIG, ScriptedPose(script)).analyze(video, exercise="agachamento")
    types = {f.type for f in analysis.findings}
    assert {"flexao_joelho_excessiva", "inclinacao_tronco_excessiva"} <= types
    assert len([f for f in analysis.findings if f.type == "flexao_joelho_excessiva"]) == 3
    assert 0 < analysis.deviation_ratio < 1


def test_fall_detected_but_not_slow_sitting(video):
    fall = VideoAnalyzer(CONFIG, ScriptedPose(fall_script())).analyze(video)
    assert [f.type for f in fall.findings] == ["fall_detected"]
    # a queda começa em 1,0 s; a mediana móvel causal (5 quadros) atrasa ~0,2–0,3 s
    assert 1.0 <= fall.findings[0].start <= 1.4

    for script in (sit_script(), sit_script(lie=True)):
        calm = VideoAnalyzer(CONFIG, ScriptedPose(script)).analyze(video)
        assert calm.findings == []


def test_unknown_exercise_or_scene(video):
    analyzer = VideoAnalyzer(CONFIG, ScriptedPose(squat_script()))
    with pytest.raises(ValueError, match="Exercício"):
        analyzer.analyze(video, exercise="nado")
    with pytest.raises(ValueError, match="Cena"):
        analyzer.analyze(video, scene="praia")


class FakeObjects:
    def detect(self, frame):
        return [Detection(label="person", confidence=0.9, bbox=(5, 100, 40, 300))]


def test_scene_rois_use_object_detections(video):
    analyzer = VideoAnalyzer(CONFIG, ScriptedPose(squat_script()), FakeObjects())
    analysis = analyzer.analyze(video, scene="quarto")
    assert [f.type for f in analysis.findings] == ["roi_area_restrita"]


def test_render_report_and_events(video, tmp_path):
    analysis = VideoAnalyzer(CONFIG, ScriptedPose(squat_script(bottom=50, lean=60))).analyze(
        video, exercise="agachamento"
    )
    out = tmp_path / "reports"
    rendered = render(analysis, CONFIG, out, "clip")
    assert rendered.video is not None and rendered.video.exists()
    assert cv2.VideoCapture(str(rendered.video)).get(cv2.CAP_PROP_FRAME_COUNT) == 70
    assert set(rendered.keyframes) == set(range(len(analysis.findings)))

    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    events = build_events(analysis, "PAC-001", start_time=t0, keyframes=rendered.keyframes)
    first = events[0]
    assert first.source is Source.VIDEO
    assert first.timestamp == t0 + timedelta(seconds=analysis.findings[0].peak)
    assert first.artifacts == (rendered.keyframes[0],)
    assert first.evidence["video"] == "clip.mp4"
    assert {e.severity for e in events} <= {Severity.MEDIUM, Severity.LOW}

    png = plot_timeline(analysis, CONFIG, out / "metricas.png")
    page = write_html(analysis, events, png, out / "clip.html").read_text(encoding="utf-8")
    assert "Flexão de joelho além do limite seguro" in page
    assert "data:image/png;base64," in page
    payload = json.loads(write_json(analysis, events, out / "clip.json").read_text())
    assert payload["summary"]["repetitions"] == 3
    assert len(payload["events"]) == len(events)
    pose = pd.read_parquet(write_pose_series(analysis, out / "pose.parquet"))
    assert {"time", "knee_angle_min", "kp_left_knee_x"} <= set(pose.columns)


def test_render_only_keyframes(video, tmp_path):
    analysis = VideoAnalyzer(CONFIG, ScriptedPose(fall_script())).analyze(video)
    rendered = render(analysis, CONFIG, tmp_path, "queda", write_video=False)
    assert rendered.video is None
    assert rendered.keyframes[0].exists()


def test_shallow_squats_are_counted_and_flagged_with_real_config(video):
    """Agachamento raso (fundo de 138°) não pode aparecer como execução correta."""
    script = squat_script(bottom=138, lean=10)
    analysis = VideoAnalyzer(CONFIG, ScriptedPose(script)).analyze(video, exercise="agachamento")
    assert len(analysis.repetitions) == 3
    shallow = [f for f in analysis.findings if f.type == "profundidade_insuficiente"]
    assert len(shallow) == 3


def test_instrument_count_does_not_count_as_exercise_deviation(video):
    analysis = VideoAnalyzer(CONFIG, ScriptedPose(squat_script())).analyze(
        video, exercise="agachamento"
    )
    assert analysis.findings == []
    times = analysis.timeline["time"]
    analysis.findings.append(
        Finding(
            type="instrument_count_mismatch",
            severity=Severity.HIGH,
            message="x",
            start=float(times.iloc[0]),
            end=float(times.iloc[-1]),
            peak=float(times.iloc[-1]),
            score=0.9,
        )
    )
    assert analysis.deviation_ratio == 0


def test_transcode_without_ffmpeg_keeps_file(monkeypatch, tmp_path):
    from medwatch.video import render as render_module

    clip = write_video(tmp_path / "clip.mp4", seconds=1)
    before = clip.read_bytes()
    monkeypatch.setattr(render_module.shutil, "which", lambda name: None)
    assert render_module.transcode_to_h264(clip) is False
    assert clip.read_bytes() == before


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg não instalado")
def test_transcode_converts_to_h264(tmp_path):
    from medwatch.video.render import transcode_to_h264

    clip = write_video(tmp_path / "clip.mp4", seconds=1)
    assert transcode_to_h264(clip) is True
    capture = cv2.VideoCapture(str(clip))
    codec = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    capture.release()
    assert codec in {"avc1", "h264"}
