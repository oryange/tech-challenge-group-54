import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from medwatch.core.events import Severity
from medwatch.video.rules import (
    Exercise,
    FallConfig,
    RepetitionConfig,
    Rule,
    count_repetitions,
    detect_falls,
    frame_findings,
    point_in_polygon,
    repetition_findings,
)
from tests.unit.video.helpers import load_config


def series(values, fps=10):
    times = pd.Series(np.arange(len(values)) / fps)
    return pd.DataFrame({"m": values}), times


def test_real_config_loads():
    config = load_config()
    assert {"agachamento", "agachamento_unipodal", "elevacao_bracos"} <= set(config.exercises)
    assert config.fall.lying_aspect_ratio > 1


def test_repetition_config_requires_one_direction():
    with pytest.raises(ValidationError):
        RepetitionConfig(metric="m", enter_below=100)
    with pytest.raises(ValidationError):
        RepetitionConfig(metric="m", enter_below=100, exit_above=150, enter_above=1, exit_below=0)


def test_frame_rule_needs_minimum_duration():
    rule = Rule(name="tronco", metric="m", max=40, severity=Severity.MEDIUM, message="x")
    metrics, times = series([10, 50, 10, 50, 55, 60, 50, 10])  # 1 quadro e depois 4 quadros
    findings = frame_findings(rule, metrics, times, min_seconds=0.3)
    assert len(findings) == 1
    finding = findings[0]
    assert (finding.start, finding.end, finding.peak) == (0.3, 0.6, 0.5)
    assert finding.evidence["value"] == 60
    assert finding.severity is Severity.MEDIUM


def test_frame_rule_min_limit_reports_lowest_value():
    rule = Rule(name="joelho", metric="m", min=60, message="x")
    metrics, times = series([90, 55, 46, 50, 90])
    [finding] = frame_findings(rule, metrics, times, min_seconds=0.2)
    assert finding.evidence["value"] == 46


def test_count_repetitions_descending_movement():
    config = RepetitionConfig(metric="m", enter_below=120, exit_above=155, min_seconds=0)
    metrics, times = series([170, 130, 100, 80, 110, 160, 170, 115, 90, 150, 165, 140])
    reps = count_repetitions(config, metrics, times)
    assert [r.extreme_value for r in reps] == [80, 90]
    assert reps[0].start == 0.2 and reps[0].end == 0.5


def test_count_repetitions_ascending_movement():
    config = RepetitionConfig(metric="m", enter_above=130, exit_below=60, min_seconds=0)
    metrics, times = series([20, 90, 140, 165, 100, 40, 30, 135, 150, 50])
    assert [r.extreme_value for r in count_repetitions(config, metrics, times)] == [165, 150]


def test_repetition_rule_flags_insufficient_amplitude():
    config = RepetitionConfig(metric="m", enter_below=120, exit_above=155, min_seconds=0)
    metrics, times = series([170, 110, 90, 160, 170, 115, 112, 160])
    reps = count_repetitions(config, metrics, times)
    rule = Rule(name="raso", metric="m", kind="repetition", target_max=100, message="raso")
    findings = repetition_findings(rule, reps)
    assert [f.evidence["repetition"] for f in findings] == [2]


def fall_inputs(drop_seconds: float, final_aspect: float, fps: int = 10):
    """Quadril desce 60% da altura em `drop_seconds` e a postura final tem `final_aspect`."""
    times = pd.Series(np.arange(0, 5, 1 / fps))
    progress = ((times - 1.0) / drop_seconds).clip(0, 1)
    hip_y = 100 + 60 * progress
    aspect = pd.Series(np.where(progress >= 1, final_aspect, 0.4))
    return times, hip_y, aspect


def test_fast_drop_ending_horizontal_is_a_fall():
    times, hip_y, aspect = fall_inputs(drop_seconds=0.4, final_aspect=2.2)
    [fall] = detect_falls(FallConfig(), times, hip_y, 100, aspect)
    assert fall.type == "fall_detected"
    assert fall.severity is Severity.CRITICAL
    assert 0.9 <= fall.start <= 1.1
    assert fall.evidence["posture_aspect_ratio"] == pytest.approx(2.2)


def test_slow_descent_is_not_a_fall_even_if_lying_down():
    times, hip_y, aspect = fall_inputs(drop_seconds=3.0, final_aspect=2.2)
    assert detect_falls(FallConfig(), times, hip_y, 100, aspect) == []


def test_fast_descent_staying_upright_is_not_a_fall():
    times, hip_y, aspect = fall_inputs(drop_seconds=0.4, final_aspect=0.8)  # agachar rápido
    assert detect_falls(FallConfig(), times, hip_y, 100, aspect) == []


def test_fall_tolerates_frames_without_person():
    times, hip_y, aspect = fall_inputs(drop_seconds=0.4, final_aspect=2.2)
    aspect.iloc[16:40:2] = np.nan  # a pose some em quadros alternados com a pessoa no chão
    assert len(detect_falls(FallConfig(), times, hip_y, 100, aspect)) == 1


def test_point_in_polygon():
    square = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    assert point_in_polygon(0.25, 0.25, square)
    assert not point_in_polygon(0.75, 0.25, square)


def test_exercise_rejects_repetition_target_that_can_never_fire():
    """enter_below 120 x target_max 130: toda repetição contada já atinge a meta."""
    with pytest.raises(ValidationError, match="nunca dispara"):
        Exercise(
            description="x",
            repetition=RepetitionConfig(metric="m", enter_below=120, exit_above=155),
            rules=[Rule(name="raso", kind="repetition", metric="m", target_max=130, message="x")],
        )
    with pytest.raises(ValidationError, match="nunca dispara"):
        Exercise(
            description="x",
            repetition=RepetitionConfig(metric="m", enter_above=150, exit_below=60),
            rules=[Rule(name="baixo", kind="repetition", metric="m", target_min=150, message="x")],
        )


def test_exercise_rejects_repetition_rule_without_matching_repetition():
    with pytest.raises(ValidationError, match="precisa de `repetition`"):
        Exercise(
            description="x",
            rules=[Rule(name="raso", kind="repetition", metric="m", target_max=130, message="x")],
        )


def test_partial_arm_raise_counts_and_is_flagged_with_real_config():
    exercise = load_config().exercises["elevacao_bracos"]
    assert exercise.repetition is not None
    # sobe até 125° (abaixo da meta de 150°) e desce, duas vezes
    values = [30, 60, 90, 115, 125, 115, 80, 40, 30, 70, 110, 125, 100, 50, 30]
    metrics, times = series(values, fps=4)  # ~1 s por elevação
    metrics = metrics.rename(columns={"m": exercise.repetition.metric})
    reps = count_repetitions(exercise.repetition, metrics, times)
    assert len(reps) == 2
    [rule] = [r for r in exercise.rules if r.name == "amplitude_insuficiente"]
    assert len(repetition_findings(rule, reps)) == 2


def test_short_low_confidence_gap_does_not_split_a_deviation():
    rule = Rule(name="tronco", metric="m", max=40, severity=Severity.MEDIUM, message="x")
    # 0,7 s acima do limite com um quadro sem keypoint no meio
    metrics, times = series([10, 50, 52, 55, np.nan, 55, 52, 50, 10])
    assert frame_findings(rule, metrics, times, min_seconds=0.3) == []  # sem interpolação
    [finding] = frame_findings(rule, metrics, times, min_seconds=0.3, max_gap_frames=2)
    assert finding.evidence["duration_s"] == pytest.approx(0.6)


def test_long_gap_is_not_interpolated():
    rule = Rule(name="tronco", metric="m", max=40, severity=Severity.MEDIUM, message="x")
    metrics, times = series([10, 50, 55, np.nan, np.nan, np.nan, 55, 50, 10])
    assert frame_findings(rule, metrics, times, min_seconds=0.3, max_gap_frames=2) == []


def test_wide_box_cut_at_the_top_is_not_a_fall():
    """Pessoa perto da câmera: a caixa cortada em cima fica larga sem a pessoa estar deitada."""
    times, hip_y, aspect = fall_inputs(drop_seconds=0.4, final_aspect=2.2)
    at_edge = pd.Series(aspect > 1)
    trunk = pd.Series(10.0, index=times.index)
    assert detect_falls(FallConfig(), times, hip_y, 100, aspect, trunk, at_edge) == []


def test_box_cut_at_the_top_counts_when_trunk_is_tilted():
    times, hip_y, aspect = fall_inputs(drop_seconds=0.4, final_aspect=2.2)
    at_edge = pd.Series(aspect > 1)
    trunk = pd.Series(np.where(aspect > 1, 80.0, 5.0))
    [fall] = detect_falls(FallConfig(), times, hip_y, 100, aspect, trunk, at_edge)
    assert fall.evidence["trunk_lean"] == pytest.approx(80.0)


def test_upright_trunk_in_image_does_not_veto_a_fall():
    """Queda em direção à câmera: o tronco deitado aparece vertical na imagem (2D)."""
    times, hip_y, aspect = fall_inputs(drop_seconds=0.4, final_aspect=2.2)
    trunk = pd.Series(3.0, index=times.index)
    at_edge = pd.Series(False, index=times.index)
    assert len(detect_falls(FallConfig(), times, hip_y, 100, aspect, trunk, at_edge)) == 1


def test_repetition_shorter_than_minimum_is_ignored():
    """O vídeo começa com a pessoa terminando de subir: 0,2 s não é uma repetição."""
    config = RepetitionConfig(metric="m", enter_below=145, exit_above=155)
    metrics, times = series([143, 150, 157, 165, 170, 120, 90, 70, 90, 120, 160, 170])
    reps = count_repetitions(config, metrics, times)
    assert [r.extreme_value for r in reps] == [70]


def test_repetition_already_in_progress_at_start_still_counts():
    """Vídeo que começa com a pessoa já no fundo: a repetição é real e conta."""
    config = RepetitionConfig(metric="m", enter_below=145, exit_above=155)
    metrics, times = series([60, 55, 70, 100, 130, 160, 170])
    assert [r.extreme_value for r in count_repetitions(config, metrics, times)] == [55]
