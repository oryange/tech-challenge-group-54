import pytest
from pydantic import ValidationError

from medwatch.audio.acoustic import AcousticFeatures
from medwatch.audio.config import AudioConfig, IndicatorSeverityConfig
from medwatch.audio.indicators import baseline_from_features, score_indicators
from medwatch.core.events import Severity
from tests.unit.audio.helpers import CONFIGS, audio_config


def features(**values):
    base = {
        "duration_seconds": 60.0,
        "speech_seconds": 45.0,
        "pause_count": 12,
        "pause_ratio": 0.2,
        "speech_rate_wpm": 150,
        "articulation_rate_wpm": 190,
        "f0_std_semitones": 3.0,
        "mean_pause_seconds": 0.6,
        "pauses_per_minute": 12,
        "mean_phrase_seconds": 3.0,
        "jitter_local_pct": 0.5,
        "shimmer_local_pct": 3.0,
        "hnr_db": 18,
    }
    return AcousticFeatures(**{**base, **values})


def by_name(results):
    return {r.name: r for r in results}


def test_consulta_no_baseline_nao_gera_indicador():
    results = by_name(score_indicators(features(), audio_config()))
    assert set(results) == {"fatigue", "dysarthria", "respiratory_distress"}
    for result in results.values():
        assert result.zscore == pytest.approx(0.0)
        assert result.severity is None


def test_fala_lenta_monotona_e_com_pausas_longas_indica_fadiga():
    slow = features(speech_rate_wpm=90, f0_std_semitones=1.0, mean_pause_seconds=1.3)
    fatigue = by_name(score_indicators(slow, audio_config()))["fatigue"]
    # z orientados: (150-90)/25=2.4, (3-1)/1=2.0, (1.3-0.6)/0.25=2.8 → média 2.4
    assert fatigue.zscore == pytest.approx(2.4, abs=0.01)
    assert fatigue.severity == Severity.MEDIUM
    assert fatigue.features["speech_rate_wpm"].zscore == pytest.approx(2.4)


def test_feature_melhor_que_o_baseline_nao_compensa_outra():
    # pausas frequentes e frases curtas, mas HNR muito acima do baseline
    mixed = features(pauses_per_minute=24, mean_phrase_seconds=1.2, hnr_db=40)
    result = by_name(score_indicators(mixed, audio_config()))["respiratory_distress"]
    # (2.4 + 1.5 + 0.5 * 0) / 2.5; sem o piso, o HNR (z=-5.5) zeraria o indicador
    assert result.zscore == pytest.approx(1.56, abs=0.01)
    assert result.severity == Severity.LOW
    assert result.features["hnr_db"].zscore == pytest.approx(-5.5)


def test_direcao_oposta_nao_indica():
    fast = features(speech_rate_wpm=200, f0_std_semitones=5.0, mean_pause_seconds=0.3)
    assert by_name(score_indicators(fast, audio_config()))["fatigue"].severity is None


def test_termo_relacionado_sobe_um_nivel():
    breathy = features(pauses_per_minute=21, mean_phrase_seconds=1.2, hnr_db=12)
    config = audio_config()
    alone = by_name(score_indicators(breathy, config))["respiratory_distress"]
    with_terms = by_name(score_indicators(breathy, config, term_categories={"dyspnea"}))[
        "respiratory_distress"
    ]
    assert alone.severity == Severity.LOW
    assert with_terms.severity == Severity.MEDIUM
    assert with_terms.related_terms_present


def test_escalonamento_para_em_high():
    severe = features(speech_rate_wpm=50, f0_std_semitones=0.2, mean_pause_seconds=2.0)
    fatigue = by_name(score_indicators(severe, audio_config(), term_categories={"fatigue"}))
    assert fatigue["fatigue"].severity == Severity.HIGH


def test_sem_features_suficientes_nao_calcula():
    sparse = features(speech_rate_wpm=None, f0_std_semitones=None)
    fatigue = by_name(score_indicators(sparse, audio_config()))["fatigue"]
    assert fatigue.zscore is None
    assert fatigue.severity is None
    assert fatigue.missing_features == ["f0_std_semitones", "speech_rate_wpm"]


def test_baseline_do_paciente_compara_com_ele_mesmo():
    config = audio_config()
    reference = features(speech_rate_wpm=110, f0_std_semitones=2.0, mean_pause_seconds=0.9)
    baseline = baseline_from_features(reference, config.baseline)
    assert baseline["speech_rate_wpm"].mean == 110
    assert baseline["speech_rate_wpm"].std == config.baseline["speech_rate_wpm"].std

    # contra a população seria fadiga; contra o próprio paciente, não
    same_as_usual = features(speech_rate_wpm=105, f0_std_semitones=1.9, mean_pause_seconds=0.95)
    population = by_name(score_indicators(same_as_usual, config))["fatigue"]
    patient = by_name(score_indicators(same_as_usual, config, baseline))["fatigue"]
    assert population.zscore > patient.zscore
    assert patient.severity is None


def test_limiares_precisam_ser_crescentes():
    with pytest.raises(ValidationError, match="crescentes"):
        IndicatorSeverityConfig(low=2.0, medium=1.5, high=3.0)


def test_config_recusa_feature_desconhecida_e_baseline_incompleto():
    import yaml

    raw = yaml.safe_load((CONFIGS / "audio.yaml").read_text("utf-8"))
    raw["indicators"]["fatigue"]["features"]["volume_db"] = {"direction": 1}
    with pytest.raises(ValidationError, match="desconhecidas"):
        AudioConfig.model_validate(raw)

    raw = yaml.safe_load((CONFIGS / "audio.yaml").read_text("utf-8"))
    del raw["baseline"]["hnr_db"]
    with pytest.raises(ValidationError, match="baseline"):
        AudioConfig.model_validate(raw)
