import pytest

from medwatch.audio.acoustic import extract_features, pause_features, speech_intervals
from medwatch.audio.config import AcousticConfig
from tests.unit.audio.helpers import clip, phrases, voiced

CONFIG = AcousticConfig()


@pytest.mark.parametrize("f0", [120.0, 150.0, 220.0])
def test_f0_de_tom_sintetico_dentro_de_5_por_cento(f0):
    features = extract_features(clip(voiced(2.0, f0=f0)), CONFIG)
    assert features.f0_mean_hz == pytest.approx(f0, rel=0.05)


def test_tom_estavel_tem_jitter_baixo_e_hnr_alto():
    features = extract_features(clip(voiced(2.0)), CONFIG)
    assert features.jitter_local_pct is not None and features.jitter_local_pct < 1.0
    assert features.hnr_db is not None and features.hnr_db > 15


def test_vibrato_aumenta_variacao_de_f0_em_semitons():
    flat = extract_features(clip(voiced(4.0)), CONFIG)
    varied = extract_features(clip(voiced(4.0, vibrato_semitones=4.0)), CONFIG)
    assert flat.f0_std_semitones is not None and varied.f0_std_semitones is not None
    assert flat.f0_std_semitones < 0.5
    assert varied.f0_std_semitones == pytest.approx(4.0 / 2**0.5, rel=0.2)


def test_pausas_e_frases():
    features = extract_features(clip(phrases(3, phrase_seconds=1.0, pause_seconds=0.8)), CONFIG)
    assert features.pause_count == 2
    assert features.mean_pause_seconds == pytest.approx(0.8, abs=0.1)
    assert features.mean_phrase_seconds == pytest.approx(1.0, abs=0.15)
    # 2 pausas em ~4,6 s entre o início da primeira frase e o fim da última
    assert features.pauses_per_minute == pytest.approx(2 / (4.6 / 60), rel=0.1)


def test_silencio_curto_nao_conta_como_pausa():
    intervals = speech_intervals(clip(phrases(3, phrase_seconds=1.0, pause_seconds=0.1)), CONFIG)
    assert len(intervals) == 1


def test_taxas_de_fala_usam_palavras_da_transcricao():
    audio = clip(phrases(2, phrase_seconds=1.5, pause_seconds=1.0))
    features = extract_features(audio, CONFIG, word_count=10)
    # 10 palavras em 4 s de extensão (3 s de fala)
    assert features.speech_rate_wpm == pytest.approx(150, rel=0.1)
    assert features.articulation_rate_wpm == pytest.approx(200, rel=0.1)
    assert extract_features(audio, CONFIG).speech_rate_wpm is None


def test_voz_curta_demais_nao_gera_features_de_voz():
    import numpy as np

    # 50 ms de voz: menos de 10 quadros de pitch, F0 e perturbações não são confiáveis
    audio = np.concatenate([voiced(0.05), np.zeros(16000, dtype=np.float32)])
    features = extract_features(clip(audio), CONFIG)
    assert features.f0_mean_hz is None
    assert features.jitter_local_pct is None
    assert features.hnr_db is None


def test_pause_features_sem_fala():
    result = pause_features([])
    assert result["pause_count"] == 0
    assert result["pauses_per_minute"] is None
