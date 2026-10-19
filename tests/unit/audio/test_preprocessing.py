import numpy as np
import pytest

from medwatch.audio.config import PreprocessingConfig
from medwatch.audio.preprocessing import AudioValidationError, load_audio
from tests.unit.audio.helpers import clip, voiced, write_wav

CONFIG = PreprocessingConfig(min_seconds=1, max_seconds=60, allowed_extensions=[".wav", ".flac"])


def test_stereo_44k_vira_mono_16k_com_mesma_duracao(tmp_path):
    mono = voiced(2.0, sr=44100)
    stereo = np.stack([mono, mono * 0.5], axis=1)
    path = write_wav(tmp_path / "consulta.wav", stereo, sr=44100)

    result = load_audio(path, CONFIG)

    assert result.sample_rate == 16000
    assert result.samples.ndim == 1
    assert result.duration == pytest.approx(2.0, abs=0.01)


def test_flac_e_aceito(tmp_path):
    path = write_wav(tmp_path / "consulta.flac", voiced(1.5))
    assert load_audio(path, CONFIG).duration == pytest.approx(1.5, abs=0.01)


@pytest.mark.parametrize("name", ["consulta.m4a", "consulta", "consulta.wav.exe"])
def test_extensao_fora_da_lista_e_recusada(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"0" * 100)
    with pytest.raises(AudioValidationError, match="não aceito"):
        load_audio(path, CONFIG)


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(AudioValidationError, match="não encontrado"):
        load_audio(tmp_path / "nada.wav", CONFIG)


def test_arquivo_grande_demais(tmp_path):
    path = write_wav(tmp_path / "consulta.wav", voiced(2.0))
    with pytest.raises(AudioValidationError, match="excede"):
        load_audio(path, CONFIG.model_copy(update={"max_file_mb": 0.001}))


def test_conteudo_que_nao_e_audio(tmp_path):
    path = tmp_path / "consulta.wav"
    path.write_bytes(b"isto nao e um wav")
    with pytest.raises(AudioValidationError, match="decodificar"):
        load_audio(path, CONFIG)


def test_duracao_minima_e_maxima(tmp_path):
    short = write_wav(tmp_path / "curto.wav", voiced(0.5))
    with pytest.raises(AudioValidationError, match="mínimo"):
        load_audio(short, CONFIG)
    long = write_wav(tmp_path / "longo.wav", voiced(3.0))
    with pytest.raises(AudioValidationError, match="máximo"):
        load_audio(long, CONFIG.model_copy(update={"max_seconds": 2}))


def test_silencio_e_recusado(tmp_path):
    path = write_wav(tmp_path / "mudo.wav", np.zeros(32000, dtype=np.float32))
    with pytest.raises(AudioValidationError, match="silêncio"):
        load_audio(path, CONFIG)


def test_pcm16_com_saturacao():
    pcm = clip(np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)).to_pcm16()
    values = np.frombuffer(pcm, dtype="<i2")
    assert values.tolist() == [0, 32767, -32767, 32767]
