import json
import shutil

import pytest
from typer.testing import CliRunner

from medwatch.cli import app
from medwatch.commands.audio import parse_start_time, recording_key
from medwatch.config import get_settings
from tests.unit.audio.helpers import FIXTURES, phrases, write_wav

runner = CliRunner()


@pytest.fixture
def mock_azure(monkeypatch):
    monkeypatch.setenv("MEDWATCH_AZURE_MOCK", "true")
    get_settings.cache_clear()


@pytest.fixture
def real_azure(monkeypatch):
    # o CI exporta MEDWATCH_AZURE_MOCK=true; estes testes cobrem o caminho sem mock
    monkeypatch.setenv("MEDWATCH_AZURE_MOCK", "false")
    get_settings.cache_clear()


@pytest.fixture
def consulta(tmp_path):
    samples = phrases(4, phrase_seconds=2.5, pause_seconds=0.5, vibrato_semitones=4.0)
    return write_wav(tmp_path / "consulta.wav", samples)


def install_recording(audio_file, scenario: str) -> None:
    """Copia as respostas gravadas do cenário para data/samples/azure/<hash do áudio>."""
    target = get_settings().data_dir / "samples" / "azure"
    target.mkdir(parents=True, exist_ok=True)
    key = recording_key(audio_file)
    shutil.copy(FIXTURES / f"{scenario}.stt.json", target / f"{key}.stt.json")
    shutil.copy(FIXTURES / f"{scenario}.language.json", target / f"{key}.language.json")


def test_analyze_em_modo_mock_gera_relatorio_e_alerta(mock_azure, consulta):
    install_recording(consulta, "consulta_dispneia")
    result = runner.invoke(
        app,
        [
            "audio",
            "analyze",
            str(consulta),
            "--patient",
            "PAC-001",
            "--start-time",
            "2026-10-19T14:30:00-03:00",
            "--alert",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "dyspnea_terms" in result.output
    assert "Em perguntas (sem evento): dor no peito" in result.output
    assert "Resumo (extrativo): Acordei várias vezes com falta de ar." in result.output
    assert "Frases-chave: falta de ar, dormir sentada" in result.output
    report = get_settings().outputs_dir / "reports" / "audio_PAC-001_20261019T173000Z.json"
    data = json.loads(report.read_text("utf-8"))
    assert data["mock"] is True
    alerts = get_settings().alerts_file.read_text("utf-8").splitlines()
    assert any(json.loads(line)["alert_type"] == "audio.dyspnea_terms" for line in alerts)


def test_consulta_normal_em_modo_mock(mock_azure, consulta):
    install_recording(consulta, "consulta_normal")
    result = runner.invoke(app, ["audio", "analyze", str(consulta), "--patient", "PAC-002"])
    assert result.exit_code == 0, result.output
    assert "Termos críticos afirmados: nenhum" in result.output
    assert "Negados (sem evento): falta de ar" in result.output


def test_mock_sem_resposta_gravada(mock_azure, consulta):
    result = runner.invoke(app, ["audio", "analyze", str(consulta), "--patient", "PAC-001"])
    assert result.exit_code == 1
    assert "--record" in result.output


def test_record_exige_azure_real(mock_azure, consulta):
    result = runner.invoke(
        app, ["audio", "analyze", str(consulta), "--patient", "PAC-001", "--record"]
    )
    assert result.exit_code == 1
    assert "MEDWATCH_AZURE_MOCK=false" in result.output


def test_sem_azure_configurado(real_azure, consulta):
    result = runner.invoke(app, ["audio", "analyze", str(consulta), "--patient", "PAC-001"])
    assert result.exit_code == 1
    assert "AZURE_SPEECH_KEY" in result.output


def test_paciente_com_nome_e_recusado(mock_azure, consulta):
    install_recording(consulta, "consulta_normal")
    result = runner.invoke(app, ["audio", "analyze", str(consulta), "--patient", "Maria Silva"])
    assert result.exit_code == 1
    assert "pseudonimizado" in result.output


def test_formato_nao_aceito(mock_azure, tmp_path):
    path = tmp_path / "consulta.m4a"
    path.write_bytes(b"0" * 10)
    result = runner.invoke(app, ["audio", "analyze", str(path), "--patient", "PAC-001"])
    assert result.exit_code == 1
    assert "não aceito" in result.output


def test_start_time():
    assert parse_start_time("2026-10-19T14:30:00-03:00").utcoffset().total_seconds() == -10800
    assert parse_start_time(None).tzinfo is not None
    with pytest.raises(ValueError, match="fuso"):
        parse_start_time("2026-10-19T14:30:00")
    with pytest.raises(ValueError, match="ISO 8601"):
        parse_start_time("ontem")


def test_recording_key_depende_do_conteudo(tmp_path):
    a = write_wav(tmp_path / "a.wav", phrases(1, 1.0, 0.1))
    b = tmp_path / "b.wav"
    shutil.copy(a, b)
    c = write_wav(tmp_path / "c.wav", phrases(2, 1.0, 0.1))
    assert recording_key(a) == recording_key(b) != recording_key(c)


def test_synth_samples_exige_azure_real(real_azure):
    result = runner.invoke(app, ["audio", "synth-samples"])
    assert result.exit_code == 1
    assert "AZURE_SPEECH_KEY" in result.output


def test_synth_samples_amostra_desconhecida(real_azure, monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "chave-fake")
    get_settings.cache_clear()
    result = runner.invoke(app, ["audio", "synth-samples", "--name", "nao_existe"])
    assert result.exit_code == 1
    assert "nao_existe" in result.output
