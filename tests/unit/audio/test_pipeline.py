import json
from datetime import UTC, datetime, timedelta

import pytest

from medwatch.audio.config import CriticalTermsConfig
from medwatch.audio.pipeline import AudioAnalyzer, max_severity
from medwatch.audio.report import load_baseline, write_report
from medwatch.core.events import Severity, Source
from tests.unit.audio.helpers import (
    FakeLanguage,
    FakeTranscriber,
    audio_config,
    load_language,
    load_transcript,
    phrases,
    terms_config,
    write_wav,
)

START = datetime(2026, 10, 19, 14, 30, tzinfo=UTC)


@pytest.fixture
def audio_file(tmp_path):
    # 4 frases de 2,5 s com pausas de 0,5 s e F0 variando como em fala natural
    samples = phrases(4, phrase_seconds=2.5, pause_seconds=0.5, vibrato_semitones=4.0)
    return write_wav(tmp_path / "consulta.wav", samples)


def analyzer(scenario: str) -> AudioAnalyzer:
    return AudioAnalyzer(
        audio_config(),
        terms_config(),
        FakeTranscriber(load_transcript(scenario)),
        FakeLanguage(load_language(scenario)),
    )


def test_consulta_dispneica_gera_evento_medium_ou_maior(audio_file):
    report = analyzer("consulta_dispneia").analyze(audio_file, "PAC-001", START)

    types = {e.type: e for e in report.events}
    assert "dyspnea_terms" in types
    assert "fatigue_terms" in types
    assert max_severity(report.events) >= Severity.MEDIUM
    dyspnea = types["dyspnea_terms"]
    assert dyspnea.source == Source.AUDIO
    assert dyspnea.severity == Severity.MEDIUM
    assert dyspnea.evidence["terms"] == ["cansaco para respirar", "falta de ar", "sem ar"]
    assert dyspnea.evidence["sentiment"] == "mixed"
    # a dor no peito foi perguntada e negada: nenhum evento de dor torácica
    assert "chest_pain_terms" not in types
    chest_pain = [m.status for m in report.critical_terms if m.category == "chest_pain"]
    assert chest_pain == ["question", "negated"]


def test_consulta_normal_nao_gera_evento_medium(audio_file):
    report = analyzer("consulta_normal").analyze(audio_file, "PAC-002", START)

    assert all(e.severity < Severity.MEDIUM for e in report.events)
    assert not [e for e in report.events if e.type.endswith("_terms")]
    assert {m.status for m in report.critical_terms} <= {"negated", "question"}


def test_evento_e_datado_pelo_inicio_da_gravacao(audio_file):
    report = analyzer("consulta_dispneia").analyze(audio_file, "PAC-001", START)
    dyspnea = next(e for e in report.events if e.type == "dyspnea_terms")
    transcript = load_transcript("consulta_dispneia")
    first = transcript.text.index("falta de ar")
    assert dyspnea.timestamp == START + timedelta(seconds=transcript.time_at(first))
    assert dyspnea.timestamp.tzinfo is not None


def test_evidencia_nao_carrega_trechos_da_transcricao(audio_file):
    report = analyzer("consulta_dispneia").analyze(audio_file, "PAC-001", START)
    evidence = json.dumps([e.evidence for e in report.events], ensure_ascii=False)
    for segment in load_transcript("consulta_dispneia").segments:
        assert segment.text not in evidence
    # resumo e frases-chave ficam no relatório local, nunca nos eventos/alertas
    assert report.language.summary is not None
    for sentence in report.language.summary.sentences:
        assert sentence not in evidence
    assert "dormir sentada" not in evidence


def test_transcricao_vazia_nao_chama_o_language(audio_file):
    from medwatch.audio.azure_language import LanguageAnalysis
    from medwatch.audio.transcript import Transcript

    language = FakeLanguage(LanguageAnalysis())
    pipeline = AudioAnalyzer(
        audio_config(), terms_config(), FakeTranscriber(Transcript(language="pt-BR")), language
    )
    report = pipeline.analyze(audio_file, "PAC-001", START)
    assert language.texts == []
    assert report.language.health_available is False
    assert report.critical_terms == []


@pytest.mark.parametrize("patient", ["Maria Silva", "../../etc", "", "a" * 65])
def test_patient_id_precisa_ser_pseudonimizado(audio_file, patient):
    with pytest.raises(ValueError, match="pseudonimizado"):
        analyzer("consulta_normal").analyze(audio_file, patient, START)


def test_horario_sem_fuso_e_recusado(audio_file):
    with pytest.raises(ValueError, match="fuso"):
        analyzer("consulta_normal").analyze(audio_file, "PAC-001", datetime(2026, 10, 19, 14))


def test_related_terms_precisa_existir():
    terms = terms_config()
    without_dyspnea = CriticalTermsConfig(
        negation=terms.negation,
        categories={k: v for k, v in terms.categories.items() if k != "dyspnea"},
    )
    with pytest.raises(ValueError, match="related_terms"):
        AudioAnalyzer(
            audio_config(),
            without_dyspnea,
            FakeTranscriber(load_transcript("consulta_normal")),
            FakeLanguage(load_language("consulta_normal")),
        )


def test_relatorio_e_baseline(audio_file, tmp_path):
    report = analyzer("consulta_normal").analyze(audio_file, "PAC-002", START)
    path = write_report(report, tmp_path / "reports")

    assert path.name == "audio_PAC-002_20261019T143000Z.json"
    data = json.loads(path.read_text("utf-8"))
    assert {"transcript", "language", "critical_terms", "features", "indicators", "events"} <= set(
        data
    )

    baseline = load_baseline(path, audio_config().baseline)
    rerun = analyzer("consulta_normal").analyze(audio_file, "PAC-002", START, baseline=baseline)
    assert rerun.baseline == "patient"
    for indicator in rerun.indicators:
        assert indicator.zscore is None or indicator.zscore == pytest.approx(0.0, abs=0.01)


def test_baseline_invalido(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="relatório de áudio"):
        load_baseline(bad, audio_config().baseline)
    with pytest.raises(ValueError, match="--baseline"):
        load_baseline(tmp_path / "nada.txt", audio_config().baseline)
