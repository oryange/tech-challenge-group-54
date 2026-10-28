from types import SimpleNamespace as NS

import pytest
from azure.core.exceptions import HttpResponseError

from medwatch.audio.azure_language import (
    AzureLanguageAnalyzer,
    RecordedLanguageAnalyzer,
    RecordingLanguageAnalyzer,
    batched,
    split_documents,
    validate_endpoint,
)
from medwatch.audio.config import LanguageConfig
from medwatch.config import AzureSettings
from tests.unit.audio.helpers import FakeLanguage, load_language


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://medwatch-g54.cognitiveservices.azure.com/",
        "https://brazilsouth.api.cognitive.microsoft.com",
    ],
)
def test_endpoint_do_azure_e_aceito(endpoint):
    assert validate_endpoint(endpoint) == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://medwatch-g54.cognitiveservices.azure.com/",
        "https://exemplo.com/",
        "https://medwatch.cognitiveservices.azure.com.exemplo.com/",
        "https://exemplo.com/medwatch.cognitiveservices.azure.com",
        "https://<recurso>.cognitiveservices.azure.com/",
    ],
)
def test_endpoint_fora_do_azure_e_recusado(endpoint):
    with pytest.raises(ValueError, match="AZURE_LANGUAGE_ENDPOINT"):
        validate_endpoint(endpoint)


def test_split_documents_respeita_limite_e_offsets():
    text = " ".join(f"Frase número {i} da consulta." for i in range(200))
    documents = split_documents(text, max_chars=300)

    assert len(documents) > 1
    assert all(len(doc) <= 300 for _, doc in documents)
    for offset, doc in documents:
        assert text[offset : offset + len(doc)] == doc
    # corta em fim de frase sempre que possível
    assert all(doc.rstrip().endswith(".") for _, doc in documents)
    assert "".join(doc for _, doc in documents) == text


def test_split_documents_sem_pontuacao_corta_em_espaco():
    text = "palavra " * 100
    documents = split_documents(text, max_chars=50)
    assert all(len(doc) <= 50 for _, doc in documents)
    assert all(not doc.startswith("alavra") for _, doc in documents)


def test_split_texto_vazio():
    assert split_documents("   ", 100) == []


def test_batched():
    assert [list(b) for b in batched([1, 2, 3, 4, 5], 2)] == [[1, 2], [3, 4], [5]]


def _scores(pos, neu, neg):
    return NS(positive=pos, neutral=neu, negative=neg)


class FakeClient:
    """Imita o TextAnalyticsClient: guarda as chamadas e devolve objetos com os mesmos campos."""

    def __init__(
        self, health_error: Exception | None = None, summary_error: Exception | None = None
    ) -> None:
        self.sentiment_calls: list[list[str]] = []
        self.key_phrase_calls: list[list[str]] = []
        self.summary_calls: list[tuple[str, list[str], int]] = []
        self.health_calls: list[list[str]] = []
        self.health_error = health_error
        self.summary_error = summary_error

    def analyze_sentiment(self, docs, language, show_opinion_mining, disable_service_logs):
        assert show_opinion_mining is True
        assert disable_service_logs is True
        assert language == "pt-BR"
        self.sentiment_calls.append(list(docs))
        results = []
        for doc in docs:
            negative = "ar" in doc
            sentence = NS(
                text=doc,
                sentiment="negative" if negative else "positive",
                confidence_scores=_scores(0.1, 0.1, 0.8) if negative else _scores(0.8, 0.1, 0.1),
                mined_opinions=[
                    NS(
                        target=NS(text="noite", sentiment=NS(value="negative")),
                        assessments=[NS(text="mal")],
                    )
                ]
                if negative
                else [],
            )
            results.append(NS(is_error=False, sentences=[sentence]))
        return results

    def extract_key_phrases(self, docs, language, disable_service_logs):
        assert disable_service_logs is True
        assert language == "pt-BR"
        self.key_phrase_calls.append(list(docs))
        results = []
        for doc in docs:
            phrases = [w.strip(".") for w in ("falta de ar", "Falta de Ar", "noite") if w in doc]
            results.append(NS(is_error=False, key_phrases=phrases))
        return results

    def begin_extract_summary(
        self, docs, language, max_sentence_count, order_by, disable_service_logs
    ):
        assert disable_service_logs is True
        assert (language, order_by) == ("pt", "Offset")
        self.summary_calls.append(("extractive", list(docs), max_sentence_count))
        if self.summary_error:
            raise self.summary_error
        sentences = [NS(text=s + ".") for s in docs[0].split(". ")[:max_sentence_count]]
        return NS(result=lambda: iter([NS(is_error=False, sentences=sentences)]))

    def begin_abstract_summary(self, docs, language, sentence_count, disable_service_logs):
        assert disable_service_logs is True
        self.summary_calls.append(("abstractive", list(docs), sentence_count))
        summary = NS(text="Paciente relata falta de ar noturna.")
        return NS(result=lambda: iter([NS(is_error=False, summaries=[summary])]))

    def begin_analyze_healthcare_entities(self, docs, language, disable_service_logs):
        assert disable_service_logs is True
        assert language == "pt"
        if self.health_error:
            raise self.health_error
        self.health_calls.append(list(docs))
        results = []
        for doc in docs:
            entities = []
            if "falta de ar" in doc:
                entities.append(
                    NS(
                        text="falta de ar",
                        category=NS(value="SymptomOrSign"),
                        subcategory=None,
                        confidence_score=0.95,
                        offset=doc.index("falta de ar"),
                        length=len("falta de ar"),
                        normalized_text="Dyspnea",
                        assertion=NS(certainty=NS(value="negative")) if "Não" in doc else None,
                    )
                )
            results.append(NS(is_error=False, entities=entities))
        return NS(result=lambda: iter(results))


def _analyzer(client, **config):
    return AzureLanguageAnalyzer(AzureSettings(), LanguageConfig(**config), client=client)


def test_sentimento_e_entidades_com_offset_no_texto_completo():
    text = "Passei bem a noite. " * 3 + "Não tenho falta de ar. Hoje estou com falta de ar."
    client = FakeClient()
    analysis = _analyzer(client, max_chars_per_document=40).analyze(text)

    assert analysis.health_available is True
    assert analysis.sentiment is not None
    assert analysis.sentiment.sentiment == "mixed"
    assert analysis.sentiment.sentences[-1].opinions[0].sentiment == "negative"
    assert len(analysis.entities) == 2
    for entity in analysis.entities:
        assert text[entity.offset : entity.offset + entity.length] == "falta de ar"
    assert [e.negated for e in analysis.entities] == [True, False]
    assert analysis.entities[0].category == "SymptomOrSign"


def test_lotes_respeitam_limites_do_servico():
    text = " ".join(f"Frase {i} da consulta." for i in range(60))
    client = FakeClient()
    _analyzer(client, max_chars_per_document=20, sentiment_batch_size=10).analyze(text)

    assert all(len(call) <= 10 for call in client.sentiment_calls)
    assert all(len(call) <= 25 for call in client.health_calls)
    assert sum(map(len, client.sentiment_calls)) == sum(map(len, client.health_calls))


def test_frases_chave_sem_repeticao_e_com_limite():
    text = "Não tenho falta de ar. " * 30 + "Passei a noite com Falta de Ar."
    client = FakeClient()
    analysis = _analyzer(client, max_chars_per_document=60, max_key_phrases=2).analyze(text)

    assert analysis.key_phrases == ["falta de ar", "noite"]
    assert all(len(call) <= 10 for call in client.key_phrase_calls)


def test_resumo_extrativo_na_ordem_da_consulta():
    text = "Acordei com falta de ar. Tive que dormir sentada. Estou cansada. Obrigada."
    client = FakeClient()
    analysis = _analyzer(client, summary_sentences=2).analyze(text)

    assert analysis.summary is not None
    assert analysis.summary.mode == "extractive"
    assert analysis.summary.sentences == ["Acordei com falta de ar.", "Tive que dormir sentada."]
    # o resumo vai inteiro numa chamada assíncrona, sem a quebra de 5.000 caracteres
    mode, docs, count = client.summary_calls[0]
    assert (mode, docs, count) == ("extractive", [text], 2)


def test_resumo_abstrativo_quando_configurado():
    client = FakeClient()
    analysis = _analyzer(client, summary_mode="abstractive").analyze("Estou com falta de ar.")
    assert analysis.summary is not None
    assert analysis.summary.mode == "abstractive"
    assert analysis.summary.text == "Paciente relata falta de ar noturna."


def test_resumo_desligado_nao_chama_o_servico():
    client = FakeClient()
    analysis = _analyzer(client, summary_mode="off").analyze("Estou com falta de ar.")
    assert analysis.summary is None
    assert client.summary_calls == []


def test_resumo_indisponivel_na_regiao_nao_derruba_a_analise():
    error = HttpResponseError(message="FeatureNotSupportedInRegion")
    client = FakeClient(summary_error=error)
    analysis = _analyzer(client).analyze("Estou com falta de ar.")
    assert analysis.summary is None
    assert analysis.sentiment is not None
    assert any(e.startswith("summary:") for e in analysis.errors)


def test_health_indisponivel_segue_sem_entidades():
    error = HttpResponseError(message="UnsupportedLanguageCode")
    client = FakeClient(health_error=error)
    analysis = _analyzer(client).analyze("Estou com falta de ar.")

    assert analysis.health_available is False
    assert analysis.entities == []
    assert analysis.sentiment is not None
    assert analysis.errors and analysis.errors[0].startswith("health:")


def test_documento_com_erro_e_registrado():
    class ErrorClient(FakeClient):
        def analyze_sentiment(self, docs, language, show_opinion_mining, disable_service_logs):
            return [NS(is_error=True, error=NS(code="InvalidDocument")) for _ in docs]

    analysis = _analyzer(ErrorClient()).analyze("Texto qualquer.")
    assert analysis.sentiment is None
    assert "sentiment:InvalidDocument" in analysis.errors


def test_sem_chave_recusa_criar_cliente():
    with pytest.raises(ValueError, match="AZURE_LANGUAGE_KEY"):
        AzureLanguageAnalyzer(AzureSettings(language_key=None), LanguageConfig())


def test_resposta_gravada_ida_e_volta(tmp_path):
    analysis = load_language("consulta_dispneia")
    path = tmp_path / "abc.language.json"
    RecordingLanguageAnalyzer(FakeLanguage(analysis), path).analyze("texto")
    assert RecordedLanguageAnalyzer(path).analyze("texto") == analysis
    with pytest.raises(FileNotFoundError):
        RecordedLanguageAnalyzer(tmp_path / "nada.json").analyze("texto")
