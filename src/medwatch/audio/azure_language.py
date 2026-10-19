"""Análise da transcrição com Azure AI Language: sentimento (com opinion mining), frases-chave,
resumo da consulta e Text Analytics for Health (sintomas, medicamentos e negação).

O texto é dividido em documentos dentro dos limites síncronos do serviço (5.120 caracteres por
documento; 10 documentos por chamada no Sentiment e nas frases-chave e 25 no Health). O resumo
é assíncrono e aceita até 125 mil caracteres por chamada. O retry em 429/5xx fica com
o azure-core, que respeita o Retry-After devolvido pelo serviço.

Todas as chamadas vão com `disable_service_logs=True` (LoggingOptOut): sem isso o serviço pode
guardar o texto por até 48 h, e no Sentiment o padrão é guardar.
"""

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar
from urllib.parse import urlparse

from loguru import logger
from pydantic import BaseModel, Field

from medwatch.audio.config import LanguageConfig

if TYPE_CHECKING:
    from medwatch.config import AzureSettings

T = TypeVar("T")

# a chave só é enviada a endpoints do próprio Azure AI Services
ALLOWED_ENDPOINT_SUFFIXES = (".cognitiveservices.azure.com", ".api.cognitive.microsoft.com")
_HOSTNAME = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")

SentimentLabel = Literal["positive", "neutral", "negative", "mixed"]

# limite das chamadas assíncronas (resumo): 125 mil caracteres no total, até 25 documentos
SUMMARY_MAX_CHARS = 120_000
SUMMARY_MAX_DOCUMENTS = 25


class SentimentScores(BaseModel):
    positive: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)


class Opinion(BaseModel):
    target: str
    sentiment: str
    assessments: list[str] = Field(default_factory=list)


class SentenceSentiment(BaseModel):
    text: str
    sentiment: SentimentLabel
    scores: SentimentScores
    opinions: list[Opinion] = Field(default_factory=list)


class SentimentResult(BaseModel):
    sentiment: SentimentLabel
    scores: SentimentScores
    sentences: list[SentenceSentiment] = Field(default_factory=list)


class HealthEntity(BaseModel):
    """Entidade do Text Analytics for Health, com offset no texto completo."""

    text: str
    category: str
    subcategory: str | None = None
    confidence: float = Field(ge=0, le=1)
    offset: int = Field(ge=0, description="posição no texto completo da transcrição")
    length: int = Field(ge=0)
    normalized_text: str | None = None
    certainty: str | None = None

    @property
    def negated(self) -> bool:
        return (self.certainty or "").startswith("negative")


class ConsultationSummary(BaseModel):
    """Resumo da consulta. Contém trechos da fala: fica só no relatório local."""

    mode: Literal["extractive", "abstractive"]
    sentences: list[str] = Field(min_length=1)

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


class LanguageAnalysis(BaseModel):
    """Resultado do Azure Language para a transcrição inteira."""

    sentiment: SentimentResult | None = None
    key_phrases: list[str] = Field(default_factory=list)
    summary: ConsultationSummary | None = None
    entities: list[HealthEntity] = Field(default_factory=list)
    health_available: bool = True
    errors: list[str] = Field(default_factory=list, description="códigos de erro do serviço")


class LanguageAnalyzer(Protocol):
    """Contrato comum ao cliente Azure e às respostas gravadas (modo mock)."""

    def analyze(self, text: str) -> LanguageAnalysis: ...


def validate_endpoint(endpoint: str) -> str:
    """Aceita só HTTPS em domínio do Azure AI Services (evita enviar a chave a outro host)."""
    parsed = urlparse(endpoint)
    host = (parsed.hostname or "").lower()
    valid_host = (
        len(host) <= 253  # limite de DNS; também limita o custo da regex
        and bool(_HOSTNAME.fullmatch(host))
        and host.endswith(ALLOWED_ENDPOINT_SUFFIXES)
    )
    if parsed.scheme != "https" or not valid_host:
        raise ValueError(
            "AZURE_LANGUAGE_ENDPOINT deve ser https://<recurso>.cognitiveservices.azure.com/"
        )
    return endpoint


_SENTENCE_END = re.compile(r"[.!?…]\s+")


def split_documents(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Divide o texto em documentos de até `max_chars`, preferindo fim de frase.

    Retorna pares (offset no texto original, documento) para remapear as entidades.
    """
    documents: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundaries = [m.end() for m in _SENTENCE_END.finditer(text, start, end)]
            if boundaries:
                end = boundaries[-1]
            else:
                space = text.rfind(" ", start, end)
                end = space + 1 if space > start else end
        chunk = text[start:end]
        if chunk.strip():
            documents.append((start, chunk))
        start = end
    return documents


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Fatias de até `size` itens, para respeitar o limite de documentos por chamada."""
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _overall_sentiment(sentences: list[SentenceSentiment]) -> SentimentResult:
    count = len(sentences)
    scores = SentimentScores(
        positive=sum(s.scores.positive for s in sentences) / count,
        neutral=sum(s.scores.neutral for s in sentences) / count,
        negative=sum(s.scores.negative for s in sentences) / count,
    )
    labels = {s.sentiment for s in sentences}
    label: SentimentLabel
    if {"positive", "negative"} <= labels:
        label = "mixed"
    else:
        label = max(("positive", "neutral", "negative"), key=lambda k: getattr(scores, k))
    return SentimentResult(sentiment=label, scores=scores, sentences=sentences)


def _enum_value(value: Any) -> str:
    """Enums do SDK herdam de str, mas no Python 3.11+ str(enum) devolve 'Classe.MEMBRO'."""
    return str(getattr(value, "value", value))


def _scores(raw: Any) -> SentimentScores:
    return SentimentScores(positive=raw.positive, neutral=raw.neutral, negative=raw.negative)


class AzureLanguageAnalyzer:
    """Cliente do Azure AI Language. `client` permite injetar um dublê nos testes."""

    def __init__(
        self, azure: "AzureSettings", config: LanguageConfig, client: Any | None = None
    ) -> None:
        self._config = config
        if client is not None:
            self._client = client
            return
        if not (azure.language_key and azure.language_endpoint):
            raise ValueError("AZURE_LANGUAGE_KEY e AZURE_LANGUAGE_ENDPOINT não configurados")
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential

        self._client = TextAnalyticsClient(
            endpoint=validate_endpoint(azure.language_endpoint),
            credential=AzureKeyCredential(azure.language_key.get_secret_value()),
            retry_total=config.retry_total,
            connection_timeout=config.timeout_seconds,
            read_timeout=config.timeout_seconds,
        )

    def analyze(self, text: str) -> LanguageAnalysis:
        documents = split_documents(text, self._config.max_chars_per_document)
        if not documents:
            return LanguageAnalysis(health_available=False)
        errors: list[str] = []
        sentiment = self._sentiment(documents, errors)
        key_phrases = self._key_phrases(documents, errors)
        summary = self._summary(text, errors)
        entities, health_available = self._health(documents, errors)
        logger.info(
            "Language: {} documentos, {} entidades, {} frases-chave, resumo {}, sentimento {}",
            len(documents),
            len(entities),
            len(key_phrases),
            f"{len(summary.sentences)} frases" if summary else "indisponível",
            sentiment.sentiment if sentiment else "indisponível",
        )
        return LanguageAnalysis(
            sentiment=sentiment,
            key_phrases=key_phrases,
            summary=summary,
            entities=entities,
            health_available=health_available,
            errors=errors,
        )

    def _key_phrases(self, documents: list[tuple[int, str]], errors: list[str]) -> list[str]:
        """Frases-chave de todos os documentos, sem repetição e na ordem em que aparecem."""
        phrases: dict[str, str] = {}
        for batch in batched(documents, self._config.key_phrase_batch_size):
            results = self._client.extract_key_phrases(
                [doc for _, doc in batch],
                language=self._config.key_phrase_language,
                disable_service_logs=True,
            )
            for result in results:
                if result.is_error:
                    errors.append(f"key_phrases:{result.error.code}")
                    continue
                for phrase in result.key_phrases:
                    phrases.setdefault(phrase.casefold(), phrase)
        return list(phrases.values())[: self._config.max_key_phrases]

    def _summary(self, text: str, errors: list[str]) -> ConsultationSummary | None:
        mode = self._config.summary_mode
        if mode == "off":
            return None
        from azure.core.exceptions import HttpResponseError

        documents = [doc for _, doc in split_documents(text, SUMMARY_MAX_CHARS)]
        documents = documents[:SUMMARY_MAX_DOCUMENTS]
        count = self._config.summary_sentences
        try:
            if mode == "abstractive":
                poller = self._client.begin_abstract_summary(
                    documents,
                    language=self._config.summary_language,
                    sentence_count=count,
                    disable_service_logs=True,
                )
            else:
                poller = self._client.begin_extract_summary(
                    documents,
                    language=self._config.summary_language,
                    max_sentence_count=count,
                    order_by="Offset",  # na ordem da consulta, não por relevância
                    disable_service_logs=True,
                )
            results = list(poller.result())
        except HttpResponseError as exc:
            # ex.: resumo abstrativo pedido num recurso de Brazil South
            errors.append(f"summary:{exc.error.code if exc.error else exc.status_code}")
            logger.warning("Resumo {} indisponível neste recurso Azure", mode)
            return None
        sentences: list[str] = []
        for result in results:
            if result.is_error:
                errors.append(f"summary:{result.error.code}")
                continue
            parts = result.summaries if mode == "abstractive" else result.sentences
            sentences.extend(part.text.strip() for part in parts if part.text.strip())
        return ConsultationSummary(mode=mode, sentences=sentences) if sentences else None

    def _sentiment(
        self, documents: list[tuple[int, str]], errors: list[str]
    ) -> SentimentResult | None:
        sentences: list[SentenceSentiment] = []
        for batch in batched(documents, self._config.sentiment_batch_size):
            results = self._client.analyze_sentiment(
                [doc for _, doc in batch],
                language=self._config.sentiment_language,
                show_opinion_mining=True,
                disable_service_logs=True,
            )
            for result in results:
                if result.is_error:
                    errors.append(f"sentiment:{result.error.code}")
                    continue
                for sentence in result.sentences:
                    opinions = [
                        Opinion(
                            target=op.target.text,
                            sentiment=_enum_value(op.target.sentiment),
                            assessments=[a.text for a in op.assessments],
                        )
                        for op in (sentence.mined_opinions or [])
                    ]
                    sentences.append(
                        SentenceSentiment(
                            text=sentence.text,
                            sentiment=sentence.sentiment,
                            scores=_scores(sentence.confidence_scores),
                            opinions=opinions,
                        )
                    )
        return _overall_sentiment(sentences) if sentences else None

    def _health(
        self, documents: list[tuple[int, str]], errors: list[str]
    ) -> tuple[list[HealthEntity], bool]:
        from azure.core.exceptions import HttpResponseError

        entities: list[HealthEntity] = []
        for batch in batched(documents, self._config.health_batch_size):
            try:
                poller = self._client.begin_analyze_healthcare_entities(
                    [doc for _, doc in batch],
                    language=self._config.health_language,
                    disable_service_logs=True,
                )
                results = list(poller.result())
            except HttpResponseError as exc:
                # sem o Health, os termos críticos seguem pelo dicionário local
                errors.append(f"health:{exc.error.code if exc.error else exc.status_code}")
                logger.warning("Text Analytics for Health indisponível; usando só o dicionário")
                return [], False
            for (doc_offset, _), result in zip(batch, results, strict=True):
                if result.is_error:
                    errors.append(f"health:{result.error.code}")
                    continue
                for entity in result.entities:
                    assertion = entity.assertion
                    entities.append(
                        HealthEntity(
                            text=entity.text,
                            category=_enum_value(entity.category),
                            subcategory=entity.subcategory,
                            confidence=entity.confidence_score,
                            offset=doc_offset + entity.offset,
                            length=entity.length,
                            normalized_text=entity.normalized_text,
                            certainty=_enum_value(assertion.certainty)
                            if assertion and assertion.certainty
                            else None,
                        )
                    )
        return entities, True


class RecordedLanguageAnalyzer:
    """Devolve uma análise gravada (modo mock: CI e demonstração offline)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def analyze(self, text: str) -> LanguageAnalysis:
        if not self._path.is_file():
            raise FileNotFoundError(
                f"Resposta gravada não encontrada: {self._path.name}. "
                "Rode uma vez com o Azure real e --record para gravá-la."
            )
        return LanguageAnalysis.model_validate_json(self._path.read_text(encoding="utf-8"))


class RecordingLanguageAnalyzer:
    """Chama o provedor real e grava a resposta para o modo mock."""

    def __init__(self, inner: LanguageAnalyzer, path: Path) -> None:
        self._inner = inner
        self._path = path

    def analyze(self, text: str) -> LanguageAnalysis:
        analysis = self._inner.analyze(text)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        return analysis
