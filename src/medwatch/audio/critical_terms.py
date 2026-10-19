"""Termos críticos na transcrição (configs/critical_terms.yaml), com respeito à negação.

A busca é palavra a palavra sobre o texto normalizado (minúsculas, sem acento), sem regex
montada a partir da configuração. Uma menção só conta como afirmada quando:
- não está negada, nem antes ("não tenho falta de ar") nem logo depois ("dor no peito, não"),
  pelo dicionário local ou pela asserção do Text Analytics for Health;
- não está numa pergunta: sem separar falantes, "Sentiu falta de ar?" é a fala do médico.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from medwatch.audio.azure_language import HealthEntity
from medwatch.audio.config import CriticalTermsConfig

_TOKEN = re.compile(r"\w+")
_SENTENCE_BREAK = re.compile(r"[.;!?…:]")
_SENTENCE_END = re.compile(r"[.!?…]")


def _normalize_char(char: str) -> str:
    base = "".join(c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c))
    lowered = base.lower()
    return lowered[0] if lowered else " "


def normalize(text: str) -> str:
    """Minúsculas e sem acento, preservando o comprimento (os offsets continuam válidos)."""
    return "".join(_normalize_char(char) for char in text)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    """Palavras do texto normalizado, com offsets no texto original."""
    return [Token(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(normalize(text))]


class TermMention(BaseModel):
    """Ocorrência de um termo crítico: afirmada, negada ou em pergunta."""

    category: str
    term: str
    start: int
    end: int
    status: Literal["affirmed", "negated", "question"] = "affirmed"
    negated_by: Literal["dictionary", "health"] | None = None

    @property
    def affirmed(self) -> bool:
        return self.status == "affirmed"


class CriticalTermDetector:
    """Encontra os termos do dicionário e classifica cada menção."""

    def __init__(self, config: CriticalTermsConfig) -> None:
        self._config = config
        self._cues = {normalize(cue) for cue in config.negation.cues}
        self._breakers = {normalize(word) for word in config.negation.scope_breakers}
        self._terms: list[tuple[str, str, tuple[str, ...]]] = []
        for name, category in config.categories.items():
            for term in category.terms:
                words = tuple(t.text for t in tokenize(term))
                if words:
                    self._terms.append((name, " ".join(words), words))

    def detect(self, text: str, entities: Sequence[HealthEntity] = ()) -> list[TermMention]:
        """Menções no texto; `entities` do Health complementam a negação."""
        tokens = tokenize(text)
        words = [token.text for token in tokens]
        matches: list[tuple[str, str, int, int]] = []
        for index in range(len(tokens)):
            for category, term, sequence in self._terms:
                last = index + len(sequence) - 1
                if tuple(words[index : last + 1]) == sequence and self._contiguous(
                    text, tokens, index, last
                ):
                    matches.append((category, term, index, last))
        # palavras de um sintoma ("sem" em "sem ar") não negam o termo seguinte
        covered = {i for _, _, first, last in matches for i in range(first, last + 1)}

        mentions: list[TermMention] = []
        for category, term, index, last in matches:
            start, end = tokens[index].start, tokens[last].end
            mention = TermMention(category=category, term=term, start=start, end=end)
            if self._negated_before(text, tokens, index, covered) or self._negated_after(
                text, tokens, last
            ):
                mention.status, mention.negated_by = "negated", "dictionary"
            elif any(e.negated and _overlaps(e, start, end) for e in entities):
                mention.status, mention.negated_by = "negated", "health"
            elif _in_question(text, end):
                mention.status = "question"
            mentions.append(mention)
        return mentions

    @staticmethod
    def _contiguous(text: str, tokens: list[Token], first: int, last: int) -> bool:
        """Termo de várias palavras não atravessa pontuação ("não, consigo respirar")."""
        return all(
            not text[tokens[i].end : tokens[i + 1].start].strip(" -") for i in range(first, last)
        )

    def _negated_before(
        self, text: str, tokens: list[Token], index: int, covered: set[int]
    ) -> bool:
        """Negação até `window_tokens` palavras antes do termo, na mesma frase.

        Palavras de outros termos não contam para a janela: em "não tenho falta de ar e
        cansaço" a negação alcança "cansaço".
        """
        remaining = self._config.negation.window_tokens
        j = index - 1
        while j >= 0 and remaining > 0:
            gap_after = text[tokens[j].end : tokens[j + 1].start]
            if _SENTENCE_BREAK.search(gap_after):
                return False
            word = tokens[j].text
            if j not in covered:
                if word in self._breakers:
                    return False
                if word in self._cues:
                    # "Não, estou com falta de ar": a negação é uma resposta, não nega o termo
                    return "," not in gap_after
                remaining -= 1
            j -= 1
        return False

    def _negated_after(self, text: str, tokens: list[Token], last: int) -> bool:
        """Negação posposta, comum na fala: "dor no peito, não." / "falta de ar nenhuma"."""
        if last + 1 >= len(tokens) or tokens[last + 1].text not in self._cues:
            return False
        if _SENTENCE_BREAK.search(text[tokens[last].end : tokens[last + 1].start]):
            return False
        if last + 2 >= len(tokens):
            return True  # a negação fecha o texto
        return bool(_SENTENCE_BREAK.search(text[tokens[last + 1].end : tokens[last + 2].start]))


def _in_question(text: str, end: int) -> bool:
    """A frase que contém o termo termina em "?"."""
    match = _SENTENCE_END.search(text, end)
    return match is not None and match.group() == "?"


def _overlaps(entity: HealthEntity, start: int, end: int) -> bool:
    return entity.offset < end and start < entity.offset + entity.length
