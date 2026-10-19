"""Transcrição de uma consulta, independente do provedor (Azure ou resposta gravada)."""

from bisect import bisect_right

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    start: float = Field(ge=0, description="segundos desde o início do áudio")
    end: float = Field(ge=0)


class Segment(BaseModel):
    """Trecho reconhecido (uma frase ou um turno de fala)."""

    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    words: list[Word] = Field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.words) if self.words else len(self.text.split())


class Transcript(BaseModel):
    language: str
    segments: list[Segment] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Texto corrido; os offsets de `time_at` se referem a esta string."""
        return " ".join(segment.text for segment in self.segments)

    @property
    def word_count(self) -> int:
        return sum(segment.word_count for segment in self.segments)

    @property
    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments if s.speaker})

    def time_at(self, char_offset: int) -> float:
        """Instante aproximado (s) do caractere `char_offset` de `text`.

        Interpola linearmente dentro do segmento: suficiente para datar um termo crítico.
        """
        if not self.segments:
            return 0.0
        starts = []
        position = 0
        for segment in self.segments:
            starts.append(position)
            position += len(segment.text) + 1
        index = max(0, bisect_right(starts, char_offset) - 1)
        segment = self.segments[index]
        relative = (char_offset - starts[index]) / max(len(segment.text), 1)
        relative = min(max(relative, 0.0), 1.0)
        return segment.start + relative * (segment.end - segment.start)
