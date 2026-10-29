"""Contratos de dados compartilhados por todos os módulos: eventos clínicos e alertas.

Cada pipeline (vídeo, áudio, vitais, prescrições, movimentação) emite `ClinicalEvent`s;
o RiskEngine (fusão) e o AlertManager os transformam em `Alert`s.
"""

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
)


class Source(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    VITALS = "vitals"
    PRESCRIPTION = "prescription"
    MOVEMENT = "movement"
    FUSION = "fusion"


class Severity(StrEnum):
    """Severidade ordenável: INFO < LOW < MEDIUM < HIGH < CRITICAL."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return list(type(self)).index(self)

    @classmethod
    def _coerce(cls, other: object) -> "Severity | None":
        """Aceita `Severity` ou o texto ("high"), como chega de YAML e de variáveis de ambiente."""
        if isinstance(other, Severity):
            return other
        if isinstance(other, str):
            # texto desconhecido levanta ValueError: devolver NotImplemented faria o Python
            # cair na comparação alfabética do str, que é justamente o que queremos evitar
            return cls(other.lower())
        return None

    # str já define a comparação lexicográfica; sobrescrevemos para usar a ordem clínica,
    # inclusive contra texto: sem isso, `Severity.CRITICAL >= "high"` seria False
    def __lt__(self, other: object) -> bool:
        if (rhs := self._coerce(other)) is None:
            return NotImplemented
        return self.rank < rhs.rank

    def __le__(self, other: object) -> bool:
        if (rhs := self._coerce(other)) is None:
            return NotImplemented
        return self.rank <= rhs.rank

    def __gt__(self, other: object) -> bool:
        if (rhs := self._coerce(other)) is None:
            return NotImplemented
        return self.rank > rhs.rank

    def __ge__(self, other: object) -> bool:
        if (rhs := self._coerce(other)) is None:
            return NotImplemented
        return self.rank >= rhs.rank


RECOMMENDED_ACTIONS: dict[Severity, str] = {
    Severity.INFO: "Registrar no prontuário.",
    Severity.LOW: "Reavaliar na próxima ronda.",
    Severity.MEDIUM: "Avaliar o paciente em até 1 hora.",
    Severity.HIGH: "Avaliação imediata pela enfermagem e comunicar o médico responsável.",
    Severity.CRITICAL: "Acionar a equipe de resposta rápida imediatamente.",
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _read_only(self: object, *args: object, **kwargs: object) -> NoReturn:
    raise TypeError("evidence é imutável: o mesmo evento é entregue a vários assinantes")


class FrozenDict(dict[str, Any]):
    """dict que recusa alterações. Continua sendo um dict: `json.dumps` e `==` funcionam."""

    __setitem__ = __delitem__ = __ior__ = _read_only
    clear = pop = popitem = setdefault = update = _read_only

    def __reduce__(self) -> tuple[type, tuple[dict[str, Any]]]:
        # copy/deepcopy/pickle reconstroem pelo construtor, sem passar pelo __setitem__
        return type(self), (dict(self),)


class FrozenList(list[Any]):
    """list que recusa alterações. Continua sendo uma list: `json.dumps` e `==` funcionam."""

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _read_only
    append = extend = insert = pop = remove = clear = sort = reverse = _read_only

    def __reduce__(self) -> tuple[type, tuple[list[Any]]]:
        return type(self), (list(self),)


def _freeze(value: Any) -> Any:
    """Congela recursivamente dicts e listas do `evidence`."""
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return FrozenList(_freeze(item) for item in value)
    return value


class ClinicalEvent(BaseModel):
    """Sinal detectado por um pipeline. `patient_id` deve ser pseudonimizado (nunca o nome)."""

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    patient_id: str = Field(min_length=1)
    source: Source
    type: str = Field(min_length=1, description="Tipo do evento em snake_case, ex.: spo2_drop")
    timestamp: AwareDatetime = Field(default_factory=_utcnow)
    score: float = Field(ge=0.0, le=1.0, description="Confiança/intensidade da anomalia")
    severity: Severity
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: tuple[Path, ...] = ()

    _normalize_timestamp = field_validator("timestamp")(_to_utc)

    @field_validator("evidence")
    @classmethod
    def _freeze_evidence(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        # o frozen do pydantic só bloqueia a atribuição; sem isto, um assinante que
        # alterasse `event.evidence` mudaria o evento para todos os outros
        return cast(dict[str, JsonValue], _freeze(value))


class Alert(BaseModel):
    """Alerta destinado à equipe médica, com os eventos que o motivaram."""

    model_config = ConfigDict(frozen=True)

    alert_id: UUID = Field(default_factory=uuid4)
    patient_id: str = Field(min_length=1)
    alert_type: str = Field(min_length=1, description="Chave de deduplicação junto com o paciente")
    severity: Severity
    title: str = Field(min_length=1)
    reasons: tuple[ClinicalEvent, ...] = Field(min_length=1)
    recommended_action: str
    created_at: AwareDatetime = Field(default_factory=_utcnow)
    acked: bool = False

    _normalize_created_at = field_validator("created_at")(_to_utc)

    @property
    def dedup_key(self) -> str:
        return f"{self.patient_id}:{self.alert_type}"

    @classmethod
    def from_event(cls, event: ClinicalEvent, title: str | None = None) -> "Alert":
        """Alerta direto de um único evento; os alertas de fusão vêm do RiskEngine."""
        return cls(
            patient_id=event.patient_id,
            alert_type=f"{event.source}.{event.type}",
            severity=event.severity,
            title=title or f"{event.source.value}: {event.type.replace('_', ' ')}",
            reasons=(event,),
            recommended_action=RECOMMENDED_ACTIONS[event.severity],
        )
