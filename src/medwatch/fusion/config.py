"""Configuração tipada da fusão multimodal (configs/fusion.yaml)."""

from datetime import timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from medwatch.core.events import ClinicalEvent, Severity, Source


class SourceCondition(BaseModel):
    """Uma perna da regra: evento de uma das fontes com um dos tipos listados."""

    source: list[Source]
    types: list[str] = Field(min_length=1)

    @field_validator("source", mode="before")
    @classmethod
    def _as_list(cls, value: object) -> object:
        # no YAML, `source: audio` e `source: [video, movement]` são aceitos
        return [value] if isinstance(value, str) else value

    def matches(self, event: ClinicalEvent) -> bool:
        return event.source in self.source and event.type in self.types


class FusionRule(BaseModel):
    """Escalonamento cross-modal: dispara quando todas as condições ocorrem na janela."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str
    severity: Severity
    window_minutes: float = Field(gt=0)
    all_of: list[SourceCondition] = Field(min_length=2)
    # as condições precisam ocorrer na ordem do `all_of` (ex.: prescrição antes da piora)
    ordered: bool = False

    @property
    def window(self) -> timedelta:
        return timedelta(minutes=self.window_minutes)


class RiskBands(BaseModel):
    """Faixas do risco agregado R ∈ [0, 1] para severidade."""

    low: float = Field(gt=0, le=1)
    medium: float = Field(gt=0, le=1)
    high: float = Field(gt=0, le=1)
    critical: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _increasing(self) -> "RiskBands":
        if not self.low < self.medium < self.high < self.critical:
            raise ValueError("risk_bands precisa ser crescente: low < medium < high < critical")
        return self

    def severity(self, score: float) -> Severity:
        for severity, threshold in (
            (Severity.CRITICAL, self.critical),
            (Severity.HIGH, self.high),
            (Severity.MEDIUM, self.medium),
            (Severity.LOW, self.low),
        ):
            if score >= threshold:
                return severity
        return Severity.INFO


class FusionConfig(BaseModel):
    window_minutes: float = Field(default=30, gt=0)
    source_weights: dict[Source, float]
    risk_bands: RiskBands
    min_sources: int = Field(default=2, ge=2)
    rules: list[FusionRule] = Field(default_factory=list)

    @field_validator("source_weights")
    @classmethod
    def _weights_in_unit_interval(cls, value: dict[Source, float]) -> dict[Source, float]:
        bad = {str(source): w for source, w in value.items() if not 0 <= w <= 1}
        if bad:
            raise ValueError(f"pesos fora de [0, 1]: {bad}")
        return value

    @model_validator(mode="after")
    def _unique_rule_names(self) -> "FusionConfig":
        names = [rule.name for rule in self.rules]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(f"regras com nome repetido: {duplicated}")
        return self

    @property
    def window(self) -> timedelta:
        return timedelta(minutes=self.window_minutes)

    @property
    def history(self) -> timedelta:
        """Quanto histórico manter por paciente: a maior janela entre risco e regras."""
        return max([self.window, *(rule.window for rule in self.rules)])

    def weight(self, source: Source) -> float:
        return self.source_weights.get(source, 0.0)

    @classmethod
    def load(cls, path: Path) -> "FusionConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))
