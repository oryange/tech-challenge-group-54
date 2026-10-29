"""Simulador de tempo real: reproduz cenários roteirizados e relatórios dos módulos.

Um cenário (`configs/scenarios/*.yaml`) é uma lista de eventos com o minuto em que ocorrem,
mais relatórios JSON já gerados pelos módulos (vitais, áudio, vídeo, prescrições,
movimentação) para reproduzir. Os horários são rebaseados para o início da simulação e os
eventos são publicados com a cadência real dividida por `speed` (60 = um minuto por segundo).
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, Field, JsonValue

from medwatch.core.events import ClinicalEvent, Severity, Source

Publish = Callable[[ClinicalEvent], Awaitable[None]]


class ScriptedEvent(BaseModel):
    """Evento do roteiro; `note` é a narração mostrada na demo."""

    at_minutes: float = Field(ge=0)
    patient_id: str = Field(min_length=1)
    source: Source
    type: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    severity: Severity
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    note: str | None = None


class Replay(BaseModel):
    """Relatório JSON de um módulo; o primeiro evento cai em `at_minutes`."""

    path: Path
    at_minutes: float = Field(default=0, ge=0)
    patient_id: str | None = None
    types: list[str] | None = None


class ExpectedAlert(BaseModel):
    at_minutes: float
    patient_id: str
    alert_type: str
    severity: Severity


class Scenario(BaseModel):
    name: str
    description: str = ""
    events: list[ScriptedEvent] = Field(default_factory=list)
    replay: list[Replay] = Field(default_factory=list)
    # sequência documentada (README); `medwatch simulate` confere o resultado contra ela
    expected_alerts: list[ExpectedAlert] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Scenario":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


@dataclass(frozen=True)
class TimedEvent:
    offset: timedelta
    event: ClinicalEvent
    note: str | None = None


def _iter_events(node: Any) -> Iterator[dict[str, Any]]:
    """Eventos em qualquer nível do relatório (vitais guardam por internação, vídeo na raiz)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "events" and isinstance(value, list):
                yield from (item for item in value if isinstance(item, dict))
            else:
                yield from _iter_events(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_events(item)


def load_report_events(path: Path) -> list[ClinicalEvent]:
    """Lê os `ClinicalEvent`s de um relatório JSON gerado por qualquer módulo."""
    data = json.loads(path.read_text(encoding="utf-8"))
    events = [ClinicalEvent.model_validate(item) for item in _iter_events(data)]
    return sorted(events, key=lambda e: e.timestamp)


def build_timeline(scenario: Scenario, start: datetime, base_dir: Path) -> list[TimedEvent]:
    """Eventos do cenário com horário real a partir de `start`, em ordem cronológica."""
    timeline = [
        TimedEvent(
            offset=timedelta(minutes=item.at_minutes),
            event=ClinicalEvent(
                patient_id=item.patient_id,
                source=item.source,
                type=item.type,
                timestamp=start + timedelta(minutes=item.at_minutes),
                score=item.score,
                severity=item.severity,
                evidence=item.evidence,
            ),
            note=item.note,
        )
        for item in scenario.events
    ]
    for replay in scenario.replay:
        path = replay.path if replay.path.is_absolute() else base_dir / replay.path
        events = [
            e for e in load_report_events(path) if replay.types is None or e.type in replay.types
        ]
        if not events:
            continue
        first = events[0].timestamp
        for event in events:
            offset = timedelta(minutes=replay.at_minutes) + (event.timestamp - first)
            # novo event_id: o mesmo relatório pode ser reproduzido mais de uma vez
            update: dict[str, Any] = {"timestamp": start + offset}
            if replay.patient_id:
                update["patient_id"] = replay.patient_id
            fresh = ClinicalEvent.model_validate(event.model_dump(exclude={"event_id"}) | update)
            timeline.append(TimedEvent(offset=offset, event=fresh))
    return sorted(timeline, key=lambda item: item.offset)


@dataclass(frozen=True)
class Published:
    item: TimedEvent
    monotonic: float


class Simulator:
    """Publica a linha do tempo respeitando a cadência; `speed=0` publica sem esperar."""

    def __init__(
        self,
        timeline: list[TimedEvent],
        publish: Publish,
        speed: float = 60.0,
        on_publish: Callable[[TimedEvent], None] | None = None,
    ) -> None:
        if speed < 0:
            raise ValueError("speed precisa ser ≥ 0")
        self.timeline = timeline
        self._publish = publish
        self._speed = speed
        self._on_publish = on_publish

    async def run(self) -> list[Published]:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        published = []
        for item in self.timeline:
            if self._speed > 0:
                due = item.offset.total_seconds() / self._speed
                await asyncio.sleep(max(0.0, due - (loop.time() - t0)))
            if self._on_publish:
                self._on_publish(item)
            published.append(Published(item, loop.time()))
            await self._publish(item.event)
        return published


class HttpPublisher:
    """Envia cada evento ao `POST /ingest/events` da API (simulador como processo separado)."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def __call__(self, event: ClinicalEvent) -> None:
        response = await self._client.post("/ingest/events", json=[event.model_dump(mode="json")])
        response.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()
