"""Estado em memória da API: alertas (com ACK), linha do tempo dos eventos e assinantes do
WebSocket.

`AlertStore` é um canal do AlertManager como os demais: recebe só os alertas entregues
(depois do filtro de severidade e do cooldown), guarda os mais recentes e os repassa aos
WebSockets conectados.
"""

import asyncio
import json
from collections import OrderedDict, defaultdict, deque
from collections.abc import AsyncIterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from loguru import logger
from pydantic import AwareDatetime

from medwatch.core.events import Alert, ClinicalEvent


class AlertView(Alert):
    """Alerta como a API o expõe: o contrato `Alert` mais o instante do ACK."""

    acked_at: AwareDatetime | None = None


class AlertStore:
    name = "store"

    def __init__(
        self, max_alerts: int = 1000, ack_log: Path | None = None, queue_size: int = 100
    ) -> None:
        self._alerts: OrderedDict[UUID, AlertView] = OrderedDict()
        self._max_alerts = max_alerts
        self._ack_log = ack_log
        self._queue_size = queue_size
        self._listeners: set[asyncio.Queue[dict[str, Any]]] = set()

    # --- canal do AlertManager -----------------------------------------------------------

    async def send(self, alert: Alert) -> None:
        view = AlertView(**dict(alert))
        self._alerts[view.alert_id] = view
        while len(self._alerts) > self._max_alerts:
            self._alerts.popitem(last=False)
        self._broadcast("alert", view)

    # --- consultas e ACK -----------------------------------------------------------------

    def recent(
        self, patient_id: str | None = None, acked: bool | None = None, limit: int = 100
    ) -> list[AlertView]:
        """Alertas do mais recente para o mais antigo."""
        selected = [
            a
            for a in reversed(self._alerts.values())
            if (patient_id is None or a.patient_id == patient_id)
            and (acked is None or a.acked == acked)
        ]
        return selected[:limit]

    def __len__(self) -> int:
        return len(self._alerts)

    def ack(self, alert_id: UUID) -> AlertView | None:
        """Marca o alerta como reconhecido pela equipe. Repetir o ACK não muda o horário."""
        current = self._alerts.get(alert_id)
        if current is None:
            return None
        if current.acked:
            return current
        acked = current.model_copy(update={"acked": True, "acked_at": datetime.now(UTC)})
        self._alerts[alert_id] = acked
        self._audit(acked)
        self._broadcast("ack", acked)
        return acked

    def _audit(self, alert: AlertView) -> None:
        # trilha de auditoria: quem reprocessa o alerts.jsonl sabe o que foi reconhecido
        if self._ack_log is None:
            return
        self._ack_log.parent.mkdir(parents=True, exist_ok=True)
        record = {"alert_id": str(alert.alert_id), "acked_at": alert.acked_at}
        with self._ack_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    # --- WebSocket -----------------------------------------------------------------------

    def listen(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._queue_size)
        self._listeners.add(queue)
        return queue

    def unlisten(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._listeners.discard(queue)

    def _broadcast(self, kind: str, alert: AlertView) -> None:
        message = {"type": kind, "alert": alert.model_dump(mode="json")}
        for queue in list(self._listeners):
            if queue.full():
                # cliente lento: perde o mais antigo, como as assinaturas do barramento
                queue.get_nowait()
                logger.warning("WebSocket lento: mensagem antiga descartada")
            queue.put_nowait(message)


class EventLog:
    """Últimos eventos de cada paciente, para a linha do tempo multimodal do dashboard."""

    def __init__(self, per_patient: int = 500) -> None:
        self._events: dict[str, deque[ClinicalEvent]] = defaultdict(
            lambda: deque(maxlen=per_patient)
        )
        self.total = 0

    def add(self, event: ClinicalEvent) -> None:
        self._events[event.patient_id].append(event)
        self.total += 1

    def for_patient(self, patient_id: str) -> list[ClinicalEvent]:
        return sorted(self._events.get(patient_id, ()), key=lambda e: e.timestamp)

    async def consume(self, events: AsyncIterable[ClinicalEvent]) -> None:
        async for event in events:
            self.add(event)
