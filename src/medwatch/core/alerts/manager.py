"""Entrega de alertas com deduplicação, cooldown e isolamento de falhas entre canais."""

import asyncio
from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from loguru import logger

from medwatch.core.alerts.channels import AlertChannel
from medwatch.core.events import Alert, ClinicalEvent, Severity


class DispatchOutcome(StrEnum):
    SENT = "sent"
    SUPPRESSED = "suppressed"
    FAILED = "failed"


@dataclass
class DispatchStats:
    sent: int = 0
    suppressed: int = 0
    ignored: int = 0
    failed: int = 0


class AlertManager:
    """Envia cada alerta a todos os canais.

    Um alerta com a mesma chave (paciente + tipo) dentro do cooldown é suprimido, exceto se a
    severidade subir: escalonamento sempre passa, para a equipe não perder uma piora. O cooldown
    só passa a valer quando pelo menos um canal entregou o alerta.

    `clock=None` mede o cooldown pelo horário dos eventos em vez do relógio do sistema, para
    reprocessar dados históricos (replay) com o mesmo comportamento do tempo real.
    """

    def __init__(
        self,
        channels: Sequence[AlertChannel],
        cooldown: timedelta = timedelta(minutes=15),
        clock: Callable[[], datetime] | None = lambda: datetime.now(UTC),
        max_in_flight: int = 32,
    ) -> None:
        self._channels = list(channels)
        self._cooldown = cooldown
        self._clock = clock
        self._max_in_flight = max_in_flight
        self._last_sent: dict[str, tuple[datetime, Severity]] = {}

    async def dispatch(self, alert: Alert) -> bool:
        """Entrega o alerta. Retorna False se foi suprimido ou se nenhum canal o entregou."""
        return await self.deliver(alert) is DispatchOutcome.SENT

    async def deliver(self, alert: Alert) -> DispatchOutcome:
        """Como `dispatch`, mas diferencia alerta suprimido de alerta que nenhum canal entregou."""
        now = self._clock() if self._clock else max(e.timestamp for e in alert.reasons)
        last = self._last_sent.get(alert.dedup_key)
        if last is not None:
            last_at, last_severity = last
            if now - last_at < self._cooldown and alert.severity <= last_severity:
                logger.debug("Alerta {} suprimido (cooldown)", alert.dedup_key)
                return DispatchOutcome.SUPPRESSED
        # reserva antes do envio, para que uma duplicata que chegue durante a entrega seja
        # suprimida; se nenhum canal entregar, a reserva é desfeita logo abaixo
        self._last_sent[alert.dedup_key] = (now, alert.severity)

        results = await asyncio.gather(
            *(channel.send(alert) for channel in self._channels), return_exceptions=True
        )
        delivered = False
        for channel, result in zip(self._channels, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "Canal {} falhou ao entregar {}: {}", channel.name, alert.alert_id, result
                )
            else:
                delivered = True

        if not delivered:
            # sem a reversão, as repetições do cooldown seriam suprimidas e a equipe não
            # receberia nenhum alerta enquanto os canais estivessem fora
            if self._last_sent.get(alert.dedup_key) == (now, alert.severity):
                if last is None:
                    del self._last_sent[alert.dedup_key]
                else:
                    self._last_sent[alert.dedup_key] = last
            logger.error("Alerta {} não foi entregue por nenhum canal", alert.alert_id)
            return DispatchOutcome.FAILED
        return DispatchOutcome.SENT

    async def consume(
        self, events: AsyncIterable[ClinicalEvent], min_severity: Severity = Severity.MEDIUM
    ) -> DispatchStats:
        """Transforma eventos do barramento em alertas, um por evento acima do limiar.

        As entregas correm em paralelo (até `max_in_flight`), para que um CRITICAL não espere
        os retries de um webhook fora do ar nos alertas anteriores. Caminho direto até o
        RiskEngine (PR 7) assumir a fusão dos eventos.
        """
        stats = DispatchStats()
        slots = asyncio.Semaphore(self._max_in_flight)

        async def deliver(event: ClinicalEvent) -> None:
            async with slots:
                outcome = await self.deliver(Alert.from_event(event))
            if outcome is DispatchOutcome.SENT:
                stats.sent += 1
            elif outcome is DispatchOutcome.SUPPRESSED:
                stats.suppressed += 1
            else:
                stats.failed += 1

        async with asyncio.TaskGroup() as group:
            async for event in events:
                if event.severity < min_severity:
                    stats.ignored += 1
                    continue
                group.create_task(deliver(event))
        return stats
