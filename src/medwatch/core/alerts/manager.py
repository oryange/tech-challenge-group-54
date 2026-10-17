"""Entrega de alertas com deduplicação, cooldown e isolamento de falhas entre canais."""

import asyncio
from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from loguru import logger

from medwatch.core.alerts.channels import AlertChannel
from medwatch.core.events import Alert, ClinicalEvent, Severity


@dataclass
class DispatchStats:
    sent: int = 0
    suppressed: int = 0
    ignored: int = 0


class AlertManager:
    """Envia cada alerta a todos os canais.

    Um alerta com a mesma chave (paciente + tipo) dentro do cooldown é suprimido, exceto se a
    severidade subir: escalonamento sempre passa, para a equipe não perder uma piora.
    """

    def __init__(
        self,
        channels: Sequence[AlertChannel],
        cooldown: timedelta = timedelta(minutes=15),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._channels = list(channels)
        self._cooldown = cooldown
        self._clock = clock
        self._last_sent: dict[str, tuple[datetime, Severity]] = {}

    async def dispatch(self, alert: Alert) -> bool:
        """Entrega o alerta. Retorna False se foi suprimido pelo cooldown."""
        now = self._clock()
        last = self._last_sent.get(alert.dedup_key)
        if last is not None:
            last_at, last_severity = last
            if now - last_at < self._cooldown and alert.severity <= last_severity:
                logger.debug("Alerta {} suprimido (cooldown)", alert.dedup_key)
                return False
        self._last_sent[alert.dedup_key] = (now, alert.severity)

        results = await asyncio.gather(
            *(channel.send(alert) for channel in self._channels), return_exceptions=True
        )
        for channel, result in zip(self._channels, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "Canal {} falhou ao entregar {}: {}", channel.name, alert.alert_id, result
                )
        return True

    async def consume(
        self, events: AsyncIterable[ClinicalEvent], min_severity: Severity = Severity.MEDIUM
    ) -> DispatchStats:
        """Transforma eventos do barramento em alertas, um por evento acima do limiar.

        Caminho direto até o RiskEngine (PR 7) assumir a fusão dos eventos.
        """
        stats = DispatchStats()
        async for event in events:
            if event.severity < min_severity:
                stats.ignored += 1
                continue
            if await self.dispatch(Alert.from_event(event)):
                stats.sent += 1
            else:
                stats.suppressed += 1
        return stats
