"""Execução local do fluxo completo: simulador → barramento → RiskEngine → AlertManager.

Mede a latência de cada alerta: do instante em que o evento que o disparou foi publicado no
barramento até a entrega nos canais.
"""

import asyncio
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from medwatch.core.alerts.channels import AlertChannel
from medwatch.core.alerts.manager import AlertManager, DispatchStats
from medwatch.core.bus import InMemoryEventBus
from medwatch.core.events import Alert, Severity
from medwatch.fusion.config import FusionConfig
from medwatch.fusion.risk_engine import RiskEngine
from medwatch.streaming.simulator import ExpectedAlert, Published, Simulator, TimedEvent


class AlertProbe:
    """Canal que só anota o alerta e o instante da entrega (para medir latência)."""

    name = "probe"

    def __init__(self) -> None:
        self.received: list[tuple[Alert, float]] = []

    async def send(self, alert: Alert) -> None:
        self.received.append((alert, asyncio.get_running_loop().time()))


@dataclass(frozen=True)
class DeliveredAlert:
    alert: Alert
    at_minutes: float
    latency_s: float


@dataclass
class SimulationResult:
    stats: DispatchStats
    delivered: list[DeliveredAlert] = field(default_factory=list)
    published: list[Published] = field(default_factory=list)

    @property
    def latencies(self) -> list[float]:
        return [d.latency_s for d in self.delivered]

    def latency_summary(self) -> tuple[float, float]:
        """(mediana, máximo) da latência evento→alerta, em segundos."""
        if not self.latencies:
            return 0.0, 0.0
        return statistics.median(self.latencies), max(self.latencies)

    def mismatches(self, expected: Sequence[ExpectedAlert]) -> list[str]:
        """Diferenças entre os alertas entregues e a sequência documentada no cenário."""

        def key(at: float, patient: str, alert_type: str, severity: Severity) -> str:
            return f"min {at:g} · {patient} · {alert_type} · {severity.value}"

        got = sorted(
            key(d.at_minutes, d.alert.patient_id, d.alert.alert_type, d.alert.severity)
            for d in self.delivered
        )
        want = sorted(key(e.at_minutes, e.patient_id, e.alert_type, e.severity) for e in expected)
        missing = [f"faltou: {k}" for k in want if k not in got]
        extra = [f"a mais: {k}" for k in got if k not in want]
        return missing + extra


async def run_local(
    timeline: list[TimedEvent],
    fusion: FusionConfig,
    channels: Sequence[AlertChannel] = (),
    cooldown: timedelta = timedelta(minutes=15),
    min_severity: Severity = Severity.MEDIUM,
    speed: float = 60.0,
    on_publish: Callable[[TimedEvent], None] | None = None,
) -> SimulationResult:
    """Roda o cenário no processo atual. Cooldown e janela usam o horário dos eventos."""
    bus = InMemoryEventBus()
    engine = RiskEngine(fusion)
    probe = AlertProbe()
    manager = AlertManager([*channels, probe], cooldown=cooldown, clock=None)
    consumer = asyncio.create_task(
        manager.consume(bus.subscribe(), min_severity=min_severity, source=engine)
    )
    published = await Simulator(timeline, bus.publish, speed=speed, on_publish=on_publish).run()
    await bus.close()
    stats = await consumer

    by_event = {str(p.item.event.event_id): p for p in published}
    delivered = []
    for alert, delivered_at in probe.received:
        # o evento que disparou o alerta é o mais recente entre os motivos
        trigger = by_event[str(max(alert.reasons, key=lambda e: e.timestamp).event_id)]
        delivered.append(
            DeliveredAlert(
                alert=alert,
                at_minutes=round(trigger.item.offset.total_seconds() / 60, 2),
                latency_s=delivered_at - trigger.monotonic,
            )
        )
    delivered.sort(key=lambda d: (d.at_minutes, d.alert.alert_type.startswith("fusion.")))
    return SimulationResult(stats=stats, delivered=delivered, published=published)
