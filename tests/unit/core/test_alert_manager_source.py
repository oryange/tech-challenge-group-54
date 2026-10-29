import asyncio

from medwatch.core.alerts import AlertManager
from medwatch.core.bus import InMemoryEventBus
from medwatch.core.events import Alert, ClinicalEvent, Severity
from tests.unit.core.test_alert_manager import RecordingChannel
from tests.unit.core.test_events import make_event


class EchoTwice:
    """AlertSource de teste: devolve o alerta direto e um segundo, CRITICAL."""

    def alerts_for(self, event: ClinicalEvent) -> list[Alert]:
        direct = Alert.from_event(event)
        return [direct, direct.model_copy(update={"alert_type": "x.y", "severity": "critical"})]


def test_consume_with_source_filters_each_alert_by_severity():
    async def scenario():
        bus = InMemoryEventBus()
        channel = RecordingChannel()
        manager = AlertManager([channel], clock=None)
        consumer = asyncio.create_task(
            manager.consume(bus.subscribe(), min_severity=Severity.MEDIUM, source=EchoTwice())
        )
        # evento LOW: o direto é descartado, o CRITICAL da fonte passa
        await bus.publish(make_event(severity=Severity.LOW))
        await bus.close()
        return await consumer, channel

    stats, channel = asyncio.run(scenario())
    assert [a.alert_type for a in channel.alerts] == ["x.y"]
    assert (stats.sent, stats.ignored) == (1, 0)
