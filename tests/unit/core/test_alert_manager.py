import asyncio
from datetime import UTC, datetime, timedelta

from medwatch.core.alerts import AlertManager, DispatchOutcome
from medwatch.core.bus import InMemoryEventBus
from medwatch.core.events import Alert, Severity
from tests.unit.core.test_events import make_event


class RecordingChannel:
    name = "recording"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    async def send(self, alert: Alert) -> None:
        self.alerts.append(alert)


class BrokenChannel:
    name = "broken"

    async def send(self, alert: Alert) -> None:
        raise RuntimeError("canal fora do ar")


class FlakyChannel(RecordingChannel):
    name = "flaky"

    def __init__(self) -> None:
        super().__init__()
        self.down = False

    async def send(self, alert: Alert) -> None:
        if self.down:
            raise RuntimeError("canal fora do ar")
        await super().send(alert)


class GatedChannel(RecordingChannel):
    """Segura a entrega de um tipo de alerta até `release` ser liberado (webhook lento)."""

    name = "gated"

    def __init__(self, release: asyncio.Event, blocked_type: str) -> None:
        super().__init__()
        self._release = release
        self._blocked = blocked_type

    async def send(self, alert: Alert) -> None:
        if alert.alert_type.endswith(f".{self._blocked}"):
            await self._release.wait()
        await super().send(alert)


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def alert(severity: Severity = Severity.HIGH, **overrides) -> Alert:
    return Alert.from_event(make_event(severity=severity, **overrides))


def make_manager(*channels, cooldown_min: int = 15):
    clock = FakeClock()
    return AlertManager(channels, cooldown=timedelta(minutes=cooldown_min), clock=clock), clock


def test_duplicate_within_cooldown_is_suppressed():
    channel = RecordingChannel()
    manager, _ = make_manager(channel)
    assert asyncio.run(manager.dispatch(alert())) is True
    assert asyncio.run(manager.dispatch(alert())) is False
    assert len(channel.alerts) == 1


def test_same_type_other_patient_is_not_suppressed():
    channel = RecordingChannel()
    manager, _ = make_manager(channel)
    asyncio.run(manager.dispatch(alert(patient_id="PAC-001")))
    asyncio.run(manager.dispatch(alert(patient_id="PAC-002")))
    assert len(channel.alerts) == 2


def test_resent_after_cooldown():
    channel = RecordingChannel()
    manager, clock = make_manager(channel)
    asyncio.run(manager.dispatch(alert()))
    clock.now += timedelta(minutes=15)
    assert asyncio.run(manager.dispatch(alert())) is True
    assert len(channel.alerts) == 2


def test_escalation_bypasses_cooldown():
    channel = RecordingChannel()
    manager, _ = make_manager(channel)
    asyncio.run(manager.dispatch(alert(Severity.HIGH)))
    assert asyncio.run(manager.dispatch(alert(Severity.CRITICAL))) is True
    # depois do escalonamento, um HIGH volta a ser duplicata
    assert asyncio.run(manager.dispatch(alert(Severity.HIGH))) is False
    assert [a.severity for a in channel.alerts] == [Severity.HIGH, Severity.CRITICAL]


def test_broken_channel_does_not_block_others():
    first, second = RecordingChannel(), RecordingChannel()
    manager, _ = make_manager(first, BrokenChannel(), second)
    assert asyncio.run(manager.dispatch(alert())) is True
    assert len(first.alerts) == len(second.alerts) == 1


def test_consume_filters_by_min_severity_and_counts():
    async def scenario(channel):
        bus = InMemoryEventBus()
        manager, _ = make_manager(channel)
        consumer = asyncio.create_task(manager.consume(bus.subscribe(), Severity.MEDIUM))
        for event in (
            make_event(severity=Severity.LOW, type="low"),
            make_event(severity=Severity.MEDIUM, type="medium"),
            make_event(severity=Severity.MEDIUM, type="medium"),
            make_event(severity=Severity.CRITICAL, type="critical"),
        ):
            await bus.publish(event)
        await bus.close()
        return await consumer

    channel = RecordingChannel()
    stats = asyncio.run(scenario(channel))
    assert (stats.sent, stats.suppressed, stats.ignored) == (2, 1, 1)
    assert [a.alert_type for a in channel.alerts] == ["vitals.medium", "vitals.critical"]


def test_failed_delivery_does_not_start_cooldown():
    """Com todos os canais fora, a repetição do alerta precisa ser tentada de novo."""
    manager, _ = make_manager(BrokenChannel())
    assert asyncio.run(manager.deliver(alert())) is DispatchOutcome.FAILED
    assert asyncio.run(manager.dispatch(alert())) is False

    channel = RecordingChannel()
    manager._channels = [channel]  # o canal volta
    assert asyncio.run(manager.dispatch(alert())) is True
    assert len(channel.alerts) == 1


def test_failed_escalation_keeps_previous_cooldown():
    channel = FlakyChannel()
    manager, _ = make_manager(channel)
    assert asyncio.run(manager.dispatch(alert(Severity.HIGH))) is True
    channel.down = True
    assert asyncio.run(manager.deliver(alert(Severity.CRITICAL))) is DispatchOutcome.FAILED
    channel.down = False
    # o HIGH ainda está no cooldown; o CRITICAL, que não foi entregue, passa de novo
    assert asyncio.run(manager.dispatch(alert(Severity.HIGH))) is False
    assert asyncio.run(manager.dispatch(alert(Severity.CRITICAL))) is True


def test_consume_counts_failed_deliveries():
    async def scenario():
        bus = InMemoryEventBus()
        manager, _ = make_manager(BrokenChannel())
        consumer = asyncio.create_task(manager.consume(bus.subscribe(), Severity.MEDIUM))
        await bus.publish(make_event(severity=Severity.HIGH))
        await bus.close()
        return await consumer

    stats = asyncio.run(scenario())
    assert (stats.sent, stats.failed) == (0, 1)


def test_consume_does_not_queue_critical_behind_slow_delivery():
    """Um webhook lento num alerta não pode atrasar a entrega de um CRITICAL seguinte."""

    async def scenario():
        release = asyncio.Event()
        channel = GatedChannel(release, blocked_type="slow")
        bus = InMemoryEventBus()
        manager, _ = make_manager(channel)
        consumer = asyncio.create_task(manager.consume(bus.subscribe(), Severity.MEDIUM))
        await bus.publish(make_event(severity=Severity.HIGH, type="slow"))
        await bus.publish(make_event(severity=Severity.CRITICAL, type="fall"))
        await bus.close()
        for _ in range(20):
            await asyncio.sleep(0)
        delivered_before_release = [a.alert_type for a in channel.alerts]
        release.set()
        stats = await consumer
        return delivered_before_release, stats

    delivered, stats = asyncio.run(scenario())
    assert delivered == ["vitals.fall"]
    assert stats.sent == 2


def test_min_severity_as_text_is_respected():
    async def scenario(channel):
        bus = InMemoryEventBus()
        manager, _ = make_manager(channel)
        consumer = asyncio.create_task(
            manager.consume(bus.subscribe(), "high")  # type: ignore[arg-type]
        )
        await bus.publish(make_event(severity=Severity.CRITICAL, type="critical"))
        await bus.publish(make_event(severity=Severity.MEDIUM, type="medium"))
        await bus.close()
        return await consumer

    channel = RecordingChannel()
    stats = asyncio.run(scenario(channel))
    assert (stats.sent, stats.ignored) == (1, 1)
    assert [a.alert_type for a in channel.alerts] == ["vitals.critical"]


def test_event_time_cooldown_for_replay():
    """Com clock=None, o cooldown segue o horário clínico dos eventos (reprocessamento)."""
    channel = RecordingChannel()
    manager = AlertManager([channel], cooldown=timedelta(minutes=15), clock=None)
    t0 = datetime(2180, 1, 1, 12, 0, tzinfo=UTC)

    def at(minutes: int) -> Alert:
        return Alert.from_event(make_event(timestamp=t0 + timedelta(minutes=minutes)))

    assert asyncio.run(manager.dispatch(at(0))) is True
    assert asyncio.run(manager.dispatch(at(10))) is False
    assert asyncio.run(manager.dispatch(at(30))) is True
    assert len(channel.alerts) == 2
