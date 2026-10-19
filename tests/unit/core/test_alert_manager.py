import asyncio
from datetime import UTC, datetime, timedelta

from medwatch.core.alerts import AlertManager
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
