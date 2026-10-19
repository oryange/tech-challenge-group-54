import asyncio

from medwatch.core.bus import InMemoryEventBus
from tests.unit.core.test_events import make_event


async def collect(subscription) -> list:
    return [event async for event in subscription]


def test_fan_out_to_all_subscribers():
    async def scenario():
        bus = InMemoryEventBus()
        first, second = bus.subscribe(), bus.subscribe()
        events = [make_event(type="a"), make_event(type="b")]
        for event in events:
            await bus.publish(event)
        await bus.close()
        return events, await collect(first), await collect(second)

    events, first, second = asyncio.run(scenario())
    assert first == events
    assert second == events


def test_closed_subscription_stops_receiving():
    async def scenario():
        bus = InMemoryEventBus()
        kept, closed = bus.subscribe(), bus.subscribe()
        closed.close()
        await bus.publish(make_event())
        await bus.close()
        return await collect(kept), await collect(closed)

    kept, closed = asyncio.run(scenario())
    assert len(kept) == 1
    assert closed == []


def test_subscriber_only_sees_events_after_subscribing():
    async def scenario():
        bus = InMemoryEventBus()
        await bus.publish(make_event(type="before"))
        subscription = bus.subscribe()
        await bus.publish(make_event(type="after"))
        await bus.close()
        return await collect(subscription)

    assert [event.type for event in asyncio.run(scenario())] == ["after"]


def test_bounded_subscription_drops_oldest():
    async def scenario():
        bus = InMemoryEventBus()
        subscription = bus.subscribe(maxsize=2)
        for name in ("e1", "e2", "e3"):
            await bus.publish(make_event(type=name))
        received = [await anext(subscription), await anext(subscription)]
        await bus.close()
        return received

    assert [event.type for event in asyncio.run(scenario())] == ["e2", "e3"]
