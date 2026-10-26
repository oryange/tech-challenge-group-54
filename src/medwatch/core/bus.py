"""Barramento de eventos clínicos.

`EventBus` é a interface; `InMemoryEventBus` atende a execução local. Um adapter para Azure
Event Hubs pode implementar o mesmo protocolo sem mudar produtores e consumidores.
"""

import asyncio
from typing import Final, Protocol, cast

from loguru import logger

from medwatch.core.events import ClinicalEvent

_CLOSED: Final = object()


class Subscription:
    """Fila própria de um consumidor. Iterável com `async for`; termina quando fechada."""

    def __init__(self, bus: "InMemoryEventBus", maxsize: int) -> None:
        self._bus = bus
        self._queue: asyncio.Queue[object] = asyncio.Queue(maxsize=maxsize)

    def _offer(self, item: object) -> None:
        if self._queue.full():
            # consumidor lento: descarta o mais antigo para não travar os produtores
            dropped = self._queue.get_nowait()
            logger.warning("Fila de assinatura cheia; evento descartado: {}", dropped)
        self._queue.put_nowait(item)

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> ClinicalEvent:
        item = await self._queue.get()
        if item is _CLOSED:
            raise StopAsyncIteration
        return cast(ClinicalEvent, item)

    def close(self) -> None:
        self._bus._unsubscribe(self)
        self._offer(_CLOSED)


class EventBus(Protocol):
    async def publish(self, event: ClinicalEvent) -> None: ...

    def subscribe(self, maxsize: int = 0) -> Subscription: ...

    async def close(self) -> None: ...


class InMemoryEventBus:
    """Fan-out: cada assinante recebe todos os eventos publicados após a assinatura."""

    def __init__(self) -> None:
        self._subscriptions: list[Subscription] = []

    async def publish(self, event: ClinicalEvent) -> None:
        logger.debug("Evento {} ({}.{})", event.event_id, event.source, event.type)
        for subscription in list(self._subscriptions):
            subscription._offer(event)

    def subscribe(self, maxsize: int = 0) -> Subscription:
        subscription = Subscription(self, maxsize)
        self._subscriptions.append(subscription)
        return subscription

    def _unsubscribe(self, subscription: Subscription) -> None:
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    async def close(self) -> None:
        """Encerra todas as assinaturas; os consumidores saem do `async for`."""
        for subscription in list(self._subscriptions):
            subscription.close()
