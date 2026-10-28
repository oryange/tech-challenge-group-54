"""Envio de eventos ao AlertManager a partir dos comandos da CLI."""

import asyncio

import typer

from medwatch.config import Settings
from medwatch.core.alerts.factory import build_alert_manager
from medwatch.core.alerts.manager import DispatchStats
from medwatch.core.bus import InMemoryEventBus
from medwatch.core.events import ClinicalEvent


async def _dispatch(
    settings: Settings, events: list[ClinicalEvent], event_time: bool
) -> DispatchStats:
    bus = InMemoryEventBus()
    manager = build_alert_manager(settings, event_time=event_time)
    consumer = asyncio.create_task(
        manager.consume(bus.subscribe(), min_severity=settings.alert_min_severity)
    )
    for event in sorted(events, key=lambda e: e.timestamp):
        await bus.publish(event)
    await bus.close()
    return await consumer


def send_and_report(
    settings: Settings, events: list[ClinicalEvent], event_time: bool = True
) -> DispatchStats:
    """Publica os eventos, entrega pelos canais do .env e imprime o resumo.

    `event_time=True` mede o cooldown pelo horário dos eventos (reprocessamento histórico).
    """
    stats = asyncio.run(_dispatch(settings, events, event_time))
    typer.echo(
        f"Alertas enviados: {stats.sent} · suprimidos (cooldown): {stats.suppressed} · "
        f"ignorados (< {settings.alert_min_severity}): {stats.ignored}"
    )
    if stats.failed:
        typer.echo(f"Sem entrega em nenhum canal: {stats.failed}", err=True)
    return stats
