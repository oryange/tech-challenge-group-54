"""Monta o AlertManager a partir das configurações (.env)."""

from datetime import timedelta

from medwatch.config import Settings
from medwatch.core.alerts.channels import AlertChannel, ConsoleChannel, JsonlChannel, WebhookChannel
from medwatch.core.alerts.manager import AlertManager


def build_channels(settings: Settings) -> list[AlertChannel]:
    channels: list[AlertChannel] = [ConsoleChannel(), JsonlChannel(settings.alerts_file)]
    if settings.alert_webhook_url and settings.alert_webhook_url.get_secret_value():
        channels.append(
            WebhookChannel(
                settings.alert_webhook_url.get_secret_value(), fmt=settings.alert_webhook_format
            )
        )
    return channels


def build_alert_manager(
    settings: Settings, event_time: bool = False, max_in_flight: int = 32
) -> AlertManager:
    """`event_time=True` mede o cooldown pelo horário dos eventos (reprocessamento histórico).

    `max_in_flight=1` entrega um alerta por vez, na ordem dos eventos (usado no demo, para as
    mensagens chegarem ao chat na mesma ordem da narração).
    """
    cooldown = timedelta(minutes=settings.alert_cooldown_minutes)
    channels = build_channels(settings)
    if event_time:
        return AlertManager(channels, cooldown=cooldown, clock=None, max_in_flight=max_in_flight)
    return AlertManager(channels, cooldown=cooldown, max_in_flight=max_in_flight)


def describe_channels(settings: Settings) -> str:
    """Canais ativos para exibir no terminal, sem a URL do webhook (ela é segredo)."""
    names = [channel.name for channel in build_channels(settings)]
    return ", ".join(
        f"webhook ({settings.alert_webhook_format})" if name == "webhook" else name
        for name in names
    )
