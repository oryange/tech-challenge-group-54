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


def build_alert_manager(settings: Settings) -> AlertManager:
    return AlertManager(
        build_channels(settings), cooldown=timedelta(minutes=settings.alert_cooldown_minutes)
    )
