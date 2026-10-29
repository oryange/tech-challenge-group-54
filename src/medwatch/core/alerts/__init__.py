"""Entrega de alertas à equipe médica."""

from medwatch.core.alerts.channels import (
    AlertChannel,
    ChannelError,
    ConsoleChannel,
    JsonlChannel,
    WebhookChannel,
)
from medwatch.core.alerts.manager import (
    AlertManager,
    AlertSource,
    DispatchOutcome,
    DispatchStats,
)

__all__ = [
    "AlertChannel",
    "AlertManager",
    "AlertSource",
    "ChannelError",
    "ConsoleChannel",
    "DispatchOutcome",
    "DispatchStats",
    "JsonlChannel",
    "WebhookChannel",
]
