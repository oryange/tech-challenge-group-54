"""Canais de entrega de alertas: console, arquivo JSONL e webhook (Teams/Slack/Discord)."""

import asyncio
import json
from pathlib import Path
from typing import Literal, Protocol

import httpx
from loguru import logger

from medwatch.core.events import Alert, Severity

WebhookFormat = Literal["generic", "teams", "slack", "discord"]

_LOG_LEVELS: dict[Severity, str] = {
    Severity.INFO: "INFO",
    Severity.LOW: "INFO",
    Severity.MEDIUM: "WARNING",
    Severity.HIGH: "ERROR",
    Severity.CRITICAL: "CRITICAL",
}


class ChannelError(RuntimeError):
    """Falha definitiva de entrega em um canal."""


class AlertChannel(Protocol):
    name: str

    async def send(self, alert: Alert) -> None: ...


def format_alert_text(alert: Alert) -> str:
    """Resumo legível do alerta, sem PII além do patient_id pseudonimizado."""
    lines = [
        f"🚨 [{alert.severity.value.upper()}] Paciente {alert.patient_id}: {alert.title}",
        f"Ação recomendada: {alert.recommended_action}",
        "Motivos:",
    ]
    lines += [
        f"  • {event.source.value}/{event.type} (score {event.score:.2f}) "
        f"às {event.timestamp:%H:%M:%S} UTC"
        for event in alert.reasons
    ]
    return "\n".join(lines)


class ConsoleChannel:
    name = "console"

    async def send(self, alert: Alert) -> None:
        logger.log(_LOG_LEVELS[alert.severity], "\n{}", format_alert_text(alert))


class JsonlChannel:
    """Um alerta por linha em JSON: fácil de auditar e de ler no dashboard."""

    name = "jsonl"

    def __init__(self, path: Path) -> None:
        self.path = path

    async def send(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(alert.model_dump_json() + "\n")


class WebhookChannel:
    """POST do alerta em um webhook. Repete em erro de rede, 429 e 5xx com backoff exponencial."""

    name = "webhook"

    def __init__(
        self,
        url: str,
        fmt: WebhookFormat = "generic",
        timeout: float = 5.0,
        retries: int = 3,
        backoff: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = url
        self._fmt = fmt
        self._timeout = timeout
        self._retries = retries
        self._backoff = backoff
        self._transport = transport

    def build_payload(self, alert: Alert) -> dict[str, object]:
        text = format_alert_text(alert)
        if self._fmt == "slack":
            return {"text": text}
        if self._fmt == "discord":
            return {"content": text[:2000]}  # limite de caracteres do Discord
        if self._fmt == "teams":
            # formato aceito pelo webhook do Teams Workflows (Adaptive Card)
            return {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": [
                                {
                                    "type": "TextBlock",
                                    "text": text.splitlines()[0],
                                    "weight": "Bolder",
                                    "size": "Medium",
                                    "wrap": True,
                                },
                                {
                                    "type": "TextBlock",
                                    "text": "\n\n".join(text.splitlines()[1:]),
                                    "wrap": True,
                                },
                            ],
                        },
                    }
                ],
            }
        return json.loads(alert.model_dump_json())

    async def send(self, alert: Alert) -> None:
        payload = self.build_payload(alert)
        reason = "nenhuma tentativa"
        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
            for attempt in range(1, self._retries + 1):
                try:
                    response = await client.post(self._url, json=payload)
                except httpx.TransportError as exc:
                    reason = type(exc).__name__
                else:
                    if response.is_success:
                        return
                    if response.status_code != 429 and response.status_code < 500:
                        raise ChannelError(
                            f"Webhook recusou o alerta (HTTP {response.status_code})"
                        )
                    reason = f"HTTP {response.status_code}"
                if attempt < self._retries:
                    logger.warning(
                        "Webhook falhou ({}), tentativa {}/{}", reason, attempt, self._retries
                    )
                    await asyncio.sleep(self._backoff * 2 ** (attempt - 1))
        raise ChannelError(f"Webhook indisponível após {self._retries} tentativas ({reason})")
