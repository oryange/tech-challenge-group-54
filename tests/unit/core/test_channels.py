import asyncio
import json

import httpx
import pytest

from medwatch.core.alerts import ChannelError, ConsoleChannel, JsonlChannel, WebhookChannel
from medwatch.core.events import Alert, Severity
from tests.unit.core.test_events import make_event

URL = "https://hooks.example.com/abc"


def sample_alert() -> Alert:
    return Alert.from_event(make_event(severity=Severity.CRITICAL))


def test_jsonl_appends_one_valid_json_per_line(tmp_path):
    path = tmp_path / "alerts" / "alerts.jsonl"
    channel = JsonlChannel(path)
    first, second = sample_alert(), sample_alert()
    asyncio.run(channel.send(first))
    asyncio.run(channel.send(second))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [Alert.model_validate_json(line) for line in lines] == [first, second]


def test_console_channel_logs_alert():
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{level} {message}")
    try:
        asyncio.run(ConsoleChannel().send(sample_alert()))
    finally:
        logger.remove(sink_id)
    assert messages and messages[0].startswith("CRITICAL")
    assert "Paciente PAC-001" in messages[0]


def recording_transport(statuses: list[int], requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(statuses[min(len(requests), len(statuses)) - 1])

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    ("fmt", "key"),
    [("slack", "text"), ("discord", "content"), ("teams", "attachments"), ("generic", "alert_id")],
)
def test_webhook_payload_formats(fmt, key):
    requests: list[httpx.Request] = []
    channel = WebhookChannel(URL, fmt=fmt, transport=recording_transport([200], requests))
    asyncio.run(channel.send(sample_alert()))
    body = json.loads(requests[0].content)
    assert key in body
    assert "PAC-001" in json.dumps(body, ensure_ascii=False)


def test_webhook_retries_on_server_error_then_succeeds():
    requests: list[httpx.Request] = []
    transport = recording_transport([503, 429, 200], requests)
    channel = WebhookChannel(URL, retries=3, backoff=0, transport=transport)
    asyncio.run(channel.send(sample_alert()))
    assert len(requests) == 3


def test_webhook_does_not_retry_client_error():
    requests: list[httpx.Request] = []
    channel = WebhookChannel(URL, backoff=0, transport=recording_transport([400], requests))
    with pytest.raises(ChannelError, match="HTTP 400"):
        asyncio.run(channel.send(sample_alert()))
    assert len(requests) == 1


def test_webhook_gives_up_after_retries():
    requests: list[httpx.Request] = []
    channel = WebhookChannel(
        URL, retries=2, backoff=0, transport=recording_transport([500], requests)
    )
    with pytest.raises(ChannelError, match="2 tentativas"):
        asyncio.run(channel.send(sample_alert()))
    assert len(requests) == 2


def test_webhook_retries_on_network_error():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("sem rede", request=request)
        return httpx.Response(200)

    channel = WebhookChannel(URL, backoff=0, transport=httpx.MockTransport(handler))
    asyncio.run(channel.send(sample_alert()))
    assert calls["n"] == 2
