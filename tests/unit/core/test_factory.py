from medwatch.config import get_settings
from medwatch.core.alerts import ConsoleChannel, JsonlChannel, WebhookChannel
from medwatch.core.alerts.factory import build_channels


def test_default_channels_without_webhook():
    channels = build_channels(get_settings())
    assert [type(c) for c in channels] == [ConsoleChannel, JsonlChannel]


def test_webhook_channel_enabled_by_env(monkeypatch):
    monkeypatch.setenv("MEDWATCH_ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
    monkeypatch.setenv("MEDWATCH_ALERT_WEBHOOK_FORMAT", "slack")
    get_settings.cache_clear()
    channels = build_channels(get_settings())
    assert isinstance(channels[-1], WebhookChannel)
