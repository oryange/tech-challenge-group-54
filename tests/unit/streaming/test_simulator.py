import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from medwatch.config import PROJECT_ROOT
from medwatch.core.events import Severity
from medwatch.fusion.config import FusionConfig
from medwatch.streaming.runner import run_local
from medwatch.streaming.simulator import (
    HttpPublisher,
    Replay,
    Scenario,
    Simulator,
    TimedEvent,
    build_timeline,
    load_report_events,
)
from tests.unit.core.test_events import make_event

T0 = datetime(2026, 10, 19, 14, 0, tzinfo=UTC)
DEMO = PROJECT_ROOT / "configs" / "scenarios" / "demo.yaml"


@pytest.fixture
def fusion() -> FusionConfig:
    return FusionConfig.load(PROJECT_ROOT / "configs" / "fusion.yaml")


def test_demo_scenario_produces_the_documented_alert_sequence(fusion):
    scenario = Scenario.load(DEMO)
    timeline = build_timeline(scenario, T0, PROJECT_ROOT)
    result = asyncio.run(run_local(timeline, fusion, speed=0))
    assert result.mismatches(scenario.expected_alerts) == []
    assert (result.stats.sent, result.stats.suppressed, result.stats.ignored) == (14, 2, 2)


def test_simulator_to_alert_latency_under_two_seconds(fusion):
    """Integração simulador → barramento → RiskEngine → AlertManager em tempo real."""
    scenario = Scenario.load(DEMO)
    timeline = build_timeline(scenario, T0, PROJECT_ROOT)
    # 40 min de cenário em ~0,4 s: os eventos chegam espaçados, como em produção
    result = asyncio.run(run_local(timeline, fusion, speed=6000))
    median, worst = result.latency_summary()
    assert len(result.delivered) == 14
    assert worst < 2.0
    assert median <= worst


def test_simulator_respects_speed():
    timeline = [
        TimedEvent(offset=timedelta(0), event=make_event(timestamp=T0)),
        TimedEvent(offset=timedelta(minutes=1), event=make_event(timestamp=T0)),
    ]
    received = []

    async def publish(event):
        received.append(asyncio.get_running_loop().time())

    published = asyncio.run(Simulator(timeline, publish, speed=600).run())
    assert len(published) == 2
    # 60 s / 600 = 0,1 s entre os dois eventos
    assert received[1] - received[0] == pytest.approx(0.1, abs=0.05)


def test_negative_speed_is_rejected():
    with pytest.raises(ValueError):
        Simulator([], lambda e: asyncio.sleep(0), speed=-1)


def test_replay_rebases_report_events(tmp_path):
    report = {
        "stays": [
            {
                "events": [
                    make_event(timestamp=T0 + timedelta(minutes=5)).model_dump(mode="json"),
                    make_event(timestamp=T0 + timedelta(minutes=7), type="hr_rise").model_dump(
                        mode="json"
                    ),
                ]
            }
        ]
    }
    path = tmp_path / "vitals.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert len(load_report_events(path)) == 2

    start = datetime(2027, 1, 1, tzinfo=UTC)
    scenario = Scenario(
        name="replay", replay=[Replay(path=path, at_minutes=2, patient_id="PAC-009")]
    )
    timeline = build_timeline(scenario, start, tmp_path)
    assert [item.offset for item in timeline] == [timedelta(minutes=2), timedelta(minutes=4)]
    assert {item.event.patient_id for item in timeline} == {"PAC-009"}
    assert timeline[0].event.timestamp == start + timedelta(minutes=2)

    only_hr = Scenario(name="r", replay=[Replay(path=path, types=["hr_rise"])])
    assert [i.event.type for i in build_timeline(only_hr, start, tmp_path)] == ["hr_rise"]


def test_replays_share_the_same_origin(tmp_path):
    # vitais começam às 08:00 e a prescrição às 10:00: a distância real precisa ser mantida
    vitals = tmp_path / "vitals.json"
    vitals.write_text(
        json.dumps({"events": [make_event(timestamp=T0).model_dump(mode="json")]}),
        encoding="utf-8",
    )
    prescriptions = tmp_path / "prescriptions.json"
    later = make_event(timestamp=T0 + timedelta(hours=2), type="dose_increase")
    prescriptions.write_text(
        json.dumps({"events": [later.model_dump(mode="json")]}), encoding="utf-8"
    )
    scenario = Scenario(
        name="multi",
        replay=[Replay(path=prescriptions, at_minutes=1), Replay(path=vitals, at_minutes=1)],
    )
    timeline = build_timeline(scenario, T0, tmp_path)
    assert [(item.event.type, item.offset) for item in timeline] == [
        ("spo2_drop", timedelta(minutes=1)),
        ("dose_increase", timedelta(hours=2, minutes=1)),
    ]


def test_mismatches_report_missing_and_extra(fusion):
    scenario = Scenario.load(DEMO)
    timeline = build_timeline(scenario, T0, PROJECT_ROOT)
    result = asyncio.run(run_local(timeline, fusion, speed=0))
    expected = scenario.expected_alerts[1:] + [
        scenario.expected_alerts[0].model_copy(update={"severity": Severity.HIGH})
    ]
    problems = result.mismatches(expected)
    assert any(p.startswith("faltou") for p in problems)
    assert any(p.startswith("a mais") for p in problems)


def test_http_publisher_posts_to_ingest_events():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"accepted": 1})

    async def scenario():
        publisher = HttpPublisher("http://api", transport=httpx.MockTransport(handler))
        await publisher(make_event())
        await publisher.aclose()

    asyncio.run(scenario())
    assert requests[0].url.path == "/ingest/events"
    assert json.loads(requests[0].content)[0]["type"] == "spo2_drop"
