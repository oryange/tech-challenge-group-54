"""Cenário fictício que percorre o fluxo completo: evento → barramento → alerta → canais."""

import asyncio
from datetime import UTC, datetime, timedelta

from medwatch.config import Settings
from medwatch.core.alerts.factory import build_alert_manager
from medwatch.core.alerts.manager import DispatchStats
from medwatch.core.bus import InMemoryEventBus
from medwatch.core.events import ClinicalEvent, Severity, Source


def demo_events(start: datetime | None = None) -> list[ClinicalEvent]:
    """Esperado com min_severity=medium: 4 enviados, 1 suprimido, 1 ignorado."""
    t0 = start or datetime.now(UTC)
    return [
        ClinicalEvent(
            patient_id="PAC-001",
            source=Source.VITALS,
            type="spo2_drop",
            timestamp=t0,
            score=0.82,
            severity=Severity.HIGH,
            evidence={"spo2": 88, "baseline": 96, "window_min": 5},
        ),
        ClinicalEvent(
            patient_id="PAC-001",
            source=Source.AUDIO,
            type="dyspnea_terms",
            timestamp=t0 + timedelta(minutes=1),
            score=0.64,
            severity=Severity.MEDIUM,
            evidence={"terms": ["falta de ar", "cansaço"], "sentiment": "negative"},
        ),
        # mesma chave e mesma severidade dentro do cooldown: deve ser suprimido
        ClinicalEvent(
            patient_id="PAC-001",
            source=Source.VITALS,
            type="spo2_drop",
            timestamp=t0 + timedelta(minutes=2),
            score=0.85,
            severity=Severity.HIGH,
            evidence={"spo2": 87, "baseline": 96, "window_min": 5},
        ),
        # mesma chave, severidade maior: escalonamento passa pelo cooldown
        ClinicalEvent(
            patient_id="PAC-001",
            source=Source.VITALS,
            type="spo2_drop",
            timestamp=t0 + timedelta(minutes=3),
            score=0.95,
            severity=Severity.CRITICAL,
            evidence={"spo2": 84, "baseline": 96, "window_min": 5},
        ),
        # abaixo da severidade mínima: ignorado
        ClinicalEvent(
            patient_id="PAC-002",
            source=Source.MOVEMENT,
            type="low_activity",
            timestamp=t0 + timedelta(minutes=4),
            score=0.30,
            severity=Severity.LOW,
            evidence={"active_minutes_last_hour": 4},
        ),
        ClinicalEvent(
            patient_id="PAC-002",
            source=Source.VIDEO,
            type="fall_detected",
            timestamp=t0 + timedelta(minutes=5),
            score=0.91,
            severity=Severity.CRITICAL,
            evidence={"hip_velocity_m_s": 2.4, "bbox_aspect_ratio": 1.8},
        ),
    ]


async def run_demo(settings: Settings) -> DispatchStats:
    bus = InMemoryEventBus()
    manager = build_alert_manager(settings)
    consumer = asyncio.create_task(
        manager.consume(bus.subscribe(), min_severity=settings.alert_min_severity)
    )
    for event in demo_events():
        await bus.publish(event)
    await bus.close()
    return await consumer
