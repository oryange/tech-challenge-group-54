"""API do MedWatch: ingestão multimodal, risco por paciente, alertas com ACK e WebSocket.

    uvicorn --factory medwatch.api.app:create_app      (ou `medwatch api`)

Fluxo: produtores (simulador, pipelines, `POST /ingest/*`) → EventBus → RiskEngine →
AlertManager → canais (console, alerts.jsonl, webhook) + AlertStore → `WS /ws/alerts`.
"""

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import (
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from loguru import logger
from pydantic import AwareDatetime, BaseModel

from medwatch import __version__
from medwatch.api.ingest import (
    MAX_VIDEO_MB,
    VIDEO_EXTENSIONS,
    AudioIngestor,
    IngestResult,
    VideoIngestor,
    audio_ingestor,
    video_ingestor,
)
from medwatch.api.state import AlertStore, AlertView, EventLog
from medwatch.config import Settings, get_settings
from medwatch.core.alerts.channels import AlertChannel
from medwatch.core.alerts.factory import build_channels
from medwatch.core.alerts.manager import AlertManager
from medwatch.core.bus import InMemoryEventBus
from medwatch.core.events import ClinicalEvent, Severity, Source
from medwatch.fusion.config import FusionConfig
from medwatch.fusion.risk_engine import Contribution, RiskEngine
from medwatch.video.pose.factory import POSE_BACKENDS

# vai para nomes de arquivo dos relatórios: só caracteres seguros (e sem "../")
PATIENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


@dataclass
class Monitor:
    settings: Settings
    bus: InMemoryEventBus
    engine: RiskEngine
    store: AlertStore
    log: EventLog
    audio: AudioIngestor
    video: VideoIngestor
    consumer: asyncio.Task[object]


class Health(BaseModel):
    status: str
    version: str
    consumer_running: bool
    events: int
    patients: int
    alerts: int
    min_severity: Severity
    azure_mock: bool


class Accepted(BaseModel):
    accepted: int


class IngestResponse(BaseModel):
    patient_id: str
    events: list[ClinicalEvent]
    report: str


class PatientSummary(BaseModel):
    patient_id: str
    score: float
    severity: Severity
    sources: list[Source]
    contributions: list[Contribution]
    open_alerts: int
    last_event_at: datetime | None
    as_of: datetime


class PatientDetail(PatientSummary):
    timeline: list[ClinicalEvent]


def get_monitor(request: Request) -> Monitor:
    monitor: Monitor = request.app.state.monitor
    return monitor


MonitorDep = Annotated[Monitor, Depends(get_monitor)]
PatientIdForm = Annotated[str, Form(pattern=PATIENT_ID_PATTERN)]


def create_app(
    settings: Settings | None = None,
    fusion: FusionConfig | None = None,
    audio: AudioIngestor | None = None,
    video: VideoIngestor | None = None,
    channels: list[AlertChannel] | None = None,
) -> FastAPI:
    """Monta a API. Os parâmetros permitem trocar Azure, modelos e canais nos testes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or get_settings()
        cfg.ensure_dirs()
        bus = InMemoryEventBus()
        engine = RiskEngine(
            fusion or FusionConfig.load(cfg.configs_dir / "fusion.yaml"),
            clock=lambda: datetime.now(UTC),
        )
        store = AlertStore(ack_log=cfg.outputs_dir / "alerts" / "acks.jsonl")
        log = EventLog()
        manager = AlertManager(
            [*(build_channels(cfg) if channels is None else channels), store],
            cooldown=timedelta(minutes=cfg.alert_cooldown_minutes),
            # cooldown pelo horário dos eventos: o simulador acelera o tempo
            clock=None,
        )
        # as duas assinaturas existem antes do primeiro publish: nenhum evento se perde
        consumer = asyncio.create_task(
            manager.consume(bus.subscribe(), min_severity=cfg.alert_min_severity, source=engine)
        )
        log_task = asyncio.create_task(log.consume(bus.subscribe()))
        app.state.monitor = Monitor(
            settings=cfg,
            bus=bus,
            engine=engine,
            store=store,
            log=log,
            audio=audio or audio_ingestor(cfg),
            video=video or video_ingestor(cfg),
            consumer=consumer,
        )
        logger.info("API pronta: fusão com {} regras", len(engine.config.rules))
        yield
        await bus.close()
        await asyncio.gather(consumer, log_task, return_exceptions=True)

    app = FastAPI(
        title="MedWatch API",
        version=__version__,
        description="Monitoramento multimodal de pacientes: ingestão, risco e alertas.",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health(monitor: MonitorDep) -> Health:
        running = not monitor.consumer.done()
        return Health(
            status="ok" if running else "degraded",
            version=__version__,
            consumer_running=running,
            events=monitor.log.total,
            patients=len(monitor.engine.patients()),
            alerts=len(monitor.store),
            min_severity=monitor.settings.alert_min_severity,
            azure_mock=monitor.settings.azure_mock,
        )

    @app.post("/ingest/events", status_code=202)
    async def ingest_events(events: list[ClinicalEvent], monitor: MonitorDep) -> Accepted:
        """Eventos já processados por um módulo ou pelo simulador."""
        for event in sorted(events, key=lambda e: e.timestamp):
            await monitor.bus.publish(event)
        return Accepted(accepted=len(events))

    @app.post("/ingest/audio", status_code=201)
    async def ingest_audio(
        file: UploadFile,
        patient_id: PatientIdForm,
        monitor: MonitorDep,
        start_time: Annotated[AwareDatetime | None, Form()] = None,
    ) -> IngestResponse:
        """Consulta gravada: Azure Speech to Text + Azure Language + acústica local."""
        from azure.core.exceptions import AzureError

        from medwatch.audio.azure_stt import SpeechServiceError
        from medwatch.audio.config import AudioConfig
        from medwatch.audio.preprocessing import AudioValidationError

        limits = AudioConfig.load(monitor.settings.configs_dir / "audio.yaml").preprocessing
        path = await _save_upload(file, tuple(limits.allowed_extensions), limits.max_file_mb)
        try:
            result = await asyncio.to_thread(
                monitor.audio, path, patient_id, start_time or datetime.now(UTC)
            )
        except (AudioValidationError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from None
        except (SpeechServiceError, AzureError) as exc:
            # só o tipo: a mensagem do SDK pode repetir trechos do request
            raise HTTPException(502, f"Azure indisponível ({type(exc).__name__})") from None
        finally:
            path.unlink(missing_ok=True)
        return await _publish_result(monitor, patient_id, result)

    @app.post("/ingest/video", status_code=201)
    async def ingest_video(
        file: UploadFile,
        patient_id: PatientIdForm,
        monitor: MonitorDep,
        start_time: Annotated[AwareDatetime | None, Form()] = None,
        exercise: Annotated[str | None, Form()] = None,
        scene: Annotated[str | None, Form()] = None,
        backend: Annotated[str, Form()] = "yolo",
    ) -> IngestResponse:
        """Vídeo clínico: pose (YOLOv8-pose/OpenPose), objetos por cena, quedas e desvios."""
        if backend not in POSE_BACKENDS:
            raise HTTPException(422, f"backend deve ser um de {POSE_BACKENDS}")
        path = await _save_upload(file, VIDEO_EXTENSIONS, MAX_VIDEO_MB)
        try:
            result = await asyncio.to_thread(
                monitor.video,
                path,
                patient_id,
                start_time or datetime.now(UTC),
                exercise,
                scene,
                backend,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        finally:
            path.unlink(missing_ok=True)
        return await _publish_result(monitor, patient_id, result)

    @app.get("/patients")
    async def patients(monitor: MonitorDep) -> list[PatientSummary]:
        """Pacientes do maior risco para o menor."""
        return [_summary(monitor, risk.patient_id) for risk in monitor.engine.patients()]

    @app.get("/patients/{patient_id}/risk")
    async def patient_risk(patient_id: str, monitor: MonitorDep) -> PatientDetail:
        """Risco atual, contribuições por modalidade e linha do tempo dos eventos."""
        timeline = monitor.log.for_patient(patient_id)
        if not timeline:
            raise HTTPException(404, "Paciente sem eventos")
        return PatientDetail(**dict(_summary(monitor, patient_id)), timeline=timeline)

    @app.get("/alerts")
    async def alerts(
        monitor: MonitorDep,
        patient_id: str | None = None,
        acked: bool | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> list[AlertView]:
        """Alertas entregues, do mais recente para o mais antigo."""
        return monitor.store.recent(patient_id=patient_id, acked=acked, limit=limit)

    @app.post("/alerts/{alert_id}/ack")
    async def ack_alert(alert_id: UUID, monitor: MonitorDep) -> AlertView:
        """A equipe reconhece o alerta; o ACK vai para os WebSockets e para acks.jsonl."""
        alert = monitor.store.ack(alert_id)
        if alert is None:
            raise HTTPException(404, "Alerta não encontrado")
        return alert

    @app.websocket("/ws/alerts")
    async def ws_alerts(websocket: WebSocket) -> None:
        """Feed em tempo real: `{"type": "alert" | "ack", "alert": {...}}`."""
        monitor: Monitor = websocket.app.state.monitor
        await websocket.accept()
        queue = monitor.store.listen()

        async def pump() -> None:
            while True:
                await websocket.send_json(await queue.get())

        async def drain() -> None:
            # o cliente não envia nada; ler é o jeito de perceber que ele desconectou
            while True:
                await websocket.receive_text()

        tasks = {asyncio.create_task(pump()), asyncio.create_task(drain())}
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                with suppress(WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                    task.result()
        finally:
            monitor.store.unlisten(queue)

    return app


def _summary(monitor: Monitor, patient_id: str) -> PatientSummary:
    risk = monitor.engine.risk(patient_id)
    timeline = monitor.log.for_patient(patient_id)
    return PatientSummary(
        patient_id=patient_id,
        score=risk.score,
        severity=risk.severity,
        sources=list(risk.sources),
        contributions=list(risk.contributions),
        open_alerts=len(monitor.store.recent(patient_id=patient_id, acked=False, limit=1000)),
        last_event_at=timeline[-1].timestamp if timeline else None,
        as_of=risk.as_of,
    )


async def _publish_result(
    monitor: Monitor, patient_id: str, result: IngestResult
) -> IngestResponse:
    for event in result.events:
        await monitor.bus.publish(event)
    return IngestResponse(patient_id=patient_id, events=result.events, report=str(result.report))


async def _save_upload(file: UploadFile, allowed: tuple[str, ...], max_mb: float) -> Path:
    """Grava o upload num arquivo temporário; do nome original só a extensão é usada."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(415, f"Formato não suportado: use {', '.join(allowed)}")
    fd, name = tempfile.mkstemp(prefix="medwatch_", suffix=suffix)
    path = Path(name)
    size, limit = 0, int(max_mb * 1024 * 1024)
    try:
        with os.fdopen(fd, "wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"Arquivo maior que {max_mb:g} MB")
                fh.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path
