"""medwatch simulate / api / dashboard: fusão multimodal e monitoramento em tempo real."""

import asyncio
import os
import subprocess  # noqa: S404 - sobe o Streamlit com o mesmo Python, sem shell
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from medwatch.config import PROJECT_ROOT, Settings, get_settings
from medwatch.core.alerts.factory import build_channels
from medwatch.fusion.config import FusionConfig
from medwatch.streaming.runner import run_local
from medwatch.streaming.simulator import (
    HttpPublisher,
    Replay,
    Scenario,
    Simulator,
    TimedEvent,
    build_timeline,
)

DEFAULT_API_URL = "http://localhost:8000"


def _api_url() -> str:
    return os.environ.get("MEDWATCH_API_URL", DEFAULT_API_URL)


def _scenario_path(settings: Settings, scenario: str) -> Path:
    path = Path(scenario)
    if path.suffix in (".yaml", ".yml") and path.exists():
        return path
    candidate = settings.configs_dir / "scenarios" / f"{scenario}.yaml"
    if candidate.exists():
        return candidate
    available = sorted(p.stem for p in (settings.configs_dir / "scenarios").glob("*.yaml"))
    typer.echo(f"Cenário não encontrado: {scenario}. Disponíveis: {', '.join(available)}", err=True)
    raise typer.Exit(code=1)


def _narrate(item: TimedEvent) -> None:
    event = item.event
    minute = item.offset.total_seconds() / 60
    line = (
        f"[min {minute:>5.1f}] {event.patient_id:<8} {event.source.value}/{event.type} "
        f"({event.severity.value}, score {event.score:.2f})"
    )
    typer.echo(line + (f"\n            ↳ {item.note}" if item.note else ""))


def simulate(
    scenario: Annotated[
        str, typer.Option(help="Nome em configs/scenarios/ ou caminho de um YAML.")
    ] = "demo",
    speed: Annotated[
        float, typer.Option(help="Aceleração: 60 = um minuto por segundo; 0 = sem espera.")
    ] = 60.0,
    target: Annotated[
        str,
        typer.Option(help="local (no processo), api (MEDWATCH_API_URL) ou a URL da API."),
    ] = "local",
    replay: Annotated[
        list[Path] | None,
        typer.Option(help="Relatório JSON de um módulo para reproduzir (repetível)."),
    ] = None,
    replay_patient: Annotated[
        str | None, typer.Option(help="Paciente atribuído aos eventos do --replay.")
    ] = None,
    only_replay: Annotated[
        bool, typer.Option(help="Ignora os eventos roteirizados do cenário.")
    ] = False,
) -> None:
    """Reproduz um cenário em tempo real: eventos → fusão (RiskEngine) → alertas."""
    settings = get_settings()
    loaded = Scenario.load(_scenario_path(settings, scenario))
    if only_replay:
        loaded = loaded.model_copy(update={"events": [], "expected_alerts": []})
    if replay:
        extra = [Replay(path=path, patient_id=replay_patient) for path in replay]
        loaded = loaded.model_copy(update={"replay": [*loaded.replay, *extra]})
    try:
        timeline = build_timeline(loaded, datetime.now(UTC), PROJECT_ROOT)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Erro ao montar o cenário: {exc}", err=True)
        raise typer.Exit(code=1) from None
    if not timeline:
        typer.echo("Cenário sem eventos.", err=True)
        raise typer.Exit(code=1)

    duration = timeline[-1].offset.total_seconds() / 60
    typer.echo(
        f"Cenário {loaded.name}: {len(timeline)} eventos em {duration:.0f} min"
        + (f" (~{duration * 60 / speed:.0f} s a {speed:g}×)" if speed else "")
    )
    if target == "local":
        _simulate_local(settings, loaded, timeline, speed)
    else:
        _simulate_remote(_api_url() if target == "api" else target, timeline, speed)


def _simulate_local(
    settings: Settings, scenario: Scenario, timeline: list[TimedEvent], speed: float
) -> None:
    fusion = FusionConfig.load(settings.configs_dir / "fusion.yaml")
    result = asyncio.run(
        run_local(
            timeline,
            fusion,
            channels=build_channels(settings),
            cooldown=timedelta(minutes=settings.alert_cooldown_minutes),
            min_severity=settings.alert_min_severity,
            speed=speed,
            on_publish=_narrate,
        )
    )
    typer.echo("\nAlertas entregues à equipe médica:")
    for delivered in result.delivered:
        alert = delivered.alert
        typer.echo(
            f"  [min {delivered.at_minutes:>5.1f}] {alert.severity.value.upper():<8} "
            f"{alert.patient_id:<8} {alert.alert_type:<34} {alert.title}"
        )
    stats = result.stats
    median, worst = result.latency_summary()
    typer.echo(
        f"\nAlertas enviados: {stats.sent} · suprimidos (cooldown): {stats.suppressed} · "
        f"eventos sem alerta (< {settings.alert_min_severity}): {stats.ignored}"
    )
    typer.echo(
        f"Latência evento→alerta: mediana {median * 1000:.1f} ms · máx. {worst * 1000:.1f} ms"
    )
    typer.echo(f"Histórico: {settings.alerts_file}")
    if stats.failed:
        typer.echo(f"Sem entrega em nenhum canal: {stats.failed}", err=True)

    # a sequência documentada só vale com a configuração padrão de alertas
    defaults = settings.alert_min_severity == "medium" and settings.alert_cooldown_minutes == 15
    if scenario.expected_alerts and defaults:
        problems = result.mismatches(scenario.expected_alerts)
        if problems:
            typer.echo("\n❌ Diferente da sequência documentada no cenário:", err=True)
            for problem in problems:
                typer.echo(f"  {problem}", err=True)
            raise typer.Exit(code=1)
        typer.echo("✅ Sequência de alertas igual à documentada no cenário")


def _simulate_remote(url: str, timeline: list[TimedEvent], speed: float) -> None:
    async def run() -> int:
        publisher = HttpPublisher(url)
        try:
            published = await Simulator(timeline, publisher, speed=speed, on_publish=_narrate).run()
        finally:
            await publisher.aclose()
        return len(published)

    try:
        sent = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001 - qualquer falha de rede vira mensagem curta
        typer.echo(
            f"Falha ao enviar para {url} ({type(exc).__name__}). A API está no ar?", err=True
        )
        raise typer.Exit(code=1) from None
    typer.echo(f"\n{sent} eventos enviados para {url}. Alertas: {url}/alerts e no dashboard.")


def api(
    host: Annotated[str, typer.Option(help="Interface de rede.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Porta HTTP.")] = 8000,
) -> None:
    """Sobe a API (FastAPI): ingestão, risco por paciente, alertas, ACK e WebSocket."""
    import uvicorn

    uvicorn.run("medwatch.api.app:create_app", factory=True, host=host, port=port)


def dashboard(
    port: Annotated[int, typer.Option(help="Porta do Streamlit.")] = 8501,
    api_url: Annotated[
        str | None, typer.Option(help="URL da API (padrão: MEDWATCH_API_URL ou localhost:8000).")
    ] = None,
) -> None:
    """Sobe o dashboard da equipe médica (Streamlit)."""
    app_file = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    env = os.environ | {"MEDWATCH_API_URL": api_url or _api_url()}
    command = [
        sys.executable, "-m", "streamlit", "run", str(app_file),
        "--server.port", str(port), "--server.headless", "true",
    ]  # fmt: skip
    raise typer.Exit(subprocess.call(command, env=env))  # noqa: S603 - argumentos fixos
