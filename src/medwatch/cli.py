"""CLI principal do MedWatch."""

import asyncio
import importlib.util
import os
import sys
from collections.abc import Callable

import httpx
import typer
from loguru import logger

from medwatch.commands import data as data_commands
from medwatch.commands import video as video_commands
from medwatch.commands import vitals as vitals_commands
from medwatch.config import Settings, get_settings
from medwatch.log import setup_logging

app = typer.Typer(help="MedWatch - monitoramento multimodal de pacientes", no_args_is_help=True)
alerts_app = typer.Typer(help="Alertas para a equipe médica.", no_args_is_help=True)
app.add_typer(alerts_app, name="alerts")
app.add_typer(data_commands.app, name="data")
app.add_typer(vitals_commands.app, name="vitals")
app.add_typer(video_commands.app, name="video")

OPTIONAL_LIBS = {
    "lib.ultralytics": "ultralytics",
    "lib.opencv": "cv2",
    "lib.azure.speech": "azure.cognitiveservices.speech",
    "lib.azure.textanalytics": "azure.ai.textanalytics",
    "lib.sklearn": "sklearn",
}

AZURE_SETUP_HINT = (
    "Chaves Azure ausentes: crie o .env a partir do .env.example e preencha AZURE_SPEECH_KEY, "
    "AZURE_LANGUAGE_ENDPOINT e AZURE_LANGUAGE_KEY (passo a passo no README, seção "
    "'Provisionamento Azure')."
)


@app.callback()
def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    setup_logging(settings)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _ping_speech(settings: Settings, transport: httpx.BaseTransport | None = None) -> bool:
    """Valida a chave do Speech emitindo um token de acesso (não consome cota de áudio)."""
    if not settings.azure.speech_key:
        return False
    url = f"https://{settings.azure.speech_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    with httpx.Client(transport=transport, timeout=10) as client:
        response = client.post(
            url, headers={"Ocp-Apim-Subscription-Key": settings.azure.speech_key.get_secret_value()}
        )
    if response.status_code != 200:
        logger.error("Azure Speech recusou a chave (HTTP {})", response.status_code)
    return response.status_code == 200


def _ping_language(settings: Settings) -> bool:
    """Valida endpoint e chave do Language com uma análise de sentimento mínima."""
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    if not (settings.azure.language_key and settings.azure.language_endpoint):
        return False
    client = TextAnalyticsClient(
        endpoint=settings.azure.language_endpoint,
        credential=AzureKeyCredential(settings.azure.language_key.get_secret_value()),
    )
    result = client.analyze_sentiment(["Paciente estável."], language="pt-BR")[0]
    return not result.is_error


def _run_online_check(label: str, check: Callable[[Settings], bool], settings: Settings) -> bool:
    try:
        return check(settings)
    except Exception as exc:  # noqa: BLE001 - reportar qualquer falha de rede/credencial
        logger.error(
            "{} falhou ({}): confira endpoint/região e chave no .env", label, type(exc).__name__
        )
        return False


@app.command()
def healthcheck(
    online: bool = typer.Option(False, help="Valida credenciais Azure com chamadas reais."),
    strict: bool = typer.Option(False, help="Trata Azure e libs opcionais como obrigatórios."),
) -> None:
    """Verifica ambiente, dependências e credenciais."""
    settings = get_settings()
    required = {
        "python>=3.11": sys.version_info >= (3, 11),
        "outputs.writable": os.access(settings.outputs_dir, os.W_OK),
    }
    optional = {
        "azure.speech.config": settings.azure.speech_configured,
        "azure.language.config": settings.azure.language_configured,
        **{label: _has_module(mod) for label, mod in OPTIONAL_LIBS.items()},
    }

    # --online é um pedido explícito de validação: o resultado passa a ser obrigatório
    if online:
        required["azure.speech.online"] = settings.azure.speech_configured and _run_online_check(
            "Azure Speech", _ping_speech, settings
        )
        required["azure.language.online"] = (
            settings.azure.language_configured
            and _has_module("azure.ai.textanalytics")
            and _run_online_check("Azure Language", _ping_language, settings)
        )

    for label, ok in {**required, **optional}.items():
        typer.echo(f"{'✅' if ok else '❌'} {label}")

    if not (settings.azure.speech_configured and settings.azure.language_configured):
        typer.echo(f"\nℹ️  {AZURE_SETUP_HINT}")

    must_pass = {**required, **optional} if strict else required
    if not all(must_pass.values()):
        raise typer.Exit(code=1)
    logger.info("Healthcheck concluído (env={})", settings.env)


@alerts_app.command("demo")
def alerts_demo() -> None:
    """Dispara um cenário fictício pelo fluxo completo de alertas."""
    from medwatch.core.alerts.demo import run_demo

    settings = get_settings()
    stats = asyncio.run(run_demo(settings))
    typer.echo(
        f"\nAlertas enviados: {stats.sent} · suprimidos (cooldown): {stats.suppressed} · "
        f"ignorados (< {settings.alert_min_severity}): {stats.ignored}"
    )
    typer.echo(f"Histórico: {settings.alerts_file}")
