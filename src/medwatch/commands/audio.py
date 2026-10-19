"""medwatch audio: análise de consultas com Azure Speech, Azure Language e acústica local."""

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from azure.core.exceptions import AzureError

from medwatch.audio.azure_language import (
    AzureLanguageAnalyzer,
    LanguageAnalyzer,
    RecordedLanguageAnalyzer,
    RecordingLanguageAnalyzer,
)
from medwatch.audio.azure_stt import (
    AzureSpeechTranscriber,
    RecordedTranscriber,
    RecordingTranscriber,
    SpeechServiceError,
    Transcriber,
)
from medwatch.audio.config import AudioConfig, CriticalTermsConfig
from medwatch.audio.pipeline import AudioAnalyzer, AudioReport
from medwatch.audio.preprocessing import AudioValidationError, validate_file
from medwatch.audio.report import load_baseline, write_report
from medwatch.audio.synthesis import SamplesConfig, build_ssml, synthesize
from medwatch.config import Settings, get_settings
from medwatch.core.alerts.factory import build_alert_manager
from medwatch.core.alerts.manager import DispatchStats
from medwatch.core.bus import InMemoryEventBus
from medwatch.core.events import ClinicalEvent

app = typer.Typer(
    help="Áudio de consultas: Azure Speech + Azure Language + acústica local.",
    no_args_is_help=True,
)

AZURE_HINT = (
    "Azure não configurado: preencha AZURE_SPEECH_KEY, AZURE_LANGUAGE_ENDPOINT e "
    "AZURE_LANGUAGE_KEY no .env (README, seção 'Provisionamento Azure') ou use "
    "MEDWATCH_AZURE_MOCK=true com respostas gravadas."
)

START_TIME_EXAMPLE = "2026-10-19T14:30:00-03:00"

# erros esperados viram mensagem curta, sem stack trace
EXPECTED_ERRORS = (AudioValidationError, ValueError, SpeechServiceError, FileNotFoundError)


def recordings_dir(settings: Settings) -> Path:
    """Respostas gravadas do Azure usadas no modo mock (só de consultas fictícias)."""
    return settings.data_dir / "samples" / "azure"


def recording_key(path: Path) -> str:
    """Respostas gravadas são indexadas pelo conteúdo do áudio, não pelo nome do arquivo."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def build_providers(
    settings: Settings, config: AudioConfig, audio_file: Path, diarize: bool, record: bool
) -> tuple[Transcriber, LanguageAnalyzer]:
    """Escolhe Azure real, Azure com gravação (--record) ou respostas gravadas (mock)."""
    key = recording_key(validate_file(audio_file, config.preprocessing))
    stt_path = recordings_dir(settings) / f"{key}.stt.json"
    language_path = recordings_dir(settings) / f"{key}.language.json"
    if settings.azure_mock:
        if record:
            raise ValueError("--record precisa do Azure real (MEDWATCH_AZURE_MOCK=false)")
        return RecordedTranscriber(stt_path), RecordedLanguageAnalyzer(language_path)

    if not (settings.azure.speech_configured and settings.azure.language_configured):
        raise ValueError(AZURE_HINT)
    transcriber: Transcriber = AzureSpeechTranscriber(settings.azure, config.speech, diarize)
    language: LanguageAnalyzer = AzureLanguageAnalyzer(settings.azure, config.language)
    if record:
        transcriber = RecordingTranscriber(transcriber, stt_path)
        language = RecordingLanguageAnalyzer(language, language_path)
    return transcriber, language


def parse_start_time(value: str | None) -> datetime:
    """Início da gravação em ISO 8601 com fuso; agora (UTC) se omitido."""
    if value is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"--start-time inválido; use ISO 8601, ex.: {START_TIME_EXAMPLE}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"--start-time precisa de fuso horário, ex.: {START_TIME_EXAMPLE}")
    return parsed


async def send_alerts(settings: Settings, events: list[ClinicalEvent]) -> DispatchStats:
    bus = InMemoryEventBus()
    manager = build_alert_manager(settings)
    consumer = asyncio.create_task(
        manager.consume(bus.subscribe(), min_severity=settings.alert_min_severity)
    )
    for event in events:
        await bus.publish(event)
    await bus.close()
    return await consumer


def _print_summary(report: AudioReport, path: Path) -> None:
    sentiment = report.language.sentiment
    typer.echo(
        f"Consulta {report.report_id}: {report.features.duration_seconds:.0f} s · "
        f"{report.transcript.word_count} palavras · "
        f"sentimento {sentiment.sentiment if sentiment else 'indisponível'}"
        + (" · mock" if report.mock else "")
    )
    if not report.language.health_available:
        typer.echo("  Text Analytics for Health indisponível: termos só pelo dicionário local")
    summary = report.language.summary
    if summary:
        label = "extrativo" if summary.mode == "extractive" else "abstrativo"
        typer.echo(f"  Resumo ({label}): {summary.text}")
    if report.language.key_phrases:
        typer.echo(f"  Frases-chave: {', '.join(report.language.key_phrases)}")

    by_status: dict[str, list[str]] = {"affirmed": [], "negated": [], "question": []}
    for mention in report.critical_terms:
        by_status[mention.status].append(mention.term)
    typer.echo(f"  Termos críticos afirmados: {', '.join(by_status['affirmed']) or 'nenhum'}")
    if by_status["negated"]:
        typer.echo(f"  Negados (sem evento): {', '.join(by_status['negated'])}")
    if by_status["question"]:
        typer.echo(f"  Em perguntas (sem evento): {', '.join(by_status['question'])}")

    typer.echo(f"  Indicadores (triagem, baseline {report.baseline}):")
    for indicator in report.indicators:
        zscore = f"z={indicator.zscore:+.2f}" if indicator.zscore is not None else "sem dados"
        severity = indicator.severity.value if indicator.severity else "-"
        typer.echo(f"    {indicator.label:<26} {zscore:<10} {severity}")

    typer.echo(f"  Eventos: {len(report.events)}")
    for event in report.events:
        typer.echo(f"    [{event.severity.value.upper()}] {event.type} (score {event.score:.2f})")
    typer.echo(f"Relatório: {path}")


@app.command()
def analyze(
    audio_file: Annotated[
        Path, typer.Argument(help="Áudio da consulta (.wav, .flac, .ogg ou .mp3).")
    ],
    patient: Annotated[str, typer.Option("--patient", help="Código pseudonimizado, ex.: PAC-001.")],
    start_time: Annotated[
        str | None,
        typer.Option("--start-time", help="Início da gravação, ISO 8601 com fuso (padrão: agora)."),
    ] = None,
    diarize: Annotated[
        bool, typer.Option(help="Separa os falantes (ConversationTranscriber).")
    ] = False,
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", help="Relatório de uma consulta anterior do mesmo paciente."),
    ] = None,
    alert: Annotated[
        bool, typer.Option(help="Envia os eventos ao AlertManager (canais do .env).")
    ] = False,
    record: Annotated[
        bool, typer.Option(help="Grava as respostas do Azure para o modo mock.")
    ] = False,
) -> None:
    """Transcreve e analisa uma consulta; gera outputs/reports/audio_<id>.json."""
    settings = get_settings()
    try:
        config = AudioConfig.load(settings.configs_dir / "audio.yaml")
        terms = CriticalTermsConfig.load(settings.configs_dir / "critical_terms.yaml")
        recorded_at = parse_start_time(start_time)
        patient_baseline = load_baseline(baseline, config.baseline) if baseline else None
        transcriber, language = build_providers(settings, config, audio_file, diarize, record)
        analyzer = AudioAnalyzer(config, terms, transcriber, language)
        report = analyzer.analyze(
            audio_file, patient, recorded_at, baseline=patient_baseline, mock=settings.azure_mock
        )
    except EXPECTED_ERRORS as exc:
        typer.echo(f"Erro: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except AzureError as exc:
        # só o tipo: a mensagem do SDK pode repetir trechos do request
        typer.echo(
            f"Erro no Azure Language ({type(exc).__name__}): confira endpoint e chave", err=True
        )
        raise typer.Exit(code=1) from None

    path = write_report(report, settings.outputs_dir / "reports")
    _print_summary(report, path)
    if alert:
        stats = asyncio.run(send_alerts(settings, report.events))
        typer.echo(
            f"Alertas enviados: {stats.sent} · suprimidos (cooldown): {stats.suppressed} · "
            f"ignorados (< {settings.alert_min_severity}): {stats.ignored}"
        )


@app.command("synth-samples")
def synth_samples(
    name: Annotated[
        list[str] | None, typer.Option("--name", help="Amostra (padrão: todas).")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Sobrescreve amostras já geradas.")] = False,
) -> None:
    """Sintetiza as consultas fictícias de configs/audio_samples.yaml em data/samples/."""
    settings = get_settings()
    if settings.azure_mock or not settings.azure.speech_configured:
        typer.echo("A síntese precisa do Azure Speech real (AZURE_SPEECH_KEY no .env).", err=True)
        raise typer.Exit(code=1)
    samples = SamplesConfig.load(settings.configs_dir / "audio_samples.yaml")
    selected = name or list(samples.samples)
    unknown = sorted(set(selected) - set(samples.samples))
    if unknown:
        typer.echo(f"Amostras desconhecidas: {', '.join(unknown)}", err=True)
        raise typer.Exit(code=1)

    for sample in selected:
        output = settings.data_dir / "samples" / f"{sample}.wav"
        if output.exists() and not overwrite:
            typer.echo(f"= {output.name} já existe (use --overwrite)")
            continue
        try:
            synthesize(build_ssml(samples.samples[sample], samples.voices), output, settings.azure)
        except SpeechServiceError as exc:
            typer.echo(f"Erro em {sample}: {exc}", err=True)
            raise typer.Exit(code=1) from None
        typer.echo(f"✓ {output}")
