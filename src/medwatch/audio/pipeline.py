"""Pipeline de áudio: arquivo → transcrição (Azure) → linguagem (Azure) → termos críticos →
features acústicas (local) → indicadores → `ClinicalEvent(source=audio)`.
"""

import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import AwareDatetime, BaseModel, JsonValue

from medwatch.audio.acoustic import AcousticFeatures, extract_features
from medwatch.audio.azure_language import LanguageAnalysis, LanguageAnalyzer
from medwatch.audio.azure_stt import Transcriber
from medwatch.audio.config import AudioConfig, CriticalTermsConfig, FeatureStats
from medwatch.audio.critical_terms import CriticalTermDetector, TermMention
from medwatch.audio.indicators import IndicatorResult, score_indicators
from medwatch.audio.preprocessing import load_audio
from medwatch.audio.transcript import Transcript
from medwatch.core.events import ClinicalEvent, Severity, Source

# identificador pseudonimizado (ex.: PAC-001); também compõe o nome do relatório
PATIENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_patient_id(patient_id: str) -> str:
    """Recusa nomes e caracteres de caminho: o id compõe o nome do relatório."""
    if not PATIENT_ID.fullmatch(patient_id):
        raise ValueError(
            "patient_id deve ser um código pseudonimizado (letras, números, - ou _), nunca o nome"
        )
    return patient_id


class AudioReport(BaseModel):
    """Tudo o que a análise produziu; gravado em outputs/reports/audio_<id>.json."""

    report_id: str
    patient_id: str
    recorded_at: AwareDatetime
    source_file: str
    mock: bool
    baseline: Literal["population", "patient"]
    transcript: Transcript
    language: LanguageAnalysis
    critical_terms: list[TermMention]
    features: AcousticFeatures
    indicators: list[IndicatorResult]
    events: list[ClinicalEvent]


class AudioAnalyzer:
    """Orquestra as etapas do áudio a partir de provedores injetáveis (Azure ou mock)."""

    def __init__(
        self,
        config: AudioConfig,
        terms: CriticalTermsConfig,
        transcriber: Transcriber,
        language: LanguageAnalyzer,
    ) -> None:
        unknown = {
            ind.related_terms
            for ind in config.indicators.values()
            if ind.related_terms and ind.related_terms not in terms.categories
        }
        if unknown:
            raise ValueError(f"related_terms sem categoria em critical_terms.yaml: {unknown}")
        self._config = config
        self._terms = terms
        self._detector = CriticalTermDetector(terms)
        self._transcriber = transcriber
        self._language = language

    def analyze(
        self,
        path: Path,
        patient_id: str,
        recorded_at: datetime,
        baseline: dict[str, FeatureStats] | None = None,
        mock: bool = False,
    ) -> AudioReport:
        """Analisa um arquivo e devolve o relatório com os eventos clínicos."""
        validate_patient_id(patient_id)
        if recorded_at.tzinfo is None:
            raise ValueError("recorded_at precisa de fuso horário, ex.: 2026-10-19T14:30:00-03:00")
        clip = load_audio(path, self._config.preprocessing)
        transcript = self._transcriber.transcribe(clip)
        text = transcript.text
        if text.strip():
            language = self._language.analyze(text)
        else:  # consulta sem fala reconhecida: nada a enviar ao Language
            language = LanguageAnalysis(health_available=False)
        mentions = self._detector.detect(text, language.entities)
        active = {m.category for m in mentions if m.affirmed}
        features = extract_features(
            clip, self._config.acoustic, word_count=transcript.word_count or None
        )
        indicators = score_indicators(features, self._config, baseline, active)

        report_id = f"{patient_id}_{recorded_at.astimezone(UTC):%Y%m%dT%H%M%SZ}"
        events = self._term_events(patient_id, recorded_at, transcript, mentions, language)
        events += self._indicator_events(
            patient_id, recorded_at, features, indicators, baseline is not None
        )
        logger.info(
            "Áudio {}: {:.0f} s, {} palavras, {} termos críticos ({} afirmados), {} eventos",
            report_id,
            clip.duration,
            transcript.word_count,
            len(mentions),
            sum(m.affirmed for m in mentions),
            len(events),
        )
        return AudioReport(
            report_id=report_id,
            patient_id=patient_id,
            recorded_at=recorded_at,
            source_file=path.name,
            mock=mock,
            baseline="patient" if baseline is not None else "population",
            transcript=transcript,
            language=language,
            critical_terms=mentions,
            features=features,
            indicators=indicators,
            events=events,
        )

    def _term_events(
        self,
        patient_id: str,
        recorded_at: datetime,
        transcript: Transcript,
        mentions: list[TermMention],
        language: LanguageAnalysis,
    ) -> list[ClinicalEvent]:
        events = []
        sentiment = language.sentiment.sentiment if language.sentiment else None
        for name, category in self._terms.categories.items():
            found = [m for m in mentions if m.category == name]
            active = [m for m in found if m.affirmed]
            if not active:
                continue
            first = min(active, key=lambda m: m.start)
            terms: list[JsonValue] = [term for term in sorted({m.term for m in active})]
            events.append(
                ClinicalEvent(
                    patient_id=patient_id,
                    source=Source.AUDIO,
                    type=f"{name}_terms",
                    timestamp=recorded_at + timedelta(seconds=transcript.time_at(first.start)),
                    score=min(1.0, 0.5 + 0.1 * (len(active) - 1)),
                    severity=category.severity,
                    # só metadados: nenhum trecho da transcrição sai do relatório local
                    evidence={
                        "category": name,
                        "label": category.label,
                        "terms": terms,
                        "mentions": len(active),
                        "negated_mentions": sum(m.status == "negated" for m in found),
                        "question_mentions": sum(m.status == "question" for m in found),
                        "sentiment": sentiment,
                    },
                )
            )
        return events

    def _indicator_events(
        self,
        patient_id: str,
        recorded_at: datetime,
        features: AcousticFeatures,
        indicators: list[IndicatorResult],
        patient_baseline: bool,
    ) -> list[ClinicalEvent]:
        midpoint = self._config.indicator_severity.medium
        events = []
        for result in indicators:
            if result.severity is None or result.zscore is None:
                continue
            events.append(
                ClinicalEvent(
                    patient_id=patient_id,
                    source=Source.AUDIO,
                    type=f"{result.name}_indicator",
                    # o indicador só existe depois de ouvir a consulta inteira
                    timestamp=recorded_at + timedelta(seconds=features.duration_seconds),
                    score=round(1 / (1 + math.exp(-(result.zscore - midpoint))), 3),
                    severity=result.severity,
                    evidence={
                        "indicator": result.name,
                        "label": result.label,
                        "zscore": result.zscore,
                        "baseline": "patient" if patient_baseline else "population",
                        "features": {
                            name: {"value": score.value, "zscore": score.zscore}
                            for name, score in result.features.items()
                        },
                        "related_terms": result.related_terms_present,
                        "screening_only": True,
                    },
                )
            )
        return events


def max_severity(events: list[ClinicalEvent]) -> Severity | None:
    """Maior severidade entre os eventos (None se não houver eventos)."""
    return max((e.severity for e in events), default=None)
