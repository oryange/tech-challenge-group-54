"""Relatório da análise de áudio e leitura de baseline a partir de um relatório anterior.

O relatório fica em outputs/ (fora do git) e contém a transcrição completa: é dado de saúde
do paciente e não deve ser anexado a issues, PRs ou mensagens.
"""

import json
from pathlib import Path

from pydantic import ValidationError

from medwatch.audio.acoustic import AcousticFeatures
from medwatch.audio.config import FeatureStats
from medwatch.audio.indicators import baseline_from_features
from medwatch.audio.pipeline import AudioReport


def write_report(report: AudioReport, reports_dir: Path) -> Path:
    """Grava `audio_<report_id>.json`; o id é validado no pipeline (sem / nem ..)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"audio_{report.report_id}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_baseline(path: Path, population: dict[str, FeatureStats]) -> dict[str, FeatureStats]:
    """Baseline do paciente a partir do relatório de uma consulta anterior (sem achados)."""
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".json" or not resolved.is_file():
        raise ValueError("--baseline deve apontar para um relatório audio_<id>.json existente")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
        features = AcousticFeatures.model_validate(data["features"])
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        raise ValueError("--baseline não é um relatório de áudio válido") from exc
    return baseline_from_features(features, population)
