"""medwatch prescriptions: anomalias na evolução de prescrições (MIMIC-IV demo)."""

import json

import typer

from medwatch.anomaly.clinical_report import plot_prescriptions, prescriptions_payload, write_json
from medwatch.anomaly.prescription_evaluation import evaluate_prescriptions
from medwatch.anomaly.prescriptions import (
    PrescriptionAnalyzer,
    PrescriptionsConfig,
    PrescriptionStay,
    iter_prescription_stays,
)
from medwatch.commands._alerts import send_and_report
from medwatch.config import Settings, get_settings
from medwatch.core.events import ClinicalEvent
from medwatch.datasets import MIMIC_IV_DEMO

app = typer.Typer(
    help="Prescrições: salto de dose, duplicidade, suspensão precoce e padrão do dia."
)


def _load(settings: Settings) -> tuple[PrescriptionsConfig, list[PrescriptionStay]]:
    dataset_dir = settings.data_dir / "raw" / MIMIC_IV_DEMO.dirname
    if not (dataset_dir / "hosp" / "prescriptions.csv.gz").exists():
        typer.echo("Dataset não encontrado. Rode antes: medwatch data download", err=True)
        raise typer.Exit(code=1)
    config = PrescriptionsConfig.load(settings.configs_dir / "prescriptions.yaml")
    return config, iter_prescription_stays(dataset_dir, config)


def _describe(event: ClinicalEvent) -> str:
    ev = event.evidence
    if event.type in ("dose_increase", "dose_decrease"):
        return f"{ev['drug']} {ev['previous_dose']} → {ev['new_dose']} ({ev['ratio']}×)"
    if event.type == "duplicate_therapy":
        drugs = ev["drugs"] if isinstance(ev["drugs"], list) else []
        return f"{ev['class']}: {' + '.join(str(d) for d in drugs)} por {ev['overlap_hours']} h"
    if event.type == "early_stop":
        return f"{ev['drug']} suspenso após {ev['duration_hours']} h e não represcrito"
    return f"score {event.score:.2f} · dia {ev.get('day')}"


@app.command("list")
def list_patients() -> None:
    """Lista os pacientes com prescrições (subject_id, internações e prescrições)."""
    _, stays = _load(get_settings())
    by_patient: dict[int, list[PrescriptionStay]] = {}
    for stay in stays:
        by_patient.setdefault(stay.subject_id, []).append(stay)
    typer.echo(f"{'subject_id':>10}  {'internações':>11}  {'prescrições':>11}")
    for subject_id, patient_stays in sorted(by_patient.items()):
        orders = sum(len(s.orders) for s in patient_stays)
        typer.echo(f"{subject_id:>10}  {len(patient_stays):>11}  {orders:>11}")


@app.command()
def analyze(
    patient: int = typer.Option(..., "--patient", help="subject_id do MIMIC (ver: list)"),
    plot: bool = typer.Option(True, help="Gera a linha do tempo PNG de cada internação."),
    alert: bool = typer.Option(False, help="Envia os eventos ao AlertManager (canais do .env)."),
) -> None:
    """Analisa as prescrições de um paciente e gera relatório JSON e gráfico."""
    settings = get_settings()
    config, stays = _load(settings)
    selected = [s for s in stays if s.subject_id == patient]
    if not selected:
        typer.echo(
            f"Paciente {patient} não encontrado. Veja: medwatch prescriptions list", err=True
        )
        raise typer.Exit(code=1)

    analyzer = PrescriptionAnalyzer(config).fit(stays)
    analyses = [analyzer.analyze(stay) for stay in selected]
    reports = settings.outputs_dir / "reports"
    for analysis in analyses:
        stay = analysis.stay
        typer.echo(
            f"\n{stay.patient_id} · internação {stay.hadm_id} · {len(stay.orders)} prescrições · "
            f"{len(analysis.days)} dias · {len(analysis.events)} eventos"
        )
        for event in analysis.events:
            typer.echo(
                f"  {event.timestamp:%Y-%m-%d %H:%M}  {event.severity.value:<8}  "
                f"{event.type:<29}  {_describe(event)}"
            )
        if plot:
            png = reports / f"prescriptions_{patient}_{stay.hadm_id}.png"
            typer.echo(f"  gráfico: {plot_prescriptions(analysis, png)}")

    report = write_json(prescriptions_payload(analyses), reports / f"prescriptions_{patient}.json")
    typer.echo(f"\nRelatório: {report}")
    if alert:
        send_and_report(settings, [e for a in analyses for e in a.events])


@app.command()
def evaluate(
    per_stay: int | None = typer.Option(None, help="Anomalias injetadas por internação."),
    seed: int = typer.Option(42, help="Semente aleatória (reprodutibilidade)."),
) -> None:
    """Precision/recall/F1 com anomalias injetadas nas prescrições reais."""
    settings = get_settings()
    config, stays = _load(settings)
    analyzer = PrescriptionAnalyzer(config).fit(stays)
    per_stay = per_stay or config.evaluation.per_stay
    metrics = evaluate_prescriptions(
        analyzer, stays, per_stay=per_stay, seed=seed, min_days=config.evaluation.min_days
    )
    typer.echo(f"\n{'método':<18} {'precision':>9} {'recall':>7} {'F1':>6}  injetadas")
    for name, metric in metrics.items():
        typer.echo(
            f"{name:<18} {metric.precision:>9.3f} {metric.recall:>7.3f} {metric.f1:>6.3f}  "
            f"{metric.injected}"
        )
    combined = metrics["combined"]
    typer.echo(
        f"\nInternações avaliadas: {combined.stays} de {len(stays)} "
        f"(≥ {config.evaluation.min_days} dias de prescrições)"
    )
    typer.echo("\nRecall por tipo de anomalia (combinado):")
    for kind, (hit, total) in sorted(combined.by_kind.items()):
        typer.echo(f"  {kind:<18} {hit / total:.3f}")

    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    result = {
        "per_stay": per_stay,
        "seed": seed,
        "stays": len(stays),
        "stays_evaluated": combined.stays,
        "metrics": {name: metric.as_dict() for name, metric in metrics.items()},
    }
    path = reports / "prescriptions_evaluation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"\nMétricas: {path}")
