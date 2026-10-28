import numpy as np
import pandas as pd
import pytest

from medwatch.anomaly.prescription_evaluation import (
    evaluate_prescriptions,
    inject_prescription_anomalies,
)
from medwatch.anomaly.prescriptions import (
    PrescriptionAnalyzer,
    PrescriptionsConfig,
    PrescriptionStay,
    iter_prescription_stays,
    normalize_orders,
    normalize_unit,
    parse_dose,
)
from medwatch.config import PROJECT_ROOT
from medwatch.core.events import Severity, Source

CONFIG = PrescriptionsConfig.load(PROJECT_ROOT / "configs" / "prescriptions.yaml")
T0 = pd.Timestamp("2180-01-01 08:00")
COLUMNS = [
    "subject_id", "hadm_id", "starttime", "stoptime", "drug_type",
    "drug", "dose_val_rx", "dose_unit_rx", "route",
]  # fmt: skip


def order(drug, start_h, stop_h, dose="1", unit="mg", route="PO", drug_type="MAIN"):
    stop = None if stop_h is None else str(T0 + pd.Timedelta(hours=stop_h))
    return (1, 10, str(T0 + pd.Timedelta(hours=start_h)), stop, drug_type, drug, dose, unit, route)


def stay(*rows) -> PrescriptionStay:
    # âncora: internação de 4 dias, para a regra de suspensão saber que ela continuou
    anchor = order("Docusate Sodium", 0, 96, "100")
    raw = pd.DataFrame([anchor, *rows, order("Senna", 95, 96, "8.6")], columns=COLUMNS)
    return PrescriptionStay("MIMIC-1", 1, 10, normalize_orders(raw, CONFIG))


def events_of(stay_: PrescriptionStay, type_: str):
    return [e for e in PrescriptionAnalyzer(CONFIG).analyze(stay_).events if e.type == type_]


@pytest.mark.parametrize(
    ("value", "expected"), [("40", 40), ("1-2", 2), ("25,000", 25000), ("0.5", 0.5)]
)
def test_parse_dose(value, expected):
    assert parse_dose(value) == expected


def test_parse_dose_without_number_is_nan():
    assert np.isnan(parse_dose("see attached")) and np.isnan(parse_dose(None))


@pytest.mark.parametrize(
    ("dose", "unit", "mg"), [(1, "g", 1000), (1, "gm", 1000), (500, "mcg", 0.5)]
)
def test_units_are_normalized_to_mg(dose, unit, mg):
    assert normalize_unit(dose, unit, CONFIG.unit_factors) == (mg, "mass_mg")


def test_base_and_additive_rows_are_ignored():
    s = stay(order("Sodium Chloride", 1, 30, "100", "mL", "IV", drug_type="BASE"))
    assert "sodium chloride" not in set(s.orders["drug_key"])


def test_dose_increase_is_flagged_with_explanation():
    s = stay(order("Furosemide", 2, 20, "20"), order("Furosemide", 10, 40, "80"))
    [event] = events_of(s, "dose_increase")
    assert event.source is Source.PRESCRIPTION
    assert event.severity is Severity.MEDIUM
    assert (event.evidence["previous_dose"], event.evidence["new_dose"]) == (20, 80)
    assert event.evidence["ratio"] == 4.0


def test_routine_doubling_is_not_flagged():
    s = stay(order("Furosemide", 2, 20, "20"), order("Furosemide", 10, 40, "40"))
    assert events_of(s, "dose_increase") == []


def test_mg_to_g_conversion_is_not_a_dose_change():
    """1000 mg → 1 g é a mesma dose; sem normalizar, pareceria uma queda de 1000×."""
    s = stay(
        order("Ceftriaxone", 2, 20, "1000", "mg", "IV"),
        order("Ceftriaxone", 10, 40, "1", "gm", "IV"),
    )
    assert events_of(s, "dose_decrease") == []


def test_high_alert_drug_raises_severity():
    s = stay(
        order("Potassium Chloride", 2, 20, "40", "mEq"),
        order("Potassium Chloride", 8, 30, "10", "mEq"),
    )
    [event] = events_of(s, "dose_decrease")
    assert event.severity is Severity.HIGH
    assert event.evidence["high_alert"] is True


def test_titratable_drug_is_not_flagged():
    s = stay(
        order("Insulin", 2, 20, "2", "UNIT", "SC"), order("Insulin", 8, 30, "10", "UNIT", "SC")
    )
    assert events_of(s, "dose_increase") == []


def test_doses_more_than_24h_apart_are_not_compared():
    s = stay(order("Furosemide", 2, 20, "20"), order("Furosemide", 40, 60, "80"))
    assert events_of(s, "dose_increase") == []


def test_duplicate_therapy_in_same_class():
    s = stay(order("Morphine Sulfate", 2, 50, "2", "mg", "IV"), order("Oxycodone", 10, 60, "5"))
    [event] = events_of(s, "duplicate_therapy")
    assert event.evidence["class"] == "opioides"
    assert event.evidence["overlap_hours"] == 40
    assert event.severity is Severity.MEDIUM  # opioides são de alto risco


def test_short_overlap_is_not_duplicate():
    s = stay(order("Morphine Sulfate", 2, 20, "2", "mg", "IV"), order("Oxycodone", 10, 30, "5"))
    assert events_of(s, "duplicate_therapy") == []


def test_early_stop_without_restart():
    s = stay(order("Cephalexin", 10, 12, "500"))
    [event] = events_of(s, "early_stop")
    assert event.evidence["duration_hours"] == 2
    # só se confirma 24 h depois da suspensão
    assert event.timestamp.hour == (T0 + pd.Timedelta(hours=36)).hour


def test_early_stop_followed_by_restart_is_not_flagged():
    s = stay(order("Cephalexin", 10, 12, "500"), order("Cephalexin", 20, 60, "500"))
    assert events_of(s, "early_stop") == []


def test_long_order_is_not_early_stop():
    s = stay(order("Cephalexin", 10, 40, "500"))
    assert events_of(s, "early_stop") == []


def test_day_features_count_activity():
    s = stay(order("Furosemide", 2, 20, "20"), order("Furosemide", 10, 40, "80"))
    days = PrescriptionAnalyzer(CONFIG).day_features(s.orders)
    first = days.iloc[0]
    assert first["new"] == 3 and first["dose_changes"] == 1
    # rótulo do dia é o fim dele
    assert days.index[0] == T0.normalize() + pd.Timedelta("1D")


def test_iforest_runs_only_after_fit():
    stays = [stay(order("Furosemide", h, h + 10, "20")) for h in range(0, 20, 4)]
    analyzer = PrescriptionAnalyzer(CONFIG)
    assert "iforest_score" not in analyzer.analyze(stays[0]).days
    analyzer.fit(stays)
    assert "iforest_score" in analyzer.analyze(stays[0]).days


def synthetic_stays(count: int = 6) -> list[PrescriptionStay]:
    rng = np.random.default_rng(0)
    drugs = ["Furosemide", "Cephalexin", "Metoprolol Tartrate", "Acetaminophen", "Pantoprazole"]
    stays = []
    for number in range(count):
        rows = [
            (number, number, str(T0 + pd.Timedelta(hours=h)), str(T0 + pd.Timedelta(hours=h + 30)),
             "MAIN", str(rng.choice(drugs)), str(int(rng.integers(10, 50))), "mg", "PO")
            for h in range(0, 120, 6)
        ]  # fmt: skip
        raw = pd.DataFrame(rows, columns=COLUMNS)
        stays.append(
            PrescriptionStay(f"MIMIC-{number}", number, number, normalize_orders(raw, CONFIG))
        )
    return stays


def test_injection_creates_labeled_anomalies():
    s = synthetic_stays(1)[0]
    analyzer = PrescriptionAnalyzer(CONFIG)
    modified, labels = inject_prescription_anomalies(
        s.orders, np.random.default_rng(1), 3, analyzer
    )
    assert len(labels) == 3
    assert all(
        abs(a.at - b.at) >= pd.Timedelta("24h") for a in labels for b in labels if a is not b
    )
    assert set(modified.dtypes.astype(str)) == set(s.orders.dtypes.astype(str))


def test_evaluation_on_synthetic_stays_detects_most_injections():
    stays = synthetic_stays()
    analyzer = PrescriptionAnalyzer(CONFIG).fit(stays)
    metrics = evaluate_prescriptions(analyzer, stays, per_stay=2, seed=0)
    rules = metrics["rules"]
    assert rules.stays == 6 and rules.injected >= 10
    assert rules.recall >= 0.8
    assert set(metrics) == {"rules", "isolation_forest", "combined"}


@pytest.mark.slow
def test_real_mimic_prescriptions_if_downloaded():
    dataset = PROJECT_ROOT / "data" / "raw" / "mimic-iv-demo-2.2"
    if not (dataset / "hosp" / "prescriptions.csv.gz").exists():
        pytest.skip("MIMIC-IV demo não baixado (medwatch data download)")
    stays = iter_prescription_stays(dataset, CONFIG)
    assert len(stays) == 250
    assert sum(len(s.orders) for s in stays) > 10_000
