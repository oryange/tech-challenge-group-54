"""Internações sintéticas e um chartevents falso no formato do MIMIC-IV."""

from pathlib import Path

import numpy as np
import pandas as pd

from medwatch.anomaly.config import VitalsConfig
from medwatch.anomaly.mimic import StayVitals
from medwatch.config import PROJECT_ROOT

BASELINE = {"hr": (80, 3), "sbp": (120, 5), "dbp": (70, 4), "rr": (16, 1), "spo2": (97, 1)}
VITAL_ITEMS = {"hr": 220045, "sbp": 220179, "dbp": 220180, "rr": 220210, "spo2": 220277}


def load_config() -> VitalsConfig:
    return VitalsConfig.load(PROJECT_ROOT / "configs" / "thresholds.yaml")


def make_raw(hours: int = 48, seed: int = 0, start: str = "2180-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=hours, freq="1h")
    raw = pd.DataFrame(index=index)
    for vital, (mean, std) in BASELINE.items():
        raw[vital] = np.round(rng.normal(mean, std, hours))
    raw["spo2"] = raw["spo2"].clip(upper=100)
    temp = np.full(hours, np.nan)
    temp[::4] = np.round(rng.normal(36.8, 0.15, len(temp[::4])), 1)  # medida a cada 4 h
    raw["temp"] = temp
    raw["on_oxygen"] = 0.0
    raw["alert"] = 1.0
    return raw


def make_stay(
    raw: pd.DataFrame | None = None, subject_id: int = 1, stay_id: int = 10
) -> StayVitals:
    return StayVitals(
        patient_id=f"MIMIC-{subject_id}",
        subject_id=subject_id,
        stay_id=stay_id,
        raw=make_raw() if raw is None else raw,
        artifacts=0,
    )


def write_fake_chartevents(dataset_dir: Path, stays: list[StayVitals]) -> Path:
    rows = []
    for stay in stays:
        for ts, record in stay.raw.iterrows():
            for vital, itemid in VITAL_ITEMS.items():
                if pd.notna(record[vital]):
                    rows.append((stay.subject_id, stay.stay_id, ts, itemid, record[vital]))
            if pd.notna(record["temp"]):
                rows.append((stay.subject_id, stay.stay_id, ts, 223762, record["temp"]))
    frame = pd.DataFrame(rows, columns=["subject_id", "stay_id", "charttime", "itemid", "valuenum"])
    frame["value"] = frame["valuenum"].astype(str)
    path = dataset_dir / "icu" / "chartevents.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
