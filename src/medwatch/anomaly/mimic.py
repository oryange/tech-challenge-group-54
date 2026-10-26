"""Leitura dos sinais vitais do MIMIC-IV (chartevents) em grades temporais por internação."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from medwatch.anomaly.config import VITALS, VitalsConfig

# itemids do MIMIC-IV (icu/d_items): pressão invasiva e não invasiva entram no mesmo sinal
ITEM_TO_VITAL: dict[int, str] = {
    220045: "hr",
    220179: "sbp",  # Non Invasive Blood Pressure systolic
    220050: "sbp",  # Arterial Blood Pressure systolic
    220180: "dbp",
    220051: "dbp",
    220210: "rr",
    224690: "rr",  # Respiratory Rate (Total)
    220277: "spo2",
    223762: "temp",  # Celsius
    223761: "temp_f",  # Fahrenheit, convertido para Celsius
    223834: "o2_flow",
    226732: "o2_device",  # texto
    220739: "gcs_eye",  # 4 = abre os olhos espontaneamente (paciente alerta)
}

ROOM_AIR_DEVICES = {"none", "room air"}


@dataclass(frozen=True)
class StayVitals:
    patient_id: str
    subject_id: int
    stay_id: int
    raw: pd.DataFrame  # grade regular com NaN onde não houve medição
    artifacts: int  # quantidade de valores descartados como artefato


def patient_id_for(subject_id: int) -> str:
    """O MIMIC já é desidentificado; o prefixo deixa explícita a origem do identificador."""
    return f"MIMIC-{subject_id}"


def load_chartevents(dataset_dir: Path, subject_ids: list[int] | None = None) -> pd.DataFrame:
    events = pd.read_csv(
        dataset_dir / "icu" / "chartevents.csv.gz",
        usecols=["subject_id", "stay_id", "charttime", "itemid", "value", "valuenum"],
        parse_dates=["charttime"],
    )
    events = events[events["itemid"].isin(ITEM_TO_VITAL)]
    if subject_ids is not None:
        events = events[events["subject_id"].isin(subject_ids)]
    events = events.assign(vital=events["itemid"].map(ITEM_TO_VITAL))
    return events.reset_index(drop=True)


def clean_values(events: pd.DataFrame, config: VitalsConfig) -> tuple[pd.DataFrame, int]:
    """Converte unidades e descarta artefatos. Retorna os eventos limpos e quantos saíram."""
    events = events.copy()
    temp_f = events["vital"] == "temp_f"
    events.loc[temp_f, "valuenum"] = (events.loc[temp_f, "valuenum"] - 32) * 5 / 9
    events.loc[temp_f, "vital"] = "temp"
    # valor em °F digitado no campo de °C (ex.: 99 "°C")
    misplaced_f = (events["vital"] == "temp") & (events["valuenum"] > 50)
    events.loc[misplaced_f, "valuenum"] = (events.loc[misplaced_f, "valuenum"] - 32) * 5 / 9

    invalid = pd.Series(False, index=events.index)
    for vital, (low, high) in config.validity.items():
        is_vital = events["vital"] == vital
        invalid |= is_vital & ~events["valuenum"].between(low, high)
    removed = int(invalid.sum())
    if removed:
        logger.debug("{} valores descartados como artefato", removed)
    return events[~invalid], removed


def build_stay_grid(events: pd.DataFrame, config: VitalsConfig) -> pd.DataFrame:
    """Grade regular (config.grid) com os vitais numéricos, `on_oxygen` e `alert`.

    Cada bin é rotulado pelo fim do intervalo, (10:00, 11:00] → 11:00: uma medida das 10:55
    vira um ponto das 11:00, nunca das 10:00. Assim nenhum evento é datado antes de o dado
    existir, o que importa na fusão com os outros módulos (PR 7).
    """
    numeric = events[events["vital"].isin(VITALS)]
    raw = (
        numeric.pivot_table(index="charttime", columns="vital", values="valuenum", aggfunc="mean")
        .resample(config.grid, label="right", closed="right")
        .median()
        .reindex(columns=list(VITALS))
    )

    # astype: sem registros de texto o pandas infere float e o acesso .str quebraria
    device = events[events["vital"] == "o2_device"].set_index("charttime")["value"].astype("string")
    flow = events[events["vital"] == "o2_flow"].set_index("charttime")["valuenum"]
    oxygen = pd.concat(
        [
            (~device.str.strip().str.lower().isin(ROOM_AIR_DEVICES)).rename("oxygen"),
            (flow > 0).rename("oxygen"),
        ]
    ).sort_index()
    eye = events[events["vital"] == "gcs_eye"].set_index("charttime")["valuenum"]

    if not oxygen.empty:
        raw["on_oxygen"] = (
            oxygen.resample(config.grid, label="right", closed="right")
            .max()
            .reindex(raw.index)
            .astype(float)
        )
    else:
        raw["on_oxygen"] = float("nan")
    raw["alert"] = (
        (eye >= 4)
        .resample(config.grid, label="right", closed="right")
        .min()
        .reindex(raw.index)
        .astype(float)
    )
    if eye.empty:
        raw["alert"] = float("nan")

    return raw


def iter_stays(
    dataset_dir: Path, config: VitalsConfig, subject_ids: list[int] | None = None
) -> list[StayVitals]:
    events = load_chartevents(dataset_dir, subject_ids)
    stays = []
    for (subject_id, stay_id), group in events.groupby(["subject_id", "stay_id"]):
        cleaned, removed = clean_values(group, config)
        if cleaned[cleaned["vital"].isin(VITALS)].empty:
            continue
        raw = build_stay_grid(cleaned, config)
        stays.append(
            StayVitals(
                patient_id=patient_id_for(int(subject_id)),
                subject_id=int(subject_id),
                stay_id=int(stay_id),
                raw=raw,
                artifacts=removed,
            )
        )
    return stays
