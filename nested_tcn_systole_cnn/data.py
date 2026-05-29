"""Dataset preparation utilities for nested patient-level validation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def load_patient_context(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "training_data.csv"
    table = pd.read_csv(path, dtype={"Patient ID": str})
    columns = {
        "Patient ID": "patient_id",
        "Murmur locations": "murmur_locations",
        "Most audible location": "most_audible_location",
        "Systolic murmur timing": "systolic_murmur_timing",
        "Systolic murmur shape": "systolic_murmur_shape",
        "Systolic murmur grading": "systolic_murmur_grading",
        "Systolic murmur pitch": "systolic_murmur_pitch",
        "Systolic murmur quality": "systolic_murmur_quality",
        "Outcome": "outcome",
        # Demographics (patient-level inputs for the optional --demographic branch).
        "Age": "age",
        "Sex": "sex",
        "Height": "height",
        "Weight": "weight",
        "Pregnancy status": "pregnancy_status",
    }
    available = [col for col in columns if col in table.columns]
    context = table[available].rename(columns=columns).copy()
    context["patient_id"] = context["patient_id"].astype(str)
    return context.drop_duplicates("patient_id")


def select_patient_subset(meta: pd.DataFrame, max_patients: int | None, seed: int) -> pd.DataFrame:
    if max_patients is None:
        return meta
    patients = meta.drop_duplicates("patient_id")[["patient_id", "target"]].copy()
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for target in [1, 0]:
        ids = patients.loc[patients["target"] == target, "patient_id"].astype(str).to_numpy()
        rng.shuffle(ids)
        quota = max(1, int(round(max_patients * len(ids) / max(len(patients), 1))))
        selected.extend(ids[:quota].tolist())
    if len(selected) < max_patients:
        remaining = patients.loc[~patients["patient_id"].astype(str).isin(set(selected)), "patient_id"].astype(str).to_numpy()
        rng.shuffle(remaining)
        selected.extend(remaining[: max_patients - len(selected)].tolist())
    selected = selected[:max_patients]
    return meta.loc[meta["patient_id"].astype(str).isin(set(selected))].copy()


def split_cnn_fit_tune_patients(
    meta: pd.DataFrame,
    train_patient_ids: set[str],
    tune_size: float,
    seed: int,
    fold: int,
) -> tuple[set[str], set[str]]:
    train_patients = meta.loc[meta["patient_id"].astype(str).isin(train_patient_ids)]
    patient_table = train_patients.drop_duplicates("patient_id")[["patient_id", "target"]].copy()
    rng = np.random.default_rng(seed + fold * 1009)
    tune_ids: list[str] = []

    for target in [1, 0]:
        ids = patient_table.loc[patient_table["target"] == target, "patient_id"].astype(str).to_numpy()
        if len(ids) < 2:
            raise RuntimeError(
                f"Fold {fold} does not have enough class {target} patients with systole spectrograms "
                "to create an internal CNN tuning split."
            )
        rng.shuffle(ids)
        count = max(1, int(round(len(ids) * tune_size)))
        count = min(count, len(ids) - 1)
        tune_ids.extend(ids[:count].tolist())

    tune_patient_ids = set(tune_ids)
    fit_patient_ids = set(patient_table["patient_id"].astype(str)) - tune_patient_ids
    for name, ids in [("fit", fit_patient_ids), ("tune", tune_patient_ids)]:
        subset = patient_table.loc[patient_table["patient_id"].astype(str).isin(ids), "target"].to_numpy(dtype=int)
        if len(np.unique(subset)) < 2:
            raise RuntimeError(f"Fold {fold} internal CNN {name} split has only one class.")
    return fit_patient_ids, tune_patient_ids


def make_tcn_subset_dataset(
    source_dataset: Path,
    subset_dir: Path,
    patient_ids: set[str],
    parse_recording_id: Callable[[Path], tuple[str, str]],
) -> None:
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    data_dir = subset_dir / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(source_dataset / "training_data.csv", dtype={"Patient ID": str})
    table = table.loc[table["Patient ID"].astype(str).isin(patient_ids)].copy()
    table.to_csv(subset_dir / "training_data.csv", index=False)

    source_data_dir = source_dataset / "training_data"
    for wav_path in sorted(source_data_dir.glob("*.wav")):
        patient_id, _location = parse_recording_id(wav_path)
        if patient_id not in patient_ids:
            continue
        for source_path in [wav_path, wav_path.with_suffix(".tsv")]:
            if not source_path.exists():
                continue
            target_path = data_dir / source_path.name
            os.symlink(source_path.resolve(), target_path)

