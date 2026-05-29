from __future__ import annotations


import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd


from .audio import parse_recording_id
from .config import RecordingItem


def load_patient_metadata(dataset_dir: Path) -> dict[str, dict[str, str]]:
    csv_path = dataset_dir / "training_data.csv"
    table = pd.read_csv(csv_path, dtype={"Patient ID": str})
    metadata: dict[str, dict[str, str]] = {}
    for _index, row in table.iterrows():
        patient_id = str(row["Patient ID"])
        metadata[patient_id] = {
            "murmur": str(row["Murmur"]),
            "outcome": str(row["Outcome"]),
        }
    return metadata


def normalize_patient_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not np.isfinite(numeric):
        return None
    return str(int(numeric))


def build_recording_index(args: argparse.Namespace) -> list[RecordingItem]:
    data_dir = args.dataset_dir / "training_data"
    metadata = load_patient_metadata(args.dataset_dir)
    items: list[RecordingItem] = []

    for wav_path in sorted(data_dir.glob("*.wav")):
        tsv_path = wav_path.with_suffix(".tsv")
        if not tsv_path.exists():
            continue
        patient_id, location = parse_recording_id(wav_path)
        patient_meta = metadata.get(patient_id, {"murmur": "Unknown", "outcome": "Unknown"})
        murmur = patient_meta["murmur"]
        if args.exclude_murmur_unknown and murmur == "Unknown":
            continue
        items.append(
            RecordingItem(
                recording_id=wav_path.stem,
                patient_id=patient_id,
                location=location,
                wav_path=str(wav_path),
                tsv_path=str(tsv_path),
                murmur=murmur,
                outcome=patient_meta["outcome"],
            )
        )

    if args.max_recordings is not None:
        rng = random.Random(args.seed)
        rng.shuffle(items)
        items = sorted(items[: args.max_recordings], key=lambda item: item.recording_id)

    if not items:
        raise RuntimeError(f"No wav+tsv recordings found under {data_dir}")
    return items


def split_by_patient(
    items: list[RecordingItem],
    dataset_dir: Path,
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[RecordingItem]]:
    if val_size < 0 or test_size < 0 or val_size + test_size >= 1:
        raise ValueError("--val-size and --test-size must be non-negative and sum to less than 1.")

    patients: dict[str, list[RecordingItem]] = {}
    murmur_by_patient: dict[str, str] = {}
    for item in items:
        patients.setdefault(item.patient_id, []).append(item)
        murmur_by_patient[item.patient_id] = item.murmur

    patient_groups = build_patient_leakage_groups(dataset_dir, set(patients))
    rng = random.Random(seed)
    split_patient_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for _murmur, groups in group_patient_groups_by_murmur(patient_groups, murmur_by_patient).items():
        groups = list(groups)
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        n_test = min(n_test, max(0, n - 2)) if n >= 3 else 0
        n_val = min(n_val, max(0, n - n_test - 1))
        split_patient_ids["test"].extend(patient_id for group in groups[:n_test] for patient_id in group)
        split_patient_ids["val"].extend(patient_id for group in groups[n_test : n_test + n_val] for patient_id in group)
        split_patient_ids["train"].extend(patient_id for group in groups[n_test + n_val :] for patient_id in group)

    splits: dict[str, list[RecordingItem]] = {}
    for split_name, patient_ids in split_patient_ids.items():
        split_items = [item for pid in patient_ids for item in patients[pid]]
        splits[split_name] = sorted(split_items, key=lambda item: item.recording_id)

    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise RuntimeError(
            "The patient split produced an empty train/val/test subset. "
            "Use more recordings or reduce --val-size/--test-size."
        )
    return splits


def build_patient_leakage_groups(dataset_dir: Path, available_patients: set[str]) -> list[tuple[str, ...]]:
    parent = {patient_id: patient_id for patient_id in available_patients}

    def find(patient_id: str) -> str:
        parent.setdefault(patient_id, patient_id)
        while parent[patient_id] != patient_id:
            parent[patient_id] = parent[parent[patient_id]]
            patient_id = parent[patient_id]
        return patient_id

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    table = pd.read_csv(dataset_dir / "training_data.csv", dtype={"Patient ID": str})
    for _index, row in table.iterrows():
        patient_id = normalize_patient_id(row["Patient ID"])
        additional_id = normalize_patient_id(row.get("Additional ID"))
        if patient_id in available_patients:
            parent.setdefault(patient_id, patient_id)
        if additional_id in available_patients:
            parent.setdefault(additional_id, additional_id)
        if patient_id in available_patients and additional_id in available_patients:
            union(patient_id, additional_id)

    groups: dict[str, list[str]] = {}
    for patient_id in sorted(available_patients):
        groups.setdefault(find(patient_id), []).append(patient_id)
    return [tuple(patient_ids) for patient_ids in groups.values()]


def group_patient_groups_by_murmur(
    patient_groups: list[tuple[str, ...]],
    murmur_by_patient: dict[str, str],
) -> dict[str, list[tuple[str, ...]]]:
    grouped: dict[str, list[tuple[str, ...]]] = {}
    for patient_group in patient_groups:
        murmurs = [murmur_by_patient[patient_id] for patient_id in patient_group if patient_id in murmur_by_patient]
        murmur = "Present" if "Present" in murmurs else ("Unknown" if "Unknown" in murmurs else "Absent")
        grouped.setdefault(murmur, []).append(patient_group)
    return grouped
