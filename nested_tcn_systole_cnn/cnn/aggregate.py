from __future__ import annotations


import numpy as np
import pandas as pd


from .config import LOCATION_ORDER


def aggregate_patient_probs(meta: pd.DataFrame, probs: np.ndarray, indices: np.ndarray, method: str = "max") -> pd.DataFrame:
    frame = meta.iloc[indices][["patient_id", "murmur", "target"]].copy()
    frame["prob"] = probs
    agg_fn = "max" if method == "max" else "mean"
    return frame.groupby("patient_id", as_index=False).agg(
        murmur=("murmur", "first"),
        target=("target", "first"),
        prob=("prob", agg_fn),
    )


def sample_weights_for_indices(
    meta: pd.DataFrame,
    indices: np.ndarray,
    weak_murmur_weight: float,
    moderate_murmur_weight: float,
) -> np.ndarray:
    weights = np.ones(len(indices), dtype=np.float32)
    if "systolic_murmur_grading" not in meta.columns:
        return weights

    frame = meta.iloc[indices].copy()
    target_column = "recording_target" if "recording_target" in frame.columns else "target"
    target = frame[target_column].to_numpy(dtype=int)
    grades = frame["systolic_murmur_grading"].fillna("").astype(str).str.strip().str.upper()
    weights[(target == 1) & (grades == "I/VI")] = float(weak_murmur_weight)
    weights[(target == 1) & (grades == "II/VI")] = float(moderate_murmur_weight)
    return weights


def aggregate_patient_location_features(
    meta: pd.DataFrame,
    probs: np.ndarray,
    indices: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    frame = meta.iloc[indices][["patient_id", "murmur", "target", "location"]].copy()
    frame["prob"] = probs

    patient = frame.groupby("patient_id", as_index=False).agg(
        murmur=("murmur", "first"),
        target=("target", "first"),
        prob_max=("prob", "max"),
        prob_mean=("prob", "mean"),
        recording_count=("prob", "count"),
    )
    pivot = frame.pivot_table(index="patient_id", columns="location", values="prob", aggfunc="max")
    pivot = pivot.reindex(columns=list(LOCATION_ORDER), fill_value=np.nan)
    has_location = pivot.notna().astype(float)
    has_location.columns = [f"has_{col}" for col in has_location.columns]
    pivot = pivot.fillna(0.0)
    pivot.columns = [f"prob_{col}" for col in pivot.columns]

    sorted_probs = np.sort(pivot[[f"prob_{loc}" for loc in LOCATION_ORDER]].to_numpy(dtype=float), axis=1)
    patient["prob_top2_mean"] = sorted_probs[:, -2:].mean(axis=1)

    features = patient.merge(pivot.reset_index(), on="patient_id", how="left").merge(
        has_location.reset_index(),
        on="patient_id",
        how="left",
    )
    feature_columns = [
        "prob_max",
        "prob_mean",
        "prob_top2_mean",
        "recording_count",
        *[f"prob_{loc}" for loc in LOCATION_ORDER],
        *[f"has_{loc}" for loc in LOCATION_ORDER],
    ]
    return features, feature_columns
