from __future__ import annotations


import math
from pathlib import Path
import numpy as np
import pandas as pd


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = int(y_sorted.sum())
    if total_pos == 0:
        return 0.0
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / total_pos)


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return tn, fp, fn, tp


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_counts(y_true, y_prob, threshold)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    return {
        "threshold": float(threshold),
        "auroc": roc_auc(y_true, y_prob),
        "auprc": average_precision(y_true, y_prob),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.05, 1.0, 0.05), 2)


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([metrics(y_true, y_prob, float(threshold)) for threshold in thresholds])


def write_threshold_tables_by_fold(
    patient_predictions: pd.DataFrame,
    output_dir: Path,
    prob_column: str,
    label: str,
) -> pd.DataFrame:
    thresholds = threshold_grid()
    rows: list[pd.DataFrame] = []
    table_dir = output_dir / "threshold_metrics_by_fold"
    table_dir.mkdir(parents=True, exist_ok=True)

    for fold in sorted(patient_predictions["fold"].unique()):
        fold_df = patient_predictions[patient_predictions["fold"] == fold]
        table = threshold_sweep(
            fold_df["target"].to_numpy(dtype=int),
            fold_df[prob_column].to_numpy(dtype=float),
            thresholds,
        )
        table.insert(0, "fold", int(fold))
        table.insert(1, "probability", label)
        table.insert(2, "patients", int(len(fold_df)))
        table.insert(3, "present", int((fold_df["target"] == 1).sum()))
        table.insert(4, "absent", int((fold_df["target"] == 0).sum()))
        table.to_csv(table_dir / f"fold_{int(fold)}_{label}_threshold_metrics.csv", index=False)
        rows.append(table)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_dir / f"threshold_metrics_by_fold_{label}.csv", index=False)
    return combined


def write_threshold_metrics_report(
    path: Path,
    raw_tables: pd.DataFrame,
    calibrated_tables: pd.DataFrame,
) -> None:
    lines = [
        "# Metricas por fold em varios thresholds",
        "",
        "As tabelas abaixo avaliam as predicoes paciente-level out-of-fold em thresholds de `0.05` a `0.95`.",
        "",
        "- `raw`: probabilidade bruta da CNN.",
        "- `calibrated`: probabilidade apos calibracao Platt por fold.",
        "",
    ]
    metric_columns = [
        "threshold",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    for fold in sorted(raw_tables["fold"].unique()):
        raw_fold = raw_tables[raw_tables["fold"] == fold]
        calibrated_fold = calibrated_tables[calibrated_tables["fold"] == fold]
        present = int(raw_fold["present"].iloc[0])
        absent = int(raw_fold["absent"].iloc[0])
        patients = int(raw_fold["patients"].iloc[0])
        lines.extend(
            [
                f"## Fold {int(fold)}",
                "",
                f"Pacientes: `{patients}` | Present: `{present}` | Absent: `{absent}`",
                "",
                "### Probabilidade bruta",
                "",
                raw_fold[metric_columns].to_markdown(index=False),
                "",
                "### Probabilidade calibrada",
                "",
                calibrated_fold[metric_columns].to_markdown(index=False),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in candidates:
        row = metrics(y_true, y_prob, float(threshold))
        score = float(row["sensitivity"]) + float(row["specificity"]) - 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = y_true.astype(float)
    p = y_prob.astype(float)
    return float(np.mean((p - y) ** 2))
