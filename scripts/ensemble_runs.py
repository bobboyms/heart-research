"""Average patient-level OOF probabilities across multiple seeded runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "nested_tcn_systole_cnn"


def _load(run_name: str) -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS_DIR / run_name / "patient_oof_predictions.csv")


def _confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tn, fp, fn, tp


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    tn, fp, fn, tp = _confusion(y, p, threshold)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return {
        "threshold": threshold,
        "auprc": average_precision_score(y, p),
        "auroc": roc_auc_score(y, p),
        "balanced_accuracy": 0.5 * (sens + spec),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None, help="Optional path to write the ensembled per-patient predictions.")
    args = parser.parse_args()

    frames = []
    for run in args.runs:
        df = _load(run)[["patient_id", "target", "prob_present_raw", "prob_present_calibrated"]].copy()
        df = df.rename(
            columns={
                "prob_present_raw": f"raw__{run}",
                "prob_present_calibrated": f"cal__{run}",
            }
        )
        frames.append(df)

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=["patient_id", "target"], how="inner")

    raw_cols = [c for c in merged.columns if c.startswith("raw__")]
    cal_cols = [c for c in merged.columns if c.startswith("cal__")]
    merged["prob_present_raw_mean"] = merged[raw_cols].mean(axis=1)
    merged["prob_present_calibrated_mean"] = merged[cal_cols].mean(axis=1)
    merged["prob_present_raw_median"] = merged[raw_cols].median(axis=1)
    merged["prob_present_calibrated_median"] = merged[cal_cols].median(axis=1)

    y = merged["target"].to_numpy(dtype=int)

    print(f"N pacientes: {len(merged)}  |  Positivos: {int(y.sum())}\n")
    print("Per-run individual (calibrado):")
    for col in cal_cols:
        m = _metrics(y, merged[col].to_numpy(dtype=float), args.threshold)
        print(f"  {col:50s}  AUPRC={m['auprc']:.4f}  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")

    print("\nEnsemble (média):")
    m = _metrics(y, merged["prob_present_calibrated_mean"].to_numpy(dtype=float), args.threshold)
    print(f"  Calibrated  AUPRC={m['auprc']:.4f}  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}  "
          f"P={m['precision']:.4f}  R={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}")
    m = _metrics(y, merged["prob_present_raw_mean"].to_numpy(dtype=float), args.threshold)
    print(f"  Raw         AUPRC={m['auprc']:.4f}  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")

    print("\nEnsemble (mediana):")
    m = _metrics(y, merged["prob_present_calibrated_median"].to_numpy(dtype=float), args.threshold)
    print(f"  Calibrated  AUPRC={m['auprc']:.4f}  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")

    if args.output:
        merged.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
