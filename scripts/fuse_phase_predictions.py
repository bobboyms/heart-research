"""Fuse systole and diastole patient-level OOF predictions and report metrics.

Reads patient_oof_predictions.csv from two runs (systole-only and diastole-only),
merges by patient_id, and evaluates several fusion strategies:

- systole only (baseline)
- diastole only (baseline)
- mean of probabilities
- max of probabilities
- weighted average (grid search of weights)
- logistic regression stacker, evaluated OOF using each run's fold assignment

Outputs a Markdown comparison to experiments/nested_tcn_systole_cnn/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "nested_tcn_systole_cnn"


def _load(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "patient_oof_predictions.csv")
    return df[["patient_id", "target", "fold", "prob_present_raw", "prob_present_calibrated"]]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auprc": average_precision_score(y, p),
        "auroc": roc_auc_score(y, p),
    }


def _oof_stacker(merged: pd.DataFrame, prob_cols: list[str]) -> np.ndarray:
    """Train a logistic regression stacker leave-one-fold-out, return OOF probs."""
    preds = np.zeros(len(merged))
    for fold in sorted(merged["fold"].unique()):
        train_mask = merged["fold"] != fold
        test_mask = merged["fold"] == fold
        x_train = merged.loc[train_mask, prob_cols].to_numpy()
        y_train = merged.loc[train_mask, "target"].to_numpy()
        x_test = merged.loc[test_mask, prob_cols].to_numpy()
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(x_train, y_train)
        preds[test_mask.to_numpy()] = clf.predict_proba(x_test)[:, 1]
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systole-run",
        default="tcn_attn_weight2_platt_thr045_ignore_multiscale_stft_dilated32_focal_reuse_tcn",
    )
    parser.add_argument(
        "--diastole-run",
        default="tcn_attn_weight2_platt_thr045_ignore_diastole_only_reuse_tcn",
    )
    parser.add_argument(
        "--output",
        default=str(EXPERIMENTS_DIR / "fusion_systole_diastole_comparison.md"),
    )
    args = parser.parse_args()

    sys_dir = EXPERIMENTS_DIR / args.systole_run
    dia_dir = EXPERIMENTS_DIR / args.diastole_run

    sys_df = _load(sys_dir).rename(
        columns={
            "prob_present_raw": "p_sys_raw",
            "prob_present_calibrated": "p_sys_cal",
            "fold": "fold_sys",
        }
    )
    dia_df = _load(dia_dir).rename(
        columns={
            "prob_present_raw": "p_dia_raw",
            "prob_present_calibrated": "p_dia_cal",
            "fold": "fold_dia",
        }
    )

    merged = sys_df.merge(dia_df.drop(columns=["target"]), on="patient_id", how="inner")
    if not (merged["fold_sys"] == merged["fold_dia"]).all():
        raise SystemExit("Fold assignments differ between runs; cannot fuse OOF safely.")
    merged = merged.rename(columns={"fold_sys": "fold"}).drop(columns=["fold_dia"])

    y = merged["target"].to_numpy()
    p_sys = merged["p_sys_cal"].to_numpy()
    p_dia = merged["p_dia_cal"].to_numpy()

    results: list[dict] = []
    results.append({"strategy": "Sístole (calibrada)", **_metrics(y, p_sys)})
    results.append({"strategy": "Diástole (calibrada)", **_metrics(y, p_dia)})
    results.append({"strategy": "Média simples", **_metrics(y, (p_sys + p_dia) / 2)})
    results.append({"strategy": "Max", **_metrics(y, np.maximum(p_sys, p_dia))})

    best_w = None
    best_auprc = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        fused = w * p_sys + (1.0 - w) * p_dia
        auprc = average_precision_score(y, fused)
        if auprc > best_auprc:
            best_auprc = auprc
            best_w = w
    fused_best = best_w * p_sys + (1.0 - best_w) * p_dia
    results.append(
        {
            "strategy": f"Média ponderada (w_sys={best_w:.2f})",
            **_metrics(y, fused_best),
        }
    )

    stack_cal = _oof_stacker(merged, ["p_sys_cal", "p_dia_cal"])
    results.append({"strategy": "Stacker LR (calibradas, OOF)", **_metrics(y, stack_cal)})

    stack_raw = _oof_stacker(merged, ["p_sys_raw", "p_dia_raw"])
    results.append({"strategy": "Stacker LR (raw, OOF)", **_metrics(y, stack_raw)})

    df = pd.DataFrame(results)
    df = df[["strategy", "auprc", "auroc"]]
    df["auprc"] = df["auprc"].round(4)
    df["auroc"] = df["auroc"].round(4)

    print(df.to_string(index=False))

    out_path = Path(args.output)
    lines = [
        "# Fusão sístole + diástole (paciente-level OOF)",
        "",
        f"- Sístole run: `{args.systole_run}`",
        f"- Diástole run: `{args.diastole_run}`",
        f"- N pacientes fundidos: {len(merged)}  (positivos: {int(y.sum())})",
        "",
        "## Resultados",
        "",
        df.to_markdown(index=False),
        "",
        "## Leitura",
        "",
        "Comparar AUPRC das estratégias de fusão contra o baseline sístole (primeira linha).",
        "Se nenhuma estratégia ultrapassar o baseline em mais de ~0.01, a diástole",
        "não carrega sinal complementar suficiente para esta combinação de modelos.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
