"""Result aggregation and reporting for nested TCN + systole CNN runs."""

from __future__ import annotations

import argparse
from types import ModuleType

import numpy as np
import pandas as pd


def format_threshold_key(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def metrics_from_binary_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold_name: str,
    cnn_module: ModuleType,
) -> dict[str, float | int | str]:
    tn, fp, fn, tp = cnn_module.confusion_counts(y_true, predictions.astype(float), 0.5)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "threshold": threshold_name,
        "auroc": cnn_module.roc_auc(y_true, probabilities),
        "auprc": cnn_module.average_precision(y_true, probabilities),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": 2 * precision * sensitivity / (precision + sensitivity) if precision and sensitivity else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def add_oof_prediction_columns(patient_oof: pd.DataFrame, decision_threshold: float) -> tuple[pd.DataFrame, str]:
    patient_oof = patient_oof.copy()
    patient_oof["pred_present_raw_threshold_05"] = (patient_oof["prob_present_raw"].to_numpy(dtype=float) >= 0.5).astype(int)
    patient_oof["pred_present_calibrated_threshold_05"] = (
        patient_oof["prob_present_calibrated"].to_numpy(dtype=float) >= 0.5
    ).astype(int)
    decision_key = format_threshold_key(decision_threshold)
    patient_oof[f"pred_present_calibrated_threshold_{decision_key}"] = (
        patient_oof["prob_present_calibrated"].to_numpy(dtype=float) >= decision_threshold
    ).astype(int)
    return patient_oof, decision_key


def tuned_metrics_by_fold(
    patient_oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    probability_column: str,
    threshold_column: str,
    threshold_name: str,
    cnn_module: ModuleType,
) -> dict[str, float | int | str]:
    predictions = []
    for row in patient_oof.itertuples(index=False):
        probability = float(getattr(row, probability_column))
        fold = int(getattr(row, "fold"))
        threshold = float(fold_metrics.loc[fold_metrics["fold"] == fold, threshold_column].iloc[0])
        predictions.append(1 if probability >= threshold else 0)
    y_true = patient_oof["target"].to_numpy(dtype=int)
    probabilities = patient_oof[probability_column].to_numpy(dtype=float)
    return metrics_from_binary_predictions(
        y_true,
        probabilities,
        np.asarray(predictions, dtype=int),
        threshold_name,
        cnn_module,
    )


def write_summary(
    output_dir,
    args: argparse.Namespace,
    patient_oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    raw_metrics_05: dict[str, float | int],
    calibrated_metrics_05: dict[str, float | int],
    tuned_metrics: dict[str, float | int | str],
    calibrated_tuned_metrics: dict[str, float | int | str],
    decision_metrics: dict[str, float | int],
) -> None:
    lines = [
        "# Grupo H - Nested TCN + CNN fase cardiaca",
        "",
        "## Objetivo",
        "",
        "Avaliar o pipeline de ponta a ponta sem permitir que o TCN veja os `.tsv` dos pacientes de validacao do classificador.",
        "",
        "## Protocolo",
        "",
        "Para cada fold:",
        "",
        "1. Treina um TCN apenas nos pacientes de treino do fold.",
        "2. Prediz segmentos de fase cardiaca em treino e validacao com esse TCN.",
        "3. Divide os pacientes de treino do fold em `cnn_fit` e `cnn_tune`.",
        "4. Treina a CNN de fase cardiaca apenas em `cnn_fit`.",
        "5. Usa `cnn_tune` para early stopping, threshold e calibracao.",
        "6. Avalia uma unica vez nos pacientes de validacao externa do fold.",
        "",
        "## Dados",
        "",
        f"- Locais: `{', '.join(args.locations)}`",
        f"- Pacientes OOF: {patient_oof['patient_id'].nunique()}",
        f"- Present: {int((patient_oof['target'] == 1).sum())}",
        f"- Absent: {int((patient_oof['target'] == 0).sum())}",
        f"- Folds: {args.folds}",
        f"- TCN target mode: {args.tcn_target_mode}",
        f"- TCN other mode: {args.tcn_other_mode}",
        f"- TCN boundary ignore ms: {args.tcn_boundary_ignore_ms}",
        f"- TCN systole weight multiplier: {args.tcn_systole_weight_multiplier}",
        f"- CNN inner validation size: {args.cnn_inner_val_size}",
        f"- Encoder block: {args.encoder_block}",
        f"- Patient fixed-location fusion (`--patient-mil-attention`): {args.patient_mil_attention}",
        "- Location slots: AV, PV, TV, MV",
        f"- Auxiliary per-location loss weight: {args.mil_instance_loss_weight}",
        f"- Weak murmur weight: {args.weak_murmur_weight}",
        f"- Moderate murmur weight: {args.moderate_murmur_weight}",
        f"- Location-aware calibration: {args.location_aware_calibration}",
        f"- SMOTE minority augmentation: {args.smote_minority_augmentation}",
        f"- SMOTE k neighbors: {args.smote_k_neighbors}",
        f"- SMOTE target ratio: {args.smote_target_ratio}",
        f"- CNN loss: {args.loss}",
        f"- Focal gamma: {args.focal_gamma}",
        f"- Focal alpha: {args.focal_alpha if args.focal_alpha is not None else 'none'}",
        f"- AUC loss weight: {args.auc_loss_weight}",
        f"- AUC loss margin: {args.auc_loss_margin}",
        f"- Stratified train batches when AUC loss is active: {args.auc_loss_weight > 0.0}",
        f"- Cleanup fold training artifacts: {args.cleanup_fold_artifacts}",
        f"- Decision threshold: {args.decision_threshold}",
        f"- Systole threshold: {args.systole_threshold if args.systole_threshold is not None else 'argmax'}",
        f"- Systole margin ms: {args.systole_margin_ms}",
        f"- CNN spectrogram type: {args.spectrogram_type}",
        f"- CNN n_mels: {args.n_mels}",
        f"- STFT low Hz: {args.low_hz}",
        f"- STFT high Hz: {args.high_hz}",
        f"- CNN phase mode: {args.cnn_phase_mode}",
        "",
        "## Metricas paciente-level OOF",
        "",
        "### Probabilidade bruta @0.5",
        "",
        pd.DataFrame([raw_metrics_05]).to_markdown(index=False),
        "",
        "### Probabilidade calibrada @0.5",
        "",
        pd.DataFrame([calibrated_metrics_05]).to_markdown(index=False),
        "",
        "### Threshold Youden por fold",
        "",
        pd.DataFrame([tuned_metrics]).to_markdown(index=False),
        "",
        "### Threshold calibrado Youden por fold",
        "",
        pd.DataFrame([calibrated_tuned_metrics]).to_markdown(index=False),
        "",
        f"### Probabilidade calibrada @{args.decision_threshold:g}",
        "",
        pd.DataFrame([decision_metrics]).to_markdown(index=False),
        "",
        "## Metricas por fold",
        "",
        fold_metrics.to_markdown(index=False),
        "",
        "## Arquivos gerados",
        "",
        "- `patient_oof_predictions.csv`",
        "- `fold_metrics.csv`",
        "- `training_history.csv`",
        "- `threshold_metrics_by_fold.md`",
        "- `mil_instance_attention_oof.csv` com diagnostico por gravacao/local quando `--patient-mil-attention` esta ativo",
        "- `fold_*/tcn/best_model.pt`",
        "- `fold_*/cnn/fold_*_best_model.pt`",
        "- `fold_*/cnn/fold_*_mil_instance_attention_validation.csv` quando `--patient-mil-attention` esta ativo",
        "- `fold_*/cnn_fit_patient_ids.txt`",
        "- `fold_*/cnn_tune_patient_ids.txt`",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
