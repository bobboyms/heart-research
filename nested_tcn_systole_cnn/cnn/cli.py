from __future__ import annotations


import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch


from nested_tcn_systole_cnn import tcn


from .audio import choose_device, set_seed
from .config import DEFAULT_PREDICTED_TSV_DIR, DEFAULT_TCN_CHECKPOINT, ModelConfig, PHASE_LABELS, REPO_ROOT, SCRIPT_DIR, StftConfig
from .dataset import build_items, load_patient_context, prepare_spectrograms, stratified_patient_folds
from .metrics import average_precision, brier_score, confusion_counts, metrics, roc_auc, write_threshold_metrics_report, write_threshold_tables_by_fold
from .training import plot_pr, train_one_fold, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dilated CNN on selected cardiac-phase STFTs from TCN segmentation.")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "circor-heart-sound-1.0.3")
    parser.add_argument("--tcn-checkpoint", type=Path, default=DEFAULT_TCN_CHECKPOINT)
    parser.add_argument("--predicted-tsv-dir", type=Path, default=DEFAULT_PREDICTED_TSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--spectrogram-cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--locations", nargs="+", default=["PV", "TV"], choices=["AV", "PV", "TV", "MV"])
    parser.add_argument("--max-recordings", type=int, default=None)

    parser.add_argument("--target-sample-rate", type=int, default=4000)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--low-hz", type=float, default=0.0)
    parser.add_argument("--high-hz", type=float, default=1000.0)
    parser.add_argument(
        "--spectrogram-type",
        choices=["stft", "log-mel"],
        default="stft",
        help="Spectrogram representation used by the CNN.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=64,
        help="Number of Mel bins used when --spectrogram-type log-mel.",
    )
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument(
        "--cnn-phase-mode",
        choices=["systole", "diastole", "both"],
        default="systole",
        help="Cardiac phase audio used by the CNN: systole only, diastole only, or systole+diastole concatenated.",
    )
    parser.add_argument("--min-systole-seconds", type=float, default=0.10)
    parser.add_argument(
        "--systole-threshold",
        type=float,
        default=None,
        help=(
            "Optional TCN probability threshold for extracting systole. "
            "For --cnn-phase-mode systole, exports only frames with p(systole) >= threshold. "
            "For diastole/both, keeps argmax phase predictions and suppresses low-confidence systole frames."
        ),
    )
    parser.add_argument(
        "--systole-margin-ms",
        type=float,
        default=0.0,
        help="Expand each selected phase segment by this many milliseconds on both sides before STFT extraction.",
    )

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--dilations", type=str, default="1,2,4,8")
    parser.add_argument(
        "--encoder-block",
        choices=["residual", "multiscale"],
        default="residual",
        help="Temporal encoder block type. 'residual' keeps the original dilated residual CNN; 'multiscale' uses parallel temporal kernels/dilations.",
    )
    parser.add_argument("--pooling", choices=["avg", "attention"], default="avg")
    parser.add_argument(
        "--patient-mil-attention",
        action="store_true",
        help="Train a patient-level fixed-location fusion head over AV/PV/TV/MV recording embeddings.",
    )
    parser.add_argument(
        "--mil-max-instances",
        type=int,
        default=8,
        help="Deprecated compatibility option; fixed-location fusion uses one slot per AV/PV/TV/MV location.",
    )
    parser.add_argument(
        "--mil-location-embedding-dim",
        type=int,
        default=4,
        help="Deprecated compatibility option.",
    )
    parser.add_argument(
        "--mil-instance-loss-weight",
        type=float,
        default=0.25,
        help="Auxiliary per-location recording loss weight used when training the fixed-location fusion model.",
    )
    parser.add_argument("--calibration", choices=["none", "platt"], default="platt")
    parser.add_argument(
        "--weak-murmur-weight",
        type=float,
        default=1.0,
        help="Per-recording loss multiplier for Present patients with systolic murmur grading I/VI.",
    )
    parser.add_argument(
        "--moderate-murmur-weight",
        type=float,
        default=1.0,
        help="Per-recording loss multiplier for Present patients with systolic murmur grading II/VI.",
    )
    parser.add_argument(
        "--location-aware-calibration",
        action="store_true",
        help="Fit the patient-level calibrated probability from per-location recording probabilities on the tuning split.",
    )
    parser.add_argument(
        "--ltsrr-prob",
        type=float,
        default=0.0,
        help=(
            "Probability of applying Local Time-Frequency Spectrum Random Replacement to each training spectrogram. "
            "0 disables the augmentation."
        ),
    )
    parser.add_argument(
        "--ltsrr-k",
        type=int,
        default=4,
        help="Number of non-overlapping time segments used by LTSRR when --ltsrr-prob > 0.",
    )
    parser.add_argument(
        "--ltsrr-frequency-ratio",
        type=float,
        default=0.25,
        help="Fraction of frequency bins replaced inside each LTSRR time segment.",
    )
    parser.add_argument(
        "--ltsrr-minority-only",
        action="store_true",
        help="Apply LTSRR only to positive/Present CNN training samples.",
    )
    parser.add_argument(
        "--smote-minority-augmentation",
        action="store_true",
        help="Use SMOTE to synthesize minority-class spectrograms for CNN training only.",
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="Number of nearest minority-class neighbors considered by SMOTE.",
    )
    parser.add_argument(
        "--smote-target-ratio",
        type=float,
        default=1.0,
        help="Target minority/majority ratio after SMOTE. 1.0 balances the training split.",
    )
    parser.add_argument(
        "--loss",
        choices=["bce", "focal"],
        default="bce",
        help="CNN training loss. BCE keeps the historical behavior; focal down-weights easy examples.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter used when --loss focal.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Optional positive-class alpha used when --loss focal. If omitted, no alpha factor is applied.",
    )
    parser.add_argument(
        "--auc-loss-weight",
        type=float,
        default=0.0,
        help="Weight for an auxiliary pairwise AUC ranking loss added to BCE/Focal. 0 disables it.",
    )
    parser.add_argument(
        "--auc-loss-margin",
        type=float,
        default=1.0,
        help="Margin used by the pairwise AUC ranking loss.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.systole_threshold is not None and not 0.0 <= args.systole_threshold <= 1.0:
        raise ValueError("--systole-threshold must be between 0.0 and 1.0.")
    if args.systole_margin_ms < 0.0:
        raise ValueError("--systole-margin-ms must be non-negative.")
    if args.cnn_phase_mode not in PHASE_LABELS:
        raise ValueError("--cnn-phase-mode must be one of: systole, diastole, both.")
    if args.low_hz < 0.0:
        raise ValueError("--low-hz must be non-negative.")
    if args.high_hz <= args.low_hz:
        raise ValueError("--high-hz must be greater than --low-hz.")
    if args.n_mels <= 0:
        raise ValueError("--n-mels must be greater than 0.")
    if args.weak_murmur_weight <= 0.0:
        raise ValueError("--weak-murmur-weight must be greater than 0.")
    if args.moderate_murmur_weight <= 0.0:
        raise ValueError("--moderate-murmur-weight must be greater than 0.")
    if args.mil_max_instances <= 0:
        raise ValueError("--mil-max-instances must be greater than 0.")
    if args.mil_location_embedding_dim <= 0:
        raise ValueError("--mil-location-embedding-dim must be greater than 0.")
    if args.mil_instance_loss_weight < 0.0:
        raise ValueError("--mil-instance-loss-weight must be non-negative.")
    if not 0.0 <= args.ltsrr_prob <= 1.0:
        raise ValueError("--ltsrr-prob must be between 0.0 and 1.0.")
    if args.ltsrr_k <= 0:
        raise ValueError("--ltsrr-k must be greater than 0.")
    if not 0.0 < args.ltsrr_frequency_ratio <= 1.0:
        raise ValueError("--ltsrr-frequency-ratio must be greater than 0 and at most 1.")
    if args.smote_k_neighbors <= 0:
        raise ValueError("--smote-k-neighbors must be greater than 0.")
    if not 0.0 < args.smote_target_ratio <= 1.0:
        raise ValueError("--smote-target-ratio must be greater than 0 and at most 1.")
    if args.patient_mil_attention and args.smote_minority_augmentation:
        raise ValueError("--smote-minority-augmentation is not supported with --patient-mil-attention.")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be non-negative.")
    if args.focal_alpha is not None and not 0.0 <= args.focal_alpha <= 1.0:
        raise ValueError("--focal-alpha must be between 0.0 and 1.0.")
    if args.auc_loss_weight < 0.0:
        raise ValueError("--auc-loss-weight must be non-negative.")
    if args.auc_loss_margin < 0.0:
        raise ValueError("--auc-loss-margin must be non-negative.")
    if args.auc_loss_weight > 0.0 and args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 when --auc-loss-weight is enabled.")
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.spectrogram_cache_dir or (output_dir / "spectrogram_cache")
    device = choose_device(args.device)
    print(f"Using device: {device}")

    tcn_device = torch.device("cpu")
    tcn_model, tcn_normalizer, tcn_cfg, _checkpoint = tcn.load_checkpoint_for_eval(args.tcn_checkpoint.resolve(), tcn_device)
    if getattr(tcn_cfg, "target_mode", "cardiac-phase") == "systole-binary" and args.cnn_phase_mode in {"diastole", "both"}:
        raise ValueError("--cnn-phase-mode diastole/both requires a TCN checkpoint trained with target_mode=cardiac-phase.")

    stft_cfg = StftConfig(
        target_sample_rate=args.target_sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        high_hz=args.high_hz,
        max_frames=args.max_frames,
        min_systole_seconds=args.min_systole_seconds,
        systole_threshold=args.systole_threshold,
        systole_margin_ms=args.systole_margin_ms,
        low_hz=args.low_hz,
        cnn_phase_mode=args.cnn_phase_mode,
        spectrogram_type=args.spectrogram_type,
        n_mels=args.n_mels,
    )

    items = build_items(args.dataset_dir.resolve(), args.locations, args.max_recordings)
    specs, labels, meta = prepare_spectrograms(
        items,
        stft_cfg,
        cache_dir,
        args.overwrite_cache,
        args.predicted_tsv_dir.resolve(),
        args.overwrite_predictions,
        tcn_model,
        tcn_normalizer,
        tcn_cfg,
        tcn_device,
        args.progress,
    )
    meta = meta.merge(load_patient_context(args.dataset_dir.resolve()), on="patient_id", how="left")
    meta.to_csv(output_dir / "recording_metadata.csv", index=False)

    dilations = tuple(int(part.strip()) for part in args.dilations.split(",") if part.strip())
    model_cfg = ModelConfig(
        freq_bins=int(specs.shape[1]),
        max_frames=int(specs.shape[2]),
        base_channels=args.base_channels,
        dropout=args.dropout,
        dilations=dilations,
        pooling=args.pooling,
        encoder_block=args.encoder_block,
    )

    patient_table = meta.drop_duplicates("patient_id")[["patient_id", "target"]].copy()
    patient_ids = patient_table["patient_id"].astype(str).to_numpy()
    y_patient = patient_table["target"].to_numpy(dtype=int)
    fold_patient_ids = stratified_patient_folds(patient_ids, y_patient, args.folds, args.seed)

    recording_oof = np.zeros(len(meta), dtype=np.float32)
    recording_threshold = np.zeros(len(meta), dtype=np.float32)
    fold_rows: list[dict[str, float | int]] = []
    history_rows: list[dict[str, float | int]] = []
    calibrated_patient_rows: list[pd.DataFrame] = []

    for fold, val_patient_ids in enumerate(fold_patient_ids, start=1):
        val_mask = meta["patient_id"].astype(str).isin(set(val_patient_ids))
        val_idx = np.flatnonzero(val_mask.to_numpy())
        train_idx = np.flatnonzero(~val_mask.to_numpy())
        val_probs, threshold, fold_metrics, history, val_patient_calibrated = train_one_fold(
            fold,
            specs,
            labels,
            meta,
            train_idx,
            val_idx,
            model_cfg,
            args,
            device,
            output_dir,
        )
        recording_oof[val_idx] = val_probs
        recording_threshold[val_idx] = threshold
        fold_rows.append(fold_metrics)
        history_rows.extend(history)
        calibrated_patient_rows.append(val_patient_calibrated)
        print(
            f"Fold {fold}/{args.folds}: "
            f"AUPRC={fold_metrics['auprc']:.3f} AUROC={fold_metrics['auroc']:.3f} "
            f"BA={fold_metrics['balanced_accuracy']:.3f} sens={fold_metrics['sensitivity']:.3f} "
            f"spec={fold_metrics['specificity']:.3f}"
        )

    recording_df = meta[["recording_id", "patient_id", "location", "murmur", "target"]].copy()
    recording_df["prob_present"] = recording_oof
    recording_df["threshold"] = recording_threshold
    recording_df.to_csv(output_dir / "recording_oof_predictions.csv", index=False)

    calibrated_patient_oof = pd.concat(calibrated_patient_rows, ignore_index=True)
    if args.patient_mil_attention:
        threshold_by_fold = {int(row["fold"]): float(row["threshold"]) for row in fold_rows}
        recording_count = recording_df.groupby("patient_id")["recording_id"].count()
        patient_oof = calibrated_patient_oof[
            ["patient_id", "murmur", "target", "fold", "prob_present_raw"]
        ].rename(columns={"prob_present_raw": "prob_present"})
        patient_oof["threshold"] = patient_oof["fold"].map(threshold_by_fold).astype(float)
        patient_oof["recording_count"] = patient_oof["patient_id"].map(recording_count).astype(int)
    else:
        patient_oof = recording_df.groupby("patient_id", as_index=False).agg(
            murmur=("murmur", "first"),
            target=("target", "first"),
            prob_present=("prob_present", "max"),
            threshold=("threshold", "first"),
            recording_count=("recording_id", "count"),
        )
    y_true = patient_oof["target"].to_numpy(dtype=int)
    y_prob = patient_oof["prob_present"].to_numpy(dtype=float)
    patient_oof["pred_present_threshold_05"] = (y_prob >= 0.5).astype(int)
    patient_oof["pred_present_threshold_tuned"] = (y_prob >= patient_oof["threshold"].to_numpy()).astype(int)
    patient_oof.to_csv(output_dir / "patient_oof_predictions.csv", index=False)

    calibrated_patient_oof["pred_present_calibrated_threshold_05"] = (
        calibrated_patient_oof["prob_present_calibrated"].to_numpy(dtype=float) >= 0.5
    ).astype(int)
    calibrated_patient_oof.to_csv(output_dir / "patient_oof_predictions_calibrated.csv", index=False)
    if args.patient_mil_attention:
        attention_paths = sorted(output_dir.glob("fold_*_mil_instance_attention_validation.csv"))
        if attention_paths:
            attention_oof = pd.concat([pd.read_csv(path, dtype={"patient_id": str}) for path in attention_paths], ignore_index=True)
            attention_oof.to_csv(output_dir / "mil_instance_attention_oof.csv", index=False)
    raw_threshold_tables = write_threshold_tables_by_fold(
        calibrated_patient_oof,
        output_dir,
        prob_column="prob_present_raw",
        label="raw",
    )
    calibrated_threshold_tables = write_threshold_tables_by_fold(
        calibrated_patient_oof,
        output_dir,
        prob_column="prob_present_calibrated",
        label="calibrated",
    )
    write_threshold_metrics_report(
        output_dir / "threshold_metrics_by_fold.md",
        raw_threshold_tables,
        calibrated_threshold_tables,
    )
    y_calibrated_true = calibrated_patient_oof["target"].to_numpy(dtype=int)
    y_calibrated_prob = calibrated_patient_oof["prob_present_calibrated"].to_numpy(dtype=float)

    tuned_pred = patient_oof["pred_present_threshold_tuned"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_counts(y_true, tuned_pred.astype(float), 0.5)
    tuned_metrics = {
        "threshold": "per_fold_youden",
        "auroc": roc_auc(y_true, y_prob),
        "auprc": average_precision(y_true, y_prob),
        "balanced_accuracy": 0.5 * ((tp / (tp + fn) if (tp + fn) else 0.0) + (tn / (tn + fp) if (tn + fp) else 0.0)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "f1": 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))) if tp and (tp + fp) and (tp + fn) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    oof_metrics_05 = metrics(y_true, y_prob, 0.5)
    calibrated_metrics_05 = metrics(y_calibrated_true, y_calibrated_prob, 0.5)
    calibrated_metrics_05["brier_score"] = brier_score(y_calibrated_true, y_calibrated_prob)
    calibrated_metrics_05["raw_brier_score"] = brier_score(y_true, y_prob)

    fold_metrics = pd.DataFrame(fold_rows)
    history = pd.DataFrame(history_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)
    plot_pr(y_true, y_prob, output_dir / "precision_recall_oof.png")
    plot_pr(y_calibrated_true, y_calibrated_prob, output_dir / "precision_recall_oof_calibrated.png")
    (output_dir / "config.json").write_text(
        json.dumps({"stft_config": asdict(stft_cfg), "model_config": asdict(model_cfg), "args": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )
    write_summary(
        output_dir / "summary.md",
        args,
        stft_cfg,
        model_cfg,
        meta,
        fold_metrics,
        oof_metrics_05,
        tuned_metrics,
        calibrated_metrics_05,
    )
    print(f"Patient OOF AUPRC={oof_metrics_05['auprc']:.3f} AUROC={oof_metrics_05['auroc']:.3f}")
    print(f"Outputs: {output_dir}")
