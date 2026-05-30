"""Adapter for the systole STFT CNN architecture."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch

from ..data import split_cnn_fit_tune_patients


def load_cnn_module() -> ModuleType:
    print("Loading CNN module. First run may spend time importing scipy/torch...", flush=True)
    from .. import cnn

    return cnn


def make_cnn_args(args: argparse.Namespace, fold_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        batch_size=args.cnn_batch_size,
        epochs=args.cnn_epochs,
        patience=args.cnn_patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        calibration=args.calibration,
        weak_murmur_weight=args.weak_murmur_weight,
        moderate_murmur_weight=args.moderate_murmur_weight,
        location_aware_calibration=args.location_aware_calibration,
        ltsrr_prob=args.ltsrr_prob,
        ltsrr_k=args.ltsrr_k,
        ltsrr_frequency_ratio=args.ltsrr_frequency_ratio,
        ltsrr_minority_only=args.ltsrr_minority_only,
        specaug_time_prob=getattr(args, "specaug_time_prob", 0.0),
        specaug_time_width=getattr(args, "specaug_time_width", 0),
        specaug_time_num_masks=getattr(args, "specaug_time_num_masks", 2),
        specaug_freq_prob=getattr(args, "specaug_freq_prob", 0.0),
        specaug_freq_width=getattr(args, "specaug_freq_width", 0),
        specaug_freq_num_masks=getattr(args, "specaug_freq_num_masks", 2),
        mixup_alpha=getattr(args, "mixup_alpha", 0.0),
        smote_minority_augmentation=args.smote_minority_augmentation,
        smote_k_neighbors=args.smote_k_neighbors,
        smote_target_ratio=args.smote_target_ratio,
        freq_norm=getattr(args, "freq_norm", "perbin"),
        aux_pitch_loss_weight=getattr(args, "aux_pitch_loss_weight", 0.0),
        init_encoder=getattr(args, "init_encoder", None),
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        auc_loss_weight=args.auc_loss_weight,
        auc_loss_margin=args.auc_loss_margin,
        encoder_block=args.encoder_block,
        patient_mil_attention=args.patient_mil_attention,
        mil_max_instances=args.mil_max_instances,
        mil_location_embedding_dim=args.mil_location_embedding_dim,
        mil_instance_loss_weight=args.mil_instance_loss_weight,
        progress=args.progress,
        output_dir=fold_dir / "cnn",
        locations=args.locations,
    )


def train_cnn_for_fold(
    args: argparse.Namespace,
    fold: int,
    fold_dir: Path,
    tcn_checkpoint: Path,
    all_items: list[object],
    all_meta: pd.DataFrame,
    train_patient_ids: set[str],
    val_patient_ids: set[str],
    tcn_module: ModuleType,
    cnn_module: ModuleType,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float | int], list[dict[str, float | int]], pd.DataFrame]:
    predicted_tsv_dir = fold_dir / "predicted_tsvs"
    cache_dir = fold_dir / "spectrogram_cache"
    cnn_output_dir = fold_dir / "cnn"
    cnn_output_dir.mkdir(parents=True, exist_ok=True)

    tcn_device = torch.device("cpu")
    if getattr(args, "use_ground_truth_segments", False) or tcn_checkpoint is None:
        tcn_model, tcn_normalizer, tcn_cfg = None, None, None
    else:
        tcn_model, tcn_normalizer, tcn_cfg, _checkpoint = tcn_module.load_checkpoint_for_eval(tcn_checkpoint, tcn_device)

    stft_cfg = cnn_module.StftConfig(
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
        stft_segment_mode=getattr(args, "stft_segment_mode", "concat"),
        phase_contrast=bool(getattr(args, "phase_contrast", False)),
        phase_contrast_dual=bool(getattr(args, "phase_contrast_dual", False)),
        phase_contrast_robust=bool(getattr(args, "phase_contrast_robust", False)),
        use_ground_truth_segments=bool(getattr(args, "use_ground_truth_segments", False)),
        use_temporal_features=bool(getattr(args, "use_temporal_features", False)),
        window_mode=getattr(args, "window_mode", "phase"),
        peak_window_seconds=float(getattr(args, "peak_window_seconds", 1.0)),
    )
    specs, labels, meta = cnn_module.prepare_spectrograms(
        all_items,
        stft_cfg,
        cache_dir,
        args.overwrite_cache,
        predicted_tsv_dir,
        False,
        tcn_model,
        tcn_normalizer,
        tcn_cfg,
        tcn_device,
        args.progress,
    )
    # All spectrograms are now in memory; we no longer need the on-disk feature cache or the
    # predicted-TSV scratch directory. Freeing these here keeps peak fold disk usage low.
    del tcn_model
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    if predicted_tsv_dir.exists():
        shutil.rmtree(predicted_tsv_dir)
    patient_context_columns = [
        "patient_id",
        "murmur_locations",
        "most_audible_location",
        "systolic_murmur_timing",
        "systolic_murmur_shape",
        "systolic_murmur_grading",
        "systolic_murmur_pitch",
        "systolic_murmur_quality",
        "outcome",
        "age",
        "sex",
        "height",
        "weight",
        "pregnancy_status",
    ]
    context = all_meta[[col for col in patient_context_columns if col in all_meta.columns]].drop_duplicates("patient_id")
    meta = meta.merge(context, on="patient_id", how="left")
    meta.to_csv(fold_dir / "recording_metadata.csv", index=False)

    cnn_fit_patient_ids, cnn_tune_patient_ids = split_cnn_fit_tune_patients(
        meta,
        train_patient_ids,
        args.cnn_inner_val_size,
        args.seed,
        fold,
    )
    train_idx = np.flatnonzero(meta["patient_id"].astype(str).isin(cnn_fit_patient_ids).to_numpy())
    tune_idx = np.flatnonzero(meta["patient_id"].astype(str).isin(cnn_tune_patient_ids).to_numpy())
    val_idx = np.flatnonzero(meta["patient_id"].astype(str).isin(val_patient_ids).to_numpy())
    if len(train_idx) == 0 or len(tune_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(f"Fold {fold} has empty fit, tune, or validation recordings after systole extraction.")
    (fold_dir / "cnn_fit_patient_ids.txt").write_text("\n".join(sorted(cnn_fit_patient_ids)) + "\n", encoding="utf-8")
    (fold_dir / "cnn_tune_patient_ids.txt").write_text("\n".join(sorted(cnn_tune_patient_ids)) + "\n", encoding="utf-8")

    dilations = tuple(int(part.strip()) for part in args.dilations.split(",") if part.strip())
    model_cfg = cnn_module.ModelConfig(
        freq_bins=int(specs.shape[1]),
        max_frames=int(specs.shape[2]),
        base_channels=args.base_channels,
        dropout=args.dropout,
        dilations=dilations,
        pooling=args.pooling,
        encoder_block=args.encoder_block,
        n_temporal_features=(cnn_module.N_TEMPORAL_FEATURES
                             if bool(getattr(args, "use_temporal_features", False)) else 0),
        arch=getattr(args, "model_arch", "cnn"),
        rnn_hidden=int(getattr(args, "rnn_hidden", 64)),
        rnn_layers=int(getattr(args, "rnn_layers", 2)),
        rnn_type=getattr(args, "rnn_type", "gru"),
        freq_emphasis=bool(getattr(args, "freq_emphasis", False)),
        freq_attention=bool(getattr(args, "freq_attention", False)),
        freq_low_hz=float(getattr(args, "freq_low_hz", 100.0)),
        freq_high_hz=float(getattr(args, "freq_high_hz", 600.0)),
        freq_emphasis_alpha_init=float(getattr(args, "freq_emphasis_alpha_init", 0.0)),
        freq_sample_rate=int(args.target_sample_rate),
        freq_fmax=float(args.high_hz),
        freq_mel_scale=(getattr(args, "spectrogram_type", "stft") == "log-mel"),
        freq_linear_branch=bool(getattr(args, "freq_linear_branch", False)),
        freq_linear_hidden=int(getattr(args, "freq_linear_hidden", 32)),
        freq_linear_arch=getattr(args, "freq_linear_arch", "transformer"),
        freq_linear_heads=int(getattr(args, "freq_linear_heads", 4)),
        freq_linear_layers=int(getattr(args, "freq_linear_layers", 2)),
        aux_pitch_classes=int(getattr(args, "aux_pitch_classes", 0)),
        n_demographic_features=(cnn_module.DEMOGRAPHIC_FEATURE_DIM
                                if bool(getattr(args, "demographic", False)) else 0),
    )
    cnn_args = make_cnn_args(args, fold_dir)
    device = cnn_module.choose_device(args.cnn_device)
    val_probs, threshold, fold_metrics, history, val_patient_calibrated = cnn_module.train_one_fold(
        fold,
        specs,
        labels,
        meta,
        train_idx,
        val_idx,
        model_cfg,
        cnn_args,
        device,
        cnn_output_dir,
        tune_indices=tune_idx,
    )
    payload = {
        "stft_config": asdict(stft_cfg),
        "model_config": asdict(model_cfg),
        "tcn_train_patients": sorted(train_patient_ids),
        "cnn_fit_patients": sorted(cnn_fit_patient_ids),
        "cnn_tune_patients": sorted(cnn_tune_patient_ids),
        "outer_val_patients": sorted(val_patient_ids),
        "tcn_checkpoint": str(tcn_checkpoint),
        "selected_threshold": float(threshold),
    }
    (fold_dir / "fold_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return val_idx, val_probs, meta, fold_metrics, history, val_patient_calibrated
