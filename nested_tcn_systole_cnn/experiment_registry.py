"""Persistent registry for named experiment runs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import DEFAULT_OUTPUT_DIR
from .scoring import parse_score_weights, weighted_mean_score


REGISTRY_CSV = "registry.csv"
REGISTRY_JSONL = "registry.jsonl"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "experiment"


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resolve_experiment_paths(args: Any) -> None:
    args.experiments_dir = args.experiments_dir.resolve()
    args.experiments_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or "manual_run"
    if args.run_name and args.output_dir is None:
        run_id = f"{slugify(args.run_name)}__{run_id_timestamp()}"
        args.output_dir = args.experiments_dir / run_id
    else:
        args.output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).resolve()
        run_id = slugify(args.output_dir.name)

    args.run_name = run_name
    args.run_id = run_id
    args.registry_csv = args.experiments_dir / REGISTRY_CSV
    args.registry_jsonl = args.experiments_dir / REGISTRY_JSONL
    args.score_weight_values = parse_score_weights(args.score_weights)


def selected_args(args: Any) -> dict[str, Any]:
    fields = [
        "run_id",
        "run_name",
        "output_dir",
        "dataset_dir",
        "locations",
        "folds",
        "seed",
        "max_patients",
        "tcn_epochs",
        "tcn_batch_size",
        "tcn_device",
        "tcn_val_size",
        "tcn_test_size",
        "tcn_pooling",
        "tcn_boundary_ignore_ms",
        "tcn_systole_weight_multiplier",
        "tcn_target_mode",
        "tcn_other_mode",
        "cnn_epochs",
        "cnn_patience",
        "cnn_batch_size",
        "cnn_inner_val_size",
        "cnn_device",
        "pooling",
        "calibration",
        "decision_threshold",
        "weak_murmur_weight",
        "moderate_murmur_weight",
        "location_aware_calibration",
        "smote_minority_augmentation",
        "smote_k_neighbors",
        "smote_target_ratio",
        "loss",
        "focal_gamma",
        "focal_alpha",
        "auc_loss_weight",
        "auc_loss_margin",
        "lr",
        "weight_decay",
        "base_channels",
        "dropout",
        "dilations",
        "encoder_block",
        "patient_mil_attention",
        "mil_instance_loss_weight",
        "target_sample_rate",
        "n_fft",
        "hop_length",
        "low_hz",
        "high_hz",
        "spectrogram_type",
        "n_mels",
        "max_frames",
        "cnn_phase_mode",
        "min_systole_seconds",
        "systole_threshold",
        "systole_margin_ms",
        "cleanup_fold_artifacts",
        "score_weights",
    ]
    payload = {}
    for field in fields:
        if hasattr(args, field):
            value = getattr(args, field)
            payload[field] = str(value) if isinstance(value, Path) else value
    return payload


def build_registry_row(args: Any, status: str, metrics: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    row = {
        "run_id": args.run_id,
        "run_name": args.run_name,
        "status": status,
        "updated_at": utc_timestamp(),
        "output_dir": str(args.output_dir),
        "summary_path": str(args.output_dir / "summary.md"),
        "metrics_summary_path": str(args.output_dir / "metrics_summary.json"),
        "tcn_target_mode": args.tcn_target_mode,
        "tcn_other_mode": args.tcn_other_mode,
        "tcn_systole_weight_multiplier": args.tcn_systole_weight_multiplier,
        "tcn_boundary_ignore_ms": args.tcn_boundary_ignore_ms,
        "cnn_epochs": args.cnn_epochs,
        "cnn_inner_val_size": args.cnn_inner_val_size,
        "encoder_block": args.encoder_block,
        "pooling": args.pooling,
        "patient_mil_attention": args.patient_mil_attention,
        "cnn_phase_mode": args.cnn_phase_mode,
        "weak_murmur_weight": args.weak_murmur_weight,
        "moderate_murmur_weight": args.moderate_murmur_weight,
        "decision_threshold": args.decision_threshold,
        "score_weights": args.score_weights,
        "error": error or "",
    }
    if metrics:
        row.update(metrics)
    return row


def upsert_registry_row(registry_csv: Path, row: dict[str, Any]) -> None:
    registry_csv.parent.mkdir(parents=True, exist_ok=True)
    if registry_csv.exists():
        table = pd.read_csv(registry_csv, dtype={"run_id": str})
        table = table.loc[table["run_id"].astype(str) != str(row["run_id"])].copy()
    else:
        table = pd.DataFrame()
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(registry_csv, index=False)


def append_registry_event(registry_jsonl: Path, row: dict[str, Any]) -> None:
    registry_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with registry_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def update_registry(args: Any, status: str, metrics: dict[str, Any] | None = None, error: str | None = None) -> None:
    row = build_registry_row(args, status, metrics, error)
    upsert_registry_row(args.registry_csv, row)
    append_registry_event(args.registry_jsonl, row)


def build_metrics_summary(
    args: Any,
    raw_metrics_05: dict[str, Any],
    calibrated_metrics_05: dict[str, Any],
    tuned_metrics: dict[str, Any],
    calibrated_tuned_metrics: dict[str, Any],
    decision_metrics: dict[str, Any],
) -> dict[str, Any]:
    comparison_metrics = dict(decision_metrics)
    comparison_metrics["precision_ppv"] = comparison_metrics.get("precision", 0.0)
    comparison_metrics["score_weights"] = args.score_weight_values
    comparison_metrics["mean_score"] = weighted_mean_score(comparison_metrics, args.score_weight_values)
    return {
        "run_id": args.run_id,
        "run_name": args.run_name,
        "status": "completed",
        "output_dir": str(args.output_dir),
        "comparison_metric": "calibrated_decision_threshold",
        "comparison_metrics": comparison_metrics,
        "raw_metrics_threshold_05": raw_metrics_05,
        "calibrated_metrics_threshold_05": calibrated_metrics_05,
        "raw_tuned_metrics_by_fold": tuned_metrics,
        "calibrated_tuned_metrics_by_fold": calibrated_tuned_metrics,
        "config": selected_args(args),
    }


def flatten_metrics_for_registry(metrics_summary: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics_summary["comparison_metrics"]
    keys = [
        "threshold",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "precision_ppv",
        "f1",
        "brier_score",
        "mean_score",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    return {key: metrics.get(key, "") for key in keys}
