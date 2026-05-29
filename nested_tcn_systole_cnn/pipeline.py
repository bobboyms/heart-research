"""End-to-end nested validation pipeline."""

from __future__ import annotations

import argparse
import json
from types import ModuleType

import numpy as np
import pandas as pd

from .artifacts import cleanup_fold_training_artifacts
from .cli import validate_args
from .data import load_patient_context, select_patient_subset
from .evaluation import add_oof_prediction_columns, tuned_metrics_by_fold, write_summary
from .experiment_registry import (
    build_metrics_summary,
    flatten_metrics_for_registry,
    resolve_experiment_paths,
    update_registry,
)
from .models.systole_cnn import load_cnn_module, train_cnn_for_fold
from .models.tcn_segmenter import load_tcn_module, train_tcn_for_fold


def load_project_modules() -> tuple[ModuleType, ModuleType]:
    tcn_module = load_tcn_module()
    cnn_module = load_cnn_module()
    print("Modules loaded. Starting nested experiment setup...", flush=True)
    return tcn_module, cnn_module


def build_recording_metadata(cnn_module: ModuleType, args: argparse.Namespace) -> tuple[list[object], pd.DataFrame]:
    all_items = cnn_module.build_items(args.dataset_dir.resolve(), args.locations, None)
    all_meta = pd.DataFrame(
        {
            "recording_id": [item.recording_id for item in all_items],
            "patient_id": [item.patient_id for item in all_items],
            "location": [item.location for item in all_items],
            "murmur": [item.murmur for item in all_items],
            "target": [1 if item.murmur == "Present" else 0 for item in all_items],
        }
    )
    all_meta = all_meta.merge(load_patient_context(args.dataset_dir.resolve()), on="patient_id", how="left")

    # Irreducible-floor experiment: drop Present patients whose systolic murmur grading is in
    # --exclude-present-grades (e.g. "I/VI"), so the positive class becomes only audible murmurs.
    excluded = [g.strip().upper() for g in str(getattr(args, "exclude_present_grades", "") or "").split(",") if g.strip()]
    if excluded and "systolic_murmur_grading" in all_meta.columns:
        grading = all_meta["systolic_murmur_grading"].fillna("").astype(str).str.strip().str.upper()
        drop_mask = (all_meta["target"] == 1) & grading.isin(excluded)
        dropped_patients = all_meta.loc[drop_mask, "patient_id"].nunique()
        all_meta = all_meta.loc[~drop_mask].copy()
        print(f"Excluded grades {excluded}: dropped {dropped_patients} Present patients.", flush=True)

    all_meta = select_patient_subset(all_meta, args.max_patients, args.seed)
    allowed_patients = set(all_meta["patient_id"].astype(str))
    all_items = [item for item in all_items if item.patient_id in allowed_patients]
    return all_items, all_meta


def write_attention_oof_if_available(output_dir, args: argparse.Namespace, patient_oof: pd.DataFrame, decision_key: str) -> None:
    if not args.patient_mil_attention:
        return
    attention_paths = sorted(output_dir.glob("fold_*/cnn/fold_*_mil_instance_attention_validation.csv"))
    if not attention_paths:
        return
    attention_oof = pd.concat([pd.read_csv(path, dtype={"patient_id": str}) for path in attention_paths], ignore_index=True)
    prediction_columns = [
        "patient_id",
        "pred_present_raw_threshold_05",
        "pred_present_calibrated_threshold_05",
        f"pred_present_calibrated_threshold_{decision_key}",
    ]
    attention_oof = attention_oof.merge(patient_oof[prediction_columns], on="patient_id", how="left")
    decision_column = f"pred_present_calibrated_threshold_{decision_key}"
    target = attention_oof["target"].to_numpy(dtype=int)
    pred = attention_oof[decision_column].to_numpy(dtype=int)
    attention_oof["patient_error_type_at_decision_threshold"] = np.select(
        [
            (target == 1) & (pred == 1),
            (target == 1) & (pred == 0),
            (target == 0) & (pred == 1),
            (target == 0) & (pred == 0),
        ],
        ["TP", "FN", "FP", "TN"],
        default="Unknown",
    )
    attention_oof.to_csv(output_dir / "mil_instance_attention_oof.csv", index=False)


def run_nested_experiment(args: argparse.Namespace) -> None:
    validate_args(args)
    resolve_experiment_paths(args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    update_registry(args, "running")

    try:
        _run_nested_experiment(args, output_dir)
    except Exception as exc:
        update_registry(args, "failed", error=str(exc))
        raise


def _run_nested_experiment(args: argparse.Namespace, output_dir) -> None:
    tcn_module, cnn_module = load_project_modules()
    cnn_module.set_seed(args.seed)

    all_items, all_meta = build_recording_metadata(cnn_module, args)
    patient_table = all_meta.drop_duplicates("patient_id")[["patient_id", "target"]].copy()
    patient_ids = patient_table["patient_id"].astype(str).to_numpy()
    y_patient = patient_table["target"].to_numpy(dtype=int)
    fold_patient_ids = cnn_module.stratified_patient_folds(patient_ids, y_patient, args.folds, args.seed)

    patient_oof_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | int]] = []
    history_rows: list[dict[str, float | int]] = []

    for fold, val_patient_ids_array in enumerate(fold_patient_ids, start=1):
        val_patient_ids = set(str(pid) for pid in val_patient_ids_array)
        train_patient_ids = set(str(pid) for pid in patient_ids if str(pid) not in val_patient_ids)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "train_patient_ids.txt").write_text("\n".join(sorted(train_patient_ids)) + "\n", encoding="utf-8")
        (fold_dir / "val_patient_ids.txt").write_text("\n".join(sorted(val_patient_ids)) + "\n", encoding="utf-8")

        if getattr(args, "use_ground_truth_segments", False):
            print(
                f"Fold {fold}/{args.folds}: skipping TCN (using ground-truth .tsv segments); "
                f"validating CNN on {len(val_patient_ids)} patients",
                flush=True,
            )
            tcn_checkpoint = None
        else:
            print(
                f"Fold {fold}/{args.folds}: training TCN on {len(train_patient_ids)} patients; "
                f"validating CNN on {len(val_patient_ids)} patients",
                flush=True,
            )
            tcn_checkpoint = train_tcn_for_fold(args, fold_dir, train_patient_ids, cnn_module.parse_recording_id)
        _val_idx, _val_probs, _meta, fold_metrics, history, val_patient_calibrated = train_cnn_for_fold(
            args,
            fold,
            fold_dir,
            tcn_checkpoint,
            all_items,
            all_meta,
            train_patient_ids,
            val_patient_ids,
            tcn_module,
            cnn_module,
        )
        patient_oof_rows.append(val_patient_calibrated)
        fold_rows.append(fold_metrics)
        history_rows.extend(history)
        print(
            f"Fold {fold}/{args.folds}: "
            f"AUPRC={fold_metrics['auprc']:.3f} AUROC={fold_metrics['auroc']:.3f} "
            f"BA={fold_metrics['balanced_accuracy']:.3f}",
            flush=True,
        )
        # Persist partial results after each fold so long runs can be inspected mid-flight.
        # The canonical files are (re)written complete at the end; these are the running view.
        pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics_partial.csv", index=False)
        pd.concat(patient_oof_rows, ignore_index=True).to_csv(
            output_dir / "patient_oof_partial.csv", index=False
        )
        if args.cleanup_fold_artifacts:
            removed_paths = cleanup_fold_training_artifacts(fold_dir)
            if removed_paths:
                removed = ", ".join(str(path.relative_to(fold_dir)) for path in removed_paths)
                print(f"Fold {fold}/{args.folds}: removed training artifacts: {removed}", flush=True)

    patient_oof = pd.concat(patient_oof_rows, ignore_index=True)
    patient_oof, decision_key = add_oof_prediction_columns(patient_oof, args.decision_threshold)
    patient_oof.to_csv(output_dir / "patient_oof_predictions.csv", index=False)
    write_attention_oof_if_available(output_dir, args, patient_oof, decision_key)

    fold_metrics = pd.DataFrame(fold_rows)
    history = pd.DataFrame(history_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)
    # Drop the running partial views now that the complete canonical files exist.
    for partial in ("fold_metrics_partial.csv", "patient_oof_partial.csv"):
        (output_dir / partial).unlink(missing_ok=True)

    raw_tables = cnn_module.write_threshold_tables_by_fold(patient_oof, output_dir, "prob_present_raw", "raw")
    calibrated_tables = cnn_module.write_threshold_tables_by_fold(
        patient_oof,
        output_dir,
        "prob_present_calibrated",
        "calibrated",
    )
    cnn_module.write_threshold_metrics_report(output_dir / "threshold_metrics_by_fold.md", raw_tables, calibrated_tables)

    y_true = patient_oof["target"].to_numpy(dtype=int)
    raw_prob = patient_oof["prob_present_raw"].to_numpy(dtype=float)
    calibrated_prob = patient_oof["prob_present_calibrated"].to_numpy(dtype=float)
    raw_metrics_05 = cnn_module.metrics(y_true, raw_prob, 0.5)
    calibrated_metrics_05 = cnn_module.metrics(y_true, calibrated_prob, 0.5)
    calibrated_metrics_05["brier_score"] = cnn_module.brier_score(y_true, calibrated_prob)
    calibrated_metrics_05["raw_brier_score"] = cnn_module.brier_score(y_true, raw_prob)
    decision_metrics = cnn_module.metrics(y_true, calibrated_prob, args.decision_threshold)
    decision_metrics["brier_score"] = cnn_module.brier_score(y_true, calibrated_prob)

    tuned_metrics = tuned_metrics_by_fold(
        patient_oof,
        fold_metrics,
        "prob_present_raw",
        "threshold",
        "per_fold_youden",
        cnn_module,
    )
    calibrated_tuned_metrics = tuned_metrics_by_fold(
        patient_oof,
        fold_metrics,
        "prob_present_calibrated",
        "calibrated_threshold",
        "calibrated_per_fold_youden",
        cnn_module,
    )
    cnn_module.plot_pr(y_true, raw_prob, output_dir / "precision_recall_oof_raw.png")
    cnn_module.plot_pr(y_true, calibrated_prob, output_dir / "precision_recall_oof_calibrated.png")

    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    metrics_summary = build_metrics_summary(
        args,
        raw_metrics_05,
        calibrated_metrics_05,
        tuned_metrics,
        calibrated_tuned_metrics,
        decision_metrics,
    )
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2, default=str), encoding="utf-8")
    write_summary(
        output_dir,
        args,
        patient_oof,
        fold_metrics,
        raw_metrics_05,
        calibrated_metrics_05,
        tuned_metrics,
        calibrated_tuned_metrics,
        decision_metrics,
    )
    update_registry(args, "completed", flatten_metrics_for_registry(metrics_summary))
    print(f"Done. Outputs: {output_dir}")
