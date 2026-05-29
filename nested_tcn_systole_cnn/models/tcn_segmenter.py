"""Adapter for the fold-specific TCN segmenter architecture."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch

from ..data import make_tcn_subset_dataset
from ..paths import REPO_ROOT


def load_tcn_module() -> ModuleType:
    print("Loading TCN module. First run may spend time importing scipy/torch...", flush=True)
    from .. import tcn

    return tcn


def build_tcn_train_command(args: argparse.Namespace, subset_dir: Path, tcn_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "nested_tcn_systole_cnn.tcn",
        "--dataset-dir",
        str(subset_dir),
        "--output-dir",
        str(tcn_dir),
        "--epochs",
        str(args.tcn_epochs),
        "--batch-size",
        str(args.tcn_batch_size),
        "--device",
        args.tcn_device,
        "--val-size",
        str(args.tcn_val_size),
        "--test-size",
        str(args.tcn_test_size),
        "--label-mode",
        "overlap",
        "--target-mode",
        args.tcn_target_mode,
        "--other-mode",
        args.tcn_other_mode,
        "--boundary-ignore-ms",
        str(args.tcn_boundary_ignore_ms),
        "--systole-weight-multiplier",
        str(args.tcn_systole_weight_multiplier),
        "--no-causal",
        "--pooling",
        args.tcn_pooling,
        "--exclude-murmur-unknown",
    ]
    tcn_specaug_time_prob = float(getattr(args, "tcn_specaug_time_prob", 0.0))
    tcn_specaug_freq_prob = float(getattr(args, "tcn_specaug_freq_prob", 0.0))
    if tcn_specaug_time_prob > 0.0 or tcn_specaug_freq_prob > 0.0:
        command += [
            "--specaug-time-prob", str(tcn_specaug_time_prob),
            "--specaug-time-width", str(int(getattr(args, "tcn_specaug_time_width", 20))),
            "--specaug-time-num-masks", str(int(getattr(args, "tcn_specaug_time_num_masks", 2))),
            "--specaug-freq-prob", str(tcn_specaug_freq_prob),
            "--specaug-freq-width", str(int(getattr(args, "tcn_specaug_freq_width", 6))),
            "--specaug-freq-num-masks", str(int(getattr(args, "tcn_specaug_freq_num_masks", 2))),
        ]
    if not args.progress:
        command.append("--no-progress")
    return command


def checkpoint_matches_args(payload: Any, args: argparse.Namespace) -> bool:
    if not isinstance(payload, dict):
        return False
    feature_config = payload.get("feature_config", {})
    checkpoint_args = payload.get("args", {})
    checkpoint_target_mode = feature_config.get("target_mode", "cardiac-phase")
    checkpoint_other_mode = feature_config.get("other_mode", "keep")
    checkpoint_boundary_ignore_ms = float(feature_config.get("boundary_ignore_ms", 0.0))
    checkpoint_systole_weight = float(checkpoint_args.get("systole_weight_multiplier", 1.0))
    checkpoint_specaug_time_prob = float(checkpoint_args.get("specaug_time_prob", 0.0))
    checkpoint_specaug_freq_prob = float(checkpoint_args.get("specaug_freq_prob", 0.0))
    requested_specaug_time_prob = float(getattr(args, "tcn_specaug_time_prob", 0.0))
    requested_specaug_freq_prob = float(getattr(args, "tcn_specaug_freq_prob", 0.0))
    return (
        checkpoint_target_mode == args.tcn_target_mode
        and checkpoint_other_mode == args.tcn_other_mode
        and abs(checkpoint_boundary_ignore_ms - args.tcn_boundary_ignore_ms) < 1e-12
        and abs(checkpoint_systole_weight - args.tcn_systole_weight_multiplier) < 1e-12
        and abs(checkpoint_specaug_time_prob - requested_specaug_time_prob) < 1e-12
        and abs(checkpoint_specaug_freq_prob - requested_specaug_freq_prob) < 1e-12
    )


def describe_checkpoint_mismatch(payload: Any, args: argparse.Namespace) -> str:
    feature_config = payload.get("feature_config", {}) if isinstance(payload, dict) else {}
    checkpoint_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    checkpoint_target_mode = feature_config.get("target_mode", "cardiac-phase")
    checkpoint_other_mode = feature_config.get("other_mode", "keep")
    checkpoint_boundary_ignore_ms = float(feature_config.get("boundary_ignore_ms", 0.0))
    checkpoint_systole_weight = float(checkpoint_args.get("systole_weight_multiplier", 1.0))
    return (
        f"Existing TCN checkpoint uses target_mode={checkpoint_target_mode}, other_mode={checkpoint_other_mode}; "
        f"boundary_ignore_ms={checkpoint_boundary_ignore_ms:g}, "
        f"systole_weight_multiplier={checkpoint_systole_weight:g}; retraining with "
        f"target_mode={args.tcn_target_mode}, other_mode={args.tcn_other_mode}, "
        f"boundary_ignore_ms={args.tcn_boundary_ignore_ms:g}, "
        f"systole_weight_multiplier={args.tcn_systole_weight_multiplier:g}."
    )


def train_tcn_for_fold(
    args: argparse.Namespace,
    fold_dir: Path,
    train_patient_ids: set[str],
    parse_recording_id: Callable[[Path], tuple[str, str]],
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> Path:
    tcn_dir = fold_dir / "tcn"
    checkpoint = tcn_dir / "best_model.pt"
    if checkpoint.exists() and not args.force_retrain_tcn:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint_matches_args(payload, args):
            return checkpoint
        print(describe_checkpoint_mismatch(payload, args))

    subset_dir = fold_dir / "tcn_dataset_train_patients"
    make_tcn_subset_dataset(args.dataset_dir.resolve(), subset_dir, train_patient_ids, parse_recording_id)
    if tcn_dir.exists():
        shutil.rmtree(tcn_dir)
    tcn_dir.mkdir(parents=True, exist_ok=True)

    command = build_tcn_train_command(args, subset_dir, tcn_dir)
    try:
        runner(command, check=True, cwd=REPO_ROOT)
    finally:
        # Free the TCN feature cache (~1.7 GB) and the symlink subset whether training
        # succeeds or fails; only `best_model.pt` and the small plots are needed downstream.
        # Running this in `finally` prevents orphaned caches from filling the disk on crash.
        tcn_cache = tcn_dir / "cache"
        if tcn_cache.exists():
            shutil.rmtree(tcn_cache)
        if subset_dir.exists():
            shutil.rmtree(subset_dir)
    return checkpoint

