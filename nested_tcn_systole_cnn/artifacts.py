"""Cleanup helpers for fold-level intermediate artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path


FOLD_TRAINING_ARTIFACT_DIRS = (
    "tcn_dataset_train_patients",
    "predicted_tsvs",
    "spectrogram_cache",
    "tcn/cache",
)


def cleanup_fold_training_artifacts(fold_dir: Path) -> list[Path]:
    """Remove large fold-local artifacts that are not needed after metrics are collected."""
    removed: list[Path] = []
    for relative_path in FOLD_TRAINING_ARTIFACT_DIRS:
        path = fold_dir / relative_path
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return removed

