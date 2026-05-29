"""Shared filesystem paths for the nested TCN + systole CNN experiment."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

TCN_SCRIPT = REPO_ROOT / "modeling" / "Grupo E TCN segmentacao frame a frame" / "train_tcn_frame_segmenter.py"
CNN_SCRIPT = REPO_ROOT / "modeling" / "Grupo G CNN dilatada systole TCN STFT" / "train_systole_stft_dilated_cnn.py"

DEFAULT_DATASET_DIR = REPO_ROOT / "circor-heart-sound-1.0.3"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs_nested"
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "nested_tcn_systole_cnn"
