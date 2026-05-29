# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.4",
#   "scipy>=1.12",
#   "torch>=2.2",
#   "tqdm>=4.66",
# ]
# ///
"""Train a TCN + temporal self-attention cardiac phase segmenter on CirCor.

Run from the repository root:

    uv run "modeling/Grupo F TCN attention segmentacao frame a frame/train_tcn_attention_frame_segmenter.py"

Quick smoke test:

    uv run "modeling/Grupo F TCN attention segmentacao frame a frame/train_tcn_attention_frame_segmenter.py" \
        --max-recordings 40 --epochs 1 --batch-size 4

Outputs are written to:

    modeling/Grupo F TCN attention segmentacao frame a frame/outputs/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import get_window
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm


LABEL_NAMES = {
    0: "other",
    1: "s1",
    2: "systole",
    3: "s2",
    4: "diastole",
}
IGNORE_INDEX = -100


@dataclass(frozen=True)
class FeatureConfig:
    frame_ms: float
    hop_ms: float
    n_mels: int
    low_hz: float
    high_hz: float
    add_deltas: bool


@dataclass(frozen=True)
class RecordingItem:
    recording_id: str
    patient_id: str
    location: str
    wav_path: str
    tsv_path: str
    murmur: str
    outcome: str


@dataclass
class Normalizer:
    mean: list[float]
    std: list[float]

    def apply(self, features: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return (features - mean) / std


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    default_dataset = repo_root / "circor-heart-sound-1.0.3"
    default_output = script_dir / "outputs"

    parser = argparse.ArgumentParser(
        description=(
            "Train a supervised dilated causal TCN + temporal self-attention model "
            "to predict CirCor cardiac phase labels for every spectrogram frame: "
            "0=other, 1=S1, 2=systole, 3=S2, 4=diastole."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--reuse-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse output normalization.json and train_label_counts.json when present. "
            "This avoids rereading every cached feature file after interrupted runs."
        ),
    )
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--exclude-murmur-unknown", action="store_true")

    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--low-hz", type=float, default=20.0)
    parser.add_argument("--high-hz", type=float, default=1800.0)
    parser.add_argument("--no-deltas", action="store_true")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build/reuse feature cache plus normalization/label statistics, then exit before training.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--train-window-seconds",
        type=float,
        default=6.0,
        help="Train on fixed-length windows instead of full recordings. Use 0 to train on full recordings.",
    )
    parser.add_argument(
        "--train-window-hop-seconds",
        type=float,
        default=3.0,
        help="Hop between training windows. Ignored when --train-window-seconds is 0.",
    )
    parser.add_argument(
        "--max-frames-per-batch",
        type=int,
        default=12000,
        help=(
            "Upper bound for padded frames per batch: max_sequence_frames * batch_items. "
            "Use 0 to disable. This prevents very long recordings from making attention batches too slow."
        ),
    )
    parser.add_argument(
        "--length-aware-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Group recordings with similar frame lengths in the same batch. "
            "This reduces padding and is especially important for attention."
        ),
    )
    parser.add_argument("--bucket-size-multiplier", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--levels", type=int, default=7)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--attention-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument(
        "--attention-window",
        type=int,
        default=65,
        help=(
            "Number of past/current frames used by local causal temporal self-attention. "
            "65 frames is about 0.65 s with the default 10 ms hop."
        ),
    )
    parser.add_argument("--attention-ff-multiplier", type=int, default=4)
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss", choices=["ce", "ce_dice", "focal", "focal_dice"], default="ce_dice")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--median-filter-frames", type=int, default=5)
    parser.add_argument("--min-segment-frames", type=int, default=3)

    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars during feature preparation, training, and evaluation.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "mps":
        print("MPS requested but not available; falling back to CPU.")
    return torch.device("cpu")


def parse_recording_id(path: Path) -> tuple[str, str]:
    match = re.match(r"(?P<patient>\d+)_(?P<location>[A-Za-z]+)(?:_\d+)?$", path.stem)
    if not match:
        raise ValueError(f"Unexpected recording name: {path.name}")
    return match.group("patient"), match.group("location")


def load_patient_metadata(dataset_dir: Path) -> dict[str, dict[str, str]]:
    csv_path = dataset_dir / "training_data.csv"
    table = pd.read_csv(csv_path, dtype={"Patient ID": str})
    metadata: dict[str, dict[str, str]] = {}
    for _index, row in table.iterrows():
        patient_id = str(row["Patient ID"])
        metadata[patient_id] = {
            "murmur": str(row["Murmur"]),
            "outcome": str(row["Outcome"]),
        }
    return metadata


def normalize_patient_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not np.isfinite(numeric):
        return None
    return str(int(numeric))


def build_recording_index(args: argparse.Namespace) -> list[RecordingItem]:
    data_dir = args.dataset_dir / "training_data"
    metadata = load_patient_metadata(args.dataset_dir)
    items: list[RecordingItem] = []

    for wav_path in sorted(data_dir.glob("*.wav")):
        tsv_path = wav_path.with_suffix(".tsv")
        if not tsv_path.exists():
            continue
        patient_id, location = parse_recording_id(wav_path)
        patient_meta = metadata.get(patient_id, {"murmur": "Unknown", "outcome": "Unknown"})
        murmur = patient_meta["murmur"]
        if args.exclude_murmur_unknown and murmur == "Unknown":
            continue
        items.append(
            RecordingItem(
                recording_id=wav_path.stem,
                patient_id=patient_id,
                location=location,
                wav_path=str(wav_path),
                tsv_path=str(tsv_path),
                murmur=murmur,
                outcome=patient_meta["outcome"],
            )
        )

    if args.max_recordings is not None:
        rng = random.Random(args.seed)
        rng.shuffle(items)
        items = sorted(items[: args.max_recordings], key=lambda item: item.recording_id)

    if not items:
        raise RuntimeError(f"No wav+tsv recordings found under {data_dir}")
    return items


def split_by_patient(
    items: list[RecordingItem],
    dataset_dir: Path,
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[RecordingItem]]:
    if val_size < 0 or test_size < 0 or val_size + test_size >= 1:
        raise ValueError("--val-size and --test-size must be non-negative and sum to less than 1.")

    patients: dict[str, list[RecordingItem]] = {}
    murmur_by_patient: dict[str, str] = {}
    for item in items:
        patients.setdefault(item.patient_id, []).append(item)
        murmur_by_patient[item.patient_id] = item.murmur

    patient_groups = build_patient_leakage_groups(dataset_dir, set(patients))
    rng = random.Random(seed)
    split_patient_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for _murmur, groups in group_patient_groups_by_murmur(patient_groups, murmur_by_patient).items():
        groups = list(groups)
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        n_test = min(n_test, max(0, n - 2)) if n >= 3 else 0
        n_val = min(n_val, max(0, n - n_test - 1))
        split_patient_ids["test"].extend(patient_id for group in groups[:n_test] for patient_id in group)
        split_patient_ids["val"].extend(patient_id for group in groups[n_test : n_test + n_val] for patient_id in group)
        split_patient_ids["train"].extend(patient_id for group in groups[n_test + n_val :] for patient_id in group)

    splits: dict[str, list[RecordingItem]] = {}
    for split_name, patient_ids in split_patient_ids.items():
        split_items = [item for pid in patient_ids for item in patients[pid]]
        splits[split_name] = sorted(split_items, key=lambda item: item.recording_id)

    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise RuntimeError(
            "The patient split produced an empty train/val/test subset. "
            "Use more recordings or reduce --val-size/--test-size."
        )
    return splits


def build_patient_leakage_groups(dataset_dir: Path, available_patients: set[str]) -> list[tuple[str, ...]]:
    parent = {patient_id: patient_id for patient_id in available_patients}

    def find(patient_id: str) -> str:
        parent.setdefault(patient_id, patient_id)
        while parent[patient_id] != patient_id:
            parent[patient_id] = parent[parent[patient_id]]
            patient_id = parent[patient_id]
        return patient_id

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    table = pd.read_csv(dataset_dir / "training_data.csv", dtype={"Patient ID": str})
    for _index, row in table.iterrows():
        patient_id = normalize_patient_id(row["Patient ID"])
        additional_id = normalize_patient_id(row.get("Additional ID"))
        if patient_id in available_patients:
            parent.setdefault(patient_id, patient_id)
        if additional_id in available_patients:
            parent.setdefault(additional_id, additional_id)
        if patient_id in available_patients and additional_id in available_patients:
            union(patient_id, additional_id)

    groups: dict[str, list[str]] = {}
    for patient_id in sorted(available_patients):
        groups.setdefault(find(patient_id), []).append(patient_id)
    return [tuple(patient_ids) for patient_ids in groups.values()]


def group_patient_groups_by_murmur(
    patient_groups: list[tuple[str, ...]],
    murmur_by_patient: dict[str, str],
) -> dict[str, list[tuple[str, ...]]]:
    grouped: dict[str, list[tuple[str, ...]]] = {}
    for patient_group in patient_groups:
        murmurs = [murmur_by_patient[patient_id] for patient_id in patient_group if patient_id in murmur_by_patient]
        murmur = "Present" if "Present" in murmurs else ("Unknown" if "Unknown" in murmurs else "Absent")
        grouped.setdefault(murmur, []).append(patient_group)
    return grouped


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    original_dtype = audio.dtype
    audio = audio.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        audio = audio / float(max(abs(info.min), info.max))
    else:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak

    audio = audio - float(np.mean(audio))
    return int(sample_rate), audio.astype(np.float32)


def read_segments(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=["start_time", "end_time", "label"],
        dtype={"start_time": float, "end_time": float, "label": int},
    )


def frame_audio(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame_len = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))

    if len(audio) == 0:
        audio = np.zeros(frame_len, dtype=np.float32)

    n_frames = max(1, int(math.ceil(max(0, len(audio) - frame_len) / hop_len)) + 1)
    total_len = (n_frames - 1) * hop_len + frame_len
    if total_len > len(audio):
        padded = np.zeros(total_len, dtype=np.float32)
        padded[: len(audio)] = audio
        audio = padded

    starts = hop_len * np.arange(n_frames, dtype=np.int64)
    indices = starts[:, None] + np.arange(frame_len, dtype=np.int64)[None, :]
    frames = audio[indices]
    centers_s = (starts + frame_len / 2.0) / float(sample_rate)
    duration_s = len(audio) / float(sample_rate)
    centers_s = np.minimum(centers_s, duration_s)
    return frames, centers_s.astype(np.float32)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft_bins: int,
    n_mels: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    high_hz = min(high_hz, sample_rate / 2.0 - 1.0)
    low_hz = max(1.0, min(low_hz, high_hz - 1.0))
    mel_points = np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft_bins - 1) * hz_points / (sample_rate / 2.0)).astype(int)
    filters = np.zeros((n_mels, n_fft_bins), dtype=np.float32)

    for i in range(1, n_mels + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_fft_bins - 1)

        for j in range(left, center):
            if 0 <= j < n_fft_bins:
                filters[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            if 0 <= j < n_fft_bins:
                filters[i - 1, j] = (right - j) / max(right - center, 1)

    return filters


def extract_frame_features(
    audio: np.ndarray,
    sample_rate: int,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    frames, centers_s = frame_audio(audio, sample_rate, cfg.frame_ms, cfg.hop_ms)
    window = get_window("hann", frames.shape[1], fftbins=True).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    filters = mel_filterbank(sample_rate, power.shape[1], cfg.n_mels, cfg.low_hz, cfg.high_hz)
    mel_power = np.maximum(power @ filters.T, 1e-10)
    log_mel = np.log(mel_power).astype(np.float32)

    if cfg.add_deltas:
        deltas = np.zeros_like(log_mel)
        if len(log_mel) > 1:
            deltas[1:-1] = 0.5 * (log_mel[2:] - log_mel[:-2])
            deltas[0] = log_mel[1] - log_mel[0]
            deltas[-1] = log_mel[-1] - log_mel[-2]
        features = np.concatenate([log_mel, deltas], axis=1)
    else:
        features = log_mel

    return features.astype(np.float32), centers_s


def labels_for_frame_centers(centers_s: np.ndarray, segments: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(centers_s), dtype=np.int64)
    for row in segments.itertuples(index=False):
        label = int(row.label)
        if label not in LABEL_NAMES:
            continue
        mask = (centers_s >= float(row.start_time)) & (centers_s < float(row.end_time))
        labels[mask] = label
    return labels


def cache_path_for_item(cache_dir: Path, item: RecordingItem, cfg: FeatureConfig) -> Path:
    cfg_key = (
        f"fm{cfg.frame_ms:g}_hm{cfg.hop_ms:g}_m{cfg.n_mels}_"
        f"lo{cfg.low_hz:g}_hi{cfg.high_hz:g}_d{int(cfg.add_deltas)}"
    )
    return cache_dir / cfg_key / f"{item.recording_id}.npz"


def load_or_create_features(
    item: RecordingItem,
    cfg: FeatureConfig,
    cache_dir: Path,
    overwrite_cache: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_path_for_item(cache_dir, item, cfg)
    if cache_path.exists() and not overwrite_cache:
        try:
            with np.load(cache_path) as data:
                return data["x"].astype(np.float32), data["y"].astype(np.int64)
        except Exception as exc:
            print(f"Warning: ignoring corrupt feature cache {cache_path.name}: {exc}")
            cache_path.unlink(missing_ok=True)

    sample_rate, audio = read_audio(Path(item.wav_path))
    features, centers_s = extract_frame_features(audio, sample_rate, cfg)
    labels = labels_for_frame_centers(centers_s, read_segments(Path(item.tsv_path)))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        x=features.astype(np.float32),
        y=labels.astype(np.int64),
        recording_id=item.recording_id,
        sample_rate=sample_rate,
        feature_config=json.dumps(asdict(cfg)),
    )
    return features, labels


class CirCorFrameDataset(Dataset):
    def __init__(
        self,
        items: list[RecordingItem],
        cfg: FeatureConfig,
        cache_dir: Path,
        overwrite_cache: bool = False,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.items = items
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.items[index]
        features, labels = load_or_create_features(item, self.cfg, self.cache_dir, self.overwrite_cache)
        if self.normalizer is not None:
            features = self.normalizer.apply(features)
        return {
            "x": torch.from_numpy(features.T.copy()),  # channels, frames
            "y": torch.from_numpy(labels.copy()),
            "recording_id": item.recording_id,
        }


class WindowedCirCorFrameDataset(Dataset):
    def __init__(
        self,
        items: list[RecordingItem],
        cfg: FeatureConfig,
        cache_dir: Path,
        overwrite_cache: bool,
        normalizer: Normalizer,
        window_seconds: float,
        hop_seconds: float,
        show_progress: bool = True,
    ) -> None:
        self.items = items
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache
        self.normalizer = normalizer
        hop_seconds = hop_seconds if hop_seconds > 0 else window_seconds
        self.window_frames = max(1, int(round(window_seconds * 1000.0 / cfg.hop_ms)))
        self.hop_frames = max(1, int(round(hop_seconds * 1000.0 / cfg.hop_ms)))
        self.windows: list[tuple[int, int, int]] = []

        iterator = progress_iter(
            enumerate(items),
            show_progress,
            total=len(items),
            desc="Indexing train windows",
            unit="rec",
            leave=False,
        )
        for item_index, item in iterator:
            features, _labels = load_or_create_features(item, cfg, cache_dir, overwrite_cache)
            n_frames = int(features.shape[0])
            if n_frames <= self.window_frames:
                self.windows.append((item_index, 0, n_frames))
                continue

            starts = list(range(0, n_frames - self.window_frames + 1, self.hop_frames))
            last_start = n_frames - self.window_frames
            if starts[-1] != last_start:
                starts.append(last_start)
            self.windows.extend((item_index, start, start + self.window_frames) for start in starts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item_index, start, end = self.windows[index]
        item = self.items[item_index]
        features, labels = load_or_create_features(item, self.cfg, self.cache_dir, self.overwrite_cache)
        features = features[start:end]
        labels = labels[start:end]
        features = self.normalizer.apply(features)
        return {
            "x": torch.from_numpy(features.T.copy()),
            "y": torch.from_numpy(labels.copy()),
            "recording_id": f"{item.recording_id}:{start}-{end}",
        }


class LengthAwareBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_size_multiplier: int,
        max_frames_per_batch: int,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
        self.max_frames_per_batch = max_frames_per_batch
        self.epoch = 0

    def make_batches_from_sorted_indices(self, indices: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_max_len = 0

        for index in indices:
            item_len = self.lengths[index]
            candidate_max_len = max(current_max_len, item_len)
            candidate_size = len(current) + 1
            exceeds_count = candidate_size > self.batch_size
            exceeds_frames = (
                self.max_frames_per_batch > 0
                and current
                and candidate_max_len * candidate_size > self.max_frames_per_batch
            )
            if exceeds_count or exceeds_frames:
                batches.append(current)
                current = []
                current_max_len = 0

            current.append(index)
            current_max_len = max(current_max_len, item_len)

        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterable[list[int]]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        if self.shuffle:
            rng.shuffle(indices)
            buckets = [
                indices[start : start + self.bucket_size]
                for start in range(0, len(indices), self.bucket_size)
            ]
            batches: list[list[int]] = []
            for bucket in buckets:
                bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)
                batches.extend(self.make_batches_from_sorted_indices(bucket))
            rng.shuffle(batches)
            yield from batches
            return

        indices.sort(key=lambda idx: self.lengths[idx], reverse=True)
        yield from self.make_batches_from_sorted_indices(indices)

    def __len__(self) -> int:
        indices = sorted(range(len(self.lengths)), key=lambda idx: self.lengths[idx], reverse=True)
        return len(self.make_batches_from_sorted_indices(indices))


def collate_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    channels = int(batch[0]["x"].shape[0])  # type: ignore[index, union-attr]
    lengths = [int(sample["y"].shape[0]) for sample in batch]  # type: ignore[index, union-attr]
    max_len = max(lengths)

    x = torch.zeros((len(batch), channels, max_len), dtype=torch.float32)
    y = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    recording_ids: list[str] = []

    for idx, sample in enumerate(batch):
        sample_x = sample["x"]  # type: ignore[assignment]
        sample_y = sample["y"]  # type: ignore[assignment]
        length = int(sample_y.shape[0])
        x[idx, :, :length] = sample_x
        y[idx, :length] = sample_y
        recording_ids.append(str(sample["recording_id"]))

    return {"x": x, "y": y, "lengths": torch.tensor(lengths), "recording_ids": recording_ids}


def feature_lengths_for_items(
    items: list[RecordingItem],
    cfg: FeatureConfig,
    cache_dir: Path,
    overwrite_cache: bool,
    show_progress: bool,
    desc: str,
) -> list[int]:
    lengths: list[int] = []
    iterator = progress_iter(items, show_progress, desc=desc, unit="rec", leave=False)
    for item in iterator:
        features, _labels = load_or_create_features(item, cfg, cache_dir, overwrite_cache)
        lengths.append(int(features.shape[0]))
    return lengths


def progress_iter(iterable: Iterable, enabled: bool, **kwargs: object) -> Iterable:
    if enabled:
        return tqdm(iterable, **kwargs)
    return iterable


def compute_normalizer(dataset: CirCorFrameDataset, show_progress: bool = True) -> Normalizer:
    total_count = 0
    total_sum: np.ndarray | None = None
    total_sumsq: np.ndarray | None = None

    iterator = progress_iter(
        range(len(dataset)),
        show_progress,
        desc="Preparing train features",
        unit="rec",
        leave=False,
    )
    for idx in iterator:
        item = dataset.items[idx]
        features, _labels = load_or_create_features(
            item,
            dataset.cfg,
            dataset.cache_dir,
            dataset.overwrite_cache,
        )
        features64 = features.astype(np.float64)
        if total_sum is None:
            total_sum = np.zeros(features.shape[1], dtype=np.float64)
            total_sumsq = np.zeros(features.shape[1], dtype=np.float64)
        total_sum += features64.sum(axis=0)
        total_sumsq += (features64**2).sum(axis=0)
        total_count += features.shape[0]

    if total_sum is None or total_sumsq is None or total_count == 0:
        raise RuntimeError("Could not compute feature normalization statistics.")

    mean = total_sum / total_count
    var = np.maximum(total_sumsq / total_count - mean**2, 1e-8)
    std = np.sqrt(var)
    return Normalizer(mean=mean.astype(float).tolist(), std=std.astype(float).tolist())


def compute_training_stats(
    dataset: CirCorFrameDataset,
    show_progress: bool = True,
) -> tuple[Normalizer, np.ndarray]:
    total_count = 0
    total_sum: np.ndarray | None = None
    total_sumsq: np.ndarray | None = None
    label_counts = np.zeros(len(LABEL_NAMES), dtype=np.int64)

    iterator = progress_iter(
        range(len(dataset)),
        show_progress,
        desc="Preparing train features/stats",
        unit="rec",
        leave=False,
    )
    for idx in iterator:
        item = dataset.items[idx]
        features, labels = load_or_create_features(
            item,
            dataset.cfg,
            dataset.cache_dir,
            dataset.overwrite_cache,
        )
        features64 = features.astype(np.float64)
        if total_sum is None:
            total_sum = np.zeros(features.shape[1], dtype=np.float64)
            total_sumsq = np.zeros(features.shape[1], dtype=np.float64)
        total_sum += features64.sum(axis=0)
        total_sumsq += (features64**2).sum(axis=0)
        total_count += features.shape[0]
        label_counts += np.bincount(labels, minlength=len(LABEL_NAMES))[: len(LABEL_NAMES)]

    if total_sum is None or total_sumsq is None or total_count == 0:
        raise RuntimeError("Could not compute feature normalization statistics.")

    mean = total_sum / total_count
    var = np.maximum(total_sumsq / total_count - mean**2, 1e-8)
    std = np.sqrt(var)
    normalizer = Normalizer(mean=mean.astype(float).tolist(), std=std.astype(float).tolist())
    return normalizer, label_counts


def count_train_labels(dataset: CirCorFrameDataset, show_progress: bool = True) -> np.ndarray:
    counts = np.zeros(len(LABEL_NAMES), dtype=np.int64)
    iterator = progress_iter(
        dataset.items,
        show_progress,
        desc="Counting train labels",
        unit="rec",
        leave=False,
    )
    for item in iterator:
        _features, labels = load_or_create_features(item, dataset.cfg, dataset.cache_dir, dataset.overwrite_cache)
        counts += np.bincount(labels, minlength=len(LABEL_NAMES))[: len(LABEL_NAMES)]
    return counts


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.downsample(x)


class LocalCausalTemporalSelfAttention(nn.Module):
    """Multi-head temporal attention over a fixed past window.

    Full self-attention over complete recordings can become too expensive for
    long PCG files. This layer keeps the causal constraint and limits memory by
    attending only to the previous `window_size - 1` frames plus the current one.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        window_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}.")
        if window_size < 1:
            raise ValueError("attention window size must be >= 1.")

        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.window_size = window_size
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, _channels, n_frames = x.shape
        window_size = min(self.window_size, n_frames)
        left_pad = window_size - 1

        q = self.q_proj(x).view(batch_size, self.heads, self.head_dim, n_frames)
        k = self.k_proj(x).view(batch_size, self.heads, self.head_dim, n_frames)
        v = self.v_proj(x).view(batch_size, self.heads, self.head_dim, n_frames)

        if left_pad > 0:
            k = F.pad(k, (left_pad, 0))
            v = F.pad(v, (left_pad, 0))

        k_windows = k.unfold(dimension=-1, size=window_size, step=1)
        v_windows = v.unfold(dimension=-1, size=window_size, step=1)
        scores = torch.einsum("bhdt,bhdtw->bhtw", q, k_windows) * self.scale

        if lengths is not None:
            frame_ids = torch.arange(n_frames, device=x.device)[None, :]
            valid = frame_ids < lengths[:, None]
            if left_pad > 0:
                valid = F.pad(valid, (left_pad, 0), value=False)
            valid_windows = valid.unfold(dimension=-1, size=window_size, step=1)
            scores = scores.masked_fill(~valid_windows[:, None, :, :], torch.finfo(scores.dtype).min)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.einsum("bhtw,bhdtw->bhdt", attention, v_windows)
        context = context.reshape(batch_size, self.channels, n_frames)
        return self.out_proj(context)


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        window_size: int,
        dropout: float,
        ff_multiplier: int,
    ) -> None:
        super().__init__()
        ff_channels = channels * ff_multiplier
        self.norm1 = nn.GroupNorm(1, channels)
        self.attention = LocalCausalTemporalSelfAttention(channels, heads, window_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, ff_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ff_channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), lengths=lengths))
        x = x + self.ffn(self.norm2(x))
        return x


class TCNAttentionFrameSegmenter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        levels: int,
        kernel_size: int,
        dropout: float,
        attention_layers: int,
        attention_heads: int,
        attention_window: int,
        attention_ff_multiplier: int,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for level in range(levels):
            dilation = 2**level
            block_in = in_channels if level == 0 else hidden_channels
            blocks.append(TemporalBlock(block_in, hidden_channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.attention_blocks = nn.ModuleList(
            [
                TemporalSelfAttentionBlock(
                    channels=hidden_channels,
                    heads=attention_heads,
                    window_size=attention_window,
                    dropout=dropout,
                    ff_multiplier=attention_ff_multiplier,
                )
                for _ in range(attention_layers)
            ]
        )
        self.classifier = nn.Conv1d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.tcn(x)
        for attention_block in self.attention_blocks:
            x = attention_block(x, lengths=lengths)
        return self.classifier(x)


def make_loaders(
    splits: dict[str, list[RecordingItem]],
    cfg: FeatureConfig,
    cache_dir: Path,
    overwrite_cache: bool,
    normalizer: Normalizer,
    batch_size: int,
    num_workers: int,
    length_aware_batches: bool,
    bucket_size_multiplier: int,
    max_frames_per_batch: int,
    train_window_seconds: float,
    train_window_hop_seconds: float,
    seed: int,
    show_progress: bool,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split_name, split_items in splits.items():
        if split_name == "train" and train_window_seconds > 0:
            dataset = WindowedCirCorFrameDataset(
                split_items,
                cfg,
                cache_dir,
                overwrite_cache,
                normalizer,
                window_seconds=train_window_seconds,
                hop_seconds=train_window_hop_seconds,
                show_progress=show_progress,
            )
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_batch,
            )
        elif length_aware_batches:
            dataset = CirCorFrameDataset(split_items, cfg, cache_dir, overwrite_cache, normalizer)
            lengths = feature_lengths_for_items(
                split_items,
                cfg,
                cache_dir,
                overwrite_cache,
                show_progress and split_name != "train",
                desc=f"Preparing {split_name} lengths",
            )
            batch_sampler = LengthAwareBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                seed=seed,
                bucket_size_multiplier=bucket_size_multiplier,
                max_frames_per_batch=max_frames_per_batch,
            )
            loaders[split_name] = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=collate_batch,
            )
        else:
            dataset = CirCorFrameDataset(split_items, cfg, cache_dir, overwrite_cache, normalizer)
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                collate_fn=collate_batch,
            )
    return loaders


def class_weights_from_counts(counts: np.ndarray) -> torch.Tensor:
    counts = np.maximum(counts.astype(np.float64), 1.0)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        mode: str,
        class_weights: torch.Tensor | None,
        dice_weight: float,
        focal_gamma: float,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=IGNORE_INDEX,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        valid = target != IGNORE_INDEX
        if self.mode.startswith("focal"):
            pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
            ce = ((1.0 - pt) ** self.focal_gamma) * ce
        base_loss = ce[valid].mean() if valid.any() else ce.mean()

        if "dice" not in self.mode:
            return base_loss
        dice = multiclass_dice_loss(logits, target)
        return base_loss + self.dice_weight * dice


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0.0

    probs = torch.softmax(logits, dim=1).permute(0, 2, 1)
    safe_target = target.clamp_min(0)
    one_hot = F.one_hot(safe_target, num_classes=len(LABEL_NAMES)).to(dtype=probs.dtype)
    valid_f = valid.unsqueeze(-1).to(dtype=probs.dtype)
    probs = probs * valid_f
    one_hot = one_hot * valid_f

    intersection = (probs * one_hot).sum(dim=(0, 1))
    denominator = probs.sum(dim=(0, 1)) + one_hot.sum(dim=(0, 1))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def create_loss(args: argparse.Namespace, class_weights: torch.Tensor | None) -> SegmentationLoss:
    return SegmentationLoss(
        mode=args.loss,
        class_weights=class_weights,
        dice_weight=args.dice_weight,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float,
    show_progress: bool,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_frames = 0

    iterator = progress_iter(
        loader,
        show_progress,
        desc=f"Epoch {epoch:03d} train",
        unit="batch",
        leave=False,
    )
    for batch in iterator:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lengths = batch["lengths"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths=lengths)
        loss = loss_fn(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        valid_frames = int((y != IGNORE_INDEX).sum().item())
        total_loss += float(loss.item()) * valid_frames
        total_frames += valid_frames
        running_loss = total_loss / max(total_frames, 1)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{running_loss:.4f}", frames=total_frames)

    return total_loss / max(total_frames, 1)


def median_smooth_labels(labels: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or len(labels) < 3:
        return labels
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, len(labels) if len(labels) % 2 == 1 else len(labels) - 1)
    if kernel_size <= 1:
        return labels
    radius = kernel_size // 2
    padded = np.pad(labels, (radius, radius), mode="edge")
    return np.asarray(
        [int(np.median(padded[index : index + kernel_size])) for index in range(len(labels))],
        dtype=np.int64,
    )


def remove_short_segments(labels: np.ndarray, min_segment_frames: int) -> np.ndarray:
    if min_segment_frames <= 1 or len(labels) == 0:
        return labels
    labels = labels.copy()
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and labels[end] == labels[start]:
            end += 1
        if end - start < min_segment_frames:
            left_label = labels[start - 1] if start > 0 else None
            right_label = labels[end] if end < len(labels) else None
            if left_label is not None:
                labels[start:end] = left_label
            elif right_label is not None:
                labels[start:end] = right_label
        start = end
    return labels


def postprocess_prediction(labels: np.ndarray, median_filter_frames: int, min_segment_frames: int) -> np.ndarray:
    labels = median_smooth_labels(labels, median_filter_frames)
    labels = remove_short_segments(labels, min_segment_frames)
    return labels


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    show_progress: bool = True,
    desc: str = "Evaluating",
    postprocess: bool = True,
    median_filter_frames: int = 5,
    min_segment_frames: int = 3,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_frames = 0
    confusion = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=np.int64)

    iterator = progress_iter(loader, show_progress, desc=desc, unit="batch", leave=False)
    for batch in iterator:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(x, lengths=lengths)
        loss = loss_fn(logits, y)
        pred = logits.argmax(dim=1)
        mask = y != IGNORE_INDEX

        valid_frames = int(mask.sum().item())
        total_loss += float(loss.item()) * valid_frames
        total_frames += valid_frames

        y_cpu = y.detach().cpu().numpy()
        pred_cpu = pred.detach().cpu().numpy()
        lengths_np = batch["lengths"].detach().cpu().numpy()
        for sample_index, length in enumerate(lengths_np):
            y_np = y_cpu[sample_index, :length]
            pred_np = pred_cpu[sample_index, :length]
            if postprocess:
                pred_np = postprocess_prediction(pred_np, median_filter_frames, min_segment_frames)
            confusion += confusion_matrix(y_np, pred_np, labels=list(LABEL_NAMES.keys()))
        if show_progress and hasattr(iterator, "set_postfix"):
            running_loss = total_loss / max(total_frames, 1)
            iterator.set_postfix(loss=f"{running_loss:.4f}", frames=total_frames)

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(total_frames, 1)
    metrics["frames"] = int(total_frames)
    metrics["confusion"] = confusion.tolist()
    return metrics


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    iou_values: list[float] = []
    weighted_f1_sum = 0.0

    for label, name in LABEL_NAMES.items():
        tp = float(confusion[label, label])
        fp = float(confusion[:, label].sum() - tp)
        fn = float(confusion[label, :].sum() - tp)
        support = int(confusion[label, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        f1_values.append(f1)
        recall_values.append(recall)
        iou_values.append(iou)
        weighted_f1_sum += f1 * support
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "support": support,
        }

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": weighted_f1_sum / total if total else 0.0,
        "balanced_accuracy": float(np.mean(recall_values)),
        "mean_iou": float(np.mean(iou_values)),
        "per_class": per_class,
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cached_normalizer(output_dir: Path) -> Normalizer | None:
    path = output_dir / "normalization.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "mean" not in payload or "std" not in payload:
        return None
    return Normalizer(mean=list(payload["mean"]), std=list(payload["std"]))


def load_cached_label_counts(output_dir: Path) -> np.ndarray | None:
    path = output_dir / "train_label_counts.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    try:
        return np.asarray([int(payload[LABEL_NAMES[i]]) for i in range(len(LABEL_NAMES))], dtype=np.int64)
    except (KeyError, TypeError, ValueError):
        return None


def save_split_manifest(output_dir: Path, splits: dict[str, list[RecordingItem]]) -> None:
    manifest = {
        split_name: [asdict(item) for item in split_items]
        for split_name, split_items in splits.items()
    }
    save_json(output_dir / "split_manifest.json", manifest)


def save_confusion_matrix(output_dir: Path, split_name: str, confusion: list[list[int]]) -> None:
    labels = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    matrix = np.asarray(confusion, dtype=np.int64)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output_dir / f"{split_name}_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{split_name} confusion matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_history_plot(output_dir: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="val accuracy")
    axes[1].plot(frame["epoch"], frame["val_macro_f1"], label="val macro F1")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    cfg: FeatureConfig,
    splits: dict[str, list[RecordingItem]],
    label_counts: np.ndarray,
    metrics: dict[str, dict[str, object]],
    best_epoch: int | None,
) -> None:
    lines: list[str] = []
    lines.append("# Grupo F - TCN + attention segmentacao frame a frame")
    lines.append("")
    lines.append("## Configuracao")
    lines.append("")
    lines.append(f"- Dataset: `{args.dataset_dir}`")
    lines.append(f"- Features: {cfg.n_mels} log-mel bins, frame={cfg.frame_ms} ms, hop={cfg.hop_ms} ms, deltas={cfg.add_deltas}")
    lines.append(
        f"- Modelo: TCN causal dilatada + self-attention temporal causal local, "
        f"hidden={args.hidden_channels}, levels={args.levels}, kernel={args.kernel_size}, "
        f"attention_layers={args.attention_layers}, attention_heads={args.attention_heads}, "
        f"attention_window={args.attention_window}, dropout={args.dropout}"
    )
    lines.append(f"- Batches por duracao: {args.length_aware_batches}")
    lines.append(f"- Limite de frames preenchidos por batch: {args.max_frames_per_batch}")
    lines.append(f"- Treino por janelas: {args.train_window_seconds}s com hop {args.train_window_hop_seconds}s")
    lines.append(f"- Loss: {args.loss}, dice_weight={args.dice_weight}, focal_gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}")
    lines.append(
        f"- Pos-processamento: {args.postprocess}, median_filter_frames={args.median_filter_frames}, "
        f"min_segment_frames={args.min_segment_frames}"
    )
    lines.append(f"- Device solicitado/usado: `{args.device}`")
    lines.append(f"- Melhor epoca por macro F1 de validacao: {best_epoch if best_epoch is not None else 'n/a'}")
    lines.append("")
    lines.append("## Split por paciente")
    lines.append("")
    for split_name, split_items in splits.items():
        patients = {item.patient_id for item in split_items}
        lines.append(f"- {split_name}: {len(split_items)} gravacoes, {len(patients)} pacientes")
    lines.append("")
    lines.append("## Distribuicao dos rotulos no treino")
    lines.append("")
    lines.append("| Classe | Frames |")
    lines.append("|---|---:|")
    for label, count in enumerate(label_counts):
        lines.append(f"| {label} = {LABEL_NAMES[label]} | {int(count)} |")
    lines.append("")
    lines.append("## Metricas")
    lines.append("")
    lines.append("| Split | Loss | Accuracy | Macro F1 | Weighted F1 | Balanced accuracy | Mean IoU | Frames |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split_name in ["val", "test"]:
        metric = metrics[split_name]
        lines.append(
            f"| {split_name} | {metric['loss']:.4f} | {metric['accuracy']:.4f} | "
            f"{metric['macro_f1']:.4f} | {metric['weighted_f1']:.4f} | "
            f"{metric['balanced_accuracy']:.4f} | {metric['mean_iou']:.4f} | {metric['frames']} |"
        )
    lines.append("")
    lines.append("## Metricas por classe no teste")
    lines.append("")
    lines.append("| Classe | Precision | Recall | F1 | IoU | Support |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    per_class = metrics["test"]["per_class"]
    assert isinstance(per_class, dict)
    for label in LABEL_NAMES.values():
        row = per_class[label]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['iou']:.4f} | {row['support']} |"
        )
    lines.append("")
    lines.append("Arquivos principais: `best_model.pt`, `metrics.json`, `training_history.csv`, matrizes de confusao CSV/PNG.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_checkpoint_for_eval(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TCNAttentionFrameSegmenter, Normalizer, FeatureConfig, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = FeatureConfig(**checkpoint["feature_config"])
    normalizer = Normalizer(**checkpoint["normalizer"])
    model_config = checkpoint["model_config"]
    model = TCNAttentionFrameSegmenter(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, normalizer, cfg, checkpoint


def checkpoint_payload(
    model: nn.Module,
    args: argparse.Namespace,
    cfg: FeatureConfig,
    normalizer: Normalizer,
    epoch: int,
    val_metrics: dict[str, object],
) -> dict[str, object]:
    in_channels = cfg.n_mels * (2 if cfg.add_deltas else 1)
    return {
        "epoch": epoch,
        "val_metrics": val_metrics,
        "feature_config": asdict(cfg),
        "normalizer": asdict(normalizer),
        "model_config": {
            "in_channels": in_channels,
            "hidden_channels": args.hidden_channels,
            "levels": args.levels,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "attention_layers": args.attention_layers,
            "attention_heads": args.attention_heads,
            "attention_window": args.attention_window,
            "attention_ff_multiplier": args.attention_ff_multiplier,
            "num_classes": len(LABEL_NAMES),
        },
        "label_names": LABEL_NAMES,
        "args": vars(args),
        "model_state_dict": model.state_dict(),
    }


def main() -> None:
    args = parse_args()
    if args.hidden_channels % args.attention_heads != 0:
        raise ValueError("--hidden-channels must be divisible by --attention-heads.")
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (output_dir / "cache")

    device = choose_device(args.device)
    print(f"Using device: {device}")

    items = build_recording_index(args)
    splits = split_by_patient(items, args.dataset_dir, args.val_size, args.test_size, args.seed)
    save_split_manifest(output_dir, splits)

    cfg = FeatureConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        n_mels=args.n_mels,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        add_deltas=not args.no_deltas,
    )

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        model, normalizer, cfg, _checkpoint = load_checkpoint_for_eval(args.checkpoint, device)
        train_dataset = CirCorFrameDataset(splits["train"], cfg, cache_dir, args.overwrite_cache)
        label_counts = load_cached_label_counts(output_dir) if args.reuse_stats and not args.overwrite_cache else None
        if label_counts is None:
            label_counts = count_train_labels(train_dataset, show_progress=args.progress)
    else:
        train_dataset_for_stats = CirCorFrameDataset(splits["train"], cfg, cache_dir, args.overwrite_cache)
        normalizer = load_cached_normalizer(output_dir) if args.reuse_stats and not args.overwrite_cache else None
        label_counts = load_cached_label_counts(output_dir) if args.reuse_stats and not args.overwrite_cache else None

        if normalizer is None and label_counts is None:
            print("Computing/caching training features, normalization statistics, and label counts...")
            normalizer, label_counts = compute_training_stats(train_dataset_for_stats, show_progress=args.progress)
            save_json(output_dir / "normalization.json", asdict(normalizer))
            save_json(
                output_dir / "train_label_counts.json",
                {LABEL_NAMES[i]: int(count) for i, count in enumerate(label_counts)},
            )
        elif normalizer is None:
            print("Computing/caching training features and normalization statistics...")
            normalizer = compute_normalizer(train_dataset_for_stats, show_progress=args.progress)
            save_json(output_dir / "normalization.json", asdict(normalizer))
        else:
            print("Reusing normalization statistics from existing normalization.json.")

        if label_counts is not None:
            print("Train label counts are available.")
        else:
            label_counts = count_train_labels(train_dataset_for_stats, show_progress=args.progress)
            save_json(
                output_dir / "train_label_counts.json",
                {LABEL_NAMES[i]: int(count) for i, count in enumerate(label_counts)},
            )

        if args.prepare_only:
            print(f"Preparation complete. Stats and cache are ready under {output_dir}.")
            return

        in_channels = cfg.n_mels * (2 if cfg.add_deltas else 1)
        model = TCNAttentionFrameSegmenter(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            levels=args.levels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            attention_layers=args.attention_layers,
            attention_heads=args.attention_heads,
            attention_window=args.attention_window,
            attention_ff_multiplier=args.attention_ff_multiplier,
            num_classes=len(LABEL_NAMES),
        ).to(device)

    loaders = make_loaders(
        splits=splits,
        cfg=cfg,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        normalizer=normalizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        length_aware_batches=args.length_aware_batches,
        bucket_size_multiplier=args.bucket_size_multiplier,
        max_frames_per_batch=args.max_frames_per_batch,
        train_window_seconds=args.train_window_seconds,
        train_window_hop_seconds=args.train_window_hop_seconds,
        seed=args.seed,
        show_progress=args.progress,
    )

    class_weights = class_weights_from_counts(label_counts).to(device) if args.use_class_weights else None
    loss_fn = create_loss(args, class_weights)

    best_epoch: int | None = None
    history: list[dict[str, float]] = []
    best_val_macro_f1 = -1.0

    if not args.eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                loss_fn,
                device,
                args.grad_clip,
                show_progress=args.progress,
                epoch=epoch,
            )
            val_metrics = evaluate(
                model,
                loaders["val"],
                loss_fn,
                device,
                show_progress=args.progress,
                desc=f"Epoch {epoch:03d} val",
                postprocess=args.postprocess,
                median_filter_frames=args.median_filter_frames,
                min_segment_frames=args.min_segment_frames,
            )
            val_macro_f1 = float(val_metrics["macro_f1"])
            val_accuracy = float(val_metrics["accuracy"])
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_metrics["loss"]),
                    "val_accuracy": val_accuracy,
                    "val_macro_f1": val_macro_f1,
                    "val_mean_iou": float(val_metrics["mean_iou"]),
                }
            )
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_accuracy:.4f} val_macro_f1={val_macro_f1:.4f}"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                torch.save(
                    checkpoint_payload(model, args, cfg, normalizer, epoch, val_metrics),
                    output_dir / "best_model.pt",
                )

        print(f"Training finished in {(time.time() - start) / 60.0:.1f} minutes.")
        save_history_plot(output_dir, history)

        model, normalizer, cfg, _checkpoint = load_checkpoint_for_eval(output_dir / "best_model.pt", device)

    final_metrics = {
        "val": evaluate(
            model,
            loaders["val"],
            loss_fn,
            device,
            show_progress=args.progress,
            desc="Final val",
            postprocess=args.postprocess,
            median_filter_frames=args.median_filter_frames,
            min_segment_frames=args.min_segment_frames,
        ),
        "test": evaluate(
            model,
            loaders["test"],
            loss_fn,
            device,
            show_progress=args.progress,
            desc="Final test",
            postprocess=args.postprocess,
            median_filter_frames=args.median_filter_frames,
            min_segment_frames=args.min_segment_frames,
        ),
    }
    save_json(output_dir / "metrics.json", final_metrics)
    save_confusion_matrix(output_dir, "val", final_metrics["val"]["confusion"])  # type: ignore[arg-type]
    save_confusion_matrix(output_dir, "test", final_metrics["test"]["confusion"])  # type: ignore[arg-type]
    write_summary(output_dir, args, cfg, splits, label_counts, final_metrics, best_epoch)
    print(f"Done. Summary written to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
