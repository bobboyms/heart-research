from __future__ import annotations


import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm


from .audio import parse_recording_id, read_audio
from .config import LABEL_DIASTOLE, LABEL_SYSTOLE, N_TEMPORAL_FEATURES, RecordingItem, StftConfig
from .segments import extract_phase_audio, format_float_key, get_segments, parse_murmur_locations, phase_label_counts, phase_seconds_by_label, selected_phase_labels
from .spectrogram import compute_temporal_features, peak_window_specs, phase_contrast_spectrogram, phase_spectrogram, phase_spectrogram_per_segment


def build_items(dataset_dir: Path, locations: list[str], max_recordings: int | None) -> list[RecordingItem]:
    metadata = pd.read_csv(dataset_dir / "training_data.csv", dtype={"Patient ID": str})
    metadata = metadata.loc[metadata["Murmur"].isin(["Present", "Absent"])].copy()
    meta_by_patient = metadata.set_index("Patient ID")["Murmur"].to_dict()
    murmur_locs_by_patient = (
        metadata.set_index("Patient ID")["Murmur locations"].to_dict()
        if "Murmur locations" in metadata.columns
        else {}
    )
    data_dir = dataset_dir / "training_data"
    items: list[RecordingItem] = []
    for wav_path in sorted(data_dir.glob("*.wav")):
        try:
            patient_id, location = parse_recording_id(wav_path)
        except ValueError:
            continue
        if location not in locations or patient_id not in meta_by_patient:
            continue
        patient_murmur = str(meta_by_patient[patient_id])
        if patient_murmur == "Present":
            murmur_locations = parse_murmur_locations(murmur_locs_by_patient.get(patient_id))
            recording_present = location in murmur_locations
        else:
            recording_present = False
        items.append(
            RecordingItem(
                recording_id=wav_path.stem,
                patient_id=patient_id,
                location=location,
                wav_path=wav_path,
                tsv_path=wav_path.with_suffix(".tsv"),
                murmur=patient_murmur,
                recording_present=recording_present,
            )
        )
    if max_recordings is not None:
        items = items[:max_recordings]
    return items


def load_patient_context(dataset_dir: Path) -> pd.DataFrame:
    table = pd.read_csv(dataset_dir / "training_data.csv", dtype={"Patient ID": str})
    columns = {
        "Patient ID": "patient_id",
        "Murmur locations": "murmur_locations",
        "Most audible location": "most_audible_location",
        "Systolic murmur timing": "systolic_murmur_timing",
        "Systolic murmur shape": "systolic_murmur_shape",
        "Systolic murmur grading": "systolic_murmur_grading",
        "Systolic murmur pitch": "systolic_murmur_pitch",
        "Systolic murmur quality": "systolic_murmur_quality",
        "Outcome": "outcome",
    }
    available = [col for col in columns if col in table.columns]
    context = table[available].rename(columns=columns).copy()
    context["patient_id"] = context["patient_id"].astype(str)
    return context.drop_duplicates("patient_id")


def cache_path(cache_dir: Path, item: RecordingItem, cfg: StftConfig) -> Path:
    low_hz = float(getattr(cfg, "low_hz", 0.0))
    band_key = f"hi{cfg.high_hz:g}" if low_hz == 0.0 else f"lo{low_hz:g}_hi{cfg.high_hz:g}"
    key = f"sr{cfg.target_sample_rate}_fft{cfg.n_fft}_hop{cfg.hop_length}_{band_key}_frames{cfg.max_frames}"
    if getattr(cfg, "spectrogram_type", "stft") == "log-mel":
        key = f"logmel_mels{int(getattr(cfg, 'n_mels', 64))}_{key}"
    if cfg.cnn_phase_mode != "systole":
        key = f"{key}_phase{cfg.cnn_phase_mode}"
    if cfg.systole_threshold is not None or cfg.systole_margin_ms != 0:
        threshold_key = "argmax" if cfg.systole_threshold is None else f"thr{format_float_key(cfg.systole_threshold)}"
        margin_key = f"margin{format_float_key(cfg.systole_margin_ms)}ms"
        key = f"{key}_{threshold_key}_{margin_key}"
    if getattr(cfg, "stft_segment_mode", "concat") == "per-segment":
        key = f"{key}_segmode-per-segment"
    if getattr(cfg, "phase_contrast", False):
        key = f"{key}_phasecontrast"
        if getattr(cfg, "phase_contrast_dual", False):
            key = f"{key}-dual"
        if getattr(cfg, "phase_contrast_robust", False):
            key = f"{key}-robust"
    if getattr(cfg, "use_ground_truth_segments", False):
        key = f"{key}_gtseg"
    if getattr(cfg, "use_temporal_features", False):
        key = f"{key}_tf"
    if getattr(cfg, "window_mode", "phase") == "peak1s":
        key = f"{key}_peak{format_float_key(float(getattr(cfg, 'peak_window_seconds', 1.0)))}s"
    return cache_dir / key / f"{item.recording_id}.npz"


def prepare_spectrograms(
    items: list[RecordingItem],
    stft_cfg: StftConfig,
    cache_dir: Path,
    overwrite_cache: bool,
    predicted_tsv_dir: Path,
    overwrite_predictions: bool,
    tcn_model: nn.Module,
    tcn_normalizer: object,
    tcn_cfg: object,
    tcn_device: torch.device,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    specs: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, object]] = []
    phase_labels = selected_phase_labels(stft_cfg)
    iterator = tqdm(
        items,
        desc=f"Preparing {stft_cfg.cnn_phase_mode} STFTs",
        unit="rec",
        disable=not show_progress,
    )
    use_temporal = bool(getattr(stft_cfg, "use_temporal_features", False))
    window_mode = getattr(stft_cfg, "window_mode", "phase")
    if window_mode == "peak1s":
        for item in iterator:
            path = cache_path(cache_dir, item, stft_cfg)
            if path.exists() and not overwrite_cache:
                with np.load(path) as data:
                    window_specs = data["window_specs"].astype(np.float32)
            else:
                sample_rate, audio = read_audio(item.wav_path)
                segments = get_segments(item.wav_path, predicted_tsv_dir, overwrite_predictions,
                                        stft_cfg, tcn_model, tcn_normalizer, tcn_cfg, tcn_device)
                window_specs = peak_window_specs(audio, sample_rate, segments, stft_cfg)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, window_specs=window_specs.astype(np.float32))
            if window_specs.size == 0:
                continue
            for w in range(window_specs.shape[0]):
                specs.append(window_specs[w])
                labels.append(1 if item.recording_present else 0)
                rows.append({
                    "recording_id": item.recording_id,
                    "patient_id": item.patient_id,
                    "location": item.location,
                    "murmur": item.murmur,
                    "target": 1 if item.murmur == "Present" else 0,
                    "recording_target": 1 if item.recording_present else 0,
                    "phase_mode": "peak1s",
                    "window_index": w,
                    "phase_seconds": 0.0,
                    "phase_segments": 0,
                    "systole_seconds": 0.0,
                    "systole_segments": 0,
                    "diastole_seconds": 0.0,
                    "diastole_segments": 0,
                })
        if not specs:
            raise RuntimeError("No peak-window spectrograms were prepared.")
        return np.stack(specs), np.asarray(labels, dtype=np.float32), pd.DataFrame(rows)

    for item in iterator:
        path = cache_path(cache_dir, item, stft_cfg)
        temporal_vec = np.zeros(N_TEMPORAL_FEATURES, dtype=np.float32)
        if path.exists() and not overwrite_cache:
            with np.load(path) as data:
                spec = data["spec"].astype(np.float32)
                phase_seconds = float(data["phase_seconds"] if "phase_seconds" in data else data["systole_seconds"])
                phase_segments = int(data["phase_segments"] if "phase_segments" in data else data["systole_segments"])
                systole_seconds = float(data["systole_seconds"]) if "systole_seconds" in data else 0.0
                systole_segments = int(data["systole_segments"]) if "systole_segments" in data else 0
                diastole_seconds = float(data["diastole_seconds"]) if "diastole_seconds" in data else 0.0
                diastole_segments = int(data["diastole_segments"]) if "diastole_segments" in data else 0
                if use_temporal and "temporal" in data:
                    temporal_vec = data["temporal"].astype(np.float32)
        else:
            sample_rate, audio = read_audio(item.wav_path)
            segments = get_segments(
                item.wav_path,
                predicted_tsv_dir,
                overwrite_predictions,
                stft_cfg,
                tcn_model,
                tcn_normalizer,
                tcn_cfg,
                tcn_device,
            )
            phase_audio = extract_phase_audio(audio, sample_rate, segments, phase_labels, stft_cfg.systole_margin_ms)
            seconds_by_label = phase_seconds_by_label(audio, sample_rate, segments, phase_labels, stft_cfg.systole_margin_ms)
            counts_by_label = phase_label_counts(segments)
            phase_seconds = len(phase_audio) / float(sample_rate) if sample_rate else 0.0
            phase_segments = sum(counts_by_label[label] for label in phase_labels)
            systole_seconds = seconds_by_label.get(LABEL_SYSTOLE, 0.0)
            systole_segments = counts_by_label[LABEL_SYSTOLE] if LABEL_SYSTOLE in phase_labels else 0
            diastole_seconds = seconds_by_label.get(LABEL_DIASTOLE, 0.0)
            diastole_segments = counts_by_label[LABEL_DIASTOLE] if LABEL_DIASTOLE in phase_labels else 0
            if phase_seconds < stft_cfg.min_systole_seconds:
                spec = np.zeros((0, 0), dtype=np.float32)
            elif getattr(stft_cfg, "phase_contrast", False):
                spec = phase_contrast_spectrogram(
                    audio, sample_rate, segments, stft_cfg,
                    dual=bool(getattr(stft_cfg, "phase_contrast_dual", False)),
                    robust=bool(getattr(stft_cfg, "phase_contrast_robust", False)),
                )
            elif getattr(stft_cfg, "stft_segment_mode", "concat") == "per-segment":
                spec = phase_spectrogram_per_segment(audio, sample_rate, segments, phase_labels, stft_cfg)
            else:
                spec = phase_spectrogram(phase_audio, sample_rate, stft_cfg)
            if use_temporal and spec.size > 0:
                temporal_vec = compute_temporal_features(audio, sample_rate, segments, phase_labels, stft_cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                spec=spec.astype(np.float32),
                phase_mode=np.asarray(stft_cfg.cnn_phase_mode),
                spectrogram_type=np.asarray(getattr(stft_cfg, "spectrogram_type", "stft")),
                n_mels=np.asarray(getattr(stft_cfg, "n_mels", 64), dtype=np.int32),
                phase_seconds=np.asarray(phase_seconds, dtype=np.float32),
                phase_segments=np.asarray(phase_segments, dtype=np.int32),
                systole_seconds=np.asarray(systole_seconds, dtype=np.float32),
                systole_segments=np.asarray(systole_segments, dtype=np.int32),
                diastole_seconds=np.asarray(diastole_seconds, dtype=np.float32),
                diastole_segments=np.asarray(diastole_segments, dtype=np.int32),
                temporal=temporal_vec.astype(np.float32),
            )
        if spec.size == 0:
            continue
        specs.append(spec)
        labels.append(1 if item.recording_present else 0)
        row = {
            "recording_id": item.recording_id,
            "patient_id": item.patient_id,
            "location": item.location,
            "murmur": item.murmur,
            "target": 1 if item.murmur == "Present" else 0,
            "recording_target": 1 if item.recording_present else 0,
            "phase_mode": stft_cfg.cnn_phase_mode,
            "phase_seconds": phase_seconds,
            "phase_segments": phase_segments,
            "systole_seconds": systole_seconds,
            "systole_segments": systole_segments,
            "diastole_seconds": diastole_seconds,
            "diastole_segments": diastole_segments,
        }
        if use_temporal:
            for j in range(N_TEMPORAL_FEATURES):
                row[f"tf_{j}"] = float(temporal_vec[j])
        rows.append(row)
    if not specs:
        raise RuntimeError(f"No {stft_cfg.cnn_phase_mode} spectrograms were prepared.")
    return np.stack(specs), np.asarray(labels, dtype=np.float32), pd.DataFrame(rows)


def stratified_patient_folds(patient_ids: np.ndarray, y_patient: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = patient_ids[y_patient == 1].copy()
    neg = patient_ids[y_patient == 0].copy()
    rng.shuffle(pos)
    rng.shuffle(neg)
    fold_lists: list[list[str]] = [[] for _ in range(folds)]
    for idx, pid in enumerate(pos):
        fold_lists[idx % folds].append(str(pid))
    for idx, pid in enumerate(neg):
        fold_lists[idx % folds].append(str(pid))
    return [np.asarray(fold, dtype=str) for fold in fold_lists]


class StratifiedBinaryBatchSampler(Sampler[list[int]]):
    def __init__(self, labels: np.ndarray, batch_size: int) -> None:
        self.labels = labels.astype(int)
        self.batch_size = int(batch_size)
        self.pos_indices = np.flatnonzero(self.labels == 1)
        self.neg_indices = np.flatnonzero(self.labels == 0)
        self.pos_per_batch = max(1, self.batch_size // 2)
        self.neg_per_batch = max(1, self.batch_size - self.pos_per_batch)
        self.batch_count = max(
            math.ceil(len(self.pos_indices) / self.pos_per_batch),
            math.ceil(len(self.neg_indices) / self.neg_per_batch),
        )

    def __len__(self) -> int:
        return int(self.batch_count)

    def __iter__(self):
        for _ in range(self.batch_count):
            pos = np.random.choice(
                self.pos_indices,
                size=self.pos_per_batch,
                replace=len(self.pos_indices) < self.pos_per_batch,
            )
            neg = np.random.choice(
                self.neg_indices,
                size=self.neg_per_batch,
                replace=len(self.neg_indices) < self.neg_per_batch,
            )
            batch = np.concatenate([pos, neg]).astype(int)
            np.random.shuffle(batch)
            yield batch.tolist()


def use_stratified_batches(args: argparse.Namespace, labels: np.ndarray) -> bool:
    if float(getattr(args, "auc_loss_weight", 0.0)) <= 0.0:
        return False
    labels_int = labels.astype(int)
    return int(getattr(args, "batch_size", 0)) >= 2 and np.any(labels_int == 1) and np.any(labels_int == 0)


def build_train_loader(dataset: Dataset, labels: np.ndarray, args: argparse.Namespace) -> DataLoader:
    if use_stratified_batches(args, labels):
        return DataLoader(dataset, batch_sampler=StratifiedBinaryBatchSampler(labels, args.batch_size))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


def _broadcast_stats(value: float | np.ndarray, freq_bins: int) -> np.ndarray:
    """Return normalization stats shaped (freq, 1) so it broadcasts against (freq, time)."""
    if np.isscalar(value):
        return np.full((freq_bins, 1), float(value), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def compute_freq_norm_stats(train_subset: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Normalization mean/std from the training spectrograms (shape N, freq, time).

    'perbin' (legacy): per-frequency-bin stats over samples+time -> shape (freq,). Z-scores each
    bin independently, whitening the cross-band energy ratio that encodes murmur pitch.
    'global': a single scalar mean/std preserving the spectral shape across bins.
    SpectrogramDataset broadcasts a scalar to (freq, 1), so either return type is accepted.
    """
    if mode == "global":
        mean = np.float32(train_subset.mean())
        std = np.float32(train_subset.std() + 1e-6)
        return mean, std
    mean = train_subset.mean(axis=(0, 2)).astype(np.float32)
    std = (train_subset.std(axis=(0, 2)) + 1e-6).astype(np.float32)
    return mean, std


_PITCH_CLASS = {"low": 0, "medium": 1, "high": 2}


def encode_pitch_targets(meta, labels: np.ndarray) -> np.ndarray:
    """Per-sample systolic murmur pitch class (Low=0, Medium=1, High=2) for the aux head.

    Supervised only on Present recordings: samples whose binary label != 1 (Absent location or
    Absent patient) and any missing/unknown pitch get -1, which CrossEntropyLoss(ignore_index=-1)
    drops. `meta` is 1:1 with `labels`/`specs` by row index.
    """
    pitch = meta["systolic_murmur_pitch"].fillna("").astype(str).str.strip().str.lower() \
        if "systolic_murmur_pitch" in meta.columns else None
    classes = np.full(len(labels), -1, dtype=np.int64)
    if pitch is not None:
        classes = pitch.map(_PITCH_CLASS).fillna(-1).to_numpy().astype(np.int64)
    classes[labels.astype(int) != 1] = -1
    return classes


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        specs: np.ndarray,
        labels: np.ndarray,
        mean: float | np.ndarray,
        std: float | np.ndarray,
        sample_weights: np.ndarray | None = None,
        augmenter: object | None = None,
        augmenter_minority_only: bool = False,
        temporal_features: np.ndarray | None = None,
        aux_targets: np.ndarray | None = None,
    ) -> None:
        self.augmenter = augmenter
        self.augmenter_minority_only = bool(augmenter_minority_only)
        freq_bins = int(specs.shape[1])
        mean_arr = _broadcast_stats(mean, freq_bins)
        std_arr = _broadcast_stats(std, freq_bins)
        self.mean = mean_arr
        self.std = std_arr
        if augmenter is not None:
            self.specs = specs.astype(np.float32)
        else:
            self.specs = ((specs - mean_arr[None, :, :]) / std_arr[None, :, :]).astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.sample_weights = None if sample_weights is None else sample_weights.astype(np.float32)
        self.temporal_features = None if temporal_features is None else temporal_features.astype(np.float32)
        self.aux_targets = None if aux_targets is None else aux_targets.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        if self.augmenter is None:
            x_array = self.specs[index]
        else:
            should_augment = not self.augmenter_minority_only or int(self.labels[index]) == 1
            augmented = self.augmenter(self.specs[index]) if should_augment else self.specs[index]  # type: ignore[operator]
            x_array = ((augmented - self.mean) / self.std).astype(np.float32)  # broadcasts (freq,1) over (freq,time)
        x = torch.from_numpy(x_array)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        weight = (
            torch.tensor(self.sample_weights[index], dtype=torch.float32)
            if self.sample_weights is not None
            else torch.tensor(1.0, dtype=torch.float32)
        )
        # Preserve legacy tuple shapes: (x, y) when no weights/temporal/aux; otherwise (x, y, weight,
        # [temporal,] [aux]). The aux target, when present, is always the last element.
        if self.sample_weights is None and self.temporal_features is None and self.aux_targets is None:
            return x, y
        result: list[torch.Tensor] = [x, y, weight]
        if self.temporal_features is not None:
            result.append(torch.from_numpy(self.temporal_features[index]))
        if self.aux_targets is not None:
            result.append(torch.tensor(self.aux_targets[index], dtype=torch.long))
        return tuple(result)
