from __future__ import annotations


import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import get_window


from .audio import read_audio, read_segments
from .config import FeatureConfig, IGNORE_INDEX, LABEL_NAMES, RecordingItem, SYSTOLE_LABEL


def frame_audio(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_len = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    original_len = len(audio)
    original_duration_s = original_len / float(sample_rate) if sample_rate else 0.0

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
    starts_s = starts / float(sample_rate)
    ends_s = (starts + frame_len) / float(sample_rate)
    centers_s = 0.5 * (starts_s + ends_s)
    centers_s = np.minimum(centers_s, original_duration_s)
    ends_s = np.minimum(ends_s, original_duration_s)
    return frames, centers_s.astype(np.float32), starts_s.astype(np.float32), ends_s.astype(np.float32)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames, centers_s, starts_s, ends_s = frame_audio(audio, sample_rate, cfg.frame_ms, cfg.hop_ms)
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

    return features.astype(np.float32), centers_s, starts_s, ends_s


def labels_for_frames(
    centers_s: np.ndarray,
    starts_s: np.ndarray,
    ends_s: np.ndarray,
    segments: pd.DataFrame,
    cfg: FeatureConfig,
) -> np.ndarray:
    labels = np.zeros(len(centers_s), dtype=np.int64)

    valid_frame = ends_s > starts_s
    labels[~valid_frame] = IGNORE_INDEX

    if cfg.label_mode == "center":
        for row in segments.itertuples(index=False):
            label = int(row.label)
            if label not in LABEL_NAMES:
                continue
            mask = valid_frame & (centers_s >= float(row.start_time)) & (centers_s < float(row.end_time))
            labels[mask] = label
    else:
        best_overlap = np.zeros(len(centers_s), dtype=np.float32)
        for row in segments.itertuples(index=False):
            label = int(row.label)
            if label not in LABEL_NAMES:
                continue
            overlap = np.maximum(
                0.0,
                np.minimum(ends_s, float(row.end_time)) - np.maximum(starts_s, float(row.start_time)),
            )
            mask = valid_frame & (overlap > best_overlap)
            labels[mask] = label
            best_overlap[mask] = overlap[mask]

    if cfg.boundary_ignore_ms > 0:
        margin_s = cfg.boundary_ignore_ms / 1000.0
        near_boundary = np.zeros(len(centers_s), dtype=bool)
        for row in segments.itertuples(index=False):
            near_boundary |= np.abs(centers_s - float(row.start_time)) <= margin_s
            near_boundary |= np.abs(centers_s - float(row.end_time)) <= margin_s
        labels[near_boundary & valid_frame] = IGNORE_INDEX
    if cfg.other_mode == "ignore":
        labels[(labels == 0) & valid_frame] = IGNORE_INDEX
    if cfg.target_mode == "systole-binary":
        labels = np.where(labels == IGNORE_INDEX, IGNORE_INDEX, (labels == SYSTOLE_LABEL).astype(np.int64))
    return labels


def cache_path_for_item(cache_dir: Path, item: RecordingItem, cfg: FeatureConfig) -> Path:
    cfg_key = (
        f"fm{cfg.frame_ms:g}_hm{cfg.hop_ms:g}_m{cfg.n_mels}_"
        f"lo{cfg.low_hz:g}_hi{cfg.high_hz:g}_d{int(cfg.add_deltas)}_"
        f"lm{cfg.label_mode}_bi{cfg.boundary_ignore_ms:g}"
        f"_tm{cfg.target_mode}_om{cfg.other_mode}"
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
    features, centers_s, starts_s, ends_s = extract_frame_features(audio, sample_rate, cfg)
    labels = labels_for_frames(centers_s, starts_s, ends_s, read_segments(Path(item.tsv_path)), cfg)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        x=features.astype(np.float32),
        y=labels.astype(np.int64),
        frame_starts_s=starts_s.astype(np.float32),
        frame_ends_s=ends_s.astype(np.float32),
        recording_id=item.recording_id,
        sample_rate=sample_rate,
        feature_config=json.dumps(asdict(cfg)),
    )
    return features, labels
