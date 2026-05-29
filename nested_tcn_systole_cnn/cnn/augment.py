from __future__ import annotations


import argparse
import math
import numpy as np


class LTSRRAugmenter:
    """Local Time-Frequency Spectrum Random Replacement for log-spectrograms."""

    def __init__(self, probability: float, k_segments: int, frequency_ratio: float) -> None:
        self.probability = float(probability)
        self.k_segments = int(k_segments)
        self.frequency_ratio = float(frequency_ratio)

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        if self.probability <= 0.0 or np.random.random() >= self.probability:
            return spec.astype(np.float32, copy=True)
        if spec.ndim != 2:
            raise ValueError("LTSRR expects a 2D spectrogram shaped as frequency x time.")

        freq_bins, frames = spec.shape
        if freq_bins == 0 or frames == 0:
            return spec.astype(np.float32, copy=True)

        output = spec.astype(np.float32, copy=True)
        source = spec.astype(np.float32, copy=False)
        freq_width = max(1, min(freq_bins, int(round(freq_bins * self.frequency_ratio))))

        for segment_indices in np.array_split(np.arange(frames), self.k_segments):
            if segment_indices.size == 0:
                continue
            segment_start = int(segment_indices[0])
            segment_end = int(segment_indices[-1]) + 1
            segment_width = segment_end - segment_start
            if segment_width <= 0:
                continue

            time_width = int(np.random.randint(1, segment_width + 1))
            time_start = int(np.random.randint(segment_start, segment_end - time_width + 1))
            freq_start = int(np.random.randint(0, freq_bins - freq_width + 1))

            local_slice = source[:, segment_start:segment_end]
            local_mean = float(local_slice.mean())
            local_std = float(local_slice.std())
            if local_std < 1e-6:
                local_std = float(source.std() + 1e-6)
            replacement = np.random.normal(
                loc=local_mean,
                scale=local_std,
                size=(freq_width, time_width),
            ).astype(np.float32)
            output[freq_start : freq_start + freq_width, time_start : time_start + time_width] = replacement

        return output


def build_ltsrr_augmenter(args: argparse.Namespace) -> LTSRRAugmenter | None:
    probability = float(getattr(args, "ltsrr_prob", 0.0))
    if probability <= 0.0:
        return None
    return LTSRRAugmenter(
        probability=probability,
        k_segments=int(getattr(args, "ltsrr_k", 4)),
        frequency_ratio=float(getattr(args, "ltsrr_frequency_ratio", 0.25)),
    )


class SpecAugmenter:
    """SpecAugment with frequency masking and time masking on log-spectrograms."""

    def __init__(
        self,
        time_prob: float,
        time_max_width: int,
        time_num_masks: int,
        freq_prob: float,
        freq_max_width: int,
        freq_num_masks: int,
    ) -> None:
        self.time_prob = float(time_prob)
        self.time_max_width = max(0, int(time_max_width))
        self.time_num_masks = max(0, int(time_num_masks))
        self.freq_prob = float(freq_prob)
        self.freq_max_width = max(0, int(freq_max_width))
        self.freq_num_masks = max(0, int(freq_num_masks))

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        if spec.size == 0:
            return spec
        out = spec.copy()
        freq_bins, time_frames = out.shape
        fill = float(out.mean()) if out.size else 0.0
        if self.time_prob > 0.0 and self.time_max_width > 0:
            for _ in range(self.time_num_masks):
                if np.random.rand() > self.time_prob:
                    continue
                width = int(np.random.randint(1, self.time_max_width + 1))
                width = min(width, time_frames)
                start = int(np.random.randint(0, max(1, time_frames - width + 1)))
                out[:, start : start + width] = fill
        if self.freq_prob > 0.0 and self.freq_max_width > 0:
            for _ in range(self.freq_num_masks):
                if np.random.rand() > self.freq_prob:
                    continue
                width = int(np.random.randint(1, self.freq_max_width + 1))
                width = min(width, freq_bins)
                start = int(np.random.randint(0, max(1, freq_bins - width + 1)))
                out[start : start + width, :] = fill
        return out


class ComposeAugmenters:
    """Apply a sequence of augmenters in order. None entries are skipped."""

    def __init__(self, augmenters: list[object]) -> None:
        self.augmenters = [aug for aug in augmenters if aug is not None]

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        for aug in self.augmenters:
            spec = aug(spec)
        return spec


def build_specaug_augmenter(args: argparse.Namespace) -> SpecAugmenter | None:
    time_prob = float(getattr(args, "specaug_time_prob", 0.0))
    freq_prob = float(getattr(args, "specaug_freq_prob", 0.0))
    if time_prob <= 0.0 and freq_prob <= 0.0:
        return None
    return SpecAugmenter(
        time_prob=time_prob,
        time_max_width=int(getattr(args, "specaug_time_width", 0)),
        time_num_masks=int(getattr(args, "specaug_time_num_masks", 2)),
        freq_prob=freq_prob,
        freq_max_width=int(getattr(args, "specaug_freq_width", 0)),
        freq_num_masks=int(getattr(args, "specaug_freq_num_masks", 2)),
    )


def build_cnn_augmenter(args: argparse.Namespace) -> object | None:
    augmenters = [build_ltsrr_augmenter(args), build_specaug_augmenter(args)]
    augmenters = [aug for aug in augmenters if aug is not None]
    if not augmenters:
        return None
    if len(augmenters) == 1:
        return augmenters[0]
    return ComposeAugmenters(augmenters)


def augmentation_config_from_args(args: argparse.Namespace) -> dict[str, float | int | bool]:
    return {
        "ltsrr_prob": float(getattr(args, "ltsrr_prob", 0.0)),
        "ltsrr_k": int(getattr(args, "ltsrr_k", 4)),
        "ltsrr_frequency_ratio": float(getattr(args, "ltsrr_frequency_ratio", 0.25)),
        "ltsrr_minority_only": bool(getattr(args, "ltsrr_minority_only", False)),
        "smote_minority_augmentation": bool(getattr(args, "smote_minority_augmentation", False)),
        "smote_k_neighbors": int(getattr(args, "smote_k_neighbors", 5)),
        "smote_target_ratio": float(getattr(args, "smote_target_ratio", 1.0)),
        "specaug_time_prob": float(getattr(args, "specaug_time_prob", 0.0)),
        "specaug_time_width": int(getattr(args, "specaug_time_width", 0)),
        "specaug_time_num_masks": int(getattr(args, "specaug_time_num_masks", 2)),
        "specaug_freq_prob": float(getattr(args, "specaug_freq_prob", 0.0)),
        "specaug_freq_width": int(getattr(args, "specaug_freq_width", 0)),
        "specaug_freq_num_masks": int(getattr(args, "specaug_freq_num_masks", 2)),
        "mixup_alpha": float(getattr(args, "mixup_alpha", 0.0)),
    }


def apply_smote_minority_augmentation(
    specs: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    k_neighbors: int,
    target_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    labels_int = labels.astype(int)
    classes, counts = np.unique(labels_int, return_counts=True)
    info: dict[str, float | int] = {
        "enabled": 1,
        "minority_class": -1,
        "original_minority_count": 0,
        "original_majority_count": 0,
        "synthetic_count": 0,
        "target_ratio": float(target_ratio),
        "k_neighbors": int(k_neighbors),
    }
    if len(classes) != 2:
        return specs, labels, sample_weights, info

    order = np.argsort(counts)
    minority_class = int(classes[order[0]])
    majority_count = int(counts[order[-1]])
    minority_count = int(counts[order[0]])
    target_minority_count = int(math.ceil(majority_count * float(target_ratio)))
    synthetic_count = max(0, target_minority_count - minority_count)
    info.update(
        {
            "minority_class": minority_class,
            "original_minority_count": minority_count,
            "original_majority_count": majority_count,
            "synthetic_count": synthetic_count,
        }
    )
    if synthetic_count == 0 or minority_count < 2:
        return specs, labels, sample_weights, info

    minority_positions = np.flatnonzero(labels_int == minority_class)
    minority_specs = specs[minority_positions].astype(np.float32, copy=False)
    flat = minority_specs.reshape(minority_count, -1)
    norms = np.sum(flat * flat, axis=1, keepdims=True)
    distances = norms + norms.T - 2.0 * (flat @ flat.T)
    np.fill_diagonal(distances, np.inf)
    neighbor_count = min(int(k_neighbors), minority_count - 1)
    nearest = np.argsort(distances, axis=1)[:, :neighbor_count]

    synthetic_specs = np.empty((synthetic_count, *specs.shape[1:]), dtype=np.float32)
    synthetic_labels = np.full(synthetic_count, minority_class, dtype=np.float32)
    synthetic_weights = np.empty(synthetic_count, dtype=np.float32)
    for synthetic_idx in range(synthetic_count):
        anchor_local = int(np.random.randint(0, minority_count))
        neighbor_local = int(np.random.choice(nearest[anchor_local]))
        ratio = float(np.random.random())
        anchor = minority_specs[anchor_local]
        neighbor = minority_specs[neighbor_local]
        synthetic_specs[synthetic_idx] = anchor + ratio * (neighbor - anchor)
        anchor_weight = float(sample_weights[minority_positions[anchor_local]])
        neighbor_weight = float(sample_weights[minority_positions[neighbor_local]])
        synthetic_weights[synthetic_idx] = anchor_weight + ratio * (neighbor_weight - anchor_weight)

    return (
        np.concatenate([specs, synthetic_specs], axis=0).astype(np.float32),
        np.concatenate([labels.astype(np.float32), synthetic_labels], axis=0).astype(np.float32),
        np.concatenate([sample_weights.astype(np.float32), synthetic_weights], axis=0).astype(np.float32),
        info,
    )
