from __future__ import annotations


from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


from .augment import TCNSpecAugmenter
from .config import FeatureConfig, IGNORE_INDEX, Normalizer, RecordingItem, label_names_for_cfg
from .features import load_or_create_features


class CirCorFrameDataset(Dataset):
    def __init__(
        self,
        items: list[RecordingItem],
        cfg: FeatureConfig,
        cache_dir: Path,
        overwrite_cache: bool = False,
        normalizer: Normalizer | None = None,
        augmenter: TCNSpecAugmenter | None = None,
    ) -> None:
        self.items = items
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache
        self.normalizer = normalizer
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.items[index]
        features, labels = load_or_create_features(item, self.cfg, self.cache_dir, self.overwrite_cache)
        if self.normalizer is not None:
            features = self.normalizer.apply(features)
        feat_cf = features.T  # (n_mels, frames)
        if self.augmenter is not None:
            feat_cf, labels = self.augmenter(feat_cf, labels)
        return {
            "x": torch.from_numpy(feat_cf.copy()),  # channels, frames
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
        augmenter: TCNSpecAugmenter | None = None,
    ) -> None:
        self.items = items
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache
        self.normalizer = normalizer
        self.augmenter = augmenter
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
        feat_cf = features.T
        if self.augmenter is not None:
            feat_cf, labels = self.augmenter(feat_cf, labels)
        return {
            "x": torch.from_numpy(feat_cf.copy()),
            "y": torch.from_numpy(labels.copy()),
            "recording_id": f"{item.recording_id}:{start}-{end}",
        }


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
    label_names = label_names_for_cfg(dataset.cfg)
    label_counts = np.zeros(len(label_names), dtype=np.int64)

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
        valid_labels = labels[labels != IGNORE_INDEX]
        label_counts += np.bincount(valid_labels, minlength=len(label_names))[: len(label_names)]

    if total_sum is None or total_sumsq is None or total_count == 0:
        raise RuntimeError("Could not compute feature normalization statistics.")

    mean = total_sum / total_count
    var = np.maximum(total_sumsq / total_count - mean**2, 1e-8)
    std = np.sqrt(var)
    normalizer = Normalizer(mean=mean.astype(float).tolist(), std=std.astype(float).tolist())
    return normalizer, label_counts


def count_train_labels(dataset: CirCorFrameDataset, show_progress: bool = True) -> np.ndarray:
    label_names = label_names_for_cfg(dataset.cfg)
    counts = np.zeros(len(label_names), dtype=np.int64)
    iterator = progress_iter(
        dataset.items,
        show_progress,
        desc="Counting train labels",
        unit="rec",
        leave=False,
    )
    for item in iterator:
        _features, labels = load_or_create_features(item, dataset.cfg, dataset.cache_dir, dataset.overwrite_cache)
        valid_labels = labels[labels != IGNORE_INDEX]
        counts += np.bincount(valid_labels, minlength=len(label_names))[: len(label_names)]
    return counts


def make_loaders(
    splits: dict[str, list[RecordingItem]],
    cfg: FeatureConfig,
    cache_dir: Path,
    overwrite_cache: bool,
    normalizer: Normalizer,
    batch_size: int,
    num_workers: int,
    train_window_seconds: float,
    train_window_hop_seconds: float,
    show_progress: bool,
    train_augmenter: TCNSpecAugmenter | None = None,
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
                augmenter=train_augmenter if split_name == "train" else None,
            )
        else:
            dataset = CirCorFrameDataset(
                split_items,
                cfg,
                cache_dir,
                overwrite_cache,
                normalizer,
                augmenter=train_augmenter if split_name == "train" else None,
            )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
    return loaders
