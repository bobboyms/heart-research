from __future__ import annotations


import argparse
import numpy as np


from .config import IGNORE_INDEX


class TCNSpecAugmenter:
    """SpecAugment for TCN features. Time masks also mark the affected labels as IGNORE_INDEX."""

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

    def __call__(self, features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # features expected as (n_mels, frames); labels (frames,)
        if features.size == 0:
            return features, labels
        feat = features.copy()
        lbl = labels.copy()
        n_mels, n_frames = feat.shape
        fill = float(feat.mean()) if feat.size else 0.0
        if self.time_prob > 0.0 and self.time_max_width > 0:
            for _ in range(self.time_num_masks):
                if np.random.rand() > self.time_prob:
                    continue
                width = int(np.random.randint(1, self.time_max_width + 1))
                width = min(width, n_frames)
                start = int(np.random.randint(0, max(1, n_frames - width + 1)))
                feat[:, start : start + width] = fill
                lbl[start : start + width] = IGNORE_INDEX
        if self.freq_prob > 0.0 and self.freq_max_width > 0:
            for _ in range(self.freq_num_masks):
                if np.random.rand() > self.freq_prob:
                    continue
                width = int(np.random.randint(1, self.freq_max_width + 1))
                width = min(width, n_mels)
                start = int(np.random.randint(0, max(1, n_mels - width + 1)))
                feat[start : start + width, :] = fill
        return feat, lbl


def build_tcn_specaug_augmenter(args: argparse.Namespace) -> TCNSpecAugmenter | None:
    time_prob = float(getattr(args, "specaug_time_prob", 0.0))
    freq_prob = float(getattr(args, "specaug_freq_prob", 0.0))
    if time_prob <= 0.0 and freq_prob <= 0.0:
        return None
    return TCNSpecAugmenter(
        time_prob=time_prob,
        time_max_width=int(getattr(args, "specaug_time_width", 0)),
        time_num_masks=int(getattr(args, "specaug_time_num_masks", 2)),
        freq_prob=freq_prob,
        freq_max_width=int(getattr(args, "specaug_freq_width", 0)),
        freq_num_masks=int(getattr(args, "specaug_freq_num_masks", 2)),
    )
