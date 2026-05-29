from __future__ import annotations


import numpy as np


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
