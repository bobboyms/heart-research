"""Experiment-level scoring helpers."""

from __future__ import annotations

from collections.abc import Mapping


SCORE_KEYS = ("sensitivity", "specificity", "precision", "f1")
DEFAULT_SCORE_WEIGHTS = {key: 1.0 for key in SCORE_KEYS}


def parse_score_weights(value: str | None) -> dict[str, float]:
    if not value:
        return DEFAULT_SCORE_WEIGHTS.copy()
    weights = DEFAULT_SCORE_WEIGHTS.copy()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("--score-weights must use key=value pairs separated by commas.")
        key, raw_weight = [item.strip() for item in part.split("=", 1)]
        if key not in SCORE_KEYS:
            supported = ", ".join(SCORE_KEYS)
            raise ValueError(f"Unsupported score weight '{key}'. Supported keys: {supported}.")
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError("--score-weights values must be non-negative.")
        weights[key] = weight
    if sum(weights.values()) <= 0.0:
        raise ValueError("At least one --score-weights value must be greater than 0.")
    return weights


def weighted_mean_score(metrics: Mapping[str, object], weights: Mapping[str, float]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for key in SCORE_KEYS:
        weight = float(weights.get(key, 0.0))
        if weight <= 0.0:
            continue
        weighted_sum += float(metrics.get(key, 0.0)) * weight
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight

