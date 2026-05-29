from __future__ import annotations


import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def clip_prob(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob.astype(float), 1e-6, 1.0 - 1e-6)


def logit(prob: np.ndarray) -> np.ndarray:
    clipped = clip_prob(prob)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def fit_platt_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y = y_true.astype(float)
    x = logit(y_prob)
    if len(np.unique(y)) < 2:
        return {"scale": 1.0, "bias": 0.0}

    def objective(params: np.ndarray) -> float:
        log_scale, bias = float(params[0]), float(params[1])
        scale = math.exp(log_scale)
        calibrated = clip_prob(sigmoid(scale * x + bias))
        bce = -np.mean(y * np.log(calibrated) + (1.0 - y) * np.log(1.0 - calibrated))
        penalty = 1e-3 * (log_scale**2 + bias**2)
        return float(bce + penalty)

    result = minimize(objective, x0=np.asarray([0.0, 0.0]), method="Nelder-Mead", options={"maxiter": 1000})
    if not result.success:
        return {"scale": 1.0, "bias": 0.0}
    return {"scale": float(math.exp(result.x[0])), "bias": float(result.x[1])}


def apply_platt_calibrator(y_prob: np.ndarray, calibrator: dict[str, float]) -> np.ndarray:
    return sigmoid(calibrator["scale"] * logit(y_prob) + calibrator["bias"])


def fit_location_aware_calibrator(
    patient_features: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    y = patient_features["target"].to_numpy(dtype=float)
    x_raw = patient_features[feature_columns].to_numpy(dtype=float)
    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0) + 1e-6
    x = (x_raw - mean) / std

    if len(np.unique(y)) < 2:
        bias = float(logit(np.asarray([np.clip(y.mean(), 1e-6, 1 - 1e-6)]))[0])
        return {
            "kind": "location_aware",
            "feature_columns": feature_columns,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "weights": np.zeros(len(feature_columns), dtype=float).tolist(),
            "bias": bias,
        }

    prevalence = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    initial = np.zeros(x.shape[1] + 1, dtype=float)
    initial[-1] = float(logit(np.asarray([prevalence]))[0])

    def objective(params: np.ndarray) -> float:
        weights = params[:-1]
        bias = float(params[-1])
        pred = clip_prob(sigmoid(x @ weights + bias))
        bce = -np.mean(y * np.log(pred) + (1.0 - y) * np.log(1.0 - pred))
        penalty = 1e-2 * float(np.sum(weights**2))
        return float(bce + penalty)

    result = minimize(objective, x0=initial, method="BFGS", options={"maxiter": 1000})
    params = result.x if result.success else initial
    return {
        "kind": "location_aware",
        "feature_columns": feature_columns,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "weights": params[:-1].astype(float).tolist(),
        "bias": float(params[-1]),
    }


def apply_location_aware_calibrator(patient_features: pd.DataFrame, calibrator: dict[str, object]) -> np.ndarray:
    feature_columns = list(calibrator["feature_columns"])
    mean = np.asarray(calibrator["feature_mean"], dtype=float)
    std = np.asarray(calibrator["feature_std"], dtype=float)
    weights = np.asarray(calibrator["weights"], dtype=float)
    bias = float(calibrator["bias"])
    x = (patient_features[feature_columns].to_numpy(dtype=float) - mean) / std
    return sigmoid(x @ weights + bias)
