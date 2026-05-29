from __future__ import annotations


from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn


from nested_tcn_systole_cnn import tcn


from .config import LABEL_DIASTOLE, LABEL_SYSTOLE, LOCATION_ORDER, PHASE_LABELS, StftConfig


def read_segments(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["start_time", "end_time", "label"])
    return pd.read_csv(
        path,
        sep="\t",
        names=["start_time", "end_time", "label"],
        dtype={"start_time": float, "end_time": float, "label": int},
    )


def write_segments(path: Path, segments: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if segments.empty:
        path.write_text("", encoding="utf-8")
        return
    segments[["start_time", "end_time", "label"]].to_csv(
        path,
        sep="\t",
        header=False,
        index=False,
        float_format="%.6f",
        )


def format_float_key(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def selected_phase_labels(stft_cfg: StftConfig) -> tuple[int, ...]:
    try:
        return PHASE_LABELS[stft_cfg.cnn_phase_mode]
    except KeyError as exc:
        raise ValueError("--cnn-phase-mode must be one of: systole, diastole, both.") from exc


def phase_name_for_label(label: int) -> str:
    if label == LABEL_SYSTOLE:
        return "systole"
    if label == LABEL_DIASTOLE:
        return "diastole"
    return f"label_{label}"


def phase_label_counts(segments: pd.DataFrame) -> dict[int, int]:
    if segments.empty:
        return {LABEL_SYSTOLE: 0, LABEL_DIASTOLE: 0}
    return {
        LABEL_SYSTOLE: int((segments["label"] == LABEL_SYSTOLE).sum()),
        LABEL_DIASTOLE: int((segments["label"] == LABEL_DIASTOLE).sum()),
    }


def predicted_tsv_path(predicted_tsv_dir: Path, wav_path: Path, stft_cfg: StftConfig) -> Path:
    # phase-contrast retains diastole labels in the prediction, so its TSVs must not be shared
    # with a plain systole-only run's cache.
    pc_key = "_pc" if bool(getattr(stft_cfg, "phase_contrast", False)) else ""
    if stft_cfg.systole_threshold is None and stft_cfg.systole_margin_ms == 0:
        if stft_cfg.cnn_phase_mode != "systole":
            return predicted_tsv_dir / f"{wav_path.stem}.predicted_phase{stft_cfg.cnn_phase_mode}{pc_key}.tsv"
        return predicted_tsv_dir / f"{wav_path.stem}.predicted{pc_key}.tsv"
    threshold_key = "argmax" if stft_cfg.systole_threshold is None else f"thr{format_float_key(stft_cfg.systole_threshold)}"
    margin_key = f"margin{format_float_key(stft_cfg.systole_margin_ms)}ms"
    phase_key = "" if stft_cfg.cnn_phase_mode == "systole" else f"_phase{stft_cfg.cnn_phase_mode}"
    return predicted_tsv_dir / f"{wav_path.stem}.predicted{phase_key}{pc_key}_{threshold_key}_{margin_key}.tsv"


def systole_probability_index(cfg: object) -> int:
    return 1 if getattr(cfg, "target_mode", "cardiac-phase") == "systole-binary" else LABEL_SYSTOLE


@torch.no_grad()
def predict_tcn_segments(
    wav_path: Path,
    model: nn.Module,
    normalizer: object,
    cfg: object,
    device: torch.device,
    systole_threshold: float | None,
    cnn_phase_mode: str,
    phase_contrast: bool = False,
) -> pd.DataFrame:
    sample_rate, audio = tcn.read_audio(wav_path)
    features, centers_s, starts_s, ends_s = tcn.extract_frame_features(audio, sample_rate, cfg)
    normalized = normalizer.apply(features)
    x = torch.from_numpy(normalized.T.copy()).unsqueeze(0).to(device)
    model.eval()
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).permute(1, 0).detach().cpu().numpy()
    if systole_threshold is None:
        pred = probs.argmax(axis=1).astype(np.int64)
    elif cnn_phase_mode == "systole":
        systole_index = systole_probability_index(cfg)
        pred = np.zeros(probs.shape[0], dtype=np.int64)
        pred[probs[:, systole_index] >= systole_threshold] = systole_index
        if phase_contrast:
            # Keep diastole frames (by argmax) so the phase-contrast input has a within-recording
            # reference. Does not alter the thresholded systole selection used as the main signal.
            full = probs.argmax(axis=1)
            pred[(full == LABEL_DIASTOLE) & (pred != systole_index)] = LABEL_DIASTOLE
    else:
        systole_index = systole_probability_index(cfg)
        pred = probs.argmax(axis=1).astype(np.int64)
        pred[(pred == systole_index) & (probs[:, systole_index] < systole_threshold)] = 0
    pred[ends_s <= starts_s] = tcn.IGNORE_INDEX
    valid = pred != tcn.IGNORE_INDEX
    pred[valid] = tcn.postprocess_prediction(pred[valid], median_filter_frames=5, min_segment_frames=3)
    segments = tcn.prediction_segments(pred, centers_s, starts_s, ends_s, probs, getattr(cfg, "target_mode", "cardiac-phase"))
    if segments.empty:
        return pd.DataFrame(columns=["start_time", "end_time", "label"])
    return segments[["start_time", "end_time", "label"]].copy()


def get_segments(
    wav_path: Path,
    predicted_tsv_dir: Path,
    overwrite_predictions: bool,
    stft_cfg: StftConfig,
    model: nn.Module | None,
    normalizer: object,
    cfg: object,
    device: torch.device,
) -> pd.DataFrame:
    if getattr(stft_cfg, "use_ground_truth_segments", False):
        return read_segments(wav_path.with_suffix(".tsv"))
    path = predicted_tsv_path(predicted_tsv_dir, wav_path, stft_cfg)
    if path.exists() and not overwrite_predictions:
        return read_segments(path)
    segments = predict_tcn_segments(
        wav_path,
        model,
        normalizer,
        cfg,
        device,
        stft_cfg.systole_threshold,
        stft_cfg.cnn_phase_mode,
        bool(getattr(stft_cfg, "phase_contrast", False)),
    )
    write_segments(path, segments)
    return segments


def extract_phase_audio(
    audio: np.ndarray,
    sample_rate: int,
    segments: pd.DataFrame,
    labels: tuple[int, ...],
    margin_ms: float,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_samples = len(audio)
    margin_seconds = max(0.0, margin_ms) / 1000.0
    selected = segments.loc[segments["label"].isin(labels)].sort_values("start_time")
    for row in selected.itertuples(index=False):
        start_time = float(row.start_time) - margin_seconds
        end_time = float(row.end_time) + margin_seconds
        start = max(0, min(n_samples, int(round(start_time * sample_rate))))
        end = max(0, min(n_samples, int(round(end_time * sample_rate))))
        if end > start:
            chunks.append(audio[start:end])
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def phase_seconds_by_label(
    audio: np.ndarray,
    sample_rate: int,
    segments: pd.DataFrame,
    labels: tuple[int, ...],
    margin_ms: float,
) -> dict[int, float]:
    return {
        label: len(extract_phase_audio(audio, sample_rate, segments, (label,), margin_ms)) / float(sample_rate)
        if sample_rate
        else 0.0
        for label in labels
    }


def parse_murmur_locations(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    locations: set[str] = set()
    for part in str(value).replace(",", "+").split("+"):
        location = part.strip()
        if location in LOCATION_ORDER:
            locations.add(location)
    return locations
