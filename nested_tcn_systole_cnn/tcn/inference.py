from __future__ import annotations


import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn


from .audio import read_audio
from .config import FeatureConfig, IGNORE_INDEX, Normalizer, OTHER_MODES, TARGET_MODES, label_names_for_cfg, label_names_for_target_mode, prediction_output_label, systole_probability_index
from .features import extract_frame_features
from .model import TCNFrameSegmenter
from .postprocess import postprocess_prediction


def load_checkpoint_for_eval(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TCNFrameSegmenter, Normalizer, FeatureConfig, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = FeatureConfig(**checkpoint["feature_config"])
    if cfg.target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported checkpoint target_mode: {cfg.target_mode}")
    if cfg.other_mode not in OTHER_MODES:
        raise ValueError(f"Unsupported checkpoint other_mode: {cfg.other_mode}")
    normalizer = Normalizer(**checkpoint["normalizer"])
    model_config = checkpoint["model_config"]
    model_config.setdefault("causal", True)
    model_config.setdefault("pooling", "none")
    model = TCNFrameSegmenter(**model_config)
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
    label_names = label_names_for_cfg(cfg)
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
            "num_classes": len(label_names),
            "causal": args.causal,
            "pooling": args.pooling,
        },
        "label_names": label_names,
        "args": vars(args),
        "model_state_dict": model.state_dict(),
    }


def prediction_segments(
    labels: np.ndarray,
    centers_s: np.ndarray,
    starts_s: np.ndarray,
    ends_s: np.ndarray,
    probs: np.ndarray,
    target_mode: str = "cardiac-phase",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    label_names = label_names_for_target_mode(target_mode)
    valid = (labels != IGNORE_INDEX) & (ends_s > starts_s)
    boundaries = np.zeros(len(labels) + 1, dtype=np.float32)
    if len(labels):
        boundaries[0] = starts_s[0]
        boundaries[-1] = ends_s[-1]
    if len(labels) > 1:
        boundaries[1:-1] = 0.5 * (centers_s[:-1] + centers_s[1:])

    start = 0
    while start < len(labels):
        if not valid[start] or labels[start] == 0:
            start += 1
            continue
        label = int(labels[start])
        end = start + 1
        while end < len(labels) and valid[end] and int(labels[end]) == label:
            end += 1

        segment_probs = probs[start:end, label] if start < end else np.array([], dtype=np.float32)
        rows.append(
            {
                "start_time": float(boundaries[start]),
                "end_time": float(boundaries[end]),
                "label": prediction_output_label(label, target_mode),
                "label_name": label_names[label],
                "confidence": float(np.mean(segment_probs)) if len(segment_probs) else 0.0,
                "frame_count": int(end - start),
            }
        )
        start = end
    return pd.DataFrame(rows)


def write_prediction_outputs(
    segments: pd.DataFrame,
    frame_probs: pd.DataFrame,
    output_tsv: Path,
    frame_output: Path | None,
) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    if segments.empty:
        output_tsv.write_text("", encoding="utf-8")
    else:
        segments[["start_time", "end_time", "label"]].to_csv(
            output_tsv,
            sep="\t",
            header=False,
            index=False,
            float_format="%.6f",
        )

    confidence_path = output_tsv.with_name(f"{output_tsv.stem}_segments_with_confidence.csv")
    segments.to_csv(confidence_path, index=False)

    if frame_output is not None:
        frame_output.parent.mkdir(parents=True, exist_ok=True)
        frame_probs.to_csv(frame_output, index=False)


@torch.no_grad()
def predict_wav(
    model: nn.Module,
    normalizer: Normalizer,
    cfg: FeatureConfig,
    wav_path: Path,
    output_tsv: Path,
    frame_output: Path | None,
    device: torch.device,
    postprocess: bool,
    median_filter_frames: int,
    min_segment_frames: int,
    systole_threshold: float | None,
) -> None:
    sample_rate, audio = read_audio(wav_path)
    features, centers_s, starts_s, ends_s = extract_frame_features(audio, sample_rate, cfg)
    normalized = normalizer.apply(features)
    x = torch.from_numpy(normalized.T.copy()).unsqueeze(0).to(device)

    model.eval()
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).permute(1, 0).detach().cpu().numpy()
    if systole_threshold is None:
        pred = probs.argmax(axis=1).astype(np.int64)
    else:
        systole_index = systole_probability_index(cfg.target_mode)
        pred = np.zeros(probs.shape[0], dtype=np.int64)
        pred[probs[:, systole_index] >= systole_threshold] = systole_index
    pred[ends_s <= starts_s] = IGNORE_INDEX
    if postprocess:
        valid_mask = pred != IGNORE_INDEX
        smoothed = postprocess_prediction(pred[valid_mask], median_filter_frames, min_segment_frames)
        pred[valid_mask] = smoothed

    label_names = label_names_for_cfg(cfg)
    segments = prediction_segments(pred, centers_s, starts_s, ends_s, probs, cfg.target_mode)
    frame_rows = {
        "frame_index": np.arange(len(pred), dtype=np.int64),
        "start_time": starts_s,
        "end_time": ends_s,
        "pred_label": pred,
        "pred_label_name": [label_names.get(int(label), "ignore") for label in pred],
        "confidence": np.where(pred >= 0, probs[np.arange(len(pred)), np.maximum(pred, 0)], 0.0),
    }
    for label, name in label_names.items():
        frame_rows[f"p_{name}"] = probs[:, label]
    frame_probs = pd.DataFrame(frame_rows)

    write_prediction_outputs(segments, frame_probs, output_tsv, frame_output)
    print(f"Predicted {len(segments)} cardiac phase segments: {output_tsv}")
