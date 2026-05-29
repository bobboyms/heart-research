#!/usr/bin/env python3
"""Cross TCN systole errors with real segment duration and boundary distance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


LABEL_NAMES = {
    0: "other",
    1: "s1",
    2: "systole",
    3: "s2",
    4: "diastole",
}
SPLIT_ORDER = ["val", "test"]
DURATION_BINS_MS = [0, 80, 100, 120, 150, 200, 300, np.inf]
DISTANCE_BINS_MS = [0, 10, 20, 30, 40, 60, 80, 120, np.inf]


def parse_args() -> argparse.Namespace:
    base = Path("modeling/Grupo H Nested TCN CNN systole")
    default_tcn_dir = base / "outputs_nested" / "fold_1" / "tcn"
    default_output_dir = (
        Path("analise_exploratoria_tcn")
        / "outputs"
        / "fold_1"
        / "systole_errors_by_duration_boundary"
    )
    default_train_script = (
        Path("modeling/Grupo E TCN segmentacao frame a frame")
        / "train_tcn_frame_segmenter.py"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained fold_1 TCN on cached recordings and relate "
            "systole errors to real TSV segment duration and boundary distance."
        )
    )
    parser.add_argument("--tcn-dir", type=Path, default=default_tcn_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--train-script", type=Path, default=default_train_script)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLIT_ORDER,
        help="Manifest splits to analyze. Defaults to val test.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_tcn_module(script_path: Path) -> ModuleType:
    if not script_path.exists():
        raise FileNotFoundError(f"Missing TCN training script: {script_path}")
    spec = importlib.util.spec_from_file_location("tcn_frame_segmenter", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_manifest(tcn_dir: Path) -> dict[str, list[dict]]:
    manifest_path = tcn_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def build_cache_index(tcn_dir: Path) -> dict[str, Path]:
    cache_dir = tcn_dir / "cache"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing TCN cache directory: {cache_dir}")
    index = {path.stem: path for path in cache_dir.rglob("*.npz")}
    if not index:
        raise FileNotFoundError(f"No .npz files found under {cache_dir}")
    return index


def read_systole_segments(item: dict) -> pd.DataFrame:
    tsv_path = Path(item["tsv_path"])
    table = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["start_s", "end_s", "class_id"],
    )
    table["class_id"] = table["class_id"].astype(int)
    table = table[(table["class_id"] == 2) & (table["end_s"] > table["start_s"])].copy()
    table["segment_index"] = np.arange(len(table))
    table["duration_ms"] = (table["end_s"] - table["start_s"]) * 1000.0
    return table


def load_recording_arrays(
    npz_path: Path,
    normalizer: object,
) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        features = data["x"].astype(np.float32)
        labels = data["y"].astype(np.int64)
        starts_s = data["frame_starts_s"].astype(np.float64)
        ends_s = data["frame_ends_s"].astype(np.float64)

    normalized = normalizer.apply(features)
    return {
        "x": normalized.T.copy(),
        "true": labels,
        "starts_s": starts_s,
        "ends_s": ends_s,
        "centers_s": (starts_s + ends_s) / 2.0,
    }


def predict_batch(
    batch_arrays: list[dict[str, np.ndarray]],
    model: torch.nn.Module,
    device: torch.device,
    postprocess_prediction: object,
    median_filter_frames: int,
    min_segment_frames: int,
) -> list[dict[str, np.ndarray]]:
    channels = int(batch_arrays[0]["x"].shape[0])
    lengths = [int(arr["true"].shape[0]) for arr in batch_arrays]
    max_len = max(lengths)
    x = np.zeros((len(batch_arrays), channels, max_len), dtype=np.float32)
    for idx, arr in enumerate(batch_arrays):
        length = lengths[idx]
        x[idx, :, :length] = arr["x"]
    x_tensor = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits = model(x_tensor)
        pred_batch = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

    predictions: list[dict[str, np.ndarray]] = []
    for idx, arr in enumerate(batch_arrays):
        length = lengths[idx]
        pred = pred_batch[idx, :length]
        pred = postprocess_prediction(pred, median_filter_frames, min_segment_frames)
        predictions.append({**arr, "pred": pred})
    return predictions


def classify_segment(systole_fraction: float, other_fraction: float) -> str:
    if systole_fraction >= 0.5:
        return "detected_majority"
    if other_fraction >= 0.5:
        return "missed_as_other_majority"
    return "missed_as_other_classes"


def duration_bin_label(values_ms: pd.Series) -> pd.Series:
    return pd.cut(
        values_ms,
        bins=DURATION_BINS_MS,
        right=False,
        labels=["<80", "80-99", "100-119", "120-149", "150-199", "200-299", ">=300"],
    )


def distance_bin_label(values_ms: pd.Series) -> pd.Series:
    return pd.cut(
        values_ms,
        bins=DISTANCE_BINS_MS,
        right=False,
        labels=["0-9", "10-19", "20-29", "30-39", "40-59", "60-79", "80-119", ">=120"],
    )


def analyze_recording(
    item: dict,
    split: str,
    prediction: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    segments = read_systole_segments(item)
    true_labels = prediction["true"]
    pred_labels = prediction["pred"]
    starts_s = prediction["starts_s"]
    ends_s = prediction["ends_s"]
    centers_s = prediction["centers_s"]

    segment_rows: list[dict] = []
    frame_rows: list[dict] = []
    base = {
        "split": split,
        "recording_id": item["recording_id"],
        "patient_id": str(item["patient_id"]),
        "location": item.get("location", ""),
        "murmur": item.get("murmur", ""),
        "outcome": item.get("outcome", ""),
    }

    for _, segment in segments.iterrows():
        segment_uid = f"{item['recording_id']}:{int(segment['segment_index'])}"
        start_s = float(segment["start_s"])
        end_s = float(segment["end_s"])
        duration_ms = float(segment["duration_ms"])
        overlap_mask = (
            (true_labels == 2)
            & (ends_s > start_s)
            & (starts_s < end_s)
        )
        frame_indices = np.flatnonzero(overlap_mask)
        if len(frame_indices) == 0:
            continue

        segment_preds = pred_labels[frame_indices]
        pred_counts = np.bincount(segment_preds, minlength=len(LABEL_NAMES))
        n_frames = int(len(frame_indices))
        systole_frames = int(pred_counts[2])
        other_frames = int(pred_counts[0])
        systole_fraction = systole_frames / n_frames
        other_fraction = other_frames / n_frames
        majority_pred = int(np.argmax(pred_counts))
        category = classify_segment(systole_fraction, other_fraction)

        row = {
            **base,
            "segment_uid": segment_uid,
            "segment_index": int(segment["segment_index"]),
            "start_s": start_s,
            "end_s": end_s,
            "duration_ms": duration_ms,
            "true_systole_frames": n_frames,
            "pred_systole_frames": systole_frames,
            "pred_other_frames": other_frames,
            "pred_s1_frames": int(pred_counts[1]),
            "pred_s2_frames": int(pred_counts[3]),
            "pred_diastole_frames": int(pred_counts[4]),
            "systole_frame_recall": systole_fraction,
            "other_frame_rate": other_fraction,
            "majority_pred_id": majority_pred,
            "majority_pred_name": LABEL_NAMES.get(majority_pred, str(majority_pred)),
            "detection_category": category,
        }
        segment_rows.append(row)

        segment_centers = centers_s[frame_indices]
        raw_distance_ms = np.minimum(
            segment_centers - start_s,
            end_s - segment_centers,
        ) * 1000.0
        distance_ms = np.maximum(raw_distance_ms, 0.0)
        for frame_index, pred_label, center_s, distance_to_boundary_ms in zip(
            frame_indices,
            segment_preds,
            segment_centers,
            distance_ms,
            strict=True,
        ):
            frame_rows.append(
                {
                    **base,
                    "segment_uid": segment_uid,
                    "segment_index": int(segment["segment_index"]),
                    "duration_ms": duration_ms,
                    "frame_index": int(frame_index),
                    "frame_center_s": float(center_s),
                    "distance_to_boundary_ms": float(distance_to_boundary_ms),
                    "pred_id": int(pred_label),
                    "pred_name": LABEL_NAMES.get(int(pred_label), str(pred_label)),
                    "is_pred_systole": int(pred_label == 2),
                    "is_systole_to_other": int(pred_label == 0),
                    "is_systole_to_non_systole": int(pred_label != 2),
                }
            )
    return segment_rows, frame_rows


def aggregate_segment_detection(segments: pd.DataFrame) -> pd.DataFrame:
    return (
        segments.groupby(["split", "detection_category"], as_index=False)
        .agg(
            segments=("duration_ms", "size"),
            recordings=("recording_id", "nunique"),
            patients=("patient_id", "nunique"),
            mean_duration_ms=("duration_ms", "mean"),
            p5_duration_ms=("duration_ms", lambda s: s.quantile(0.05)),
            p25_duration_ms=("duration_ms", lambda s: s.quantile(0.25)),
            median_duration_ms=("duration_ms", "median"),
            p75_duration_ms=("duration_ms", lambda s: s.quantile(0.75)),
            p95_duration_ms=("duration_ms", lambda s: s.quantile(0.95)),
            mean_recall=("systole_frame_recall", "mean"),
            mean_other_rate=("other_frame_rate", "mean"),
        )
    )


def aggregate_frames_by_duration(frames: pd.DataFrame) -> pd.DataFrame:
    data = frames.copy()
    data["duration_bin_ms"] = duration_bin_label(data["duration_ms"])
    grouped = (
        data.groupby(["split", "duration_bin_ms"], as_index=False, observed=True)
        .agg(
            true_systole_frames=("frame_index", "size"),
            pred_systole_frames=("is_pred_systole", "sum"),
            systole_to_other_frames=("is_systole_to_other", "sum"),
            systole_to_non_systole_frames=("is_systole_to_non_systole", "sum"),
            segments=("segment_uid", "nunique"),
            recordings=("recording_id", "nunique"),
            patients=("patient_id", "nunique"),
        )
    )
    grouped["frame_recall"] = grouped["pred_systole_frames"] / grouped["true_systole_frames"]
    grouped["other_error_rate"] = (
        grouped["systole_to_other_frames"] / grouped["true_systole_frames"]
    )
    grouped["non_systole_error_rate"] = (
        grouped["systole_to_non_systole_frames"] / grouped["true_systole_frames"]
    )
    return grouped


def aggregate_frames_by_distance(frames: pd.DataFrame) -> pd.DataFrame:
    data = frames.copy()
    data["distance_bin_ms"] = distance_bin_label(data["distance_to_boundary_ms"])
    grouped = (
        data.groupby(["split", "distance_bin_ms"], as_index=False, observed=True)
        .agg(
            true_systole_frames=("frame_index", "size"),
            pred_systole_frames=("is_pred_systole", "sum"),
            systole_to_other_frames=("is_systole_to_other", "sum"),
            systole_to_non_systole_frames=("is_systole_to_non_systole", "sum"),
            segments=("segment_uid", "nunique"),
            recordings=("recording_id", "nunique"),
            patients=("patient_id", "nunique"),
        )
    )
    grouped["frame_recall"] = grouped["pred_systole_frames"] / grouped["true_systole_frames"]
    grouped["other_error_rate"] = (
        grouped["systole_to_other_frames"] / grouped["true_systole_frames"]
    )
    grouped["non_systole_error_rate"] = (
        grouped["systole_to_non_systole_frames"] / grouped["true_systole_frames"]
    )
    return grouped


def aggregate_frames_by_location(frames: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frames.groupby(["split", "location"], as_index=False, observed=True)
        .agg(
            true_systole_frames=("frame_index", "size"),
            pred_systole_frames=("is_pred_systole", "sum"),
            systole_to_other_frames=("is_systole_to_other", "sum"),
            systole_to_non_systole_frames=("is_systole_to_non_systole", "sum"),
            segments=("segment_uid", "nunique"),
            recordings=("recording_id", "nunique"),
            patients=("patient_id", "nunique"),
        )
    )
    grouped["frame_recall"] = grouped["pred_systole_frames"] / grouped["true_systole_frames"]
    grouped["other_error_rate"] = (
        grouped["systole_to_other_frames"] / grouped["true_systole_frames"]
    )
    grouped["non_systole_error_rate"] = (
        grouped["systole_to_non_systole_frames"] / grouped["true_systole_frames"]
    )
    return grouped


def recompute_confusion(frames_by_recording: list[dict]) -> pd.DataFrame:
    rows = []
    for item in frames_by_recording:
        split = item["split"]
        true_labels = item["true"]
        pred_labels = item["pred"]
        matrix = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=np.int64)
        for true_id, pred_id in zip(true_labels, pred_labels, strict=True):
            if true_id in LABEL_NAMES and pred_id in LABEL_NAMES:
                matrix[int(true_id), int(pred_id)] += 1
        for true_id, true_name in LABEL_NAMES.items():
            for pred_id, pred_name in LABEL_NAMES.items():
                rows.append(
                    {
                        "split": split,
                        "true_id": true_id,
                        "true_name": true_name,
                        "pred_id": pred_id,
                        "pred_name": pred_name,
                        "frames": int(matrix[true_id, pred_id]),
                    }
                )
    df = pd.DataFrame(rows)
    return (
        df.groupby(["split", "true_id", "true_name", "pred_id", "pred_name"], as_index=False)
        ["frames"]
        .sum()
    )


def plot_segment_duration_by_category(summary: pd.DataFrame, output_dir: Path) -> None:
    categories = [
        "detected_majority",
        "missed_as_other_majority",
        "missed_as_other_classes",
    ]
    for split in SPLIT_ORDER:
        subset = summary[summary["split"] == split]
        if subset.empty:
            continue
        ordered = subset.set_index("detection_category").reindex(categories).dropna(how="all")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(ordered.index, ordered["median_duration_ms"])
        ax.set_title(f"Mediana da duracao da sistole por categoria - {split}")
        ax.set_xlabel("Categoria")
        ax.set_ylabel("Duracao mediana (ms)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{split}_segment_median_duration_by_detection.png", dpi=180)
        plt.close()


def plot_error_rates(table: pd.DataFrame, x_col: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for split in SPLIT_ORDER:
        subset = table[table["split"] == split]
        if subset.empty:
            continue
        ax.plot(
            subset[x_col].astype(str),
            subset["other_error_rate"] * 100,
            marker="o",
            label=f"{split}: systole -> other",
        )
        ax.plot(
            subset[x_col].astype(str),
            subset["frame_recall"] * 100,
            marker="s",
            linestyle="--",
            label=f"{split}: recall systole",
        )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("% dos frames reais de sistole")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def fmt_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df.loc[:, columns].copy()
    for col in table.columns:
        if col.endswith("_rate") or col == "frame_recall" or col == "mean_recall" or col == "mean_other_rate":
            table[col] = (table[col] * 100).round(2)
        elif col.endswith("_ms"):
            table[col] = table[col].round(1)
    return table.to_markdown(index=False)


def write_summary(
    output_dir: Path,
    segment_summary: pd.DataFrame,
    by_duration: pd.DataFrame,
    by_distance: pd.DataFrame,
    by_location: pd.DataFrame,
    segments: pd.DataFrame,
) -> None:
    worst_segments = segments.sort_values(
        ["other_frame_rate", "true_systole_frames", "duration_ms"],
        ascending=[False, False, True],
    ).head(25)
    lines = [
        "# Analise exploratoria TCN - erros de sistole por duracao e fronteira",
        "",
        "Esta analise roda o `best_model.pt` do TCN do `fold_1` nos caches de `val/test`.",
        "Cada segmento real de sistole do `.tsv` e cruzado com os frames verdadeiros de sistole e com a predicao pos-processada do TCN.",
        "",
        "Categorias de segmento:",
        "",
        "- `detected_majority`: pelo menos 50% dos frames do segmento foram preditos como `systole`.",
        "- `missed_as_other_majority`: pelo menos 50% dos frames do segmento foram preditos como `other`.",
        "- `missed_as_other_classes`: maioria dos frames virou outra classe cardiaca, nao `systole` nem `other`.",
        "",
        "## Duracao dos segmentos por categoria",
        "",
        fmt_table(
            segment_summary,
            [
                "split",
                "detection_category",
                "segments",
                "recordings",
                "patients",
                "mean_duration_ms",
                "p25_duration_ms",
                "median_duration_ms",
                "p75_duration_ms",
                "mean_recall",
                "mean_other_rate",
            ],
        ),
        "",
        "## Erro por faixa de duracao do segmento",
        "",
        fmt_table(
            by_duration,
            [
                "split",
                "duration_bin_ms",
                "true_systole_frames",
                "segments",
                "frame_recall",
                "other_error_rate",
                "non_systole_error_rate",
            ],
        ),
        "",
        "## Erro por distancia ate a borda da sistole",
        "",
        fmt_table(
            by_distance,
            [
                "split",
                "distance_bin_ms",
                "true_systole_frames",
                "segments",
                "frame_recall",
                "other_error_rate",
                "non_systole_error_rate",
            ],
        ),
        "",
        "## Erro por local",
        "",
        fmt_table(
            by_location,
            [
                "split",
                "location",
                "true_systole_frames",
                "segments",
                "recordings",
                "frame_recall",
                "other_error_rate",
                "non_systole_error_rate",
            ],
        ),
        "",
        "## Segmentos com maior taxa `systole -> other`",
        "",
        fmt_table(
            worst_segments,
            [
                "split",
                "location",
                "recording_id",
                "patient_id",
                "murmur",
                "segment_index",
                "duration_ms",
                "true_systole_frames",
                "systole_frame_recall",
                "other_frame_rate",
                "detection_category",
            ],
        ),
        "",
        "## Arquivos gerados",
        "",
        "- `systole_segment_error_profile.csv`",
        "- `systole_frame_error_profile.csv`",
        "- `segment_detection_duration_summary.csv`",
        "- `systole_error_by_duration_bin.csv`",
        "- `systole_error_by_boundary_distance_bin.csv`",
        "- `systole_error_by_location.csv`",
        "- `recomputed_frame_confusion.csv`",
        "- `*_segment_median_duration_by_detection.png`",
        "- `systole_error_by_duration_bin.png`",
        "- `systole_error_by_boundary_distance_bin.png`",
        "",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_tcn_module(args.train_script)
    device = torch.device(args.device)
    model, normalizer, _cfg, checkpoint = module.load_checkpoint_for_eval(
        args.tcn_dir / "best_model.pt",
        device,
    )
    model.eval()

    checkpoint_args = checkpoint.get("args", {})
    median_filter_frames = int(checkpoint_args.get("median_filter_frames", 5))
    min_segment_frames = int(checkpoint_args.get("min_segment_frames", 3))
    batch_size = int(checkpoint_args.get("batch_size", 42))

    manifest = load_manifest(args.tcn_dir)
    cache_index = build_cache_index(args.tcn_dir)

    segment_rows: list[dict] = []
    frame_rows: list[dict] = []
    confusion_inputs: list[dict] = []
    missing: list[str] = []
    for split in args.splits:
        split_items = manifest.get(split, [])
        for batch_start in range(0, len(split_items), batch_size):
            batch_items = split_items[batch_start : batch_start + batch_size]
            batch_arrays: list[dict[str, np.ndarray]] = []
            kept_items: list[dict] = []
            for item in batch_items:
                recording_id = item["recording_id"]
                npz_path = cache_index.get(recording_id)
                if npz_path is None:
                    missing.append(recording_id)
                    continue
                batch_arrays.append(load_recording_arrays(npz_path, normalizer))
                kept_items.append(item)
            if not batch_arrays:
                continue
            predictions = predict_batch(
                batch_arrays,
                model,
                device,
                module.postprocess_prediction,
                median_filter_frames,
                min_segment_frames,
            )
            for item, prediction in zip(kept_items, predictions, strict=True):
                confusion_inputs.append(
                    {
                        "split": split,
                        "true": prediction["true"],
                        "pred": prediction["pred"],
                    }
                )
                rec_segment_rows, rec_frame_rows = analyze_recording(item, split, prediction)
                segment_rows.extend(rec_segment_rows)
                frame_rows.extend(rec_frame_rows)
    if missing:
        raise FileNotFoundError(f"Missing cache files: {missing[:10]}")

    segments = pd.DataFrame(segment_rows)
    frames = pd.DataFrame(frame_rows)
    if segments.empty or frames.empty:
        raise ValueError("No systole segment/frame rows were produced")

    segment_summary = aggregate_segment_detection(segments)
    by_duration = aggregate_frames_by_duration(frames)
    by_distance = aggregate_frames_by_distance(frames)
    by_location = aggregate_frames_by_location(frames)
    confusion = recompute_confusion(confusion_inputs)

    segments.to_csv(output_dir / "systole_segment_error_profile.csv", index=False)
    frames.to_csv(output_dir / "systole_frame_error_profile.csv", index=False)
    segment_summary.to_csv(output_dir / "segment_detection_duration_summary.csv", index=False)
    by_duration.to_csv(output_dir / "systole_error_by_duration_bin.csv", index=False)
    by_distance.to_csv(output_dir / "systole_error_by_boundary_distance_bin.csv", index=False)
    by_location.to_csv(output_dir / "systole_error_by_location.csv", index=False)
    confusion.to_csv(output_dir / "recomputed_frame_confusion.csv", index=False)

    plot_segment_duration_by_category(segment_summary, output_dir)
    plot_error_rates(
        by_duration,
        "duration_bin_ms",
        output_dir / "systole_error_by_duration_bin.png",
        "Erro `systole -> other` e recall por duracao do segmento",
    )
    plot_error_rates(
        by_distance,
        "distance_bin_ms",
        output_dir / "systole_error_by_boundary_distance_bin.png",
        "Erro `systole -> other` e recall por distancia ate a borda",
    )
    write_summary(output_dir, segment_summary, by_duration, by_distance, by_location, segments)
    print(f"Wrote systole error EDA outputs to {output_dir}")


if __name__ == "__main__":
    main()
