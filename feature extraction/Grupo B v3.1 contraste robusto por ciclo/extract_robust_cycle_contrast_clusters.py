# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.4",
#   "scipy>=1.12",
#   "tabulate>=0.9",
#   "umap-learn>=0.5.7",
# ]
# ///
"""Robust cycle-level systole-minus-diastole contrast clustering.

This is v3.1 of the unsupervised contrast experiment. Compared with v3, it:

1. Uses robust z-contrast:
       (systole - median(diastole)) / MAD(diastole)
2. Computes descriptors per systolic cycle, then aggregates with mean/max/p90/top3.
3. Produces projections for all features and separated low/mid/high-band views.

Run from the repository root:

    uv run "feature extraction/Grupo B v3.1 contraste robusto por ciclo/extract_robust_cycle_contrast_clusters.py"
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler


LABEL_SYSTOLE = 2
LABEL_DIASTOLE = 4
LOCATIONS = ["AV", "PV", "TV", "MV"]

BANDS_HZ = {
    "low_25_200hz": (25.0, 200.0),
    "mid_200_600hz": (200.0, 600.0),
    "high_600_1000hz": (600.0, 1000.0),
    "very_low_25_80hz": (25.0, 80.0),
    "low_mid_80_200hz": (80.0, 200.0),
    "mid_high_200_400hz": (200.0, 400.0),
    "upper_mid_400_800hz": (400.0, 800.0),
}

FEATURE_VIEWS = {
    "all": None,
    "low": ("low_25_200hz", "very_low_25_80hz", "low_mid_80_200hz"),
    "mid": ("mid_200_600hz", "mid_high_200_400hz", "upper_mid_400_800hz"),
    "high": ("high_600_1000hz", "upper_mid_400_800hz"),
    "profile": ("profile_bin",),
}

META_COLUMNS = {
    "patient_id",
    "recording_id",
    "location",
    "wav_path",
    "murmur",
    "outcome",
    "age",
    "sex",
    "campaign",
    "systolic_murmur_grading",
    "systolic_murmur_pitch",
    "most_audible_location",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="Generate robust cycle-level systole-minus-diastole contrast clusters."
    )
    parser.add_argument("--dataset-dir", type=Path, default=repo_root / "circor-heart-sound-1.0.3")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs")
    parser.add_argument("--include-unknown", action="store_true", help="Include Murmur=Unknown.")
    parser.add_argument("--max-recordings", type=int, default=None, help="Optional quick-test limit.")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--cluster-sweep", type=str, default="2,4,6,8,10")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--target-sample-rate", type=int, default=4000)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--low-hz", type=float, default=25.0)
    parser.add_argument("--high-hz", type=float, default=1000.0)
    parser.add_argument("--profile-bins", type=int, default=24)
    parser.add_argument("--robust-scale-floor", type=float, default=0.03)
    parser.add_argument("--z-clip", type=float, default=12.0)
    return parser.parse_args()


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    original_dtype = audio.dtype
    audio = audio.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        audio = audio / max(abs(info.min), info.max)
    else:
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1.0:
            audio = audio / peak
    audio = audio - float(np.mean(audio))
    return int(sample_rate), audio


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    divisor = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def read_segments(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=["start_time", "end_time", "label"],
        dtype={"start_time": float, "end_time": float, "label": int},
    )


def parse_recording_id(path: Path) -> tuple[str, str]:
    match = re.match(r"(?P<patient>\d+)_(?P<location>[A-Za-z]+)(?:_\d+)?$", path.stem)
    if not match:
        raise ValueError(f"Unexpected recording name: {path.name}")
    return match.group("patient"), match.group("location")


def segment_log_specs(
    audio: np.ndarray,
    sample_rate: int,
    segments: pd.DataFrame,
    label: int,
    n_fft: int,
    hop_length: int,
    low_hz: float,
    high_hz: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    selected = segments.loc[segments["label"] == label].sort_values("start_time")
    specs: list[np.ndarray] = []
    kept_freqs: np.ndarray | None = None
    n_samples = len(audio)
    for row in selected.itertuples(index=False):
        start = max(0, min(n_samples, int(round(float(row.start_time) * sample_rate))))
        end = max(0, min(n_samples, int(round(float(row.end_time) * sample_rate))))
        if end <= start:
            continue
        chunk = audio[start:end].astype(np.float32)
        if len(chunk) < n_fft:
            padded = np.zeros(n_fft, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunk = padded
        freqs, _times, zxx = stft(
            chunk,
            fs=sample_rate,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if kept_freqs is None:
            kept_freqs = freqs[mask].astype(np.float32)
        spec = np.log1p(np.abs(zxx).astype(np.float32))[mask]
        if spec.shape[1] > 0:
            specs.append(spec.astype(np.float32))

    if kept_freqs is None:
        return [], np.zeros(0, dtype=np.float32)
    return specs, kept_freqs


def aggregate_values(prefix: str, values: list[float], out: dict[str, float]) -> None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        for stat in ("mean", "std", "p50", "p90", "max", "top3_mean"):
            out[f"{prefix}_{stat}"] = 0.0
        return
    top = np.sort(arr)[-min(3, arr.size) :]
    out[f"{prefix}_mean"] = float(np.mean(arr))
    out[f"{prefix}_std"] = float(np.std(arr))
    out[f"{prefix}_p50"] = float(np.percentile(arr, 50))
    out[f"{prefix}_p90"] = float(np.percentile(arr, 90))
    out[f"{prefix}_max"] = float(np.max(arr))
    out[f"{prefix}_top3_mean"] = float(np.mean(top))


def cycle_band_metrics(z: np.ndarray, freqs: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {
            "mean": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "positive_mean": 0.0,
            "positive_fraction": 0.0,
            "positive_area": 0.0,
            "frame_p90_positive": 0.0,
            "sustained_fraction": 0.0,
            "peak_pos": 0.0,
            "slope": 0.0,
        }
    band = z[mask]
    positive = np.maximum(band, 0.0)
    frame_energy = positive.mean(axis=0) if positive.ndim == 2 else np.asarray([float(np.mean(positive))])
    if frame_energy.size == 0:
        frame_energy = np.asarray([0.0])
    peak_pos = float(np.argmax(frame_energy) / max(1, frame_energy.size - 1))
    if frame_energy.size >= 2:
        tt = np.linspace(0.0, 1.0, frame_energy.size)
        slope = float(np.polyfit(tt, frame_energy, 1)[0])
    else:
        slope = 0.0
    threshold = float(np.percentile(frame_energy, 75)) if frame_energy.size else 0.0
    return {
        "mean": float(np.mean(band)),
        "p90": float(np.percentile(band, 90)),
        "max": float(np.max(band)),
        "positive_mean": float(np.mean(positive)),
        "positive_fraction": float(np.mean(band > 0.0)),
        "positive_area": float(np.mean(np.maximum(np.mean(band, axis=1), 0.0))),
        "frame_p90_positive": float(np.percentile(frame_energy, 90)),
        "sustained_fraction": float(np.mean(frame_energy >= threshold)) if threshold > 0.0 else 0.0,
        "peak_pos": peak_pos,
        "slope": slope,
    }


def robust_cycle_contrast_features(
    systole_specs: list[np.ndarray],
    diastole_specs: list[np.ndarray],
    freqs: np.ndarray,
    profile_bins: int,
    robust_scale_floor: float,
    z_clip: float,
) -> dict[str, float]:
    diastole = np.concatenate(diastole_specs, axis=1)
    reference = np.median(diastole, axis=1, keepdims=True).astype(np.float32)
    mad = np.median(np.abs(diastole - reference), axis=1, keepdims=True).astype(np.float32)
    scale = 1.4826 * mad
    finite_scale = scale[np.isfinite(scale) & (scale > 0.0)]
    floor = max(float(np.median(finite_scale)) * robust_scale_floor, 1e-3) if finite_scale.size else 1e-3
    scale = np.maximum(scale, floor).astype(np.float32)

    out: dict[str, float] = {
        "systole_cycle_count": float(len(systole_specs)),
        "diastole_cycle_count": float(len(diastole_specs)),
        "diastole_frame_count": float(diastole.shape[1]),
        "robust_scale_floor_used": float(floor),
        "freq_bin_count": float(len(freqs)),
    }

    per_metric: dict[str, list[float]] = {}

    def collect(name: str, value: float) -> None:
        per_metric.setdefault(name, []).append(float(value))

    profile_edges = np.linspace(float(freqs.min()), float(freqs.max()), profile_bins + 1)
    band_masks = {name: (freqs >= low) & (freqs < high) for name, (low, high) in BANDS_HZ.items()}

    for spec in systole_specs:
        z = ((spec - reference) / scale).astype(np.float32)
        z = np.clip(z, -z_clip, z_clip)
        positive = np.maximum(z, 0.0)
        collect("all_mean", float(np.mean(z)))
        collect("all_p90", float(np.percentile(z, 90)))
        collect("all_max", float(np.max(z)))
        collect("all_positive_mean", float(np.mean(positive)))
        collect("all_positive_fraction", float(np.mean(z > 0.0)))

        low_mask = band_masks["low_25_200hz"]
        mid_mask = band_masks["mid_200_600hz"]
        high_mask = band_masks["high_600_1000hz"]
        low_area = float(np.mean(np.maximum(z[low_mask], 0.0))) if np.any(low_mask) else 0.0
        mid_area = float(np.mean(np.maximum(z[mid_mask], 0.0))) if np.any(mid_mask) else 0.0
        high_area = float(np.mean(np.maximum(z[high_mask], 0.0))) if np.any(high_mask) else 0.0
        total_area = low_area + mid_area + high_area + 1e-12
        collect("bandshare_low", low_area / total_area)
        collect("bandshare_mid", mid_area / total_area)
        collect("bandshare_high", high_area / total_area)
        collect("ratio_mid_to_low", mid_area / (low_area + 1e-12))
        collect("ratio_high_to_low", high_area / (low_area + 1e-12))

        for band_name, mask in band_masks.items():
            metrics = cycle_band_metrics(z, freqs, mask)
            for metric_name, value in metrics.items():
                collect(f"{band_name}_{metric_name}", value)

        freq_mean = z.mean(axis=1)
        freq_p90 = np.percentile(z, 90, axis=1)
        freq_positive = positive.mean(axis=1)
        for idx in range(profile_bins):
            if idx == profile_bins - 1:
                mask = (freqs >= profile_edges[idx]) & (freqs <= profile_edges[idx + 1])
            else:
                mask = (freqs >= profile_edges[idx]) & (freqs < profile_edges[idx + 1])
            if not np.any(mask):
                collect(f"profile_bin_{idx:02d}_mean", 0.0)
                collect(f"profile_bin_{idx:02d}_p90", 0.0)
                collect(f"profile_bin_{idx:02d}_positive", 0.0)
            else:
                collect(f"profile_bin_{idx:02d}_mean", float(np.mean(freq_mean[mask])))
                collect(f"profile_bin_{idx:02d}_p90", float(np.mean(freq_p90[mask])))
                collect(f"profile_bin_{idx:02d}_positive", float(np.mean(freq_positive[mask])))

    for metric_name, values in per_metric.items():
        aggregate_values(metric_name, values, out)
    return out


def extract_recording(wav_path: Path, metadata: pd.DataFrame, args: argparse.Namespace) -> dict[str, float | str] | None:
    tsv_path = wav_path.with_suffix(".tsv")
    if not tsv_path.exists():
        return None
    patient_id, location = parse_recording_id(wav_path)
    if location not in LOCATIONS:
        return None
    row = metadata.loc[metadata["Patient ID"].astype(str) == patient_id]
    if row.empty:
        return None

    sample_rate, audio = read_audio(wav_path)
    audio = resample_audio(audio, sample_rate, args.target_sample_rate)
    segments = read_segments(tsv_path)
    systole_specs, freqs = segment_log_specs(
        audio,
        args.target_sample_rate,
        segments,
        LABEL_SYSTOLE,
        args.n_fft,
        args.hop_length,
        args.low_hz,
        args.high_hz,
    )
    diastole_specs, _ = segment_log_specs(
        audio,
        args.target_sample_rate,
        segments,
        LABEL_DIASTOLE,
        args.n_fft,
        args.hop_length,
        args.low_hz,
        args.high_hz,
    )
    if not systole_specs or not diastole_specs or freqs.size == 0:
        return None

    meta = row.iloc[0]
    features: dict[str, float | str] = {
        "patient_id": patient_id,
        "recording_id": wav_path.stem,
        "location": location,
        "wav_path": str(wav_path),
        "murmur": str(meta["Murmur"]),
        "outcome": str(meta["Outcome"]),
        "age": str(meta["Age"]),
        "sex": str(meta["Sex"]),
        "campaign": str(meta["Campaign"]),
        "systolic_murmur_grading": str(meta.get("Systolic murmur grading", "")),
        "systolic_murmur_pitch": str(meta.get("Systolic murmur pitch", "")),
        "most_audible_location": str(meta.get("Most audible location", "")),
    }
    features.update(
        robust_cycle_contrast_features(
            systole_specs,
            diastole_specs,
            freqs,
            args.profile_bins,
            args.robust_scale_floor,
            args.z_clip,
        )
    )
    return features


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def select_view_features(feature_columns: list[str], view: str) -> list[str]:
    needles = FEATURE_VIEWS[view]
    if needles is None:
        return feature_columns
    return [column for column in feature_columns if any(needle in column for needle in needles)]


def fit_projection(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int,
    skip_umap: bool,
    scaler_name: str = "standard",
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = df.copy()
    if len(df) < 3 or not feature_columns:
        result["pca_1"] = 0.0
        result["pca_2"] = 0.0
        result["cluster"] = 0
        return result, {"silhouette_pca": float("nan"), "pca12_variance": float("nan")}

    matrix = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()
    scaled = scaler.fit_transform(matrix)
    n_pca = min(10, scaled.shape[0], scaled.shape[1])
    pca_model = PCA(n_components=n_pca, random_state=42)
    pca = pca_model.fit_transform(scaled)
    k = min(max(1, n_clusters), len(df))
    clusters = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(pca)

    result["pca_1"] = pca[:, 0]
    result["pca_2"] = pca[:, 1] if pca.shape[1] > 1 else 0.0
    result["cluster"] = clusters

    if not skip_umap and len(df) >= 8:
        neighbors = min(20, max(3, len(df) // 20))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=neighbors,
            min_dist=0.12,
            metric="euclidean",
            random_state=42,
        )
        embedding = reducer.fit_transform(scaled)
        result["umap_1"] = embedding[:, 0]
        result["umap_2"] = embedding[:, 1]

    silhouette = float("nan")
    if len(np.unique(clusters)) > 1 and len(df) > k:
        silhouette = float(silhouette_score(pca, clusters))
    metrics = {
        "silhouette_pca": silhouette,
        "pca12_variance": float(np.sum(pca_model.explained_variance_ratio_[:2])),
    }
    return result, metrics


def cluster_metrics(projected: pd.DataFrame) -> dict[str, float | int]:
    cluster_stats = projected.groupby("cluster")["murmur"].agg(
        rows="size",
        present_count=lambda values: int((values == "Present").sum()),
        present_rate=lambda values: float((values == "Present").mean()),
    )
    best_cluster = int(cluster_stats["present_rate"].idxmax())
    base_rate = float((projected["murmur"] == "Present").mean())
    best_rows = int(cluster_stats.loc[best_cluster, "rows"])
    best_present = int(cluster_stats.loc[best_cluster, "present_count"])
    total_present = int((projected["murmur"] == "Present").sum())
    metrics: dict[str, float | int] = {
        "best_cluster": best_cluster,
        "best_cluster_rows": best_rows,
        "best_cluster_present_count": best_present,
        "best_cluster_present_rate": float(cluster_stats.loc[best_cluster, "present_rate"]),
        "best_cluster_present_capture": float(best_present / total_present) if total_present else float("nan"),
        "best_cluster_enrichment": float(cluster_stats.loc[best_cluster, "present_rate"] / base_rate) if base_rate else float("nan"),
    }
    largest_enriched = cluster_stats.loc[cluster_stats["present_rate"] >= max(0.5, base_rate * 2.0)]
    metrics["enriched_cluster_rows_total"] = int(largest_enriched["rows"].sum()) if not largest_enriched.empty else 0
    metrics["enriched_cluster_present_count_total"] = int(largest_enriched["present_count"].sum()) if not largest_enriched.empty else 0
    metrics["enriched_cluster_present_capture_total"] = (
        float(largest_enriched["present_count"].sum() / total_present) if total_present and not largest_enriched.empty else 0.0
    )
    for cluster_id, row in cluster_stats.iterrows():
        metrics[f"cluster_{int(cluster_id)}_rows"] = int(row["rows"])
        metrics[f"cluster_{int(cluster_id)}_present_rate"] = float(row["present_rate"])
    return metrics


def scatter(df: pd.DataFrame, x: str, y: str, output_path: Path, title: str, color_col: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    murmur_colors = {"Absent": "#2E86AB", "Present": "#D1495B", "Unknown": "#777777"}
    cluster_colors = ["#3B7EA1", "#D1495B", "#5A9367", "#F2A541", "#6A4C93", "#8C564B", "#17A398", "#777777"]
    for value, subset in df.groupby(color_col):
        if color_col == "cluster":
            color = cluster_colors[int(value) % len(cluster_colors)]
            label = f"cluster {value}"
        else:
            color = murmur_colors.get(str(value), "#777777")
            label = str(value)
        ax.scatter(
            subset[x],
            subset[y],
            s=34,
            alpha=0.78,
            linewidths=0.3,
            edgecolors="white",
            c=color,
            label=label,
        )
    ax.axhline(0, color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_projection_outputs(projected: pd.DataFrame, output_prefix: Path, title_prefix: str, skip_umap: bool) -> None:
    scatter(projected, "pca_1", "pca_2", output_prefix.with_name(output_prefix.name + "_pca_murmur.png"), f"{title_prefix}: PCA por murmur", "murmur")
    scatter(projected, "pca_1", "pca_2", output_prefix.with_name(output_prefix.name + "_pca_cluster.png"), f"{title_prefix}: PCA por cluster", "cluster")
    if not skip_umap and "umap_1" in projected.columns:
        scatter(projected, "umap_1", "umap_2", output_prefix.with_name(output_prefix.name + "_umap_murmur.png"), f"{title_prefix}: UMAP por murmur", "murmur")
        scatter(projected, "umap_1", "umap_2", output_prefix.with_name(output_prefix.name + "_umap_cluster.png"), f"{title_prefix}: UMAP por cluster", "cluster")


def aggregate_by_patient(recording_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    meta = recording_df.groupby("patient_id", as_index=False).agg(
        murmur=("murmur", "first"),
        outcome=("outcome", "first"),
        age=("age", "first"),
        sex=("sex", "first"),
        campaign=("campaign", "first"),
        location_count=("location", "nunique"),
        recording_count=("recording_id", "count"),
    )
    mean_features = recording_df.groupby("patient_id")[feature_columns].mean().add_prefix("mean_").reset_index()
    max_features = recording_df.groupby("patient_id")[feature_columns].max().add_prefix("max_").reset_index()
    p90_features = recording_df.groupby("patient_id")[feature_columns].quantile(0.90).add_prefix("p90_").reset_index()
    return meta.merge(mean_features, on="patient_id").merge(max_features, on="patient_id").merge(p90_features, on="patient_id")


def projection_rows(
    df: pd.DataFrame,
    feature_columns: list[str],
    level: str,
    location: str,
    args: argparse.Namespace,
    output_dir: Path,
    file_prefix: str,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for view in FEATURE_VIEWS:
        view_features = select_view_features(feature_columns, view)
        if not view_features:
            continue
        projected, projection_metrics = fit_projection(df, view_features, args.n_clusters, args.skip_umap)
        projected.to_csv(output_dir / f"{file_prefix}_{view}_projection.csv", index=False)
        save_projection_outputs(projected, output_dir / f"{file_prefix}_{view}", f"{file_prefix} {view}", args.skip_umap)
        rows.append(
            {
                "level": level,
                "location": location,
                "view": view,
                "n_clusters": int(args.n_clusters),
                "rows": int(len(projected)),
                "patients": int(projected["patient_id"].nunique()) if "patient_id" in projected else int(len(projected)),
                "present_rate": float((projected["murmur"] == "Present").mean()),
                "feature_count": int(len(view_features)),
                **cluster_metrics(projected),
                **projection_metrics,
            }
        )
    return rows


def sweep_cluster_counts(df: pd.DataFrame, feature_columns: list[str], level: str, location: str, args: argparse.Namespace) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    k_values = [int(value.strip()) for value in args.cluster_sweep.split(",") if value.strip()]
    for view in FEATURE_VIEWS:
        view_features = select_view_features(feature_columns, view)
        if not view_features:
            continue
        for k in k_values:
            if len(df) <= k:
                continue
            projected, projection_metrics = fit_projection(df, view_features, k, skip_umap=True)
            rows.append(
                {
                    "level": level,
                    "location": location,
                    "view": view,
                    "n_clusters": int(k),
                    "rows": int(len(projected)),
                    "patients": int(projected["patient_id"].nunique()) if "patient_id" in projected else int(len(projected)),
                    "present_rate": float((projected["murmur"] == "Present").mean()),
                    "feature_count": int(len(view_features)),
                    **cluster_metrics(projected),
                    **projection_metrics,
                }
            )
    return rows


def write_summary(
    output_path: Path,
    recording_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    feature_columns: list[str],
    patient_feature_columns: list[str],
    args: argparse.Namespace,
) -> None:
    best_patient = sweep_df.loc[sweep_df["level"].eq("patient_aggregated")].sort_values(
        ["best_cluster_present_capture", "best_cluster_present_rate", "best_cluster_rows"],
        ascending=[False, False, False],
    )
    best_patient_table = best_patient.head(12)
    lines = [
        "# Grupo B v3.1 - contraste robusto por ciclo",
        "",
        "## Objetivo",
        "",
        "Incrementar o v3 para destacar mais casos com murmurio usando contraste robusto pela diastole, features por ciclo e agregacao top-k.",
        "",
        "Feature central:",
        "",
        "`z[f,t] = (log(1 + |STFT(sistole)[f,t]|) - mediana_t(diastole[f])) / MAD_t(diastole[f])`",
        "",
        "## Configuracao",
        "",
        f"- Target sample rate: {args.target_sample_rate} Hz",
        f"- STFT: n_fft={args.n_fft}, hop_length={args.hop_length}",
        f"- Faixa analisada: {args.low_hz:.1f}-{args.high_hz:.1f} Hz",
        f"- K-means padrao: {args.n_clusters}",
        f"- Sweep de k: {args.cluster_sweep}",
        f"- Robust scale floor: {args.robust_scale_floor}",
        f"- Z clip: {args.z_clip}",
        f"- UMAP gerado: {'nao' if args.skip_umap else 'sim'}",
        "",
        "## Resumo",
        "",
        f"- Gravacoes analisadas: {len(recording_df)}",
        f"- Pacientes analisados: {recording_df['patient_id'].nunique()}",
        f"- Features por gravacao: {len(feature_columns)}",
        f"- Pacientes agregados: {len(patient_df)}",
        f"- Features por paciente: {len(patient_feature_columns)}",
        "",
        "## Murmur por local",
        "",
        pd.crosstab(recording_df["location"], recording_df["murmur"], margins=True).to_markdown(),
        "",
        "## Metricas principais",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Melhores configuracoes paciente-level no sweep",
        "",
        best_patient_table.to_markdown(index=False),
        "",
        "## Arquivos gerados",
        "",
        "- `recording_robust_cycle_contrast_features.csv`",
        "- `patient_robust_cycle_contrast_features.csv`",
        "- `projection_metrics.csv`",
        "- `cluster_sweep_metrics.csv`",
        "- `projections/*_projection.csv` e PNGs PCA/UMAP",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    data_dir = dataset_dir / "training_data"
    metadata_path = dataset_dir / "training_data.csv"
    output_dir = args.output_dir.resolve()
    projection_dir = output_dir / "projections"
    output_dir.mkdir(parents=True, exist_ok=True)
    projection_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    if not args.include_unknown:
        metadata = metadata.loc[metadata["Murmur"].isin(["Present", "Absent"])].copy()

    wav_paths = sorted(data_dir.glob("*.wav"))
    if args.max_recordings is not None:
        wav_paths = wav_paths[: args.max_recordings]

    rows: list[dict[str, float | str]] = []
    skipped = 0
    for index, wav_path in enumerate(wav_paths, start=1):
        row = extract_recording(wav_path, metadata, args)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if index % 250 == 0:
            print(f"Processed {index}/{len(wav_paths)} recordings...", flush=True)

    if not rows:
        raise RuntimeError("No rows extracted. Check dataset path and filters.")

    recording_df = pd.DataFrame(rows)
    feature_columns = numeric_feature_columns(recording_df)
    recording_df.to_csv(output_dir / "recording_robust_cycle_contrast_features.csv", index=False)

    metrics_rows: list[dict[str, float | str | int]] = []
    sweep_rows: list[dict[str, float | str | int]] = []
    metrics_rows.extend(projection_rows(recording_df, feature_columns, "recording_global", "all", args, projection_dir, "recording_all"))
    sweep_rows.extend(sweep_cluster_counts(recording_df, feature_columns, "recording_global", "all", args))

    for location in LOCATIONS:
        subset = recording_df.loc[recording_df["location"] == location].copy()
        if subset.empty:
            continue
        loc_dir = projection_dir / f"location_{location}"
        loc_dir.mkdir(parents=True, exist_ok=True)
        metrics_rows.extend(projection_rows(subset, feature_columns, "recording_location", location, args, loc_dir, location))
        sweep_rows.extend(sweep_cluster_counts(subset, feature_columns, "recording_location", location, args))

    patient_df = aggregate_by_patient(recording_df, feature_columns)
    patient_feature_columns = numeric_feature_columns(patient_df)
    patient_df.to_csv(output_dir / "patient_robust_cycle_contrast_features.csv", index=False)
    metrics_rows.extend(projection_rows(patient_df, patient_feature_columns, "patient_aggregated", "mean_max_p90", args, projection_dir, "patient"))
    sweep_rows.extend(sweep_cluster_counts(patient_df, patient_feature_columns, "patient_aggregated", "mean_max_p90", args))

    metrics_df = pd.DataFrame(metrics_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    metrics_df.to_csv(output_dir / "projection_metrics.csv", index=False)
    sweep_df.to_csv(output_dir / "cluster_sweep_metrics.csv", index=False)
    write_summary(output_dir / "summary.md", recording_df, patient_df, metrics_df, sweep_df, feature_columns, patient_feature_columns, args)

    print(f"Extracted {len(recording_df)} recordings. Skipped {skipped}.", flush=True)
    print(f"Recording feature columns: {len(feature_columns)}", flush=True)
    print(f"Patient feature columns: {len(patient_feature_columns)}", flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
