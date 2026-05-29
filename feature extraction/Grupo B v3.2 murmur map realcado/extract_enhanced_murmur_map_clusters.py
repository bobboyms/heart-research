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
"""Enhanced murmur-map clustering.

This v3.2 experiment builds on v3.1. It keeps the robust systole-vs-diastole
contrast, but enhances the region that behaves like a murmur:

1. Crop systole edges to reduce S1/S2 leakage.
2. Keep positive robust contrast only.
3. Smooth the positive contrast in time/frequency.
4. Threshold the smoothed map to remove isolated weak noise.
5. Add persistence features and a compact low-band heatmap representation.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V31_SCRIPT = (
    SCRIPT_DIR.parent
    / "Grupo B v3.1 contraste robusto por ciclo"
    / "extract_robust_cycle_contrast_clusters.py"
)

spec = importlib.util.spec_from_file_location("v31_contrast", V31_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import v3.1 helper script: {V31_SCRIPT}")
v31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v31)


FEATURE_VIEWS = {
    "all": None,
    "low": ("enh_low", "enh_very_low", "enh_low_mid", "lowres_low"),
    "mid": ("enh_mid", "enh_mid_high", "enh_upper_mid"),
    "high": ("enh_high", "enh_upper_mid"),
    "map_low": ("map_low_bin",),
}

BANDS_HZ = {
    "low_25_200hz": (25.0, 200.0),
    "mid_200_600hz": (200.0, 600.0),
    "high_600_1000hz": (600.0, 1000.0),
    "very_low_25_80hz": (25.0, 80.0),
    "low_mid_80_200hz": (80.0, 200.0),
    "mid_high_200_400hz": (200.0, 400.0),
    "upper_mid_400_800hz": (400.0, 800.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enhanced murmur-map clusters.")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "circor-heart-sound-1.0.3")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--cluster-sweep", type=str, default="2,4,6,8,10,12")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--target-sample-rate", type=int, default=4000)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--low-n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--low-hop-length", type=int, default=48)
    parser.add_argument("--low-hz", type=float, default=25.0)
    parser.add_argument("--high-hz", type=float, default=1000.0)
    parser.add_argument("--low-map-high-hz", type=float, default=260.0)
    parser.add_argument("--robust-scale-floor", type=float, default=0.03)
    parser.add_argument("--z-clip", type=float, default=12.0)
    parser.add_argument("--center-crop", type=float, default=0.15)
    parser.add_argument("--smooth-freq-sigma", type=float, default=0.75)
    parser.add_argument("--smooth-time-sigma", type=float, default=1.0)
    parser.add_argument("--map-threshold", type=float, default=1.0)
    parser.add_argument("--heatmap-freq-bins", type=int, default=16)
    parser.add_argument("--heatmap-time-bins", type=int, default=32)
    return parser.parse_args()


def crop_systole_specs(specs: list[np.ndarray], crop_fraction: float) -> list[np.ndarray]:
    cropped: list[np.ndarray] = []
    crop_fraction = min(max(float(crop_fraction), 0.0), 0.45)
    for spec in specs:
        frames = spec.shape[1]
        trim = int(round(frames * crop_fraction))
        if frames - 2 * trim >= 1:
            cropped.append(spec[:, trim : frames - trim])
        else:
            cropped.append(spec)
    return cropped


def robust_reference(diastole_specs: list[np.ndarray], robust_scale_floor: float) -> tuple[np.ndarray, np.ndarray, float]:
    diastole = np.concatenate(diastole_specs, axis=1)
    reference = np.median(diastole, axis=1, keepdims=True).astype(np.float32)
    mad = np.median(np.abs(diastole - reference), axis=1, keepdims=True).astype(np.float32)
    scale = 1.4826 * mad
    finite_scale = scale[np.isfinite(scale) & (scale > 0.0)]
    floor = max(float(np.median(finite_scale)) * robust_scale_floor, 1e-3) if finite_scale.size else 1e-3
    return reference, np.maximum(scale, floor).astype(np.float32), floor


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


def longest_run_fraction(active: np.ndarray) -> float:
    if active.size == 0:
        return 0.0
    best = 0
    current = 0
    for value in active.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best / active.size)


def resize_matrix(matrix: np.ndarray, freq_bins: int, time_bins: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((freq_bins, time_bins), dtype=np.float32)
    source_f = np.linspace(0.0, 1.0, matrix.shape[0])
    target_f = np.linspace(0.0, 1.0, freq_bins)
    freq_resized = np.vstack([np.interp(target_f, source_f, matrix[:, t]) for t in range(matrix.shape[1])]).T
    source_t = np.linspace(0.0, 1.0, matrix.shape[1])
    target_t = np.linspace(0.0, 1.0, time_bins)
    resized = np.vstack([np.interp(target_t, source_t, row) for row in freq_resized])
    return resized.astype(np.float32)


def enhanced_cycle_features(
    systole_specs: list[np.ndarray],
    diastole_specs: list[np.ndarray],
    freqs: np.ndarray,
    prefix: str,
    args: argparse.Namespace,
    include_heatmap: bool = False,
) -> dict[str, float]:
    reference, scale, floor = robust_reference(diastole_specs, args.robust_scale_floor)
    band_masks = {name: (freqs >= low) & (freqs < high) for name, (low, high) in BANDS_HZ.items()}
    out: dict[str, float] = {
        f"{prefix}_cycle_count": float(len(systole_specs)),
        f"{prefix}_robust_scale_floor_used": float(floor),
        f"{prefix}_freq_bin_count": float(len(freqs)),
    }
    per_metric: dict[str, list[float]] = {}
    heatmaps: list[tuple[float, np.ndarray]] = []

    def collect(name: str, value: float) -> None:
        per_metric.setdefault(name, []).append(float(value))

    for spec in systole_specs:
        z = ((spec - reference) / scale).astype(np.float32)
        z = np.clip(z, -args.z_clip, args.z_clip)
        positive = np.maximum(z, 0.0)
        smooth = gaussian_filter(positive, sigma=(args.smooth_freq_sigma, args.smooth_time_sigma))
        murmur_map = np.where(smooth >= args.map_threshold, smooth, 0.0).astype(np.float32)

        all_frame_energy = murmur_map.mean(axis=0) if murmur_map.shape[1] else np.zeros(1)
        collect(f"{prefix}_map_total_energy", float(np.mean(murmur_map)))
        collect(f"{prefix}_map_active_fraction", float(np.mean(murmur_map > 0.0)))
        collect(f"{prefix}_map_longest_run_fraction", longest_run_fraction(all_frame_energy > 0.0))
        collect(f"{prefix}_z_fraction_gt_15", float(np.mean(z > 1.5)))
        collect(f"{prefix}_z_fraction_gt_2", float(np.mean(z > 2.0)))
        collect(f"{prefix}_z_fraction_gt_3", float(np.mean(z > 3.0)))

        for band_name, mask in band_masks.items():
            if not np.any(mask):
                continue
            band = murmur_map[mask]
            raw_band = z[mask]
            frame_energy = band.mean(axis=0) if band.shape[1] else np.zeros(1)
            freq_energy = band.mean(axis=1) if band.shape[0] else np.zeros(1)
            active_frames = frame_energy > 0.0
            top_count = max(1, int(math.ceil(0.30 * frame_energy.size)))
            top_frames = np.sort(frame_energy)[-top_count:]
            collect(f"enh_{band_name}_energy", float(np.mean(band)))
            collect(f"enh_{band_name}_p90", float(np.percentile(band, 90)))
            collect(f"enh_{band_name}_max", float(np.max(band)))
            collect(f"enh_{band_name}_active_fraction", float(np.mean(band > 0.0)))
            collect(f"enh_{band_name}_frame_active_fraction", float(np.mean(active_frames)))
            collect(f"enh_{band_name}_longest_run_fraction", longest_run_fraction(active_frames))
            collect(f"enh_{band_name}_top30_frame_mean", float(np.mean(top_frames)))
            collect(f"enh_{band_name}_z_gt_2_fraction", float(np.mean(raw_band > 2.0)))
            collect(f"enh_{band_name}_z_gt_3_fraction", float(np.mean(raw_band > 3.0)))
            collect(f"enh_{band_name}_peak_pos", float(np.argmax(frame_energy) / max(1, frame_energy.size - 1)))
            if float(np.sum(freq_energy)) > 0.0:
                collect(f"enh_{band_name}_freq_centroid", float(np.sum(freqs[mask] * freq_energy) / np.sum(freq_energy)))
            else:
                collect(f"enh_{band_name}_freq_centroid", 0.0)

        if include_heatmap:
            low_mask = (freqs >= 25.0) & (freqs < 200.0)
            if np.any(low_mask):
                low_map = murmur_map[low_mask]
                score = float(np.mean(low_map))
                heatmaps.append((score, resize_matrix(low_map, args.heatmap_freq_bins, args.heatmap_time_bins)))

    for metric_name, values in per_metric.items():
        aggregate_values(metric_name, values, out)

    if include_heatmap:
        if heatmaps:
            heatmaps.sort(key=lambda item: item[0], reverse=True)
            top = [matrix for _score, matrix in heatmaps[: min(3, len(heatmaps))]]
            heatmap = np.mean(np.stack(top), axis=0)
        else:
            heatmap = np.zeros((args.heatmap_freq_bins, args.heatmap_time_bins), dtype=np.float32)
        for f_idx in range(args.heatmap_freq_bins):
            for t_idx in range(args.heatmap_time_bins):
                out[f"map_low_bin_f{f_idx:02d}_t{t_idx:02d}"] = float(heatmap[f_idx, t_idx])

    return out


def extract_recording(wav_path: Path, metadata: pd.DataFrame, args: argparse.Namespace) -> dict[str, float | str] | None:
    tsv_path = wav_path.with_suffix(".tsv")
    if not tsv_path.exists():
        return None
    patient_id, location = v31.parse_recording_id(wav_path)
    if location not in v31.LOCATIONS:
        return None
    row = metadata.loc[metadata["Patient ID"].astype(str) == patient_id]
    if row.empty:
        return None

    sample_rate, audio = v31.read_audio(wav_path)
    audio = v31.resample_audio(audio, sample_rate, args.target_sample_rate)
    segments = v31.read_segments(tsv_path)
    systole_specs, freqs = v31.segment_log_specs(
        audio,
        args.target_sample_rate,
        segments,
        v31.LABEL_SYSTOLE,
        args.n_fft,
        args.hop_length,
        args.low_hz,
        args.high_hz,
    )
    diastole_specs, _ = v31.segment_log_specs(
        audio,
        args.target_sample_rate,
        segments,
        v31.LABEL_DIASTOLE,
        args.n_fft,
        args.hop_length,
        args.low_hz,
        args.high_hz,
    )
    low_systole_specs, low_freqs = v31.segment_log_specs(
        audio,
        args.target_sample_rate,
        segments,
        v31.LABEL_SYSTOLE,
        args.low_n_fft,
        args.low_hop_length,
        args.low_hz,
        args.low_map_high_hz,
    )
    low_diastole_specs, _ = v31.segment_log_specs(
        audio,
        args.target_sample_rate,
        segments,
        v31.LABEL_DIASTOLE,
        args.low_n_fft,
        args.low_hop_length,
        args.low_hz,
        args.low_map_high_hz,
    )
    if not systole_specs or not diastole_specs or not low_systole_specs or not low_diastole_specs:
        return None

    systole_specs = crop_systole_specs(systole_specs, args.center_crop)
    low_systole_specs = crop_systole_specs(low_systole_specs, args.center_crop)

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
    features.update(enhanced_cycle_features(systole_specs, diastole_specs, freqs, "base", args))
    features.update(enhanced_cycle_features(low_systole_specs, low_diastole_specs, low_freqs, "lowres", args, include_heatmap=True))
    return features


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in v31.META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def select_view_features(feature_columns: list[str], view: str) -> list[str]:
    needles = FEATURE_VIEWS[view]
    if needles is None:
        return feature_columns
    return [column for column in feature_columns if any(needle in column for needle in needles)]


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
        projected, projection_metrics = v31.fit_projection(df, view_features, args.n_clusters, args.skip_umap)
        projected.to_csv(output_dir / f"{file_prefix}_{view}_projection.csv", index=False)
        v31.save_projection_outputs(projected, output_dir / f"{file_prefix}_{view}", f"{file_prefix} {view}", args.skip_umap)
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
                **v31.cluster_metrics(projected),
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
            projected, projection_metrics = v31.fit_projection(df, view_features, k, skip_umap=True)
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
                    **v31.cluster_metrics(projected),
                    **projection_metrics,
                }
            )
    return rows


def plot_heatmap(ax: plt.Axes, matrix: np.ndarray, title: str) -> None:
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("tempo normalizado na sistole")
    ax.set_ylabel("freq baixa")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_heatmap_comparison(recording_df: pd.DataFrame, patient_projection: pd.DataFrame, output_path: Path, args: argparse.Namespace) -> None:
    map_cols = sorted([column for column in recording_df.columns if column.startswith("map_low_bin_")])
    if not map_cols:
        return
    cluster_rates = patient_projection.groupby("cluster")["murmur"].apply(lambda values: float((values == "Present").mean()))
    base_rate = float((patient_projection["murmur"] == "Present").mean())
    enriched_clusters = cluster_rates.loc[cluster_rates >= max(0.5, base_rate * 2.0)].index.tolist()
    enriched_patients = set(patient_projection.loc[patient_projection["cluster"].isin(enriched_clusters), "patient_id"].astype(str))
    present_enriched = recording_df.loc[
        recording_df["patient_id"].astype(str).isin(enriched_patients) & recording_df["murmur"].eq("Present"), map_cols
    ]
    all_enriched = recording_df.loc[recording_df["patient_id"].astype(str).isin(enriched_patients), map_cols]
    absent_background = recording_df.loc[
        ~recording_df["patient_id"].astype(str).isin(enriched_patients) & recording_df["murmur"].eq("Absent"), map_cols
    ]

    def mean_map(frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.zeros((args.heatmap_freq_bins, args.heatmap_time_bins), dtype=np.float32)
        return frame.mean(axis=0).to_numpy(dtype=np.float32).reshape(args.heatmap_freq_bins, args.heatmap_time_bins)

    present_map = mean_map(present_enriched)
    enriched_map = mean_map(all_enriched)
    absent_map = mean_map(absent_background)
    diff_map = np.maximum(present_map - absent_map, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_heatmap(axes[0, 0], enriched_map, "clusters enriquecidos")
    plot_heatmap(axes[0, 1], present_map, "Present nos clusters")
    plot_heatmap(axes[1, 0], absent_map, "Absent fora dos clusters")
    plot_heatmap(axes[1, 1], diff_map, "Present - Absent")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
        ["enriched_cluster_present_capture_total", "best_cluster_present_rate", "best_cluster_rows"],
        ascending=[False, False, False],
    )
    best_single = sweep_df.loc[sweep_df["level"].eq("patient_aggregated")].sort_values(
        ["best_cluster_present_capture", "best_cluster_present_rate"],
        ascending=[False, False],
    )
    lines = [
        "# Grupo B v3.2 - murmur map realcado",
        "",
        "## Objetivo",
        "",
        "Realcar a regiao do murmurio no contraste sistole menos diastole antes de clusterizar.",
        "",
        "Incrementos sobre v3.1:",
        "",
        "- corte central da sistole para reduzir vazamento de S1/S2;",
        "- mapa positivo `max(z, 0)`;",
        "- suavizacao leve em tempo/frequencia;",
        "- threshold do mapa para remover ruido fraco;",
        "- features de persistencia temporal;",
        "- mapa compacto da banda baixa para preservar a localizacao tempo-frequencia.",
        "",
        "## Configuracao",
        "",
        f"- K-means padrao: {args.n_clusters}",
        f"- Sweep de k: {args.cluster_sweep}",
        f"- Corte central: {args.center_crop:.2f} de cada borda",
        f"- Threshold do mapa: {args.map_threshold}",
        f"- Smooth sigma freq/time: {args.smooth_freq_sigma}/{args.smooth_time_sigma}",
        f"- Low-res STFT: n_fft={args.low_n_fft}, hop={args.low_hop_length}",
        "",
        "## Resumo",
        "",
        f"- Gravacoes analisadas: {len(recording_df)}",
        f"- Pacientes analisados: {recording_df['patient_id'].nunique()}",
        f"- Features por gravacao: {len(feature_columns)}",
        f"- Pacientes agregados: {len(patient_df)}",
        f"- Features por paciente: {len(patient_feature_columns)}",
        "",
        "## Melhores configuracoes paciente-level por captura enriquecida",
        "",
        best_patient.head(15).to_markdown(index=False),
        "",
        "## Melhores configuracoes paciente-level por melhor cluster unico",
        "",
        best_single.head(15).to_markdown(index=False),
        "",
        "## Metricas principais",
        "",
        metrics_df.to_markdown(index=False),
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
    recording_df.to_csv(output_dir / "recording_enhanced_murmur_map_features.csv", index=False)

    metrics_rows: list[dict[str, float | str | int]] = []
    sweep_rows: list[dict[str, float | str | int]] = []
    metrics_rows.extend(projection_rows(recording_df, feature_columns, "recording_global", "all", args, projection_dir, "recording_all"))
    sweep_rows.extend(sweep_cluster_counts(recording_df, feature_columns, "recording_global", "all", args))

    patient_df = v31.aggregate_by_patient(recording_df, feature_columns)
    patient_feature_columns = numeric_feature_columns(patient_df)
    patient_df.to_csv(output_dir / "patient_enhanced_murmur_map_features.csv", index=False)
    metrics_rows.extend(projection_rows(patient_df, patient_feature_columns, "patient_aggregated", "mean_max_p90", args, projection_dir, "patient"))
    sweep_rows.extend(sweep_cluster_counts(patient_df, patient_feature_columns, "patient_aggregated", "mean_max_p90", args))

    metrics_df = pd.DataFrame(metrics_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    metrics_df.to_csv(output_dir / "projection_metrics.csv", index=False)
    sweep_df.to_csv(output_dir / "cluster_sweep_metrics.csv", index=False)

    patient_map_projection = pd.read_csv(projection_dir / "patient_map_low_projection.csv")
    save_heatmap_comparison(recording_df, patient_map_projection, output_dir / "murmur_map_heatmap_comparison.png", args)

    write_summary(output_dir / "summary.md", recording_df, patient_df, metrics_df, sweep_df, feature_columns, patient_feature_columns, args)
    print(f"Extracted {len(recording_df)} recordings. Skipped {skipped}.", flush=True)
    print(f"Recording feature columns: {len(feature_columns)}", flush=True)
    print(f"Patient feature columns: {len(patient_feature_columns)}", flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
