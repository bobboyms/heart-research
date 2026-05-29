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
"""Cluster recordings using explicit systole-minus-diastole spectral contrast.

This is the unsupervised visualization analogue of the best supervised
phase-contrast idea: use diastole from the same recording as the local acoustic
background, then measure what is left in systole.

Run from the repository root:

    uv run "feature extraction/Grupo B v3 contraste sistole menos diastole/extract_systole_diastole_contrast_clusters.py"

Outputs:

    feature extraction/Grupo B v3 contraste sistole menos diastole/outputs/
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
from sklearn.preprocessing import StandardScaler


LABEL_SYSTOLE = 2
LABEL_DIASTOLE = 4
LOCATIONS = ["AV", "PV", "TV", "MV"]

BANDS_HZ = {
    "25_80hz": (25.0, 80.0),
    "80_200hz": (80.0, 200.0),
    "200_400hz": (200.0, 400.0),
    "400_800hz": (400.0, 800.0),
    "800_1000hz": (800.0, 1000.0),
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
        description="Generate PCA/UMAP/k-means clusters from systole-minus-diastole contrast features."
    )
    parser.add_argument("--dataset-dir", type=Path, default=repo_root / "circor-heart-sound-1.0.3")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs")
    parser.add_argument("--include-unknown", action="store_true", help="Include Murmur=Unknown.")
    parser.add_argument("--max-recordings", type=int, default=None, help="Optional quick-test limit.")
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--target-sample-rate", type=int, default=4000)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--low-hz", type=float, default=25.0)
    parser.add_argument("--high-hz", type=float, default=1000.0)
    parser.add_argument("--profile-bins", type=int, default=24)
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


def phase_log_frames(
    audio: np.ndarray,
    sample_rate: int,
    segments: pd.DataFrame,
    label: int,
    n_fft: int,
    hop_length: int,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = segments.loc[segments["label"] == label].sort_values("start_time")
    if selected.empty:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float32)

    blocks: list[np.ndarray] = []
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
            blocks.append(spec)

    if not blocks or kept_freqs is None:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks, axis=1).astype(np.float32), kept_freqs


def add_stats(prefix: str, values: np.ndarray, out: dict[str, float]) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        out[f"{prefix}_mean"] = 0.0
        out[f"{prefix}_std"] = 0.0
        out[f"{prefix}_p75"] = 0.0
        out[f"{prefix}_p90"] = 0.0
        out[f"{prefix}_max"] = 0.0
        out[f"{prefix}_positive_fraction"] = 0.0
        out[f"{prefix}_positive_mean"] = 0.0
        return
    positive = finite[finite > 0.0]
    out[f"{prefix}_mean"] = float(np.mean(finite))
    out[f"{prefix}_std"] = float(np.std(finite))
    out[f"{prefix}_p75"] = float(np.percentile(finite, 75))
    out[f"{prefix}_p90"] = float(np.percentile(finite, 90))
    out[f"{prefix}_max"] = float(np.max(finite))
    out[f"{prefix}_positive_fraction"] = float(np.mean(finite > 0.0))
    out[f"{prefix}_positive_mean"] = float(np.mean(positive)) if positive.size else 0.0


def contrast_features(
    systole: np.ndarray,
    diastole: np.ndarray,
    freqs: np.ndarray,
    profile_bins: int,
) -> dict[str, float]:
    reference = np.median(diastole, axis=1, keepdims=True).astype(np.float32)
    contrast = (systole - reference).astype(np.float32)
    positive = np.maximum(contrast, 0.0)
    out: dict[str, float] = {
        "systole_frame_count": float(systole.shape[1]),
        "diastole_frame_count": float(diastole.shape[1]),
        "freq_bin_count": float(systole.shape[0]),
    }

    add_stats("contrast_all", contrast, out)
    add_stats("contrast_positive_all", positive, out)
    freq_mean = contrast.mean(axis=1)
    freq_p90 = np.percentile(contrast, 90, axis=1)
    freq_positive = positive.mean(axis=1)

    for band_name, (low, high) in BANDS_HZ.items():
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            continue
        add_stats(f"contrast_{band_name}", contrast[mask], out)
        out[f"contrast_{band_name}_freq_mean"] = float(np.mean(freq_mean[mask]))
        out[f"contrast_{band_name}_freq_p90_mean"] = float(np.mean(freq_p90[mask]))
        out[f"contrast_{band_name}_positive_area"] = float(np.mean(freq_positive[mask]))

    edges = np.linspace(float(freqs.min()), float(freqs.max()), profile_bins + 1)
    for idx in range(profile_bins):
        if idx == profile_bins - 1:
            mask = (freqs >= edges[idx]) & (freqs <= edges[idx + 1])
        else:
            mask = (freqs >= edges[idx]) & (freqs < edges[idx + 1])
        if not np.any(mask):
            out[f"profile_bin_{idx:02d}_mean"] = 0.0
            out[f"profile_bin_{idx:02d}_p90"] = 0.0
            out[f"profile_bin_{idx:02d}_positive"] = 0.0
            continue
        out[f"profile_bin_{idx:02d}_mean"] = float(np.mean(freq_mean[mask]))
        out[f"profile_bin_{idx:02d}_p90"] = float(np.mean(freq_p90[mask]))
        out[f"profile_bin_{idx:02d}_positive"] = float(np.mean(freq_positive[mask]))

    low_mask = (freqs >= 25.0) & (freqs < 200.0)
    mid_mask = (freqs >= 200.0) & (freqs < 600.0)
    high_mask = freqs >= 600.0
    low_area = float(np.mean(freq_positive[low_mask])) if np.any(low_mask) else 0.0
    mid_area = float(np.mean(freq_positive[mid_mask])) if np.any(mid_mask) else 0.0
    high_area = float(np.mean(freq_positive[high_mask])) if np.any(high_mask) else 0.0
    total_area = low_area + mid_area + high_area + 1e-12
    out["contrast_positive_low_share"] = low_area / total_area
    out["contrast_positive_mid_share"] = mid_area / total_area
    out["contrast_positive_high_share"] = high_area / total_area
    out["contrast_positive_mid_to_low"] = mid_area / (low_area + 1e-12)
    out["contrast_positive_high_to_low"] = high_area / (low_area + 1e-12)
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
    systole, freqs = phase_log_frames(
        audio,
        args.target_sample_rate,
        segments,
        LABEL_SYSTOLE,
        args.n_fft,
        args.hop_length,
        args.low_hz,
        args.high_hz,
    )
    diastole, _ = phase_log_frames(
        audio,
        args.target_sample_rate,
        segments,
        LABEL_DIASTOLE,
        args.n_fft,
        args.hop_length,
        args.low_hz,
        args.high_hz,
    )
    if systole.size == 0 or diastole.size == 0:
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
    features.update(contrast_features(systole, diastole, freqs, args.profile_bins))
    return features


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def project_cluster(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int,
    skip_umap: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = df.copy()
    if len(df) < 3:
        result["pca_1"] = 0.0
        result["pca_2"] = 0.0
        result["cluster"] = 0
        return result, {"silhouette_pca": float("nan"), "pca12_variance": float("nan")}

    matrix = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = StandardScaler().fit_transform(matrix)
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


def scatter(df: pd.DataFrame, x: str, y: str, output_path: Path, title: str, color_col: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    murmur_colors = {"Absent": "#2E86AB", "Present": "#D1495B", "Unknown": "#777777"}
    cluster_colors = ["#3B7EA1", "#D1495B", "#5A9367", "#F2A541", "#6A4C93", "#777777"]
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
    return meta.merge(mean_features, on="patient_id").merge(max_features, on="patient_id")


def projection_diagnostic_metrics(projected: pd.DataFrame) -> dict[str, float | int]:
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
    for cluster_id, row in cluster_stats.iterrows():
        metrics[f"cluster_{int(cluster_id)}_rows"] = int(row["rows"])
        metrics[f"cluster_{int(cluster_id)}_present_rate"] = float(row["present_rate"])
    return metrics


def write_summary(
    output_path: Path,
    recording_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    feature_columns: list[str],
    patient_feature_columns: list[str],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Grupo B v3 - contraste sistole menos diastole",
        "",
        "## Objetivo",
        "",
        "Gerar clusters para visualizar a separacao entre gravacoes/pacientes com `Murmur = Present` e `Murmur = Absent` usando a diastole como ruido de fundo da propria gravacao.",
        "",
        "Feature central:",
        "",
        "`contraste[f,t] = log(1 + |STFT(sistole)[f,t]|) - mediana_t(log(1 + |STFT(diastole)[f,t]|))`",
        "",
        "Ou seja: para cada frequencia, subtrai-se da sistole a referencia acustica da diastole do mesmo microfone/paciente.",
        "",
        "## Configuracao",
        "",
        f"- Target sample rate: {args.target_sample_rate} Hz",
        f"- STFT: n_fft={args.n_fft}, hop_length={args.hop_length}",
        f"- Faixa analisada: {args.low_hz:.1f}-{args.high_hz:.1f} Hz",
        f"- Profile bins: {args.profile_bins}",
        f"- K-means clusters: {args.n_clusters}",
        f"- UMAP gerado: {'nao' if args.skip_umap else 'sim'}",
        "",
        "## Resumo",
        "",
        f"- Gravacoes analisadas: {len(recording_df)}",
        f"- Pacientes analisados: {recording_df['patient_id'].nunique()}",
        f"- Features de contraste por gravacao: {len(feature_columns)}",
        f"- Pacientes agregados: {len(patient_df)}",
        f"- Features agregadas por paciente: {len(patient_feature_columns)}",
        "",
        "## Murmur por local",
        "",
        pd.crosstab(recording_df["location"], recording_df["murmur"], margins=True).to_markdown(),
        "",
        "## Metricas por visualizacao",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Leitura rapida",
        "",
        "- Compare `best_cluster_present_rate` contra `present_rate`: quanto maior o enriquecimento, mais o contraste conseguiu juntar sopros.",
        "- `best_cluster_present_capture` mede quanto dos casos `Present` caem no cluster enriquecido; pureza alta com captura baixa indica um subgrupo claro, mas incompleto.",
        "- A leitura mais importante e o nivel `patient_aggregated`, porque o rotulo `Murmur` e por paciente.",
        "",
        "## Arquivos gerados",
        "",
        "- `recording_contrast_features.csv`: features de contraste por gravacao.",
        "- `recording_contrast_features_with_projection.csv`: PCA/k-means global por gravacao.",
        "- `patient_contrast_features.csv`: agregacao por paciente usando media e maximo entre locais.",
        "- `patient_contrast_features_with_projection.csv`: PCA/k-means por paciente.",
        "- `*_pca_murmur.png` e `*_pca_cluster.png`: visualizacoes PCA.",
        "- `*_umap_murmur.png` e `*_umap_cluster.png`: visualizacoes UMAP, se habilitado.",
        "- `by_location/`: visualizacoes separadas por local de ausculta.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_projection_outputs(projected: pd.DataFrame, output_prefix: Path, title_prefix: str, skip_umap: bool) -> None:
    scatter(projected, "pca_1", "pca_2", output_prefix.with_name(output_prefix.name + "_pca_murmur.png"), f"{title_prefix}: PCA por murmur", "murmur")
    scatter(projected, "pca_1", "pca_2", output_prefix.with_name(output_prefix.name + "_pca_cluster.png"), f"{title_prefix}: PCA por cluster", "cluster")
    if not skip_umap and "umap_1" in projected.columns:
        scatter(projected, "umap_1", "umap_2", output_prefix.with_name(output_prefix.name + "_umap_murmur.png"), f"{title_prefix}: UMAP por murmur", "murmur")
        scatter(projected, "umap_1", "umap_2", output_prefix.with_name(output_prefix.name + "_umap_cluster.png"), f"{title_prefix}: UMAP por cluster", "cluster")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    data_dir = dataset_dir / "training_data"
    metadata_path = dataset_dir / "training_data.csv"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    recording_df.to_csv(output_dir / "recording_contrast_features.csv", index=False)

    metrics_rows: list[dict[str, float | str | int]] = []

    projected_recordings, global_metrics = project_cluster(recording_df, feature_columns, args.n_clusters, args.skip_umap)
    projected_recordings.to_csv(output_dir / "recording_contrast_features_with_projection.csv", index=False)
    save_projection_outputs(projected_recordings, output_dir / "recording", "Todas as gravacoes", args.skip_umap)
    metrics_rows.append(
        {
            "level": "recording_global",
            "location": "all",
            "rows": int(len(recording_df)),
            "patients": int(recording_df["patient_id"].nunique()),
            "present_rate": float((recording_df["murmur"] == "Present").mean()),
            **projection_diagnostic_metrics(projected_recordings),
            **global_metrics,
        }
    )

    per_location_dir = output_dir / "by_location"
    per_location_dir.mkdir(parents=True, exist_ok=True)
    for location in LOCATIONS:
        subset = recording_df.loc[recording_df["location"] == location].copy()
        if subset.empty:
            continue
        projected, metrics = project_cluster(subset, feature_columns, args.n_clusters, args.skip_umap)
        projected.to_csv(per_location_dir / f"{location}_contrast_features_with_projection.csv", index=False)
        save_projection_outputs(projected, per_location_dir / location, f"{location}", args.skip_umap)
        metrics_rows.append(
            {
                "level": "recording_location",
                "location": location,
                "rows": int(len(projected)),
                "patients": int(projected["patient_id"].nunique()),
                "present_rate": float((projected["murmur"] == "Present").mean()),
                **projection_diagnostic_metrics(projected),
                **metrics,
            }
        )

    patient_df = aggregate_by_patient(recording_df, feature_columns)
    patient_feature_columns = numeric_feature_columns(patient_df)
    patient_df.to_csv(output_dir / "patient_contrast_features.csv", index=False)
    projected_patients, patient_metrics = project_cluster(patient_df, patient_feature_columns, args.n_clusters, args.skip_umap)
    projected_patients.to_csv(output_dir / "patient_contrast_features_with_projection.csv", index=False)
    save_projection_outputs(projected_patients, output_dir / "patient", "Pacientes agregados", args.skip_umap)
    metrics_rows.append(
        {
            "level": "patient_aggregated",
            "location": "mean_max",
            "rows": int(len(patient_df)),
            "patients": int(len(patient_df)),
            "present_rate": float((patient_df["murmur"] == "Present").mean()),
            **projection_diagnostic_metrics(projected_patients),
            **patient_metrics,
        }
    )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "projection_metrics.csv", index=False)
    write_summary(output_dir / "summary.md", recording_df, patient_df, metrics_df, feature_columns, patient_feature_columns, args)

    print(f"Extracted {len(recording_df)} recordings. Skipped {skipped}.", flush=True)
    print(f"Recording feature columns: {len(feature_columns)}", flush=True)
    print(f"Patient feature columns: {len(patient_feature_columns)}", flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
