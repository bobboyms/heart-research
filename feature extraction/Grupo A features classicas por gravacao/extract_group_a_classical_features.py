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
"""Grupo A experiment: classical global features per recording.

This script extracts classical audio features from each full CirCor recording,
without using the cardiac-phase `.tsv` segmentations. It then generates PCA,
UMAP, k-means diagnostics, and patient-level mean/max aggregations.

Run from the repository root:

    uv run "feature extraction/Grupo A features classicas por gravacao/extract_group_a_classical_features.py"

Fast PCA-only run:

    uv run "feature extraction/Grupo A features classicas por gravacao/extract_group_a_classical_features.py" --skip-umap
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
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import get_window
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


BANDS_HZ = {
    "25_80hz": (25.0, 80.0),
    "80_200hz": (80.0, 200.0),
    "200_400hz": (200.0, 400.0),
    "400_800hz": (400.0, 800.0),
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="Extract Grupo A classical global features from CirCor recordings."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=repo_root / "circor-heart-sound-1.0.3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include Murmur=Unknown. Default excludes it.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Optional quick-test limit.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=2,
        help="Number of k-means clusters.",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP and generate PCA only.",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    original_dtype = audio.dtype
    audio = audio.astype(np.float64)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        audio = audio / max(abs(info.min), info.max)
    else:
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1.0:
            audio = audio / peak

    if len(audio):
        audio = audio - np.mean(audio)
    return sample_rate, audio


def parse_recording_id(path: Path) -> tuple[str, str]:
    match = re.match(r"(?P<patient>\d+)_(?P<location>[A-Za-z]+)(?:_\d+)?$", path.stem)
    if not match:
        raise ValueError(f"Unexpected recording name: {path.name}")
    return match.group("patient"), match.group("location")


def frame_signal(audio: np.ndarray, sample_rate: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_len = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if len(audio) < frame_len:
        padded = np.zeros(frame_len, dtype=np.float64)
        padded[: len(audio)] = audio
        return padded.reshape(1, frame_len)
    n_frames = 1 + (len(audio) - frame_len) // hop_len
    indices = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return audio[indices]


def power_spectrum_frames(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    frames = frame_signal(audio, sample_rate)
    window = get_window("hann", frames.shape[1], fftbins=True)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = (np.abs(spectrum) ** 2) / frames.shape[1]
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    return power, freqs


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft_bins: int,
    n_mels: int = 24,
    low_hz: float = 25.0,
    high_hz: float = 800.0,
) -> np.ndarray:
    high_hz = min(high_hz, sample_rate / 2.0)
    mel_points = np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft_bins - 1) * hz_points / (sample_rate / 2.0)).astype(int)
    filters = np.zeros((n_mels, n_fft_bins), dtype=np.float64)

    for i in range(1, n_mels + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, n_fft_bins - 1)
        for j in range(left, center):
            if 0 <= j < n_fft_bins:
                filters[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            if 0 <= j < n_fft_bins:
                filters[i - 1, j] = (right - j) / max(right - center, 1)

    return filters


def empty_feature_values(n_mfcc: int = 13, n_mels: int = 24) -> dict[str, float]:
    values = {
        "duration_sec": 0.0,
        "rms": 0.0,
        "peak_abs": 0.0,
        "crest_factor": 0.0,
        "zero_crossing_rate": 0.0,
        "clipping_fraction_099": 0.0,
        "energy": 0.0,
        "spectral_centroid": 0.0,
        "spectral_bandwidth": 0.0,
        "spectral_rolloff_85": 0.0,
        "spectral_flatness": 0.0,
        "spectral_entropy": 0.0,
    }
    for band_name in BANDS_HZ:
        values[f"band_energy_{band_name}"] = 0.0
        values[f"band_energy_ratio_{band_name}"] = 0.0
    for idx in range(1, n_mfcc + 1):
        for stat in ("mean", "std", "p10", "p90"):
            values[f"mfcc_{idx}_{stat}"] = 0.0
            values[f"delta_mfcc_{idx}_{stat}"] = 0.0
    for idx in range(1, n_mels + 1):
        for stat in ("mean", "std", "p10", "p90"):
            values[f"logmel_{idx}_{stat}"] = 0.0
    return values


def extract_classical_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if len(audio) < 8 or np.allclose(audio, 0.0):
        return empty_feature_values()

    features = empty_feature_values()
    features["duration_sec"] = float(len(audio) / sample_rate)
    features["rms"] = float(np.sqrt(np.mean(audio**2)))
    features["peak_abs"] = float(np.max(np.abs(audio)))
    features["crest_factor"] = safe_divide(features["peak_abs"], features["rms"])
    features["zero_crossing_rate"] = float(np.count_nonzero(np.diff(np.signbit(audio))) / max(len(audio) - 1, 1))
    features["clipping_fraction_099"] = float(np.mean(np.abs(audio) >= 0.99))
    features["energy"] = float(np.sum(audio**2))

    power, freqs = power_spectrum_frames(audio, sample_rate)
    mean_power = np.maximum(np.mean(power, axis=0), 1e-18)
    total_power = float(np.sum(mean_power))

    centroid = float(np.sum(freqs * mean_power) / total_power)
    features["spectral_centroid"] = centroid
    features["spectral_bandwidth"] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mean_power) / total_power))
    cumulative = np.cumsum(mean_power)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    features["spectral_rolloff_85"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    positive_power = mean_power[mean_power > 0]
    features["spectral_flatness"] = float(np.exp(np.mean(np.log(positive_power))) / np.mean(positive_power))
    probability = mean_power / total_power
    features["spectral_entropy"] = float(-np.sum(probability * np.log2(probability + 1e-18)) / math.log2(len(probability)))

    for band_name, (low_hz, high_hz) in BANDS_HZ.items():
        mask = (freqs >= low_hz) & (freqs < high_hz)
        band_energy = float(np.sum(mean_power[mask]))
        features[f"band_energy_{band_name}"] = band_energy
        features[f"band_energy_ratio_{band_name}"] = safe_divide(band_energy, total_power)

    filters = mel_filterbank(sample_rate, power.shape[1], n_mels=24)
    mel_energy = np.maximum(power @ filters.T, 1e-12)
    log_mel = np.log(mel_energy)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :13]
    delta_mfcc = np.diff(mfcc, axis=0)
    if len(delta_mfcc) == 0:
        delta_mfcc = np.zeros_like(mfcc)

    for idx in range(13):
        mfcc_values = mfcc[:, idx]
        delta_values = delta_mfcc[:, idx]
        for prefix, values in (("mfcc", mfcc_values), ("delta_mfcc", delta_values)):
            features[f"{prefix}_{idx + 1}_mean"] = float(np.mean(values))
            features[f"{prefix}_{idx + 1}_std"] = float(np.std(values))
            features[f"{prefix}_{idx + 1}_p10"] = float(np.percentile(values, 10))
            features[f"{prefix}_{idx + 1}_p90"] = float(np.percentile(values, 90))

    for idx in range(log_mel.shape[1]):
        values = log_mel[:, idx]
        features[f"logmel_{idx + 1}_mean"] = float(np.mean(values))
        features[f"logmel_{idx + 1}_std"] = float(np.std(values))
        features[f"logmel_{idx + 1}_p10"] = float(np.percentile(values, 10))
        features[f"logmel_{idx + 1}_p90"] = float(np.percentile(values, 90))

    return features


def extract_recording(wav_path: Path, metadata: pd.DataFrame) -> dict[str, float | str] | None:
    patient_id, location = parse_recording_id(wav_path)
    row = metadata.loc[metadata["Patient ID"].astype(str) == patient_id]
    if row.empty:
        return None

    sample_rate, audio = read_audio(wav_path)
    features: dict[str, float | str] = {
        "patient_id": patient_id,
        "recording_id": wav_path.stem,
        "location": location,
        "wav_path": str(wav_path),
        "sample_rate": float(sample_rate),
        "murmur": str(row.iloc[0]["Murmur"]),
        "outcome": str(row.iloc[0]["Outcome"]),
        "age": str(row.iloc[0]["Age"]),
        "sex": str(row.iloc[0]["Sex"]),
        "campaign": str(row.iloc[0]["Campaign"]),
    }
    features.update(extract_classical_features(audio, sample_rate))
    return features


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "patient_id",
        "recording_id",
        "location",
        "wav_path",
        "murmur",
        "outcome",
        "age",
        "sex",
        "campaign",
    }
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def project_cluster(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int,
    skip_umap: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if len(df) < 3:
        result = df.copy()
        result["pca_1"] = 0.0
        result["pca_2"] = 0.0
        result["cluster"] = 0
        return result, {"silhouette_pca": float("nan"), "pca12_variance": float("nan")}

    matrix = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = StandardScaler().fit_transform(matrix)

    n_pca = min(10, scaled.shape[0], scaled.shape[1])
    pca_model = PCA(n_components=n_pca, random_state=42)
    pca = pca_model.fit_transform(scaled)

    k = min(n_clusters, len(df))
    clusters = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(pca)
    result = df.copy()
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

    return result, {
        "silhouette_pca": silhouette,
        "pca12_variance": float(np.sum(pca_model.explained_variance_ratio_[:2])),
    }


def projection_diagnostic_metrics(projected: pd.DataFrame) -> dict[str, float | int]:
    ct = pd.crosstab(projected["cluster"], projected["murmur"])
    pca2_rates = projected.groupby(pd.qcut(projected["pca_2"], 5, duplicates="drop"), observed=True)["murmur"].apply(
        lambda values: float((values == "Present").mean())
    )
    metrics: dict[str, float | int] = {
        "cluster_0_rows": int(ct.loc[0].sum()) if 0 in ct.index else 0,
        "cluster_1_rows": int(ct.loc[1].sum()) if 1 in ct.index else 0,
        "cluster_0_present_rate": float(ct.loc[0, "Present"] / ct.loc[0].sum()) if 0 in ct.index and "Present" in ct.columns else float("nan"),
        "cluster_1_present_rate": float(ct.loc[1, "Present"] / ct.loc[1].sum()) if 1 in ct.index and "Present" in ct.columns else float("nan"),
        "pca2_lowest_quintile_present_rate": float(pca2_rates.iloc[0]) if len(pca2_rates) else float("nan"),
        "pca2_highest_quintile_present_rate": float(pca2_rates.iloc[-1]) if len(pca2_rates) else float("nan"),
    }
    return metrics


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
    title: str,
    color_col: str = "murmur",
    marker_col: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"Absent": "#2E86AB", "Present": "#D1495B", "Unknown": "#777777"}
    markers = {"AV": "o", "PV": "^", "TV": "s", "MV": "D", "Phc": "P"}

    if marker_col:
        for marker_value, marker in markers.items():
            marker_subset = df.loc[df[marker_col] == marker_value]
            if marker_subset.empty:
                continue
            for value, subset in marker_subset.groupby(color_col):
                ax.scatter(
                    subset[x],
                    subset[y],
                    s=34,
                    alpha=0.78,
                    linewidths=0.3,
                    edgecolors="white",
                    c=colors.get(str(value), None),
                    marker=marker,
                    label=f"{value} / {marker_value}",
                )
    else:
        for value, subset in df.groupby(color_col):
            ax.scatter(
                subset[x],
                subset[y],
                s=34,
                alpha=0.78,
                linewidths=0.3,
                edgecolors="white",
                c=colors.get(str(value), None),
                label=str(value),
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


def project_by_location(
    recording_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    n_clusters: int,
    skip_umap: bool,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    per_location_dir = output_dir / "by_location"
    per_location_dir.mkdir(parents=True, exist_ok=True)

    for location in sorted(recording_df["location"].dropna().unique()):
        subset = recording_df.loc[recording_df["location"] == location].copy()
        if len(subset) < 8:
            continue
        projected, metrics = project_cluster(subset, feature_columns, n_clusters, skip_umap)
        projected.to_csv(per_location_dir / f"{location}_classical_features_with_projection.csv", index=False)
        scatter(projected, "pca_1", "pca_2", per_location_dir / f"{location}_pca_murmur.png", f"{location}: PCA de features classicas")
        if not skip_umap and "umap_1" in projected.columns:
            scatter(projected, "umap_1", "umap_2", per_location_dir / f"{location}_umap_murmur.png", f"{location}: UMAP de features classicas")
        rows.append(
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
    return rows


def write_summary(
    output_path: Path,
    recording_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    feature_columns: list[str],
    patient_feature_columns: list[str],
    skip_umap: bool,
) -> None:
    lines = [
        "# Grupo A - features classicas por gravacao",
        "",
        "## Objetivo",
        "",
        "Extrair features classicas do audio inteiro de cada gravacao, sem usar segmentacao por fase cardiaca.",
        "",
        "Este experimento serve como baseline global por gravacao para comparar com as features segmentadas do Grupo B.",
        "",
        "## Resumo",
        "",
        f"- Gravacoes analisadas: {len(recording_df)}",
        f"- Pacientes analisados: {recording_df['patient_id'].nunique()}",
        f"- Features classicas por gravacao: {len(feature_columns)}",
        f"- Pacientes agregados: {len(patient_df)}",
        f"- Features agregadas por paciente: {len(patient_feature_columns)}",
        f"- UMAP gerado: {'nao' if skip_umap else 'sim'}",
        "",
        "## Murmur por local",
        "",
        pd.crosstab(recording_df["location"], recording_df["murmur"], margins=True).to_markdown(),
        "",
        "## Metricas por visualizacao",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Interpretacao inicial",
        "",
        "- Se os clusters deste Grupo A forem menos enriquecidos em `Present` do que os da v2 do Grupo B, isso reforca que a segmentacao por fase e as razoes sistolicas sao mais informativas.",
        "- Se o Grupo A separar bem, pode haver informacao global forte no audio inteiro, mas ainda sera necessario auditar volume, ganho, local e qualidade da gravacao.",
        "- Esta etapa nao usa `.tsv`; portanto, ela e mais simples, mas menos alinhada clinicamente ao fato de que a maioria dos sopros positivos e sistolica.",
        "",
        "## Arquivos gerados",
        "",
        "- `recording_classical_features.csv`: features classicas por gravacao.",
        "- `recording_classical_features_with_projection.csv`: PCA/k-means global por gravacao.",
        "- `patient_classical_features.csv`: agregacao por paciente usando media e maximo entre locais.",
        "- `patient_classical_features_with_projection.csv`: PCA/k-means por paciente.",
        "- `by_location/*_pca_murmur.png`: PCA separado por local.",
        "- `by_location/*_umap_murmur.png`: UMAP separado por local, se habilitado.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        row = extract_recording(wav_path, metadata)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if index % 250 == 0:
            print(f"Processed {index}/{len(wav_paths)} recordings...")

    if not rows:
        raise RuntimeError("No rows extracted. Check dataset path and filters.")

    recording_df = pd.DataFrame(rows)
    feature_columns = numeric_feature_columns(recording_df)
    recording_df.to_csv(output_dir / "recording_classical_features.csv", index=False)

    projected_recordings, global_metrics = project_cluster(recording_df, feature_columns, args.n_clusters, args.skip_umap)
    projected_recordings.to_csv(output_dir / "recording_classical_features_with_projection.csv", index=False)
    scatter(
        projected_recordings,
        "pca_1",
        "pca_2",
        output_dir / "recording_pca_murmur_by_location.png",
        "Todas as gravacoes: PCA de features classicas",
        marker_col="location",
    )
    if not args.skip_umap and "umap_1" in projected_recordings.columns:
        scatter(
            projected_recordings,
            "umap_1",
            "umap_2",
            output_dir / "recording_umap_murmur_by_location.png",
            "Todas as gravacoes: UMAP de features classicas",
            marker_col="location",
        )

    metrics_rows: list[dict[str, float | str | int]] = [
        {
            "level": "recording_global",
            "location": "all",
            "rows": int(len(projected_recordings)),
            "patients": int(projected_recordings["patient_id"].nunique()),
            "present_rate": float((projected_recordings["murmur"] == "Present").mean()),
            **projection_diagnostic_metrics(projected_recordings),
            **global_metrics,
        }
    ]
    metrics_rows.extend(project_by_location(recording_df, feature_columns, output_dir, args.n_clusters, args.skip_umap))

    patient_df = aggregate_by_patient(recording_df, feature_columns)
    patient_feature_columns = numeric_feature_columns(patient_df)
    patient_df.to_csv(output_dir / "patient_classical_features.csv", index=False)
    projected_patients, patient_metrics = project_cluster(patient_df, patient_feature_columns, args.n_clusters, args.skip_umap)
    projected_patients.to_csv(output_dir / "patient_classical_features_with_projection.csv", index=False)
    scatter(projected_patients, "pca_1", "pca_2", output_dir / "patient_pca_murmur.png", "Pacientes agregados: PCA de features classicas")
    if not args.skip_umap and "umap_1" in projected_patients.columns:
        scatter(projected_patients, "umap_1", "umap_2", output_dir / "patient_umap_murmur.png", "Pacientes agregados: UMAP de features classicas")

    metrics_rows.append(
        {
            "level": "patient_aggregated",
            "location": "mean_max",
            "rows": int(len(projected_patients)),
            "patients": int(len(projected_patients)),
            "present_rate": float((projected_patients["murmur"] == "Present").mean()),
            **projection_diagnostic_metrics(projected_patients),
            **patient_metrics,
        }
    )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "projection_metrics.csv", index=False)
    write_summary(
        output_dir / "summary.md",
        recording_df,
        patient_df,
        metrics_df,
        feature_columns,
        patient_feature_columns,
        args.skip_umap,
    )

    print(f"Extracted {len(recording_df)} recordings. Skipped {skipped}.")
    print(f"Recording feature columns: {len(feature_columns)}")
    print(f"Patient feature columns: {len(patient_feature_columns)}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
