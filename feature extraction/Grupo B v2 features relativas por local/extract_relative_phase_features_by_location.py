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
"""Second Grupo B experiment: relative phase features by auscultation location.

This experiment intentionally avoids absolute volume/energy features in the
final feature matrix. It keeps ratios and systole-vs-diastole deltas, excludes
RMS, peak, raw energy, and MFCC_1, then generates PCA/UMAP views separately for
AV, PV, TV, and MV. It also creates patient-level mean/max aggregations.

Run from the repository root:

    uv run "feature extraction/Grupo B v2 features relativas por local/extract_relative_phase_features_by_location.py"

Outputs:

    feature extraction/Grupo B v2 features relativas por local/outputs/
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


PHASE_LABELS = {
    1: "s1",
    2: "systole",
    3: "s2",
    4: "diastole",
}

LOCATIONS = ["AV", "PV", "TV", "MV"]

BANDS_HZ = {
    "25_80hz": (25.0, 80.0),
    "80_200hz": (80.0, 200.0),
    "200_400hz": (200.0, 400.0),
    "400_800hz": (400.0, 800.0),
}

SPECTRAL_KEYS = [
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff_85",
    "spectral_flatness",
    "spectral_entropy",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="Extract relative phase features and generate per-location PCA/UMAP plots."
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
        help="K-means clusters for diagnostic plots.",
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
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak
    audio = audio - np.mean(audio)
    return sample_rate, audio


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


def segment_audio(audio: np.ndarray, sample_rate: int, segments: pd.DataFrame, label: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_samples = len(audio)
    for row in segments.loc[segments["label"] == label].itertuples(index=False):
        start = max(0, min(n_samples, int(round(row.start_time * sample_rate))))
        end = max(0, min(n_samples, int(round(row.end_time * sample_rate))))
        if end > start:
            chunks.append(audio[start:end])
    if not chunks:
        return np.array([], dtype=np.float64)
    return np.concatenate(chunks)


def phase_durations(segments: pd.DataFrame, label: int) -> np.ndarray:
    rows = segments.loc[segments["label"] == label]
    return (rows["end_time"] - rows["start_time"]).to_numpy(dtype=np.float64)


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


def mel_filterbank(sample_rate: int, n_fft_bins: int, n_mels: int = 24, low_hz: float = 25.0, high_hz: float = 800.0) -> np.ndarray:
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


def zero_measurements() -> dict[str, float]:
    values = {
        "duration_sec": 0.0,
        "segment_count": 0.0,
        "segment_duration_mean": 0.0,
        "segment_duration_std": 0.0,
        "zero_crossing_rate": 0.0,
        "spectral_centroid": 0.0,
        "spectral_bandwidth": 0.0,
        "spectral_rolloff_85": 0.0,
        "spectral_flatness": 0.0,
        "spectral_entropy": 0.0,
        "energy_internal": 0.0,
    }
    for band_name in BANDS_HZ:
        values[f"band_share_{band_name}"] = 0.0
        values[f"band_energy_internal_{band_name}"] = 0.0
    for idx in range(2, 14):
        for stat in ("mean", "std", "p10", "p90"):
            values[f"mfcc_{idx}_{stat}"] = 0.0
    return values


def phase_measurements(audio: np.ndarray, sample_rate: int, durations: np.ndarray) -> dict[str, float]:
    if len(audio) < 8 or np.allclose(audio, 0.0):
        values = zero_measurements()
        if len(durations):
            values["duration_sec"] = float(np.sum(durations))
            values["segment_count"] = float(len(durations))
            values["segment_duration_mean"] = float(np.mean(durations))
            values["segment_duration_std"] = float(np.std(durations))
        return values

    values = zero_measurements()
    total_energy = float(np.sum(audio**2))
    values["duration_sec"] = float(len(audio) / sample_rate)
    values["segment_count"] = float(len(durations))
    values["segment_duration_mean"] = float(np.mean(durations)) if len(durations) else 0.0
    values["segment_duration_std"] = float(np.std(durations)) if len(durations) else 0.0
    values["energy_internal"] = total_energy
    values["zero_crossing_rate"] = float(np.count_nonzero(np.diff(np.signbit(audio))) / max(len(audio) - 1, 1))

    power, freqs = power_spectrum_frames(audio, sample_rate)
    mean_power = np.maximum(np.mean(power, axis=0), 1e-18)
    total_power = float(np.sum(mean_power))
    centroid = float(np.sum(freqs * mean_power) / total_power)
    values["spectral_centroid"] = centroid
    values["spectral_bandwidth"] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mean_power) / total_power))
    cumulative = np.cumsum(mean_power)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    values["spectral_rolloff_85"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    positive_power = mean_power[mean_power > 0]
    values["spectral_flatness"] = float(np.exp(np.mean(np.log(positive_power))) / np.mean(positive_power))
    probability = mean_power / total_power
    values["spectral_entropy"] = float(-np.sum(probability * np.log2(probability + 1e-18)) / math.log2(len(probability)))

    for band_name, (low_hz, high_hz) in BANDS_HZ.items():
        mask = (freqs >= low_hz) & (freqs < high_hz)
        band_energy = float(np.sum(mean_power[mask]))
        values[f"band_energy_internal_{band_name}"] = band_energy
        values[f"band_share_{band_name}"] = safe_divide(band_energy, total_power)

    filters = mel_filterbank(sample_rate, power.shape[1], n_mels=24)
    mel_energy = np.maximum(power @ filters.T, 1e-12)
    log_mel = np.log(mel_energy)
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :13]
    for idx in range(2, 14):
        vals = coeffs[:, idx - 1]
        values[f"mfcc_{idx}_mean"] = float(np.mean(vals))
        values[f"mfcc_{idx}_std"] = float(np.std(vals))
        values[f"mfcc_{idx}_p10"] = float(np.percentile(vals, 10))
        values[f"mfcc_{idx}_p90"] = float(np.percentile(vals, 90))

    return values


def build_relative_features(measurements: dict[str, dict[str, float]]) -> dict[str, float]:
    s1 = measurements["s1"]
    systole = measurements["systole"]
    s2 = measurements["s2"]
    diastole = measurements["diastole"]
    out: dict[str, float] = {}

    s1_s2_energy = s1["energy_internal"] + s2["energy_internal"]
    out["ratio_systole_energy_to_s1_s2"] = safe_divide(systole["energy_internal"], s1_s2_energy)
    out["ratio_diastole_energy_to_s1_s2"] = safe_divide(diastole["energy_internal"], s1_s2_energy)
    out["ratio_systole_energy_to_diastole"] = safe_divide(systole["energy_internal"], diastole["energy_internal"])

    s1_s2_duration = s1["duration_sec"] + s2["duration_sec"]
    out["ratio_systole_duration_to_s1_s2"] = safe_divide(systole["duration_sec"], s1_s2_duration)
    out["ratio_diastole_duration_to_s1_s2"] = safe_divide(diastole["duration_sec"], s1_s2_duration)
    out["ratio_systole_duration_to_diastole"] = safe_divide(systole["duration_sec"], diastole["duration_sec"])

    systole_high = systole["band_energy_internal_200_400hz"] + systole["band_energy_internal_400_800hz"]
    out["ratio_systole_high_freq_to_systole_energy"] = safe_divide(systole_high, systole["energy_internal"])

    for band_name in BANDS_HZ:
        systole_band = systole[f"band_energy_internal_{band_name}"]
        s1_s2_band = s1[f"band_energy_internal_{band_name}"] + s2[f"band_energy_internal_{band_name}"]
        diastole_band = diastole[f"band_energy_internal_{band_name}"]
        out[f"ratio_systole_{band_name}_to_s1_s2"] = safe_divide(systole_band, s1_s2_band)
        out[f"ratio_systole_{band_name}_to_diastole"] = safe_divide(systole_band, diastole_band)
        out[f"systole_band_share_{band_name}"] = systole[f"band_share_{band_name}"]
        out[f"diastole_band_share_{band_name}"] = diastole[f"band_share_{band_name}"]
        out[f"delta_systole_diastole_band_share_{band_name}"] = (
            systole[f"band_share_{band_name}"] - diastole[f"band_share_{band_name}"]
        )

    for key in SPECTRAL_KEYS:
        out[f"delta_systole_diastole_{key}"] = systole[key] - diastole[key]
        out[f"delta_systole_s1_{key}"] = systole[key] - s1[key]
        out[f"delta_systole_s2_{key}"] = systole[key] - s2[key]

    for idx in range(2, 14):
        for stat in ("mean", "std", "p10", "p90"):
            key = f"mfcc_{idx}_{stat}"
            out[f"delta_systole_diastole_{key}"] = systole[key] - diastole[key]
            out[f"delta_systole_s1_{key}"] = systole[key] - s1[key]
            out[f"delta_systole_s2_{key}"] = systole[key] - s2[key]

    out["delta_systole_diastole_segment_duration_mean"] = (
        systole["segment_duration_mean"] - diastole["segment_duration_mean"]
    )
    out["delta_systole_diastole_segment_duration_std"] = (
        systole["segment_duration_std"] - diastole["segment_duration_std"]
    )
    return out


def extract_recording(wav_path: Path, metadata: pd.DataFrame) -> dict[str, float | str] | None:
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
    segments = read_segments(tsv_path)

    measurements: dict[str, dict[str, float]] = {}
    for label, phase in PHASE_LABELS.items():
        phase_audio = segment_audio(audio, sample_rate, segments, label)
        durations = phase_durations(segments, label)
        measurements[phase] = phase_measurements(phase_audio, sample_rate, durations)

    features: dict[str, float | str] = {
        "patient_id": patient_id,
        "recording_id": wav_path.stem,
        "location": location,
        "wav_path": str(wav_path),
        "murmur": str(row.iloc[0]["Murmur"]),
        "outcome": str(row.iloc[0]["Outcome"]),
        "age": str(row.iloc[0]["Age"]),
        "sex": str(row.iloc[0]["Sex"]),
        "campaign": str(row.iloc[0]["Campaign"]),
    }
    features.update(build_relative_features(measurements))
    return features


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"patient_id", "recording_id", "location", "wav_path", "murmur", "outcome", "age", "sex", "campaign"}
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def project_cluster(df: pd.DataFrame, feature_columns: list[str], n_clusters: int, skip_umap: bool) -> tuple[pd.DataFrame, dict[str, float]]:
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
    metrics = {
        "silhouette_pca": silhouette,
        "pca12_variance": float(np.sum(pca_model.explained_variance_ratio_[:2])),
    }
    return result, metrics


def scatter(df: pd.DataFrame, x: str, y: str, output_path: Path, title: str, color_col: str = "murmur") -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"Absent": "#2E86AB", "Present": "#D1495B", "Unknown": "#777777"}
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


def projection_diagnostic_metrics(projected: pd.DataFrame) -> dict[str, float | int]:
    ct = pd.crosstab(projected["cluster"], projected["murmur"])
    present_rates = projected.groupby(pd.qcut(projected["pca_2"], 5, duplicates="drop"), observed=True)["murmur"].apply(
        lambda values: float((values == "Present").mean())
    )

    metrics: dict[str, float | int] = {
        "cluster_0_rows": int(ct.loc[0].sum()) if 0 in ct.index else 0,
        "cluster_1_rows": int(ct.loc[1].sum()) if 1 in ct.index else 0,
        "cluster_0_present_rate": float(ct.loc[0, "Present"] / ct.loc[0].sum()) if 0 in ct.index and "Present" in ct.columns else float("nan"),
        "cluster_1_present_rate": float(ct.loc[1, "Present"] / ct.loc[1].sum()) if 1 in ct.index and "Present" in ct.columns else float("nan"),
        "pca2_lowest_quintile_present_rate": float(present_rates.iloc[0]) if len(present_rates) else float("nan"),
        "pca2_highest_quintile_present_rate": float(present_rates.iloc[-1]) if len(present_rates) else float("nan"),
    }
    return metrics


def plot_per_location(recording_df: pd.DataFrame, feature_columns: list[str], output_dir: Path, n_clusters: int, skip_umap: bool) -> list[dict[str, float | str | int]]:
    metrics_rows: list[dict[str, float | str | int]] = []
    per_location_dir = output_dir / "by_location"
    per_location_dir.mkdir(parents=True, exist_ok=True)

    for location in LOCATIONS:
        subset = recording_df.loc[recording_df["location"] == location].copy()
        if subset.empty:
            continue
        projected, metrics = project_cluster(subset, feature_columns, n_clusters, skip_umap)
        projected.to_csv(per_location_dir / f"{location}_relative_features_with_projection.csv", index=False)
        scatter(projected, "pca_1", "pca_2", per_location_dir / f"{location}_pca_murmur.png", f"{location}: PCA de features relativas")
        if not skip_umap and "umap_1" in projected.columns:
            scatter(projected, "umap_1", "umap_2", per_location_dir / f"{location}_umap_murmur.png", f"{location}: UMAP de features relativas")
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
    return metrics_rows


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
        "# Grupo B v2 - features relativas por local",
        "",
        "## Objetivo",
        "",
        "Esta segunda versao remove features diretamente ligadas a volume absoluto e mantem razoes/deltas entre fases do ciclo cardiaco.",
        "",
        "Foram excluidos da matriz final: RMS, peak, energia absoluta e MFCC_1.",
        "",
        "## Resumo",
        "",
        f"- Gravacoes analisadas: {len(recording_df)}",
        f"- Pacientes analisados: {recording_df['patient_id'].nunique()}",
        f"- Features relativas por gravacao: {len(feature_columns)}",
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
        "- Esta versao deve ser comparada contra o experimento anterior para verificar se a separacao por `pca_2` permanece sem RMS, peak, energia absoluta e MFCC_1.",
        "- A interpretacao principal deve ser feita por local (`AV`, `PV`, `TV`, `MV`), porque o primeiro experimento mostrou forte efeito de local no PCA.",
        "- Se a taxa de `Present` continuar subindo nos maiores valores de `pca_2` dentro de um local, isso reforca que existe sinal de sopro nas features relativas.",
        "- Se a diferenca desaparecer, a separacao anterior provavelmente dependia demais de volume/ganho/energia absoluta.",
        "",
        "## Arquivos gerados",
        "",
        "- `recording_relative_phase_features.csv`: features relativas por gravacao.",
        "- `recording_relative_phase_features_with_projection.csv`: PCA/k-means global por gravacao.",
        "- `patient_relative_phase_features.csv`: agregacao por paciente usando media e maximo entre locais.",
        "- `patient_relative_phase_features_with_projection.csv`: PCA/k-means global por paciente.",
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
    recording_df.to_csv(output_dir / "recording_relative_phase_features.csv", index=False)

    projected_recordings, global_metrics = project_cluster(recording_df, feature_columns, args.n_clusters, args.skip_umap)
    projected_recordings.to_csv(output_dir / "recording_relative_phase_features_with_projection.csv", index=False)
    scatter(projected_recordings, "pca_1", "pca_2", output_dir / "recording_pca_murmur.png", "Todas as gravacoes: PCA de features relativas")
    if not args.skip_umap and "umap_1" in projected_recordings.columns:
        scatter(projected_recordings, "umap_1", "umap_2", output_dir / "recording_umap_murmur.png", "Todas as gravacoes: UMAP de features relativas")

    metrics_rows = [
        {
            "level": "recording_global",
            "location": "all",
            "rows": int(len(recording_df)),
            "patients": int(recording_df["patient_id"].nunique()),
            "present_rate": float((recording_df["murmur"] == "Present").mean()),
            **projection_diagnostic_metrics(projected_recordings),
            **global_metrics,
        }
    ]
    metrics_rows.extend(plot_per_location(recording_df, feature_columns, output_dir, args.n_clusters, args.skip_umap))

    patient_df = aggregate_by_patient(recording_df, feature_columns)
    patient_feature_columns = numeric_feature_columns(patient_df)
    patient_df.to_csv(output_dir / "patient_relative_phase_features.csv", index=False)
    projected_patients, patient_metrics = project_cluster(patient_df, patient_feature_columns, args.n_clusters, args.skip_umap)
    projected_patients.to_csv(output_dir / "patient_relative_phase_features_with_projection.csv", index=False)
    scatter(projected_patients, "pca_1", "pca_2", output_dir / "patient_pca_murmur.png", "Pacientes agregados: PCA de features relativas")
    if not args.skip_umap and "umap_1" in projected_patients.columns:
        scatter(projected_patients, "umap_1", "umap_2", output_dir / "patient_umap_murmur.png", "Pacientes agregados: UMAP de features relativas")

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
