# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.4",
#   "scipy>=1.12",
#   "tabulate>=0.9",
# ]
# ///
"""Extract segmented cardiac-cycle features and visualize murmur clusters.

Run from the repository root:

    uv run "feature extraction/Grupo B features segmentadas por fase cardiaca/extract_group_b_features.py"

Outputs are written to:

    feature extraction/Grupo B features segmentadas por fase cardiaca/outputs/
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import butter, get_window, sosfiltfilt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PHASES = {
    1: "s1",
    2: "systole",
    3: "s2",
    4: "diastole",
}

BANDS_HZ = {
    "25_80hz": (25.0, 80.0),
    "80_200hz": (80.0, 200.0),
    "200_400hz": (200.0, 400.0),
    "400_800hz": (400.0, 800.0),
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    default_dataset = repo_root / "circor-heart-sound-1.0.3"
    default_output = script_dir / "outputs"

    parser = argparse.ArgumentParser(
        description=(
            "Extract Grupo B cardiac-phase features from CirCor audio, "
            "cluster recordings, and generate distance visualizations."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include Murmur=Unknown in features and plots. Default excludes it.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=2,
        help="Number of k-means clusters. Default: 2.",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE plot. PCA and k-means still run.",
    )
    return parser.parse_args()


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


def safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


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


def bandpass_filter(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = sample_rate / 2.0
    low = max(low_hz / nyquist, 1e-5)
    high = min(high_hz / nyquist, 0.999)
    if high <= low or len(audio) < 32:
        return np.array([], dtype=np.float64)
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, audio)
    except ValueError:
        return np.array([], dtype=np.float64)


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
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, axis=1)
    power = (np.abs(spectrum) ** 2) / frames.shape[1]
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    return power, freqs


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft_bins: int, n_mels: int = 20, low_hz: float = 25.0, high_hz: float = 800.0) -> np.ndarray:
    high_hz = min(high_hz, sample_rate / 2.0)
    mel_points = np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft_bins - 1) * hz_points / (sample_rate / 2.0)).astype(int)
    filters = np.zeros((n_mels, n_fft_bins), dtype=np.float64)

    for i in range(1, n_mels + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_fft_bins - 1)

        for j in range(left, center):
            if 0 <= j < n_fft_bins:
                filters[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            if 0 <= j < n_fft_bins:
                filters[i - 1, j] = (right - j) / max(right - center, 1)

    return filters


def mfcc_stats(audio: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> dict[str, float]:
    if len(audio) < 8 or np.allclose(audio, 0.0):
        return {f"mfcc_{i}_{stat}": 0.0 for i in range(1, n_mfcc + 1) for stat in ("mean", "std", "p10", "p90")}

    power, _ = power_spectrum_frames(audio, sample_rate)
    filters = mel_filterbank(sample_rate, power.shape[1], n_mels=24)
    mel_energy = np.maximum(power @ filters.T, 1e-12)
    log_mel = np.log(mel_energy)
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]

    features: dict[str, float] = {}
    for idx in range(n_mfcc):
        values = coeffs[:, idx]
        prefix = f"mfcc_{idx + 1}"
        features[f"{prefix}_mean"] = float(np.mean(values))
        features[f"{prefix}_std"] = float(np.std(values))
        features[f"{prefix}_p10"] = float(np.percentile(values, 10))
        features[f"{prefix}_p90"] = float(np.percentile(values, 90))
    return features


def spectral_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if len(audio) < 8 or np.allclose(audio, 0.0):
        return {
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_rolloff_85": 0.0,
            "spectral_flatness": 0.0,
            "spectral_entropy": 0.0,
        }

    power, freqs = power_spectrum_frames(audio, sample_rate)
    mean_power = np.maximum(np.mean(power, axis=0), 1e-18)
    total_power = float(np.sum(mean_power))
    centroid = float(np.sum(freqs * mean_power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mean_power) / total_power))

    cumulative = np.cumsum(mean_power)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    positive_power = mean_power[mean_power > 0]
    flatness = float(np.exp(np.mean(np.log(positive_power))) / np.mean(positive_power))

    probability = mean_power / total_power
    entropy = float(-np.sum(probability * np.log2(probability + 1e-18)) / math.log2(len(probability)))

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff_85": rolloff,
        "spectral_flatness": flatness,
        "spectral_entropy": entropy,
    }


def basic_audio_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if len(audio) == 0:
        return {
            "duration_sec": 0.0,
            "rms": 0.0,
            "peak_abs": 0.0,
            "crest_factor": 0.0,
            "zero_crossing_rate": 0.0,
            "energy": 0.0,
        }

    rms = float(np.sqrt(np.mean(audio**2)))
    peak_abs = float(np.max(np.abs(audio)))
    zero_crossings = np.count_nonzero(np.diff(np.signbit(audio)))
    return {
        "duration_sec": float(len(audio) / sample_rate),
        "rms": rms,
        "peak_abs": peak_abs,
        "crest_factor": safe_divide(peak_abs, rms),
        "zero_crossing_rate": float(zero_crossings / max(len(audio) - 1, 1)),
        "energy": float(np.sum(audio**2)),
    }


def band_energy_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    total_energy = float(np.sum(audio**2)) if len(audio) else 0.0
    features: dict[str, float] = {}

    for band_name, (low_hz, high_hz) in BANDS_HZ.items():
        filtered = bandpass_filter(audio, sample_rate, low_hz, high_hz)
        energy = float(np.sum(filtered**2)) if len(filtered) else 0.0
        features[f"band_energy_{band_name}"] = energy
        features[f"band_energy_ratio_{band_name}"] = safe_divide(energy, total_energy)

    return features


def phase_duration_features(segments: pd.DataFrame, label: int) -> dict[str, float]:
    durations = (segments.loc[segments["label"] == label, "end_time"] - segments.loc[segments["label"] == label, "start_time"]).to_numpy()
    if len(durations) == 0:
        return {
            "segment_count": 0.0,
            "segment_duration_mean": 0.0,
            "segment_duration_std": 0.0,
            "segment_duration_p10": 0.0,
            "segment_duration_p90": 0.0,
        }
    return {
        "segment_count": float(len(durations)),
        "segment_duration_mean": float(np.mean(durations)),
        "segment_duration_std": float(np.std(durations)),
        "segment_duration_p10": float(np.percentile(durations, 10)),
        "segment_duration_p90": float(np.percentile(durations, 90)),
    }


def extract_phase_features(audio: np.ndarray, sample_rate: int, segments: pd.DataFrame, label: int, phase: str) -> dict[str, float]:
    phase_audio = segment_audio(audio, sample_rate, segments, label)

    features: dict[str, float] = {}
    phase_extractors = (
        phase_duration_features(segments, label),
        basic_audio_features(phase_audio, sample_rate),
        band_energy_features(phase_audio, sample_rate),
        spectral_features(phase_audio, sample_rate),
        mfcc_stats(phase_audio, sample_rate),
    )

    for feature_group in phase_extractors:
        for key, value in feature_group.items():
            features[f"{phase}_{key}"] = value

    return features


def add_cross_phase_features(features: dict[str, float]) -> None:
    s1_energy = features.get("s1_energy", 0.0)
    systole_energy = features.get("systole_energy", 0.0)
    s2_energy = features.get("s2_energy", 0.0)
    diastole_energy = features.get("diastole_energy", 0.0)
    heart_sound_energy = s1_energy + s2_energy

    features["ratio_systole_energy_to_s1_s2"] = safe_divide(systole_energy, heart_sound_energy)
    features["ratio_diastole_energy_to_s1_s2"] = safe_divide(diastole_energy, heart_sound_energy)
    features["ratio_systole_to_diastole_energy"] = safe_divide(systole_energy, diastole_energy)

    for band_name in BANDS_HZ:
        systole_band = features.get(f"systole_band_energy_{band_name}", 0.0)
        s1_band = features.get(f"s1_band_energy_{band_name}", 0.0)
        s2_band = features.get(f"s2_band_energy_{band_name}", 0.0)
        diastole_band = features.get(f"diastole_band_energy_{band_name}", 0.0)
        features[f"ratio_systole_{band_name}_to_s1_s2"] = safe_divide(systole_band, s1_band + s2_band)
        features[f"ratio_systole_{band_name}_to_diastole"] = safe_divide(systole_band, diastole_band)

    systole_high = features.get("systole_band_energy_200_400hz", 0.0) + features.get("systole_band_energy_400_800hz", 0.0)
    features["ratio_systole_high_freq_to_systole_energy"] = safe_divide(systole_high, systole_energy)

    features["delta_systole_diastole_centroid"] = (
        features.get("systole_spectral_centroid", 0.0) - features.get("diastole_spectral_centroid", 0.0)
    )
    features["delta_systole_diastole_entropy"] = (
        features.get("systole_spectral_entropy", 0.0) - features.get("diastole_spectral_entropy", 0.0)
    )


def extract_recording_features(wav_path: Path, metadata: pd.DataFrame) -> dict[str, float | str] | None:
    tsv_path = wav_path.with_suffix(".tsv")
    if not tsv_path.exists():
        return None

    patient_id, location = parse_recording_id(wav_path)
    row = metadata.loc[metadata["Patient ID"].astype(str) == patient_id]
    if row.empty:
        return None

    sample_rate, audio = read_audio(wav_path)
    segments = read_segments(tsv_path)

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
        "audio_duration_sec": float(len(audio) / sample_rate),
    }

    for label, phase in PHASES.items():
        features.update(extract_phase_features(audio, sample_rate, segments, label, phase))

    add_cross_phase_features(features)  # type: ignore[arg-type]
    return features


def numeric_feature_columns(features_df: pd.DataFrame) -> list[str]:
    excluded = {"patient_id", "recording_id", "location", "wav_path", "murmur", "outcome", "age", "sex", "campaign"}
    return [
        column
        for column in features_df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(features_df[column])
    ]


def fit_projection_and_clusters(features_df: pd.DataFrame, feature_columns: list[str], n_clusters: int) -> tuple[pd.DataFrame, np.ndarray]:
    matrix = features_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = StandardScaler().fit_transform(matrix)

    n_pca = min(10, scaled.shape[0], scaled.shape[1])
    pca_model = PCA(n_components=n_pca, random_state=42)
    pca = pca_model.fit_transform(scaled)

    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
    clusters = kmeans.fit_predict(pca)

    result = features_df.copy()
    result["cluster"] = clusters
    result["pca_1"] = pca[:, 0]
    result["pca_2"] = pca[:, 1] if pca.shape[1] > 1 else 0.0

    if len(np.unique(clusters)) > 1 and len(result) > n_clusters:
        result.attrs["silhouette_pca"] = float(silhouette_score(pca, clusters))
    else:
        result.attrs["silhouette_pca"] = float("nan")

    explained = np.zeros(2)
    explained[: min(2, len(pca_model.explained_variance_ratio_))] = pca_model.explained_variance_ratio_[:2]
    result.attrs["pca_explained_first_two"] = explained
    return result, pca


def add_tsne_projection(result: pd.DataFrame, pca: np.ndarray) -> pd.DataFrame:
    if len(result) < 5:
        result["tsne_1"] = 0.0
        result["tsne_2"] = 0.0
        return result

    perplexity = min(30, max(2, (len(result) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    coords = tsne.fit_transform(pca)
    result = result.copy()
    result["tsne_1"] = coords[:, 0]
    result["tsne_2"] = coords[:, 1]
    return result


def scatter_by_label(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    output_path: Path,
    title: str,
    marker_col: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {
        "Absent": "#2E86AB",
        "Present": "#D1495B",
        "Unknown": "#777777",
    }
    markers = {
        "AV": "o",
        "PV": "^",
        "TV": "s",
        "MV": "D",
        "Phc": "P",
    }

    if marker_col:
        for marker_value, marker in markers.items():
            subset = df.loc[df[marker_col] == marker_value]
            if subset.empty:
                continue
            for label_value, label_subset in subset.groupby(color_col):
                ax.scatter(
                    label_subset[x_col],
                    label_subset[y_col],
                    c=colors.get(str(label_value), "#444444"),
                    marker=marker,
                    s=34,
                    alpha=0.78,
                    linewidths=0.3,
                    edgecolors="white",
                    label=f"{label_value} / {marker_value}",
                )
    else:
        for label_value, subset in df.groupby(color_col):
            ax.scatter(
                subset[x_col],
                subset[y_col],
                s=34,
                alpha=0.78,
                linewidths=0.3,
                edgecolors="white",
                label=str(label_value),
            )

    ax.axhline(0, color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(df: pd.DataFrame, output_path: Path) -> None:
    cluster_table = pd.crosstab(df["cluster"], df["murmur"], margins=True)
    location_table = pd.crosstab(df["location"], df["murmur"], margins=True)
    silhouette = df.attrs.get("silhouette_pca", float("nan"))
    pca_explained = df.attrs.get("pca_explained_first_two", np.array([np.nan, np.nan]))

    lines = [
        "# Grupo B - features segmentadas por fase cardiaca",
        "",
        f"Gravacoes analisadas: {len(df)}",
        f"Pacientes analisados: {df['patient_id'].nunique()}",
        f"Clusters k-means: {df['cluster'].nunique()}",
        f"Silhouette em PCA: {silhouette:.4f}" if np.isfinite(silhouette) else "Silhouette em PCA: n/a",
        f"Variancia explicada PCA1+PCA2: {float(np.sum(pca_explained)):.4f}",
        "",
        "## Murmur por cluster",
        "",
        cluster_table.to_markdown(),
        "",
        "## Murmur por local de ausculta",
        "",
        location_table.to_markdown(),
        "",
        "## Arquivos gerados",
        "",
        "- `recording_phase_features.csv`: features extraidas por gravacao.",
        "- `recording_phase_features_with_clusters.csv`: features com PCA, t-SNE opcional e cluster.",
        "- `pca_murmur_by_location.png`: PCA colorido por Murmur e marcado por local.",
        "- `pca_clusters.png`: PCA colorido por cluster.",
        "- `tsne_murmur_by_location.png`: t-SNE colorido por Murmur e marcado por local, se habilitado.",
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
        features = extract_recording_features(wav_path, metadata)
        if features is None:
            skipped += 1
            continue
        rows.append(features)
        if index % 250 == 0:
            print(f"Processed {index}/{len(wav_paths)} recordings...")

    if not rows:
        raise RuntimeError("No features were extracted. Check dataset path and labels.")

    features_df = pd.DataFrame(rows)
    feature_columns = numeric_feature_columns(features_df)
    features_csv = output_dir / "recording_phase_features.csv"
    features_df.to_csv(features_csv, index=False)

    clustered_df, pca = fit_projection_and_clusters(features_df, feature_columns, args.n_clusters)
    if not args.skip_tsne:
        clustered_df = add_tsne_projection(clustered_df, pca)

    clustered_csv = output_dir / "recording_phase_features_with_clusters.csv"
    clustered_df.to_csv(clustered_csv, index=False)

    scatter_by_label(
        clustered_df,
        "pca_1",
        "pca_2",
        "murmur",
        output_dir / "pca_murmur_by_location.png",
        "PCA das features por fase: Murmur e local",
        marker_col="location",
    )
    scatter_by_label(
        clustered_df,
        "pca_1",
        "pca_2",
        "cluster",
        output_dir / "pca_clusters.png",
        "PCA das features por fase: k-means clusters",
    )

    if not args.skip_tsne:
        scatter_by_label(
            clustered_df,
            "tsne_1",
            "tsne_2",
            "murmur",
            output_dir / "tsne_murmur_by_location.png",
            "t-SNE das features por fase: Murmur e local",
            marker_col="location",
        )

    write_summary(clustered_df, output_dir / "summary.md")

    print(f"Extracted {len(features_df)} recordings. Skipped {skipped}.")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
