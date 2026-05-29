# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "panns-inference>=0.1.1",
#   "scikit-learn>=1.4",
#   "scipy>=1.12",
#   "tabulate>=0.9",
#   "torch>=2.2",
#   "umap-learn>=0.5.7",
# ]
# ///
"""Grupo C2 experiment: PANNs embeddings by cardiac phase.

This is the PANNs analogue of Grupo B v2. It uses `.tsv` cardiac-phase
segmentations to extract frozen PANNs/Cnn14 embeddings separately for S1,
systole, S2, and diastole. The final feature matrix focuses on systole-vs-other
phase deltas, absolute deltas, cosine distances, and norm ratios.

Run from the repository root:

    uv run "feature extraction/Grupo C2 PANNs embeddings por fase cardiaca/extract_panns_phase_embeddings.py"

Quick test:

    uv run "feature extraction/Grupo C2 PANNs embeddings por fase cardiaca/extract_panns_phase_embeddings.py" --max-recordings 20 --skip-umap --batch-size 1

Recommended first full run on MacBook M3 Pro:

    uv run "feature extraction/Grupo C2 PANNs embeddings por fase cardiaca/extract_panns_phase_embeddings.py" --skip-umap --batch-size 1
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PANNS_SAMPLE_RATE = 32000
EMBEDDING_SIZE = 2048
PHASE_LABELS = {
    1: "s1",
    2: "systole",
    3: "s2",
    4: "diastole",
}
LOCATIONS = ["AV", "PV", "TV", "MV"]

LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description="Extract frozen PANNs embeddings by cardiac phase.")
    parser.add_argument("--dataset-dir", type=Path, default=repo_root / "circor-heart-sound-1.0.3")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs")
    parser.add_argument("--panns-dir", type=Path, default=Path.home() / "panns_data")
    parser.add_argument("--include-unknown", action="store_true", help="Include Murmur=Unknown. Default excludes it.")
    parser.add_argument("--max-recordings", type=int, default=None, help="Optional quick-test limit.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="cpu",
        help="Default is cpu for stability. Use auto or mps to try Apple Silicon MPS.",
    )
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument(
        "--reuse-recording-features",
        action="store_true",
        help="Reuse recording_panns_phase_features.csv if present.",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def download_file(url: str, target: Path, min_size_bytes: int = 1) -> None:
    if target.exists() and target.stat().st_size >= min_size_bytes:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url} -> {target}", flush=True)
    with urllib.request.urlopen(url) as response, tmp_target.open("wb") as out:
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100.0
                print(f"\r  {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
        if total:
            print(flush=True)
    tmp_target.replace(target)


def ensure_panns_files(panns_dir: Path) -> tuple[Path, Path]:
    labels_path = panns_dir / "class_labels_indices.csv"
    checkpoint_path = panns_dir / "Cnn14_mAP=0.431.pth"
    download_file(LABELS_URL, labels_path, min_size_bytes=1000)
    download_file(CHECKPOINT_URL, checkpoint_path, min_size_bytes=300_000_000)
    return labels_path, checkpoint_path


def load_class_count(labels_path: Path) -> int:
    with labels_path.open("r", newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "mps":
        print("MPS requested but not available. Falling back to CPU.", flush=True)
    return torch.device("cpu")


def load_panns_model(checkpoint_path: Path, classes_num: int, device: torch.device) -> torch.nn.Module:
    from panns_inference.models import Cnn14

    model = Cnn14(
        sample_rate=PANNS_SAMPLE_RATE,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=classes_num,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    return model


def parse_recording_id(path: Path) -> tuple[str, str]:
    match = re.match(r"(?P<patient>\d+)_(?P<location>[A-Za-z]+)(?:_\d+)?$", path.stem)
    if not match:
        raise ValueError(f"Unexpected recording name: {path.name}")
    return match.group("patient"), match.group("location")


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    original_dtype = audio.dtype
    audio = audio.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        audio = audio / float(max(abs(info.min), info.max))
    else:
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1.0:
            audio = audio / peak
    if len(audio):
        audio = audio - float(np.mean(audio))
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak
    return sample_rate, audio.astype(np.float32)


def read_segments(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=["start_time", "end_time", "label"],
        dtype={"start_time": float, "end_time": float, "label": int},
    )


def resample_audio(audio: np.ndarray, sample_rate: int, target_rate: int = PANNS_SAMPLE_RATE) -> np.ndarray:
    if sample_rate == target_rate:
        return audio.astype(np.float32)
    gcd = math.gcd(sample_rate, target_rate)
    up = target_rate // gcd
    down = sample_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def segment_phase(audio: np.ndarray, sample_rate: int, segments: pd.DataFrame, label: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_samples = len(audio)
    for row in segments.loc[segments["label"] == label].itertuples(index=False):
        start = max(0, min(n_samples, int(round(row.start_time * sample_rate))))
        end = max(0, min(n_samples, int(round(row.end_time * sample_rate))))
        if end > start:
            chunks.append(audio[start:end])
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def pad_or_trim_for_panns(audio: np.ndarray, sample_rate: int, min_sec: float = 1.0) -> np.ndarray:
    min_samples = int(round(min_sec * sample_rate))
    if len(audio) == 0:
        return np.zeros(min_samples, dtype=np.float32)
    if len(audio) < min_samples:
        padded = np.zeros(min_samples, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded
    return audio.astype(np.float32)


def infer_batch_embeddings(
    model: torch.nn.Module,
    phase_audio_by_name: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    names = list(phase_audio_by_name.keys())
    arrays = [phase_audio_by_name[name] for name in names]
    max_len = max(len(array) for array in arrays)
    padded = np.zeros((len(arrays), max_len), dtype=np.float32)
    for idx, array in enumerate(arrays):
        padded[idx, : len(array)] = array

    embeddings: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(names), batch_size):
            batch_names = names[start : start + batch_size]
            batch = torch.from_numpy(padded[start : start + batch_size]).to(device)
            output = model(batch, None)
            batch_embeddings = output["embedding"].detach().cpu().numpy().astype(np.float32)
            for name, embedding in zip(batch_names, batch_embeddings, strict=True):
                embeddings[name] = embedding
    return embeddings


def build_phase_embedding_features(embeddings: dict[str, np.ndarray], durations: dict[str, float]) -> dict[str, float]:
    s1 = embeddings["s1"]
    systole = embeddings["systole"]
    s2 = embeddings["s2"]
    diastole = embeddings["diastole"]
    s1_s2_mean = (s1 + s2) / 2.0

    features: dict[str, float] = {}
    comparisons = {
        "systole_diastole": (systole, diastole),
        "systole_s1": (systole, s1),
        "systole_s2": (systole, s2),
        "systole_s1_s2_mean": (systole, s1_s2_mean),
    }

    for name, (left, right) in comparisons.items():
        delta = left - right
        abs_delta = np.abs(delta)
        for idx, value in enumerate(delta):
            features[f"delta_{name}_{idx:04d}"] = float(value)
        for idx, value in enumerate(abs_delta):
            features[f"abs_delta_{name}_{idx:04d}"] = float(value)
        features[f"cosine_distance_{name}"] = cosine_distance(left, right)
        features[f"l2_distance_{name}"] = float(np.linalg.norm(delta))
        features[f"norm_ratio_{name}"] = safe_divide(float(np.linalg.norm(left)), float(np.linalg.norm(right)))

    features["duration_ratio_systole_to_s1_s2"] = safe_divide(durations["systole"], durations["s1"] + durations["s2"])
    features["duration_ratio_systole_to_diastole"] = safe_divide(durations["systole"], durations["diastole"])
    features["duration_delta_systole_diastole"] = durations["systole"] - durations["diastole"]
    return features


def extract_recording_features(
    wav_path: Path,
    metadata: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | str] | None:
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

    phase_audio_32k: dict[str, np.ndarray] = {}
    durations: dict[str, float] = {}
    for label, phase_name in PHASE_LABELS.items():
        phase_audio = segment_phase(audio, sample_rate, segments, label)
        durations[phase_name] = float(len(phase_audio) / sample_rate) if sample_rate else 0.0
        phase_32k = resample_audio(phase_audio, sample_rate, PANNS_SAMPLE_RATE)
        phase_audio_32k[phase_name] = pad_or_trim_for_panns(phase_32k, PANNS_SAMPLE_RATE, min_sec=1.0)

    embeddings = infer_batch_embeddings(model, phase_audio_32k, device, batch_size)
    features: dict[str, float | str] = {
        "patient_id": patient_id,
        "recording_id": wav_path.stem,
        "location": location,
        "wav_path": str(wav_path),
        "sample_rate": float(sample_rate),
        "audio_duration_sec": float(len(audio) / sample_rate) if sample_rate else 0.0,
        "murmur": str(row.iloc[0]["Murmur"]),
        "outcome": str(row.iloc[0]["Outcome"]),
        "age": str(row.iloc[0]["Age"]),
        "sex": str(row.iloc[0]["Sex"]),
        "campaign": str(row.iloc[0]["Campaign"]),
    }
    for phase_name, duration in durations.items():
        features[f"{phase_name}_duration_sec"] = duration
    features.update(build_phase_embedding_features(embeddings, durations))
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

    matrix = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    scaled = StandardScaler().fit_transform(matrix).astype(np.float32)
    pca_dim = min(50, scaled.shape[0], scaled.shape[1])
    pca_model = PCA(n_components=pca_dim, svd_solver="randomized", random_state=42)
    pca = pca_model.fit_transform(scaled).astype(np.float32)

    k = min(n_clusters, len(df))
    pca_for_cluster = pca[:, : min(10, pca.shape[1])]
    clusters = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(pca_for_cluster)
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
        embedding = reducer.fit_transform(pca)
        result["umap_1"] = embedding[:, 0]
        result["umap_2"] = embedding[:, 1]

    silhouette = float("nan")
    if len(np.unique(clusters)) > 1 and len(df) > k:
        silhouette = float(silhouette_score(pca_for_cluster, clusters))

    return result, {
        "silhouette_pca": silhouette,
        "pca12_variance": float(np.sum(pca_model.explained_variance_ratio_[:2])),
    }


def projection_diagnostic_metrics(projected: pd.DataFrame) -> dict[str, float | int]:
    ct = pd.crosstab(projected["cluster"], projected["murmur"])
    pca2_rates = projected.groupby(pd.qcut(projected["pca_2"], 5, duplicates="drop"), observed=True)["murmur"].apply(
        lambda values: float((values == "Present").mean())
    )
    return {
        "cluster_0_rows": int(ct.loc[0].sum()) if 0 in ct.index else 0,
        "cluster_1_rows": int(ct.loc[1].sum()) if 1 in ct.index else 0,
        "cluster_0_present_rate": float(ct.loc[0, "Present"] / ct.loc[0].sum()) if 0 in ct.index and "Present" in ct.columns else float("nan"),
        "cluster_1_present_rate": float(ct.loc[1, "Present"] / ct.loc[1].sum()) if 1 in ct.index and "Present" in ct.columns else float("nan"),
        "pca2_lowest_quintile_present_rate": float(pca2_rates.iloc[0]) if len(pca2_rates) else float("nan"),
        "pca2_highest_quintile_present_rate": float(pca2_rates.iloc[-1]) if len(pca2_rates) else float("nan"),
    }


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
    markers = {"AV": "o", "PV": "^", "TV": "s", "MV": "D"}
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


def plot_per_location(
    recording_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    n_clusters: int,
    skip_umap: bool,
) -> list[dict[str, float | str | int]]:
    metrics_rows: list[dict[str, float | str | int]] = []
    per_location_dir = output_dir / "by_location"
    per_location_dir.mkdir(parents=True, exist_ok=True)

    for location in LOCATIONS:
        subset = recording_df.loc[recording_df["location"] == location].copy()
        if len(subset) < 8:
            continue
        projected, metrics = project_cluster(subset, feature_columns, n_clusters, skip_umap)
        projected.to_csv(per_location_dir / f"{location}_panns_phase_features_with_projection.csv", index=False)
        scatter(projected, "pca_1", "pca_2", per_location_dir / f"{location}_pca_murmur.png", f"{location}: PCA PANNs por fase")
        if not skip_umap and "umap_1" in projected.columns:
            scatter(projected, "umap_1", "umap_2", per_location_dir / f"{location}_umap_murmur.png", f"{location}: UMAP PANNs por fase")
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
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    lines = [
        "# Grupo C2 - PANNs embeddings por fase cardiaca",
        "",
        "## Objetivo",
        "",
        "Usar PANNs/Cnn14 pre-treinado como extrator congelado em cada fase cardiaca: `S1`, sistole, `S2` e diastole.",
        "",
        "A matriz final usa deltas, deltas absolutos, distancias cosseno/L2 e razoes de norma envolvendo a sistole, de forma analoga ao Grupo B v2.",
        "",
        "## Configuracao",
        "",
        f"- Device usado: `{device}`",
        f"- Sample rate PANNs: {PANNS_SAMPLE_RATE} Hz",
        f"- Batch size: {args.batch_size}",
        f"- UMAP gerado: {'nao' if args.skip_umap else 'sim'}",
        "",
        "## Resumo",
        "",
        f"- Gravacoes analisadas: {len(recording_df)}",
        f"- Pacientes analisados: {recording_df['patient_id'].nunique()}",
        f"- Features PANNs por fase por gravacao: {len(feature_columns)}",
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
        "## Interpretacao inicial",
        "",
        "- Compare este resultado contra Grupo B v2 e Grupo C1.",
        "- Se C2 superar C1, isso indica que segmentar por fase tambem ajuda embeddings pre-treinados.",
        "- Se C2 ainda ficar abaixo de Grupo B v2, as features relativas manuais continuam sendo a base mais forte.",
        "",
        "## Arquivos gerados",
        "",
        "- `recording_panns_phase_features.csv`: features PANNs por fase por gravacao.",
        "- `recording_panns_phase_features_with_projection.csv`: PCA/UMAP/k-means global por gravacao.",
        "- `patient_panns_phase_features.csv`: agregacao por paciente.",
        "- `patient_panns_phase_features_with_projection.csv`: agregacao por paciente com PCA/UMAP/k-means.",
        "- `by_location/*_pca_murmur.png`: PCA separado por local.",
        "- `by_location/*_umap_murmur.png`: UMAP separado por local, se habilitado.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_interpretation(output_path: Path, metrics_df: pd.DataFrame) -> None:
    def best_cluster_rate(level: str) -> float:
        row = metrics_df.loc[metrics_df["level"] == level]
        if row.empty:
            return float("nan")
        row0 = row.iloc[0]
        return float(max(row0["cluster_0_present_rate"], row0["cluster_1_present_rate"]))

    lines = [
        "# Interpretacao dos resultados - Grupo C2 PANNs por fase",
        "",
        "Este documento sera preenchido automaticamente apos a execucao do script.",
        "",
        "## Leitura rapida",
        "",
        f"- Melhor taxa `Present` em cluster global por gravacao: {best_cluster_rate('recording_global'):.1%}",
        f"- Melhor taxa `Present` em cluster agregado por paciente: {best_cluster_rate('patient_aggregated'):.1%}",
        "",
        "## Comparacao esperada",
        "",
        "- Grupo C1 PANNs global por paciente: 28.5% `Present` no melhor cluster.",
        "- Grupo B v2 por paciente: 89.5% `Present` no melhor cluster.",
        "- Este C2 testa se PANNs melhora quando respeita as fases cardiacas.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    data_dir = dataset_dir / "training_data"
    metadata_path = dataset_dir / "training_data.csv"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    recording_csv = output_dir / "recording_panns_phase_features.csv"
    if args.reuse_recording_features and recording_csv.exists():
        recording_df = pd.read_csv(recording_csv)
        device = choose_device("cpu")
    else:
        labels_path, checkpoint_path = ensure_panns_files(args.panns_dir.expanduser())
        classes_num = load_class_count(labels_path)
        device = choose_device(args.device)
        print(f"Loading PANNs/Cnn14 on {device}...", flush=True)
        model = load_panns_model(checkpoint_path, classes_num, device)

        metadata = pd.read_csv(metadata_path)
        if not args.include_unknown:
            metadata = metadata.loc[metadata["Murmur"].isin(["Present", "Absent"])].copy()

        wav_paths = sorted(data_dir.glob("*.wav"))
        if args.max_recordings is not None:
            wav_paths = wav_paths[: args.max_recordings]

        rows: list[dict[str, float | str]] = []
        skipped = 0
        for index, wav_path in enumerate(wav_paths, start=1):
            try:
                row = extract_recording_features(wav_path, metadata, model, device, args.batch_size)
            except Exception as exc:
                if device.type == "mps":
                    print(f"MPS failed on {wav_path.name}: {exc}. Falling back to CPU and retrying.", flush=True)
                    device = torch.device("cpu")
                    model.to(device)
                    row = extract_recording_features(wav_path, metadata, model, device, args.batch_size)
                else:
                    raise
            if row is None:
                skipped += 1
                continue
            rows.append(row)
            if index % 25 == 0:
                pd.DataFrame(rows).to_csv(output_dir / "recording_panns_phase_features.partial.csv", index=False)
                print(f"Processed {index}/{len(wav_paths)} recordings...", flush=True)

        if not rows:
            raise RuntimeError("No features extracted. Check dataset path and filters.")
        recording_df = pd.DataFrame(rows)
        recording_df.to_csv(recording_csv, index=False)
        partial_path = output_dir / "recording_panns_phase_features.partial.csv"
        if partial_path.exists():
            partial_path.unlink()
        print(f"Extracted {len(recording_df)} recordings. Skipped {skipped}.", flush=True)

    feature_columns = numeric_feature_columns(recording_df)
    projected_recordings, global_metrics = project_cluster(recording_df, feature_columns, args.n_clusters, args.skip_umap)
    projected_recordings.to_csv(output_dir / "recording_panns_phase_features_with_projection.csv", index=False)
    scatter(
        projected_recordings,
        "pca_1",
        "pca_2",
        output_dir / "recording_pca_murmur_by_location.png",
        "Todas as gravacoes: PCA PANNs por fase",
        marker_col="location",
    )
    if not args.skip_umap and "umap_1" in projected_recordings.columns:
        scatter(
            projected_recordings,
            "umap_1",
            "umap_2",
            output_dir / "recording_umap_murmur_by_location.png",
            "Todas as gravacoes: UMAP PANNs por fase",
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
    metrics_rows.extend(plot_per_location(recording_df, feature_columns, output_dir, args.n_clusters, args.skip_umap))

    patient_df = aggregate_by_patient(recording_df, feature_columns)
    patient_feature_columns = numeric_feature_columns(patient_df)
    patient_df.to_csv(output_dir / "patient_panns_phase_features.csv", index=False)
    projected_patients, patient_metrics = project_cluster(patient_df, patient_feature_columns, args.n_clusters, args.skip_umap)
    projected_patients.to_csv(output_dir / "patient_panns_phase_features_with_projection.csv", index=False)
    scatter(projected_patients, "pca_1", "pca_2", output_dir / "patient_pca_murmur.png", "Pacientes agregados: PCA PANNs por fase")
    if not args.skip_umap and "umap_1" in projected_patients.columns:
        scatter(projected_patients, "umap_1", "umap_2", output_dir / "patient_umap_murmur.png", "Pacientes agregados: UMAP PANNs por fase")

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
    write_summary(output_dir / "summary.md", recording_df, patient_df, metrics_df, feature_columns, patient_feature_columns, args, device)
    write_interpretation(output_dir / "interpretacao_resultados.md", metrics_df)

    print(f"Recording feature columns: {len(feature_columns)}", flush=True)
    print(f"Patient feature columns: {len(patient_feature_columns)}", flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
