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
"""Grupo C1 experiment: global PANNs embeddings per recording.

This script uses PANNs/Cnn14 as a frozen pretrained embedding extractor. It
does not use cardiac-phase `.tsv` segmentations and does not fine-tune the
model. Each full recording is split into overlapping windows, embeddings are
extracted per window, and mean/std/max pooling creates one vector per
recording. Patient-level mean/max aggregation is also generated.

Run from the repository root:

    uv run "feature extraction/Grupo C1 PANNs embeddings globais por gravacao/extract_panns_global_embeddings.py"

Recommended quick test:

    uv run "feature extraction/Grupo C1 PANNs embeddings globais por gravacao/extract_panns_global_embeddings.py" --max-recordings 20 --skip-umap

For a stable Mac run, CPU is the default. Try MPS only if you want to test it:

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run "feature extraction/Grupo C1 PANNs embeddings globais por gravacao/extract_panns_global_embeddings.py" --device mps --batch-size 1 --skip-umap
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
DEFAULT_WINDOW_SEC = 10.0
DEFAULT_HOP_SEC = 5.0
EMBEDDING_SIZE = 2048

LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description="Extract frozen PANNs/Cnn14 embeddings from CirCor recordings.")
    parser.add_argument("--dataset-dir", type=Path, default=repo_root / "circor-heart-sound-1.0.3")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs")
    parser.add_argument("--panns-dir", type=Path, default=Path.home() / "panns_data")
    parser.add_argument("--include-unknown", action="store_true", help="Include Murmur=Unknown. Default excludes it.")
    parser.add_argument("--max-recordings", type=int, default=None, help="Optional quick-test limit.")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--hop-sec", type=float, default=DEFAULT_HOP_SEC)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="cpu",
        help="Default is cpu for stability. Use auto or mps to try Apple Silicon MPS.",
    )
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument(
        "--reuse-recording-embeddings",
        action="store_true",
        help="Reuse existing recording_panns_embeddings.csv if present.",
    )
    return parser.parse_args()


def download_file(url: str, target: Path, min_size_bytes: int = 1) -> None:
    if target.exists() and target.stat().st_size >= min_size_bytes:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url} -> {target}")
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
                print(f"\r  {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.1f}%)", end="")
        if total:
            print()
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
        print("MPS requested but not available. Falling back to CPU.")
    return torch.device("cpu")


def load_panns_model(checkpoint_path: Path, classes_num: int, device: torch.device) -> torch.nn.Module:
    # Import only after labels exist, because panns_inference imports config at module import time.
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


def resample_audio(audio: np.ndarray, sample_rate: int, target_rate: int = PANNS_SAMPLE_RATE) -> np.ndarray:
    if sample_rate == target_rate:
        return audio.astype(np.float32)
    gcd = math.gcd(sample_rate, target_rate)
    up = target_rate // gcd
    down = sample_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def make_windows(audio: np.ndarray, sample_rate: int, window_sec: float, hop_sec: float) -> np.ndarray:
    window_samples = int(round(window_sec * sample_rate))
    hop_samples = int(round(hop_sec * sample_rate))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window-sec and hop-sec must be positive.")

    if len(audio) == 0:
        return np.zeros((1, window_samples), dtype=np.float32)

    if len(audio) <= window_samples:
        window = np.zeros(window_samples, dtype=np.float32)
        window[: len(audio)] = audio
        return window[None, :]

    starts = list(range(0, len(audio) - window_samples + 1, hop_samples))
    if starts[-1] != len(audio) - window_samples:
        starts.append(len(audio) - window_samples)
    windows = np.stack([audio[start : start + window_samples] for start in starts]).astype(np.float32)
    return windows


def infer_embeddings(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            output = model(batch, None)
            emb = output["embedding"].detach().cpu().numpy().astype(np.float32)
            embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)


def pool_window_embeddings(window_embeddings: np.ndarray) -> dict[str, float]:
    if len(window_embeddings) == 0:
        window_embeddings = np.zeros((1, EMBEDDING_SIZE), dtype=np.float32)
    pooled = {
        "mean": np.mean(window_embeddings, axis=0),
        "std": np.std(window_embeddings, axis=0),
        "max": np.max(window_embeddings, axis=0),
    }
    features: dict[str, float] = {}
    for pool_name, values in pooled.items():
        for idx, value in enumerate(values):
            features[f"panns_{pool_name}_{idx:04d}"] = float(value)
    return features


def extract_recording_embedding(
    wav_path: Path,
    metadata: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    window_sec: float,
    hop_sec: float,
) -> dict[str, float | str] | None:
    patient_id, location = parse_recording_id(wav_path)
    row = metadata.loc[metadata["Patient ID"].astype(str) == patient_id]
    if row.empty:
        return None

    sample_rate, audio = read_audio(wav_path)
    audio_32k = resample_audio(audio, sample_rate, PANNS_SAMPLE_RATE)
    windows = make_windows(audio_32k, PANNS_SAMPLE_RATE, window_sec, hop_sec)
    window_embeddings = infer_embeddings(model, windows, device, batch_size)

    features: dict[str, float | str] = {
        "patient_id": patient_id,
        "recording_id": wav_path.stem,
        "location": location,
        "wav_path": str(wav_path),
        "sample_rate": float(sample_rate),
        "audio_duration_sec": float(len(audio) / sample_rate) if sample_rate else 0.0,
        "panns_window_count": float(len(windows)),
        "murmur": str(row.iloc[0]["Murmur"]),
        "outcome": str(row.iloc[0]["Outcome"]),
        "age": str(row.iloc[0]["Age"]),
        "sex": str(row.iloc[0]["Sex"]),
        "campaign": str(row.iloc[0]["Campaign"]),
    }
    features.update(pool_window_embeddings(window_embeddings))
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
    clusters = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(pca[:, : min(10, pca.shape[1])])
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
        silhouette = float(silhouette_score(pca[:, : min(10, pca.shape[1])], clusters))

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
        "# Grupo C1 - PANNs embeddings globais por gravacao",
        "",
        "## Objetivo",
        "",
        "Usar PANNs/Cnn14 pre-treinado como extrator congelado de embeddings do audio inteiro de cada gravacao.",
        "",
        "Este experimento nao usa `.tsv` e nao faz fine-tuning.",
        "",
        "## Configuracao",
        "",
        f"- Device usado: `{device}`",
        f"- Sample rate PANNs: {PANNS_SAMPLE_RATE} Hz",
        f"- Janela: {args.window_sec:.2f} s",
        f"- Hop: {args.hop_sec:.2f} s",
        f"- Batch size: {args.batch_size}",
        f"- UMAP gerado: {'nao' if args.skip_umap else 'sim'}",
        "",
        "## Resumo",
        "",
        f"- Gravacoes analisadas: {len(recording_df)}",
        f"- Pacientes analisados: {recording_df['patient_id'].nunique()}",
        f"- Features PANNs por gravacao: {len(feature_columns)}",
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
        "- Compare este resultado com Grupo A e Grupo B v2.",
        "- Se os embeddings PANNs formarem clusters muito enriquecidos em `Present`, vale testar pooling por local e depois embeddings por fase.",
        "- Se forem inferiores ao Grupo B v2, os embeddings ainda podem ser usados em um modelo hibrido junto das features relativas por fase.",
        "",
        "## Arquivos gerados",
        "",
        "- `recording_panns_embeddings.csv`: embeddings agregados por gravacao.",
        "- `recording_panns_embeddings_with_projection.csv`: embeddings com PCA/UMAP/k-means.",
        "- `patient_panns_embeddings.csv`: agregacao por paciente.",
        "- `patient_panns_embeddings_with_projection.csv`: agregacao por paciente com PCA/UMAP/k-means.",
        "- `projection_metrics.csv`: metricas diagnosticas dos clusters/projecoes.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_interpretation(output_path: Path, metrics_df: pd.DataFrame) -> None:
    def metric(level: str, column: str) -> float:
        row = metrics_df.loc[metrics_df["level"] == level]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][column])

    lines = [
        "# Interpretacao dos resultados - Grupo C1 PANNs",
        "",
        "Este documento interpreta embeddings PANNs globais por gravacao.",
        "",
        "## Leitura rapida",
        "",
        f"- Melhor taxa `Present` em cluster global por gravacao: {max(metric('recording_global', 'cluster_0_present_rate'), metric('recording_global', 'cluster_1_present_rate')):.1%}",
        f"- Melhor taxa `Present` em cluster agregado por paciente: {max(metric('patient_aggregated', 'cluster_0_present_rate'), metric('patient_aggregated', 'cluster_1_present_rate')):.1%}",
        f"- Taxa `Present` no maior quintil de PCA2 por paciente: {metric('patient_aggregated', 'pca2_highest_quintile_present_rate'):.1%}",
        "",
        "## Como interpretar",
        "",
        "- Este experimento testa se um modelo geral de audio, treinado em AudioSet, encontra uma representacao util para sopro cardiaco sem treinamento adicional.",
        "- Como o modelo nao conhece fases cardiacas, um resultado fraco nao invalida embeddings; pode indicar que precisamos de pooling por fase ou combinacao com Grupo B v2.",
        "- O criterio principal e comparar a concentracao de `Present` contra Grupo A e Grupo B v2.",
        "",
        "## Proximo passo sugerido",
        "",
        "- Se este resultado for competitivo com Grupo B v2, testar PANNs por local e depois por fase usando `.tsv`.",
        "- Se for inferior, usar PANNs apenas como complemento em um baseline hibrido.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    data_dir = dataset_dir / "training_data"
    metadata_path = dataset_dir / "training_data.csv"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    recording_csv = output_dir / "recording_panns_embeddings.csv"
    if args.reuse_recording_embeddings and recording_csv.exists():
        recording_df = pd.read_csv(recording_csv)
        device = choose_device("cpu")
    else:
        labels_path, checkpoint_path = ensure_panns_files(args.panns_dir.expanduser())
        classes_num = load_class_count(labels_path)
        requested_device = choose_device(args.device)
        print(f"Loading PANNs/Cnn14 on {requested_device}...")
        model = load_panns_model(checkpoint_path, classes_num, requested_device)
        device = requested_device

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
                row = extract_recording_embedding(
                    wav_path,
                    metadata,
                    model,
                    device,
                    args.batch_size,
                    args.window_sec,
                    args.hop_sec,
                )
            except Exception as exc:
                if device.type == "mps":
                    print(f"MPS failed on {wav_path.name}: {exc}. Falling back to CPU and retrying.")
                    device = torch.device("cpu")
                    model.to(device)
                    row = extract_recording_embedding(
                        wav_path,
                        metadata,
                        model,
                        device,
                        args.batch_size,
                        args.window_sec,
                        args.hop_sec,
                    )
                else:
                    raise

            if row is None:
                skipped += 1
                continue
            rows.append(row)
            if index % 25 == 0:
                partial_df = pd.DataFrame(rows)
                partial_df.to_csv(output_dir / "recording_panns_embeddings.partial.csv", index=False)
                print(f"Processed {index}/{len(wav_paths)} recordings...")

        if not rows:
            raise RuntimeError("No embeddings extracted. Check dataset path and filters.")

        recording_df = pd.DataFrame(rows)
        recording_df.to_csv(recording_csv, index=False)
        partial_path = output_dir / "recording_panns_embeddings.partial.csv"
        if partial_path.exists():
            partial_path.unlink()
        print(f"Extracted {len(recording_df)} recordings. Skipped {skipped}.")

    feature_columns = numeric_feature_columns(recording_df)
    projected_recordings, global_metrics = project_cluster(recording_df, feature_columns, args.n_clusters, args.skip_umap)
    projected_recordings.to_csv(output_dir / "recording_panns_embeddings_with_projection.csv", index=False)
    scatter(
        projected_recordings,
        "pca_1",
        "pca_2",
        output_dir / "recording_pca_murmur_by_location.png",
        "Todas as gravacoes: PCA de embeddings PANNs",
        marker_col="location",
    )
    if not args.skip_umap and "umap_1" in projected_recordings.columns:
        scatter(
            projected_recordings,
            "umap_1",
            "umap_2",
            output_dir / "recording_umap_murmur_by_location.png",
            "Todas as gravacoes: UMAP de embeddings PANNs",
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

    patient_df = aggregate_by_patient(recording_df, feature_columns)
    patient_feature_columns = numeric_feature_columns(patient_df)
    patient_df.to_csv(output_dir / "patient_panns_embeddings.csv", index=False)
    projected_patients, patient_metrics = project_cluster(patient_df, patient_feature_columns, args.n_clusters, args.skip_umap)
    projected_patients.to_csv(output_dir / "patient_panns_embeddings_with_projection.csv", index=False)
    scatter(projected_patients, "pca_1", "pca_2", output_dir / "patient_pca_murmur.png", "Pacientes agregados: PCA de embeddings PANNs")
    if not args.skip_umap and "umap_1" in projected_patients.columns:
        scatter(projected_patients, "umap_1", "umap_2", output_dir / "patient_umap_murmur.png", "Pacientes agregados: UMAP de embeddings PANNs")

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

    print(f"Recording feature columns: {len(feature_columns)}")
    print(f"Patient feature columns: {len(patient_feature_columns)}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
