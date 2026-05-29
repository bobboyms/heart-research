"""Exploratory analysis: where do murmur (Present) systole spectrograms overlap with non-murmur (Absent)?

Pipeline:
  1. For each AV/PV/TV/MV recording, read the ground-truth .tsv and extract systole-only audio.
  2. Compute a per-segment STFT (same band/params the CNN uses) and summarize it into a compact
     spectral-envelope descriptor (per-frequency mean/std/max/p90 over all systole frames).
  3. Standardize, project with PCA + t-SNE, and cluster with KMeans.
  4. Find the Present recordings whose nearest neighbours are mostly Absent ("confusable" murmurs),
     and the Absent recordings surrounded by Present ("false-positive-prone").

Recording-level labels are location-aware: a recording is Present only if its patient is Present
AND its auscultation location appears in `Murmur locations`.

Outputs (under outputs/):
  - descriptors.npz                 cached descriptors + metadata (skip recompute on rerun)
  - projection_tsne.png             2D t-SNE colored by label
  - projection_pca.png              2D PCA colored by label
  - clusters_tsne.png               2D t-SNE colored by KMeans cluster
  - cluster_summary.csv             per-cluster size and % Present
  - confusable_present.csv          Present recordings with the most Absent neighbours
  - fp_prone_absent.csv             Absent recordings with the most Present neighbours
  - summary.md                      short written interpretation
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import stft
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "circor-heart-sound-1.0.3"
CNN_SCRIPT = REPO_ROOT / "modeling" / "Grupo G CNN dilatada systole TCN STFT" / "train_systole_stft_dilated_cnn.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

LABEL_SYSTOLE = 2
TARGET_SR = 4000
N_FFT = 128
HOP = 32
LOW_HZ = 0.0
HIGH_HZ = 1000.0


def load_cnn_module():
    spec = importlib.util.spec_from_file_location("cnn_explore", CNN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cnn_explore"] = module
    spec.loader.exec_module(module)
    return module


def systole_frames(audio: np.ndarray, sr: int, segments: pd.DataFrame, cnn) -> np.ndarray | None:
    """Per-segment STFT of systole audio; returns (freq_bins, total_frames) log-magnitude, no padding."""
    resampled = cnn.resample_audio(audio, sr, TARGET_SR)
    sel = segments.loc[segments["label"] == LABEL_SYSTOLE].sort_values("start_time")
    blocks: list[np.ndarray] = []
    freq_mask = None
    for row in sel.itertuples(index=False):
        start = max(0, int(round(float(row.start_time) * TARGET_SR)))
        end = min(len(resampled), int(round(float(row.end_time) * TARGET_SR)))
        if end - start < N_FFT:
            continue
        seg = resampled[start:end]
        freqs, _t, zxx = stft(seg, fs=TARGET_SR, window="hann", nperseg=N_FFT,
                              noverlap=N_FFT - HOP, nfft=N_FFT, boundary=None, padded=False)
        spec = np.log1p(np.abs(zxx).astype(np.float32))
        if freq_mask is None:
            freq_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
        spec = spec[freq_mask]
        if spec.shape[1] > 0:
            blocks.append(spec)
    if not blocks:
        return None
    return np.concatenate(blocks, axis=1)


def descriptor_from_frames(frames: np.ndarray) -> np.ndarray:
    """Compact, time-length-invariant spectral-envelope descriptor."""
    mean = frames.mean(axis=1)
    std = frames.std(axis=1)
    mx = frames.max(axis=1)
    p90 = np.percentile(frames, 90, axis=1)
    return np.concatenate([mean, std, mx, p90]).astype(np.float32)


def build_descriptors(cnn, max_recordings: int | None) -> pd.DataFrame:
    items = cnn.build_items(DATASET_DIR, ["AV", "PV", "TV", "MV"], max_recordings)
    context = cnn.load_patient_context(DATASET_DIR)
    grade_by_patient = (
        context.set_index("patient_id")["systolic_murmur_grading"].to_dict()
        if "systolic_murmur_grading" in context.columns else {}
    )
    rows = []
    descriptors = []
    for i, item in enumerate(items):
        if i % 200 == 0:
            print(f"  {i}/{len(items)} recordings...", flush=True)
        try:
            sr, audio = cnn.read_audio(item.wav_path)
        except Exception:
            continue
        segments = cnn.read_segments(item.wav_path.with_suffix(".tsv"))
        if segments.empty:
            continue
        frames = systole_frames(audio, sr, segments, cnn)
        if frames is None:
            continue
        descriptors.append(descriptor_from_frames(frames))
        rows.append({
            "recording_id": item.recording_id,
            "patient_id": item.patient_id,
            "location": item.location,
            "patient_murmur": item.murmur,
            "recording_present": int(item.recording_present),
            "grading": str(grade_by_patient.get(item.patient_id, "")).strip(),
            "n_systole_frames": int(frames.shape[1]),
        })
    meta = pd.DataFrame(rows)
    meta_x = np.stack(descriptors)
    return meta, meta_x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--knn", type=int, default=15, help="Neighbours used for the overlap analysis.")
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--reuse-descriptors", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT_DIR / "descriptors.npz"

    cnn = load_cnn_module()
    if args.reuse_descriptors and cache.exists():
        data = np.load(cache, allow_pickle=True)
        meta = pd.DataFrame({k: data[k] for k in data.files if k != "X"})
        X = data["X"]
        print(f"Reused {len(meta)} cached descriptors.")
    else:
        print("Extracting systole spectral-envelope descriptors...")
        meta, X = build_descriptors(cnn, args.max_recordings)
        np.savez_compressed(cache, X=X, **{c: meta[c].to_numpy() for c in meta.columns})
        print(f"Built {len(meta)} descriptors.")

    y = meta["recording_present"].to_numpy(dtype=int)
    print(f"Recordings: {len(meta)} | Present(local): {int(y.sum())} | Absent: {int((y==0).sum())}")

    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(30, X_std.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_std)
    print(f"PCA: {X_pca.shape[1]} comps explain {pca.explained_variance_ratio_.sum():.1%} variance")

    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    km = KMeans(n_clusters=args.clusters, n_init=10, random_state=42)
    clusters = km.fit_predict(X_pca)
    meta["cluster"] = clusters

    # --- Plots ---
    def scatter(xy, color, title, path, cmap=None, legend_labels=None):
        plt.figure(figsize=(9, 7))
        if legend_labels:
            for val, lab, c in legend_labels:
                m = color == val
                plt.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, label=lab, c=c)
            plt.legend()
        else:
            sc = plt.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.6, c=color, cmap=cmap)
            plt.colorbar(sc)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()

    scatter(X_tsne, y, "t-SNE da sistole (vermelho=Present local, azul=Absent)",
            OUTPUT_DIR / "projection_tsne.png",
            legend_labels=[(0, "Absent", "#3b6dd6"), (1, "Present", "#d63b3b")])
    scatter(X_pca[:, :2], y, "PCA da sistole (vermelho=Present local, azul=Absent)",
            OUTPUT_DIR / "projection_pca.png",
            legend_labels=[(0, "Absent", "#3b6dd6"), (1, "Present", "#d63b3b")])
    scatter(X_tsne, clusters, "t-SNE colorido por cluster KMeans",
            OUTPUT_DIR / "clusters_tsne.png", cmap="tab10")

    # --- Cluster summary ---
    cluster_rows = []
    for c in sorted(meta["cluster"].unique()):
        sub = meta[meta["cluster"] == c]
        cluster_rows.append({
            "cluster": c,
            "n": len(sub),
            "n_present": int(sub["recording_present"].sum()),
            "pct_present": round(100 * sub["recording_present"].mean(), 1),
        })
    cluster_summary = pd.DataFrame(cluster_rows).sort_values("pct_present", ascending=False)
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)

    # --- Overlap / nearest-neighbour analysis (in PCA space) ---
    nn = NearestNeighbors(n_neighbors=args.knn + 1).fit(X_pca)
    _dist, idx = nn.kneighbors(X_pca)
    neigh = idx[:, 1:]  # drop self
    neigh_present = y[neigh]  # (N, knn)
    frac_present_neighbors = neigh_present.mean(axis=1)
    meta["frac_present_neighbors"] = frac_present_neighbors

    # Present recordings whose neighbours are mostly Absent -> confusable murmurs
    present_mask = y == 1
    confusable = meta[present_mask].copy()
    confusable["frac_absent_neighbors"] = 1.0 - confusable["frac_present_neighbors"]
    confusable = confusable.sort_values("frac_absent_neighbors", ascending=False)
    confusable_cols = ["recording_id", "patient_id", "location", "grading",
                       "frac_absent_neighbors", "n_systole_frames", "cluster"]
    confusable[confusable_cols].to_csv(OUTPUT_DIR / "confusable_present.csv", index=False)

    # Absent recordings surrounded by Present -> false-positive prone
    absent_mask = y == 0
    fp_prone = meta[absent_mask].copy()
    fp_prone = fp_prone.sort_values("frac_present_neighbors", ascending=False)
    fp_cols = ["recording_id", "patient_id", "location", "patient_murmur",
               "frac_present_neighbors", "n_systole_frames", "cluster"]
    fp_prone[fp_cols].to_csv(OUTPUT_DIR / "fp_prone_absent.csv", index=False)

    # How many Present are "deep in Absent territory"?
    deep_confusable = int((confusable["frac_absent_neighbors"] >= 0.8).sum())
    well_separated = int((confusable["frac_absent_neighbors"] <= 0.2).sum())

    lines = [
        "# Exploração: overlap sistole Present vs Absent",
        "",
        f"- Recordings analisados: {len(meta)} (Present local: {int(y.sum())}, Absent: {int((y==0).sum())})",
        f"- Descritor: envelope espectral da sistole (mean/std/max/p90 por bin de frequencia, {X.shape[1]} dims)",
        f"- Projecao: PCA({X_pca.shape[1]}) -> t-SNE 2D; clusters: KMeans k={args.clusters}; vizinhanca k={args.knn}",
        "",
        "## Overlap (vizinhos no espaco PCA)",
        "",
        f"- Present com >=80% de vizinhos Absent (murmurio 'escondido', dificil): **{deep_confusable}** "
        f"({100*deep_confusable/max(1,int(y.sum())):.1f}% dos Present)",
        f"- Present com <=20% de vizinhos Absent (bem separado, facil): {well_separated} "
        f"({100*well_separated/max(1,int(y.sum())):.1f}% dos Present)",
        "",
        "## Distribuicao de grading entre os Present mais confundiveis (top 30)",
        "",
        confusable.head(30)["grading"].value_counts().to_frame("n").to_markdown(),
        "",
        "## Clusters (ordenado por % Present)",
        "",
        cluster_summary.to_markdown(index=False),
        "",
        "## Leitura",
        "",
        "Clusters com % Present intermediario (30-70%) sao a zona de overlap onde o classificador "
        "nao consegue separar. Os recordings em `confusable_present.csv` com `frac_absent_neighbors` "
        "alto sao murmurios que, no espaco do envelope espectral da sistole, parecem ruido normal — "
        "candidatos a olhar individualmente (grading, qualidade do .tsv, presenca real do sopro).",
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines))

    print("\n=== RESUMO ===")
    print(f"Present com >=80% vizinhos Absent (dificeis): {deep_confusable} / {int(y.sum())}")
    print(f"Present com <=20% vizinhos Absent (faceis):   {well_separated} / {int(y.sum())}")
    print("\nClusters por % Present:")
    print(cluster_summary.to_string(index=False))
    print(f"\nOutputs em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
