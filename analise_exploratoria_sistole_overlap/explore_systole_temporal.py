"""Exploratory test: do temporal-dynamics features separate the 'hidden' (I/VI) murmurs
that the static spectral envelope cannot?

Compares two descriptors on the same recordings:
  A) envelope-only   : per-frequency mean/std/max/p90 over all systole frames (time collapsed)
  B) envelope+temporal: A) plus per-beat temporal-dynamics features (fill fraction, envelope
                        shape, spectral flux, sustained high-band, temporal CV), averaged over beats.

For each, it runs the k-NN overlap analysis and reports how many Present recordings are
'difficult' (>=80% Absent neighbours). If temporal dynamics carry signal, the difficult count
should drop, especially for grade I/VI.

Temporal features are computed PER systole segment (one heartbeat) and averaged, so the
concatenation boundaries between beats never enter the flux/shape statistics.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import stft
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
FREQ_PER_BIN = TARGET_SR / N_FFT  # 31.25 Hz


def load_cnn_module():
    spec = importlib.util.spec_from_file_location("cnn_explore_t", CNN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cnn_explore_t"] = module
    spec.loader.exec_module(module)
    return module


def segment_frame_blocks(audio, sr, segments, cnn):
    """Return a list of (freq_bins, t) log-magnitude arrays, one per systole segment."""
    resampled = cnn.resample_audio(audio, sr, TARGET_SR)
    sel = segments.loc[segments["label"] == LABEL_SYSTOLE].sort_values("start_time")
    blocks = []
    freq_mask = None
    freqs_kept = None
    for row in sel.itertuples(index=False):
        start = max(0, int(round(float(row.start_time) * TARGET_SR)))
        end = min(len(resampled), int(round(float(row.end_time) * TARGET_SR)))
        if end - start < N_FFT:
            continue
        freqs, _t, zxx = stft(resampled[start:end], fs=TARGET_SR, window="hann", nperseg=N_FFT,
                              noverlap=N_FFT - HOP, nfft=N_FFT, boundary=None, padded=False)
        spec = np.log1p(np.abs(zxx).astype(np.float32))
        if freq_mask is None:
            freq_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
            freqs_kept = freqs[freq_mask]
        spec = spec[freq_mask]
        if spec.shape[1] >= 2:
            blocks.append(spec)
    return blocks, freqs_kept


def envelope_descriptor(blocks):
    frames = np.concatenate(blocks, axis=1)
    return np.concatenate([
        frames.mean(axis=1), frames.std(axis=1),
        frames.max(axis=1), np.percentile(frames, 90, axis=1),
    ]).astype(np.float32)


def temporal_descriptor(blocks, freqs):
    """Per-beat dynamics, averaged across beats. ~13 features."""
    # murmur band ~100-500 Hz; high band ~250-700 Hz
    murmur_band = (freqs >= 100) & (freqs <= 500)
    high_band = (freqs >= 250) & (freqs <= 700)
    feats_per_beat = []
    for spec in blocks:  # spec: (freq, t)
        t = spec.shape[1]
        e = spec.sum(axis=0)  # energy curve
        rng = e.max() - e.min()
        e_n = (e - e.min()) / (rng + 1e-6)
        tt = np.linspace(0.0, 1.0, t)
        # envelope shape
        fill_fraction = float((e_n > 0.5).mean())
        peak_pos = float(np.argmax(e) / max(1, t - 1))
        slope = float(np.polyfit(tt, e_n, 1)[0]) if t >= 2 else 0.0
        curv = float(np.polyfit(tt, e_n, 2)[0]) if t >= 3 else 0.0
        # spectral flux (frame-to-frame change within the beat)
        diffs = np.linalg.norm(np.diff(spec, axis=1), axis=0)
        flux_mean = float(diffs.mean()) if diffs.size else 0.0
        flux_std = float(diffs.std()) if diffs.size else 0.0
        # sustained high band: fraction of frames where high-band energy is above its own median
        hb = spec[high_band].sum(axis=0) if high_band.any() else np.zeros(t)
        hb_fill = float((hb > np.median(hb)).mean()) if t else 0.0
        hb_mean = float(hb.mean())
        # temporal coefficient of variation in murmur band (sustained noise -> lower CV)
        mb = spec[murmur_band]
        cv = float((mb.std(axis=1) / (mb.mean(axis=1) + 1e-6)).mean()) if murmur_band.any() else 0.0
        # autocorr of energy at lag 1 (sustained -> high)
        if t >= 2 and e.std() > 1e-6:
            ac1 = float(np.corrcoef(e[:-1], e[1:])[0, 1])
        else:
            ac1 = 0.0
        feats_per_beat.append([fill_fraction, peak_pos, slope, curv, flux_mean,
                               flux_std, hb_fill, hb_mean, cv, ac1])
    arr = np.asarray(feats_per_beat, dtype=np.float32)
    # average across beats + a couple of across-beat consistency measures
    mean_feats = arr.mean(axis=0)
    n_beats = np.array([len(blocks)], dtype=np.float32)
    fill_consistency = np.array([arr[:, 0].std()], dtype=np.float32)
    return np.concatenate([mean_feats, n_beats, fill_consistency]).astype(np.float32)


def build(cnn, max_recordings):
    items = cnn.build_items(DATASET_DIR, ["AV", "PV", "TV", "MV"], max_recordings)
    context = cnn.load_patient_context(DATASET_DIR)
    grade_by_patient = (context.set_index("patient_id")["systolic_murmur_grading"].to_dict()
                        if "systolic_murmur_grading" in context.columns else {})
    rows, env, tmp = [], [], []
    for i, item in enumerate(items):
        if i % 400 == 0:
            print(f"  {i}/{len(items)}...", flush=True)
        try:
            sr, audio = cnn.read_audio(item.wav_path)
        except Exception:
            continue
        segments = cnn.read_segments(item.wav_path.with_suffix(".tsv"))
        if segments.empty:
            continue
        blocks, freqs = segment_frame_blocks(audio, sr, segments, cnn)
        if not blocks:
            continue
        env.append(envelope_descriptor(blocks))
        tmp.append(temporal_descriptor(blocks, freqs))
        rows.append({
            "recording_id": item.recording_id, "patient_id": item.patient_id,
            "location": item.location, "patient_murmur": item.murmur,
            "recording_present": int(item.recording_present),
            "grading": str(grade_by_patient.get(item.patient_id, "")).strip(),
        })
    return pd.DataFrame(rows), np.stack(env), np.stack(tmp)


def difficult_analysis(X, y, knn):
    Xs = StandardScaler().fit_transform(X)
    n_comp = min(30, Xs.shape[1])
    Xp = PCA(n_components=n_comp, random_state=42).fit_transform(Xs)
    nn = NearestNeighbors(n_neighbors=knn + 1).fit(Xp)
    _d, idx = nn.kneighbors(Xp)
    frac_absent = 1.0 - y[idx[:, 1:]].mean(axis=1)
    return Xp, frac_absent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--knn", type=int, default=15)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cnn = load_cnn_module()
    print("Extracting envelope + temporal descriptors...")
    meta, X_env, X_tmp = build(cnn, args.max_recordings)
    X_env = np.nan_to_num(X_env, nan=0.0, posinf=0.0, neginf=0.0)
    X_tmp = np.nan_to_num(X_tmp, nan=0.0, posinf=0.0, neginf=0.0)
    y = meta["recording_present"].to_numpy(dtype=int)
    print(f"Recordings: {len(meta)} | Present: {int(y.sum())} | Absent: {int((y==0).sum())}")

    X_both = np.concatenate([X_env, X_tmp], axis=1)

    results = {}
    for name, X in [("envelope", X_env), ("envelope+temporal", X_both), ("temporal-only", X_tmp)]:
        _Xp, frac_absent = difficult_analysis(X, y, args.knn)
        meta[f"frac_absent_{name}"] = frac_absent
        present = frac_absent[y == 1]
        difficult = int((present >= 0.8).sum())
        easy = int((present <= 0.2).sum())
        results[name] = (difficult, easy, present.mean())

    # grade breakdown of difficulty change
    pres = meta[y == 1].copy()
    moved = pres[(pres["frac_absent_envelope"] >= 0.8) & (pres["frac_absent_envelope+temporal"] < 0.8)]

    print("\n=== Present 'difíceis' (>=80% vizinhos Absent) por descritor ===")
    print(f"{'descritor':>20} {'difíceis':>9} {'fáceis(<=20%)':>14} {'média frac_absent':>18}")
    for name, (d, e, m) in results.items():
        print(f"{name:>20} {d:>9} {e:>14} {m:>18.3f}")
    print(f"\nPresent que SAÍRAM de 'difícil' ao adicionar temporal: {len(moved)}")
    if len(moved):
        print("Grading desses recuperados:")
        print(moved["grading"].value_counts().to_string())

    # projection plot for envelope+temporal
    Xp_both, _ = difficult_analysis(X_both, y, args.knn)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(Xp_both)
    plt.figure(figsize=(9, 7))
    for val, lab, c in [(0, "Absent", "#3b6dd6"), (1, "Present", "#d63b3b")]:
        m = y == val
        plt.scatter(tsne[m, 0], tsne[m, 1], s=8, alpha=0.6, label=lab, c=c)
    plt.legend(); plt.title("t-SNE envelope+temporal"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "projection_tsne_temporal.png", dpi=130)
    plt.close()

    # write summary
    lines = [
        "# Teste: features de dinâmica temporal vs envelope estático",
        "",
        f"- Recordings: {len(meta)} (Present {int(y.sum())}, Absent {int((y==0).sum())}); vizinhança k={args.knn}",
        "",
        "## Present 'difíceis' (>=80% vizinhos Absent no espaço PCA)",
        "",
        "| descritor | difíceis | fáceis (<=20%) | média frac_absent |",
        "|---|---:|---:|---:|",
    ]
    for name, (d, e, m) in results.items():
        lines.append(f"| {name} | {d} | {e} | {m:.3f} |")
    lines += [
        "",
        f"- Present recuperados (saíram de difícil ao add temporal): **{len(moved)}**",
        "",
        "## Grading dos recuperados",
        "",
        (moved["grading"].value_counts().to_frame("n").to_markdown() if len(moved) else "_nenhum_"),
        "",
        "## Leitura",
        "",
        "Se 'envelope+temporal' reduz os difíceis vs 'envelope', a dinâmica temporal carrega sinal "
        "que o espectro médio descarta — justificando adicionar essas features ao input do CNN. "
        "Se não muda, o murmúrio suave realmente não é separável por áudio de sístole nessa representação.",
        "",
    ]
    (OUTPUT_DIR / "summary_temporal.md").write_text("\n".join(lines))
    np.savez_compressed(OUTPUT_DIR / "descriptors_temporal.npz",
                        X_env=X_env, X_tmp=X_tmp, **{c: meta[c].to_numpy() for c in meta.columns})
    print(f"\nOutputs em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
