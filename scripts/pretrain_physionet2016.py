"""POC de transfer: pré-treina o encoder da CNN sistólica no PhysioNet/CinC 2016.

Objetivo (prova de conceito leve): verificar se o encoder do `SystoleDilatedCNN`, alimentado com a
MESMA representação do melhor modelo CirCor (phase-contrast na banda baixa, `--high-hz 300`), aprende
uma representação útil de PCG na tarefa Normal/Abnormal do PhysioNet 2016. Se sim, os pesos do encoder
podem ser carregados e fine-tunados no CirCor-murmur (passo seguinte).

Segmentação: usa as anotações Springer do próprio PhysioNet (estados S1/systole/S2/diastole), em 2000 Hz,
evitando rodar o TCN pediátrico fora de domínio. Saída: `--out` com o state_dict do encoder + pool.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.io.wavfile as wavfile
import torch
from torch import nn
from torch.utils.data import DataLoader

from nested_tcn_systole_cnn.cnn import (
    ModelConfig,
    SpectrogramDataset,
    StftConfig,
    SystoleDilatedCNN,
    average_precision,
    compute_freq_norm_stats,
    phase_contrast_spectrogram,
    roc_auc,
    set_seed,
)

_STATE_TO_LABEL = {"s1": 1, "systole": 2, "s2": 3, "diastole": 4}
_ANN_SAMPLE_RATE = 2000  # annotation indices are in the wav's native 2000 Hz


def load_labels(root: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for ref in sorted(root.glob("training-?/REFERENCE.csv")):
        for line in ref.read_text().splitlines():
            rid, lab = line.split(",")
            labels[rid.strip()] = 1 if lab.strip() == "1" else 0  # Abnormal=1, Normal=0
    return labels


def _state_name(cell) -> str:
    a = np.array(cell).flatten()
    while a.size and not isinstance(a[0], (str, np.str_)):  # unwrap nested object arrays
        a = np.array(a[0]).flatten()
    return str(a[0]).strip().lower() if a.size else ""


def load_segments(mat_path: Path) -> pd.DataFrame:
    m = sio.loadmat(mat_path)
    key = next(k for k in m if not k.startswith("__"))
    rows = m[key]
    idx = np.array([int(np.array(r[0]).flatten()[0]) for r in rows], dtype=float)
    states = [_state_name(r[1]) for r in rows]
    starts, ends, labels = [], [], []
    for i in range(len(rows) - 1):
        lab = _STATE_TO_LABEL.get(states[i], 0)
        starts.append(idx[i] / _ANN_SAMPLE_RATE)
        ends.append(idx[i + 1] / _ANN_SAMPLE_RATE)
        labels.append(lab)
    return pd.DataFrame({"start_time": starts, "end_time": ends, "label": labels})


def find_annotation(root: Path, rid: str) -> Path | None:
    cands = glob.glob(str(root / "annotations" / "hand_corrected" / "*" / f"{rid}_StateAns.mat"))
    return Path(cands[0]) if cands else None


def build_dataset(root: Path, cfg: StftConfig, limit: int | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = load_labels(root)
    specs, ys, ids = [], [], []
    rids = sorted(labels)
    if limit:
        rids = rids[:limit]
    for n, rid in enumerate(rids):
        ann = find_annotation(root, rid)
        if ann is None:
            continue
        wav_path = next(iter(glob.glob(str(root / "training-?" / f"{rid}.wav"))), None)
        if wav_path is None:
            continue
        sr, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float32)
        segments = load_segments(ann)
        if not (segments["label"] == 2).any() or not (segments["label"] == 4).any():
            continue  # need both systole and diastole for the contrast
        spec = phase_contrast_spectrogram(audio, sr, segments, cfg)
        if spec.size == 0:
            continue
        specs.append(spec.astype(np.float32))
        ys.append(labels[rid])
        ids.append(rid)
        if (n + 1) % 500 == 0:
            print(f"  processados {n + 1}/{len(rids)} (válidos {len(specs)})", flush=True)
    return np.stack(specs), np.asarray(ys, dtype=np.float32), ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--physionet-dir", type=Path, default=Path("physionet-2016"))
    p.add_argument("--out", type=Path, default=Path("experiments/transfer/physionet2016_encoder.pt"))
    p.add_argument("--high-hz", type=float, default=300.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--device", default="mps")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="smoke: limita nº de gravações")
    args = p.parse_args()

    set_seed(args.seed)
    cfg = StftConfig(target_sample_rate=4000, n_fft=128, hop_length=32, high_hz=args.high_hz,
                     max_frames=256, min_systole_seconds=0.1, systole_threshold=0.45,
                     systole_margin_ms=50.0, low_hz=0.0, stft_segment_mode="per-segment")
    print("Construindo dataset PhysioNet (phase-contrast lowband)...", flush=True)
    specs, y, ids = build_dataset(args.physionet_dir, cfg, args.limit)
    print(f"Gravações válidas: {len(y)}  | Abnormal={int(y.sum())} ({100*y.mean():.0f}%)  | spec shape={specs.shape[1:]}", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * args.val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    mean, std = compute_freq_norm_stats(specs[tr_idx], "global")

    device = torch.device(args.device if torch.backends.mps.is_available() or args.device == "cpu" else "cpu")
    train_ds = SpectrogramDataset(specs[tr_idx], y[tr_idx], mean, std)
    val_ds = SpectrogramDataset(specs[val_idx], y[val_idx], mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model_cfg = ModelConfig(freq_bins=specs.shape[1], max_frames=specs.shape[2], base_channels=16,
                            dropout=0.25, dilations=(1, 2, 4, 8, 16, 32), pooling="attention",
                            encoder_block="multiscale")
    model = SystoleDilatedCNN(model_cfg).to(device)
    pos, neg = float(y[tr_idx].sum()), float((y[tr_idx] == 0).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0), device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)

    best_auroc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in ((b[0].to(device), b[1].to(device)) for b in train_loader):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        probs = []
        with torch.no_grad():
            for b in val_loader:
                probs.append(torch.sigmoid(model(b[0].to(device))).cpu().numpy())
        vp = np.concatenate(probs)
        au, ap = roc_auc(y[val_idx], vp), average_precision(y[val_idx], vp)
        print(f"Epoch {epoch:02d}/{args.epochs}  val_AUROC={au:.4f}  val_AUPRC={ap:.4f}", flush=True)
        if au > best_auroc:
            best_auroc = au
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "model_cfg": vars(model_cfg), "best_val_auroc": best_auroc,
                "norm_mean": mean, "norm_std": std, "high_hz": args.high_hz}, args.out)
    print(f"\nMELHOR val_AUROC PhysioNet Normal/Abnormal = {best_auroc:.4f}  -> encoder salvo em {args.out}", flush=True)


if __name__ == "__main__":
    main()
