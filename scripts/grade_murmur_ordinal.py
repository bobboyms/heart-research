"""Modelo dedicado: graduar a INTENSIDADE do sopro (ordinal I<II<III) nos pacientes Present.

Separado do pipeline principal (Present/Absent). Reusa a melhor representação do projeto
(phase-contrast banda baixa <=300 Hz) + o encoder SystoleDilatedCNN, mas troca a cabeça para uma
saída de REGRESSÃO ordinal (alvo numérico 0=I/VI, 1=II/VI, 2=III/VI), treinada com MSE. Respeita a
ordem I<II<III, o que é robusto com a classe do meio (II/VI, n=28) minúscula.

Segmentação: ground-truth (.tsv do CirCor), evitando o TCN. Validação: 5 folds por paciente,
estratificados por grau. Reporta MAE (em unidades de grau), correlação de Spearman, acurácia e
matriz de confusão (predição arredondada), tudo OOF paciente-level.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from nested_tcn_systole_cnn.cnn import (
    ModelConfig,
    SpectrogramDataset,
    StftConfig,
    SystoleDilatedCNN,
    build_items,
    compute_freq_norm_stats,
    phase_contrast_spectrogram,
    set_seed,
)
from nested_tcn_systole_cnn.cnn.models import SystoleRNN
from nested_tcn_systole_cnn.cnn.segments import read_segments

_GRADE_TO_LEVEL = {"I/VI": 0, "II/VI": 1, "III/VI": 2}
_LEVEL_NAME = {0: "I/VI", 1: "II/VI", 2: "III/VI"}


def build_dataset(dataset_dir: Path, cfg: StftConfig, with_absent: bool = False):
    meta = pd.read_csv(dataset_dir / "training_data.csv", dtype={"Patient ID": str})
    grade_by_patient = {str(r["Patient ID"]): r["Systolic murmur grading"] for _, r in meta.iterrows()}
    items = build_items(dataset_dir, ["AV", "PV", "TV", "MV"], None, target="murmur")
    specs, levels, patients = [], [], []
    for it in items:
        if with_absent:
            # 3 níveis ordenados: Ausente(0) < fraco I/II(1) < forte III(2), em TODOS os pacientes.
            if it.murmur == "Absent":
                level = 0
            elif it.recording_present:  # Present no foco do sopro
                grade = grade_by_patient.get(it.patient_id)
                if grade not in _GRADE_TO_LEVEL:
                    continue
                level = 2 if _GRADE_TO_LEVEL[grade] == 2 else 1  # III->forte(2); I/II->fraco(1)
            else:
                continue  # Present fora do foco do sopro: ambíguo, pula
        else:
            if not it.recording_present:  # só gravações no(s) local(is) do sopro (location-aware)
                continue
            grade = grade_by_patient.get(it.patient_id)
            if grade not in _GRADE_TO_LEVEL:
                continue
            level = _GRADE_TO_LEVEL[grade]
        segments = read_segments(it.tsv_path)
        if not (segments["label"] == 2).any() or not (segments["label"] == 4).any():
            continue
        import scipy.io.wavfile as wavfile
        sr, audio = wavfile.read(it.wav_path)
        spec = phase_contrast_spectrogram(audio.astype(np.float32), sr, segments, cfg)
        if spec.size == 0:
            continue
        specs.append(spec.astype(np.float32))
        levels.append(level)
        patients.append(it.patient_id)
    return np.stack(specs), np.asarray(levels, dtype=np.float32), np.asarray(patients)


def stratified_patient_folds(patients, levels, folds, seed):
    # nível por paciente (todas as gravações de um paciente têm o mesmo grau)
    pat_level = {p: int(l) for p, l in zip(patients, levels)}
    rng = np.random.default_rng(seed)
    assign = {}
    for lvl in [0, 1, 2]:
        pids = np.array([p for p in sorted(set(patients)) if pat_level[p] == lvl])
        pids = rng.permutation(pids)
        for i, p in enumerate(pids):
            assign[p] = i % folds
    return np.array([assign[p] for p in patients])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=Path("circor-heart-sound-1.0.3"))
    ap.add_argument("--out", type=Path, default=Path("experiments/grading"))
    ap.add_argument("--high-hz", type=float, default=300.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", choices=["cnn", "rnn"], default="rnn",
                    help="rnn = biGRU sobre a sequência de energia-da-banda-relativa-à-fase (input phase-contrast lowband)")
    ap.add_argument("--rnn-hidden", type=int, default=64)
    ap.add_argument("--rnn-layers", type=int, default=2)
    ap.add_argument("--binary-strong", action="store_true",
                    help="alvo binário forte (III/VI) vs fraco (I+II) — reporta AUROC/AUPRC")
    ap.add_argument("--with-absent", action="store_true",
                    help="ordinal 3 níveis em TODOS os pacientes: Ausente(0) < fraco I/II(1) < forte III(2)")
    ap.add_argument("--init-encoder", type=str, default=None,
                    help="checkpoint pré-treinado (PhysioNet) p/ inicializar encoder+pool")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = StftConfig(target_sample_rate=4000, n_fft=128, hop_length=32, high_hz=args.high_hz,
                     max_frames=256, min_systole_seconds=0.1, systole_threshold=0.45,
                     systole_margin_ms=50.0, low_hz=0.0, stft_segment_mode="per-segment")
    level_name = {0: "Ausente", 1: "fraco", 2: "forte"} if args.with_absent else _LEVEL_NAME
    print("Construindo dataset (phase-contrast lowband, GT seg)...", flush=True)
    specs, levels, patients = build_dataset(args.dataset_dir, cfg, with_absent=args.with_absent)
    n_pat = len(set(patients))
    print(f"Gravações: {len(levels)} de {n_pat} pacientes | níveis: "
          f"{ {level_name[k]: int((levels==k).sum()) for k in [0,1,2]} }", flush=True)

    binary = bool(args.binary_strong)
    targets = (levels == 2).astype(np.float32) if binary else levels.astype(np.float32)
    if binary:
        print(f"  alvo BINÁRIO forte(III) vs fraco(I+II): positivos={int(targets.sum())}/{len(targets)}", flush=True)

    fold_of = stratified_patient_folds(patients, levels, args.folds, args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.backends.mps.is_available()) else "cpu")

    def load_pretrained(model):
        if not args.init_encoder:
            return
        ck = torch.load(args.init_encoder, map_location="cpu", weights_only=False)
        pre = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        own = model.state_dict()
        keep = {k: v for k, v in pre.items() if k in own and own[k].shape == v.shape
                and (k.startswith("encoder.") or k.startswith("pool."))}
        model.load_state_dict({**own, **keep}, strict=False)
        return len(keep)

    rec_pred = np.zeros(len(levels), dtype=np.float32)
    for fold in range(args.folds):
        tr = np.flatnonzero(fold_of != fold)
        va = np.flatnonzero(fold_of == fold)
        mean, std = compute_freq_norm_stats(specs[tr], "global")
        tl = DataLoader(SpectrogramDataset(specs[tr], targets[tr], mean, std), batch_size=args.batch_size, shuffle=True)
        vl = DataLoader(SpectrogramDataset(specs[va], targets[va], mean, std), batch_size=args.batch_size, shuffle=False)
        mcfg = ModelConfig(freq_bins=specs.shape[1], max_frames=specs.shape[2], base_channels=16,
                           dropout=0.25, dilations=(1, 2, 4, 8, 16, 32), pooling="attention",
                           encoder_block="multiscale", arch=args.arch,
                           rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers, rnn_type="gru")
        model = (SystoleRNN(mcfg) if args.arch == "rnn" else SystoleDilatedCNN(mcfg)).to(device)
        nload = load_pretrained(model)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
        loss_fn = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()
        best_score, best_pred = -1e9, None  # ambos maximizam: binário=AUROC, ordinal=-MAE
        for epoch in range(1, args.epochs + 1):
            model.train()
            for b in tl:
                xb, yb = b[0].to(device), b[1].to(device)
                opt.zero_grad(set_to_none=True)
                loss_fn(model(xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            model.eval()
            preds = []
            with torch.no_grad():
                for b in vl:
                    preds.append(model(b[0].to(device)).cpu().numpy())
            vp = np.concatenate(preds)
            if binary:
                prob = 1 / (1 + np.exp(-vp)); score = roc_auc_score(targets[va], prob)
                better = score > best_score
            else:
                score = -float(np.mean(np.abs(vp - targets[va]))); better = score > best_score
            if better:
                best_score, best_pred = score, vp
        rec_pred[va] = best_pred
        tag = f"AUROC={best_score:.3f}" if binary else f"MAE={-best_score:.3f}"
        print(f"Fold {fold+1}/{args.folds}: val {tag}" + (f"  (init {nload} tensores)" if nload else ""), flush=True)

    # agrega gravação -> paciente e avalia OOF paciente-level
    dfp = pd.DataFrame({"patient": patients, "true_level": levels, "pred": rec_pred})
    pat = dfp.groupby("patient").agg(true_level=("true_level", "first"), pred=("pred", "mean")).reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    if binary:
        y = (pat["true_level"].to_numpy() == 2).astype(int)
        prob = 1 / (1 + np.exp(-pat["pred"].to_numpy()))
        print("\n=== OOF paciente-level — FORTE(III) vs FRACO(I+II), n=%d ===" % len(pat))
        print(f"  positivos(III)={int(y.sum())}/{len(y)}  AUPRC={average_precision_score(y,prob):.4f}  AUROC={roc_auc_score(y,prob):.4f}")
    else:
        y_true = pat["true_level"].to_numpy(); y_pred = pat["pred"].to_numpy()
        y_round = np.clip(np.rint(y_pred), 0, 2).astype(int)
        cm = confusion_matrix(y_true.astype(int), y_round, labels=[0, 1, 2])
        print("\n=== OOF paciente-level (n=%d) ===" % len(pat))
        print(f"  MAE(grau)={np.mean(np.abs(y_pred-y_true)):.3f}  Spearman={spearmanr(y_true,y_pred).correlation:.3f}"
              f"  acc-exata={(y_round==y_true.astype(int)).mean():.3f}  dentro-de-1-grau={(np.abs(y_round-y_true.astype(int))<=1).mean():.3f}")
        print(f"            pred:  {level_name[0]:>7} {level_name[1]:>7} {level_name[2]:>7}")
        for i, name in level_name.items():
            print(f"    true {name:8s}  {cm[i,0]:7d} {cm[i,1]:7d} {cm[i,2]:7d}")
    pat.to_csv(args.out / "grading_oof_patient.csv", index=False)
    print(f"\nsalvo: {args.out/'grading_oof_patient.csv'}")


if __name__ == "__main__":
    main()
