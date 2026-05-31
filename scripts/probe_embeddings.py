"""Geometria dos embeddings do lb300: os faint PERDIDOS estão misturados com os Absent (limite de
representação) ou separáveis mas mal-classificados pelo head (consertável)?

Carrega os checkpoints CNN por fold, extrai o pooled embedding (model.encode) de cada gravação,
agrega por paciente (gravação de maior prob = a que dirige a decisão por max), e testa a
separabilidade missed-faint × Absent no espaço de embedding (classificador não-linear) vs o head.
Caveat: usa segmentação GROUND-TRUTH (não o TCN) p/ montar o phase-contrast — aproximação aceitável
para a geometria; recomputa a norma global por fold a partir do treino.
"""
import sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, torch
import scipy.io.wavfile as wav
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

from nested_tcn_systole_cnn.cnn import (ModelConfig, StftConfig, SystoleDilatedCNN,
                                        build_items, compute_freq_norm_stats, phase_contrast_spectrogram,
                                        stratified_patient_folds, load_patient_context)
from nested_tcn_systole_cnn.cnn.segments import read_segments

RUN = Path("experiments/nested_tcn_systole_cnn/phase_contrast_lowband300_reuse_tcn")
cfg = StftConfig(target_sample_rate=4000, n_fft=128, hop_length=32, high_hz=300.0, max_frames=256,
                 min_systole_seconds=0.1, systole_threshold=0.45, systole_margin_ms=50.0, low_hz=0.0,
                 stft_segment_mode="per-segment")

# 1. specs phase-contrast lowband (GT) por gravação
items = build_items(Path("circor-heart-sound-1.0.3"), ["AV","PV","TV","MV"], None, target="murmur")
rec = {}  # recording_id -> (spec, patient_id, recording_present, target)
for it in items:
    seg = read_segments(it.tsv_path)
    if not (seg["label"]==2).any() or not (seg["label"]==4).any():
        continue
    sr, x = wav.read(it.wav_path)
    s = phase_contrast_spectrogram(x.astype(np.float32), sr, seg, cfg)
    if s.size == 0: continue
    rec[it.recording_id] = (s.astype(np.float32), it.patient_id, it.recording_present, 1 if it.murmur=="Present" else 0)
print(f"gravações com spec: {len(rec)}", flush=True)

ctx = load_patient_context(Path("circor-heart-sound-1.0.3"))
grade = dict(zip(ctx.patient_id.astype(str), ctx.get("systolic_murmur_grading")))

pat_meta = pd.DataFrame([(p, t) for _,(s,p,rp,t) in rec.items()], columns=["patient_id","target"]).drop_duplicates("patient_id")
pids = pat_meta.patient_id.astype(str).to_numpy(); y = pat_meta.target.to_numpy()
fold_lists = stratified_patient_folds(pids, y, 5, 42)  # lista de arrays: ids de val por fold
fold_of = {}
for f, arr in enumerate(fold_lists):
    for p in arr:
        fold_of[str(p)] = f

mcfg = ModelConfig(freq_bins=next(iter(rec.values()))[0].shape[0], max_frames=256, base_channels=16,
                   dropout=0.25, dilations=(1,2,4,8,16,32), pooling="attention", encoder_block="multiscale")
dev = torch.device("cpu")
recs = list(rec.items())

emb_pat, prob_pat, tgt_pat, pid_list = [], [], [], []
for f in range(5):
    ck = torch.load(RUN / f"fold_{f+1}" / "cnn" / f"fold_{f+1}_best_model.pt", map_location="cpu", weights_only=False)
    model = SystoleDilatedCNN(mcfg).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=True)  # strict: pega mismatch
    model.eval()
    mean, std = float(ck["spectrogram_mean"]), float(ck["spectrogram_std"])  # norma EXATA do treino
    va_ids = set(p for p in pids if fold_of[p] == f)
    by_pat = {}
    with torch.no_grad():
        for rid,(s,p,rp,t) in recs:
            if p not in va_ids: continue
            xn = ((s - mean)/std).astype(np.float32)
            xb = torch.from_numpy(xn).unsqueeze(0).to(dev)
            e = model.encode(xb).squeeze(0).numpy()
            pr = torch.sigmoid(model(xb)).item()
            if p not in by_pat or pr > by_pat[p][1]:
                by_pat[p] = (e, pr, t)
    for p,(e,pr,t) in by_pat.items():
        emb_pat.append(e); prob_pat.append(pr); tgt_pat.append(t); pid_list.append(p)

E = np.stack(emb_pat); P = np.array(prob_pat); T = np.array(tgt_pat)
G = np.array([str(grade.get(p,"")) for p in pid_list])
print(f"pacientes: {len(T)}  | sanidade AUROC(prob recomputada vs target)={roc_auc_score(T,P):.3f} (esperado ~0.9 se carregou certo)", flush=True)

absent = T==0
miss = (G=="I/VI") & (P<0.5)   # faint perdidos
caught = (G=="I/VI") & (P>=0.5)
print(f"\nAbsent={absent.sum()}  I/VI perdidos={miss.sum()}  I/VI pegos={caught.sum()}", flush=True)

# Separabilidade no EMBEDDING: missed-faint vs Absent (classificador não-linear, CV)
mask = absent | miss
Xs = E[mask]; ys = miss[mask].astype(int)
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
oof = cross_val_predict(rf, Xs, ys, cv=5, method="predict_proba")[:,1]
auc_emb = roc_auc_score(ys, oof)
print(f"\n=== Separabilidade missed-faint × Absent NO EMBEDDING (RF, CV) ===")
print(f"  AUROC={auc_emb:.3f}")
print(f"  (head linear já dá ~0.5 nesses, pois prob<0.5 ~ Absent. Se RF>>0.7 => sinal está no encoder, head perde (b);")
print(f"   se RF~0.5-0.6 => inseparável no encoder = limite de representação (a))")
# distância média no embedding
cen_ab = E[absent].mean(0); cen_ca = E[caught].mean(0)
def d(v): return np.linalg.norm(v-cen_ab)
print(f"\n  dist média ao centroide Absent: I/VI pegos={np.mean([d(E[i]) for i in np.where(caught)[0]]):.2f}  "
      f"I/VI perdidos={np.mean([d(E[i]) for i in np.where(miss)[0]]):.2f}  Absent={np.mean([d(E[i]) for i in np.where(absent)[0]]):.2f}")
