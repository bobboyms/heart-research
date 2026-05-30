"""Probe: a Wavelet Scattering Transform (WST) da sístole carrega sinal NOVO além do lb300?

Mede (1) WST sozinho (LR com CV paciente) e (2) fusão tardia lb300 + WST. Se a fusão não somar,
WST é redundante com o que o CNN-sobre-STFT já extrai.
"""
import sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scipy.special as _sp
if not hasattr(_sp, "sph_harm"):
    _sp.sph_harm = getattr(_sp, "sph_harm_y", lambda *a, **k: 0)  # kymatio 0.3 vs scipy novo

import numpy as np, pandas as pd, scipy.io.wavfile as wav, torch
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from kymatio.torch import Scattering1D
from nested_tcn_systole_cnn.cnn.segments import read_segments

L = 4096  # ~1s de sístole concatenada @4000Hz
S = Scattering1D(J=6, shape=L, Q=8)

def systole_audio(wavp, tsvp):
    fs, x = wav.read(wavp); x = x.astype(np.float32)
    if fs != 4000:
        x = resample_poly(x, 4000, fs); fs = 4000
    x = x / (np.abs(x).max() + 1e-9)
    seg = read_segments(Path(tsvp))
    chunks = [x[int(a*fs):int(b*fs)] for a, b, l in seg[["start_time","end_time","label"]].itertuples(index=False) if l == 2]
    if not chunks:
        return None
    s = np.concatenate(chunks)
    if len(s) >= L: s = s[:L]
    else: s = np.pad(s, (0, L - len(s)))
    return s

def wst_vec(s):
    with torch.no_grad():
        c = S(torch.from_numpy(s).float().unsqueeze(0)).squeeze(0).numpy()  # (126, 64)
    return np.log1p(np.abs(c)).mean(axis=1)  # média no tempo -> 126

base = "circor-heart-sound-1.0.3/training_data"
oof = pd.read_csv("experiments/nested_tcn_systole_cnn/phase_contrast_lowband300_reuse_tcn/patient_oof_predictions.csv")
oof["pid"] = oof.patient_id.astype(str)
feats = {}
for i, pid in enumerate(oof.pid):
    vecs = []
    for tsv in glob.glob(f"{base}/{pid}_*.tsv"):
        wavp = tsv[:-4] + ".wav"
        if not os.path.exists(wavp): continue
        s = systole_audio(wavp, tsv)
        if s is not None: vecs.append(wst_vec(s))
    if vecs: feats[pid] = np.mean(vecs, axis=0)
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(oof)}", flush=True)

oof = oof[oof.pid.isin(feats)].copy()
W = np.stack([feats[p] for p in oof.pid])
y = oof.target.to_numpy()
plb = oof.prob_present_calibrated.to_numpy()
logit = np.log(np.clip(plb,1e-4,1-1e-4)/(1-np.clip(plb,1e-4,1-1e-4)))

wst_only = np.zeros(len(oof)); fused = np.zeros(len(oof))
for fo in sorted(oof.fold.unique()):
    tr = (oof.fold != fo).values; va = (oof.fold == fo).values
    sc = StandardScaler().fit(W[tr])
    Wt, Wv = sc.transform(W[tr]), sc.transform(W[va])
    wst_only[va] = LogisticRegression(max_iter=2000, C=0.5).fit(Wt, y[tr]).predict_proba(Wv)[:,1]
    Xtr = np.column_stack([logit[tr], Wt]); Xv = np.column_stack([logit[va], Wv])
    fused[va] = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr, y[tr]).predict_proba(Xv)[:,1]

print(f"\nn={len(oof)} Present={int(y.sum())}")
print(f"  WST sozinho:          AUPRC={average_precision_score(y,wst_only):.4f} AUROC={roc_auc_score(y,wst_only):.4f}")
print(f"  lb300 sozinho:        AUPRC={average_precision_score(y,plb):.4f} AUROC={roc_auc_score(y,plb):.4f}")
print(f"  lb300 + WST (fusão):  AUPRC={average_precision_score(y,fused):.4f} AUROC={roc_auc_score(y,fused):.4f}")
