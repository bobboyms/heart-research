"""Juiz de qualidade (SQI): treina no PhysioNet 2016 (rótulo SQI 0/1) e pontua as gravações do CirCor.

Estágio A: features de qualidade (do sinal inteiro) -> LogisticRegression (transferível) prevê SQI; CV AUROC.
Estágio B: mesmas features no CirCor -> score de qualidade por gravação; sanidade (correlação com a
           confiança de segmentação do TCN, se disponível) + distribuição.
Não usa a representação phase-contrast (qualidade é do registro todo, não da sístole).
"""
import sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, scipy.io.wavfile as wav
from scipy.signal import welch, resample_poly
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

SR = 2000  # domínio comum (PhysioNet nativo); CirCor resampla p/ cá

def features(x, sr):
    if sr != SR:
        x = resample_poly(x, SR, sr)
    x = x.astype(np.float64)
    peak = np.abs(x).max() + 1e-9
    xn = x / peak
    f, P = welch(xn, SR, nperseg=1024)
    P = P + 1e-12; Pn = P / P.sum()
    flat = np.exp(np.mean(np.log(P))) / np.mean(P)          # spectral flatness (ruído->1)
    ent = -np.sum(Pn * np.log(Pn)) / np.log(len(Pn))         # entropia espectral norm.
    cent = np.sum(f * Pn)                                     # centroide
    lo = P[(f >= 0) & (f < 25)].sum() / P.sum()              # baseline/wander
    hi = P[f > 400].sum() / P.sum()                          # hiss/alta-freq
    mid = P[(f >= 25) & (f <= 300)].sum() / P.sum()          # banda cardíaca
    zcr = np.mean(np.abs(np.diff(np.sign(xn)))) / 2          # zero-crossing
    kurt = kurtosis(xn)                                      # impulsividade
    sat = np.mean(np.abs(xn) > 0.99)                         # saturação
    # periodicidade (batimento limpo): pico de autocorr 0.3-1.5s, via FFT (O(n log n)) e em até 15s
    seg = xn[: 15 * SR]
    n = len(seg); nfft = 1 << int(np.ceil(np.log2(2 * n)))
    fx = np.fft.rfft(seg - seg.mean(), nfft)
    ac = np.fft.irfft(fx * np.conj(fx), nfft)[:n]; ac = ac / (ac[0] + 1e-12)
    lo_lag, hi_lag = int(0.3*SR), min(int(1.5*SR), n-1)
    periodicity = ac[lo_lag:hi_lag].max() if hi_lag > lo_lag else 0.0
    rms = np.sqrt(np.mean(xn**2))
    return [flat, ent, cent, lo, hi, mid, zcr, kurt, sat, periodicity, rms]

FEAT_NAMES = ["flatness","entropy","centroid","lo_energy","hi_energy","mid_energy","zcr","kurtosis","saturation","periodicity","rms"]

def physionet_data(root):
    X, y, ids = [], [], []
    for ref in sorted(root.glob("annotations/updated/training-?/REFERENCE_withSQI.csv")):
        for line in ref.read_text().splitlines():
            rid, _cls, sqi = line.split(",")
            wavp = next(iter(glob.glob(str(root / "training-?" / f"{rid.strip()}.wav"))), None)
            if not wavp: continue
            sr, x = wav.read(wavp)
            if len(x) < SR: continue
            X.append(features(x, sr)); y.append(int(sqi.strip())); ids.append(rid.strip())
    return np.array(X), np.array(y), ids

def circor_scores(root, model, scaler):
    rows = []
    for wavp in sorted(glob.glob(str(root / "training_data" / "*.wav"))):
        sr, x = wav.read(wavp)
        if len(x) < SR: continue
        q = model.predict_proba(scaler.transform([features(x, sr)]))[0, 1]
        rows.append({"recording_id": Path(wavp).stem, "quality": q})
    return pd.DataFrame(rows)

def main():
    pn = Path("physionet-2016"); circ = Path("circor-heart-sound-1.0.3")
    print("Estágio A: features PhysioNet...", flush=True)
    X, y, _ = physionet_data(pn)
    print(f"  n={len(y)}  boas(SQI=1)={int(y.sum())}  ruins(0)={int((y==0).sum())}", flush=True)
    sc = StandardScaler().fit(X)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"),
                            sc.transform(X), y, cv=cv, method="predict_proba")[:, 1]
    # nota: SQI=1 é "boa"; AUROC mede prever boa-vs-ruim
    print(f"  CV AUROC (prever qualidade)={roc_auc_score(y, oof):.3f}  AUPRC(boa)={average_precision_score(y, oof):.3f}", flush=True)
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    coef = dict(sorted(zip(FEAT_NAMES, model.coef_[0]), key=lambda t: -abs(t[1])))
    print("  pesos (|maiores|):", {k: round(v,2) for k,v in list(coef.items())[:6]}, flush=True)

    print("\nEstágio B: pontuando CirCor...", flush=True)
    cs = circor_scores(circ, model, sc)
    out = Path("experiments/sqi"); out.mkdir(parents=True, exist_ok=True)
    cs.to_csv(out / "circor_quality_scores.csv", index=False)
    print(f"  CirCor gravações pontuadas: {len(cs)}", flush=True)
    print(f"  quality: média={cs.quality.mean():.3f}  <0.5={int((cs.quality<0.5).sum())} ({100*(cs.quality<0.5).mean():.0f}%)  <0.3={int((cs.quality<0.3).sum())}", flush=True)
    # sanidade: correlação com confiança de segmentação do TCN (se houver)
    seg = Path("feature extraction/Grupo B v2 features relativas por local com TCN predito/outputs/predicted_segmentation_quality.csv")
    if seg.exists():
        sg = pd.read_csv(seg)[["recording_id","mean_segment_confidence"]].dropna()
        m = cs.merge(sg, on="recording_id", how="inner")
        if len(m) > 10:
            print(f"  sanidade: corr(quality, conf_segmentação_TCN)={m.quality.corr(m.mean_segment_confidence):.3f} (n={len(m)})", flush=True)
    print(f"\nsalvo: {out/'circor_quality_scores.csv'}")

if __name__ == "__main__":
    main()
