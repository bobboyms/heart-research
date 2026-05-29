# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.4",
#   "scipy>=1.12",
#   "tabulate>=0.9",
#   "umap-learn>=0.5.7",
# ]
# ///
"""Grupo B v3.3 - textura na banda baixa + camada de separabilidade Present x Absent.

Duas perguntas guiam o experimento:

1. Dentro da banda baixa (<=260 Hz), quais EXTRACOES de informacao alem de energia e
   persistencia (ja cobertas pelo v3.2) descrevem o sopro? Aqui adicionamos o eixo de
   "textura ruido-vs-tonal" e "forma espectral" que estavam em falta:
   - spectral flatness / entropia do contraste (sopro = ruido de banda larga -> achatado);
   - tilt/inclinacao espectral e razoes de sub-bandas finas (25-80 / 80-150 / 150-260 Hz);
   - skew/kurtosis temporal e espectral do contraste;
   - Gini / esparsidade e flux temporal do mapa (concentracao vs espalhamento);
   - razao tonal-pico (proxy de HNR) e fracao broadband ativa;
   - razao sistole/diastole na banda.

2. QUANTO um audio com sopro se afasta de um sem sopro? O v3.x so media pureza de cluster.
   Aqui adicionamos uma camada explicita de SEPARABILIDADE:
   - por feature: AUC univariada (P[Present>Absent]), Cohen's d, Mann-Whitney U;
   - multivariada: Mahalanobis entre centroides, razao de Fisher (trace), silhueta usando o
     rotulo como particao;
   - score continuo por gravacao/paciente: distancia de Mahalanobis ao centroide Absent
     (= "o quanto este audio se afasta de um audio normal"), com AUC do proprio score.

O script NAO recomputa as features do v3.2: ele le o CSV ja salvo, extrai so a textura nova
direto dos .wav, junta por recording_id e roda a separabilidade em tres conjuntos
(v3.2 sozinho / textura nova / combinado) para mostrar se a textura ADICIONA separacao.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, mannwhitneyu, skew
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V31_SCRIPT = (
    SCRIPT_DIR.parent
    / "Grupo B v3.1 contraste robusto por ciclo"
    / "extract_robust_cycle_contrast_clusters.py"
)
V32_SCRIPT = (
    SCRIPT_DIR.parent
    / "Grupo B v3.2 murmur map realcado"
    / "extract_enhanced_murmur_map_clusters.py"
)
V32_RECORDING_CSV = (
    SCRIPT_DIR.parent
    / "Grupo B v3.2 murmur map realcado"
    / "outputs"
    / "recording_enhanced_murmur_map_features.csv"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v31 = _load_module("v31_contrast", V31_SCRIPT)
v32 = _load_module("v32_enhanced", V32_SCRIPT)


# Sub-bandas finas dentro da banda baixa onde mora o sinal do sopro.
FINE_BANDS_HZ = {
    "b25_80": (25.0, 80.0),
    "b80_150": (80.0, 150.0),
    "b150_260": (150.0, 260.0),
}

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-band texture features + separability layer.")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "circor-heart-sound-1.0.3")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--v32-recording-csv", type=Path, default=V32_RECORDING_CSV)
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--target-sample-rate", type=int, default=4000)
    # Espelha os parametros de banda baixa do v3.2.
    parser.add_argument("--low-n-fft", type=int, default=256)
    parser.add_argument("--low-hop-length", type=int, default=48)
    parser.add_argument("--low-hz", type=float, default=25.0)
    parser.add_argument("--low-map-high-hz", type=float, default=260.0)
    parser.add_argument("--robust-scale-floor", type=float, default=0.03)
    parser.add_argument("--z-clip", type=float, default=12.0)
    parser.add_argument("--center-crop", type=float, default=0.15)
    return parser.parse_args()


def gini(values: np.ndarray) -> float:
    arr = np.sort(np.abs(values.ravel()).astype(np.float64))
    n = arr.size
    if n == 0 or arr.sum() <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * arr.sum()))


def spectral_flatness(profile: np.ndarray) -> float:
    p = np.maximum(profile.astype(np.float64), 0.0) + EPS
    geo = np.exp(np.mean(np.log(p)))
    return float(geo / np.mean(p))


def spectral_entropy(profile: np.ndarray) -> float:
    p = np.maximum(profile.astype(np.float64), 0.0)
    total = p.sum()
    if total <= 0 or p.size <= 1:
        return 0.0
    p = p / total
    ent = -np.sum(p * np.log(p + EPS))
    return float(ent / np.log(p.size))


def lowband_texture_cycle(
    systole_specs: list[np.ndarray],
    diastole_specs: list[np.ndarray],
    freqs: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Features de textura por gravacao, agregadas dos ciclos sistolicos."""
    reference, scale, _floor = v32.robust_reference(diastole_specs, args.robust_scale_floor)
    diastole_mean_spec = np.concatenate(diastole_specs, axis=1).mean(axis=1)  # por freq
    band_masks = {name: (freqs >= lo) & (freqs < hi) for name, (lo, hi) in FINE_BANDS_HZ.items()}

    per_metric: dict[str, list[float]] = {}

    def collect(name: str, value: float) -> None:
        if np.isfinite(value):
            per_metric.setdefault(name, []).append(float(value))

    for spec in systole_specs:
        z = np.clip(((spec - reference) / scale).astype(np.float32), -args.z_clip, args.z_clip)
        positive = np.maximum(z, 0.0)

        freq_profile = positive.mean(axis=1)  # energia positiva por frequencia
        frame_energy = positive.mean(axis=0)  # energia positiva por tempo
        sys_mean_spec = spec.mean(axis=1)  # espectro medio bruto da sistole

        # --- Forma espectral / pitch ---
        collect("tex_flatness", spectral_flatness(freq_profile))
        collect("tex_entropy", spectral_entropy(freq_profile))
        if freqs.size >= 2 and np.any(freq_profile > 0):
            collect("tex_tilt", float(np.polyfit(freqs, freq_profile, 1)[0]))
            centroid = float(np.sum(freqs * freq_profile) / (freq_profile.sum() + EPS))
            collect("tex_freq_centroid", centroid)
            spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * freq_profile) / (freq_profile.sum() + EPS)))
            collect("tex_freq_spread", spread)

        total_band = float(freq_profile.sum()) + EPS
        for band_name, mask in band_masks.items():
            collect(f"tex_frac_{band_name}", float(freq_profile[mask].sum()) / total_band)

        # --- Textura ruido-vs-tonal ---
        collect("tex_broadband_active_fraction", float(np.mean(freq_profile > 0)))
        collect("tex_z_skew", float(skew(z, axis=None)))
        collect("tex_z_kurtosis", float(kurtosis(z, axis=None)))
        # proxy de HNR: pico tonal vs piso mediano do espectro bruto da sistole
        med = float(np.median(sys_mean_spec))
        collect("tex_tonal_peak_ratio", float(np.max(sys_mean_spec)) / (med + EPS))

        # --- Concentracao / dinamica temporal ---
        collect("tex_gini_map", gini(positive))
        if frame_energy.size >= 2:
            flux = float(np.mean(np.abs(np.diff(frame_energy))))
            collect("tex_temporal_flux", flux / (float(np.mean(frame_energy)) + EPS))
            collect("tex_frame_skew", float(skew(frame_energy)))
            collect("tex_frame_kurtosis", float(kurtosis(frame_energy)))

        # --- Excesso sistole/diastole na banda ---
        collect("tex_sys_dia_ratio", float(sys_mean_spec.mean()) / (float(diastole_mean_spec.mean()) + EPS))

    out: dict[str, float] = {"tex_cycle_count": float(len(systole_specs))}
    for name, values in per_metric.items():
        v31.aggregate_values(name, values, out)
    return out


def extract_texture(wav_path: Path, metadata: pd.DataFrame, args: argparse.Namespace) -> dict[str, float | str] | None:
    tsv_path = wav_path.with_suffix(".tsv")
    if not tsv_path.exists():
        return None
    patient_id, location = v31.parse_recording_id(wav_path)
    if location not in v31.LOCATIONS:
        return None
    if metadata.loc[metadata["Patient ID"].astype(str) == patient_id].empty:
        return None

    sample_rate, audio = v31.read_audio(wav_path)
    audio = v31.resample_audio(audio, sample_rate, args.target_sample_rate)
    segments = v31.read_segments(tsv_path)
    systole_specs, freqs = v31.segment_log_specs(
        audio, args.target_sample_rate, segments, v31.LABEL_SYSTOLE,
        args.low_n_fft, args.low_hop_length, args.low_hz, args.low_map_high_hz,
    )
    diastole_specs, _ = v31.segment_log_specs(
        audio, args.target_sample_rate, segments, v31.LABEL_DIASTOLE,
        args.low_n_fft, args.low_hop_length, args.low_hz, args.low_map_high_hz,
    )
    if not systole_specs or not diastole_specs:
        return None
    systole_specs = v32.crop_systole_specs(systole_specs, args.center_crop)

    features: dict[str, float | str] = {"recording_id": wav_path.stem}
    features.update(lowband_texture_cycle(systole_specs, diastole_specs, freqs, args))
    return features


# ------------------------------------------------------------------ #
# Camada de separabilidade
# ------------------------------------------------------------------ #

def _clean_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    sub = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = [c for c in cols if float(sub[c].std()) > 0.0]
    return sub[keep].to_numpy(dtype=np.float64), keep


def per_feature_separability(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = (df["murmur"].to_numpy() == "Present").astype(int)
    rows = []
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        if float(np.std(x)) == 0.0:
            continue
        pres, absent = x[y == 1], x[y == 0]
        auc = roc_auc_score(y, x)
        pooled = np.sqrt((np.var(pres, ddof=1) + np.var(absent, ddof=1)) / 2.0) + EPS
        cohens_d = float((pres.mean() - absent.mean()) / pooled)
        try:
            _, mwu_p = mannwhitneyu(pres, absent, alternative="two-sided")
        except ValueError:
            mwu_p = 1.0
        rows.append({
            "feature": col,
            "auc": float(auc),
            "auc_oriented": float(max(auc, 1.0 - auc)),
            "cohens_d": cohens_d,
            "mwu_p": float(mwu_p),
            "present_mean": float(pres.mean()),
            "absent_mean": float(absent.mean()),
        })
    out = pd.DataFrame(rows).sort_values("auc_oriented", ascending=False).reset_index(drop=True)
    return out


def multivariate_separation(df: pd.DataFrame, cols: list[str], label: str) -> dict[str, float | str | int]:
    y = (df["murmur"].to_numpy() == "Present").astype(int)
    X_raw, keep = _clean_matrix(df, cols)
    if not keep or X_raw.shape[1] == 0:
        return {"set": label, "n_features": 0}
    X = StandardScaler().fit_transform(X_raw)
    pres, absent = X[y == 1], X[y == 0]
    mu1, mu0 = pres.mean(axis=0), absent.mean(axis=0)
    delta = mu1 - mu0

    # Covariancia within-class (pooled) com shrinkage de Ledoit-Wolf.
    residuals = np.vstack([pres - mu1, absent - mu0])
    cov = LedoitWolf().fit(residuals)
    prec = cov.precision_
    mahalanobis = float(np.sqrt(max(delta @ prec @ delta, 0.0)))

    # Razao de Fisher por traco: scatter between / within.
    sw = np.trace(cov.covariance_)
    sb = float(delta @ delta)  # ja padronizado; |mu1-mu0|^2
    fisher_trace = float(sb / (sw + EPS))

    sil = float(silhouette_score(X, y)) if X.shape[1] >= 1 and len(np.unique(y)) == 2 else float("nan")

    # Score continuo: distancia de Mahalanobis ao centroide Absent.
    cov_absent = LedoitWolf().fit(absent - mu0)
    diff = X - mu0
    dist_to_absent = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, cov_absent.precision_, diff), 0.0))
    score_auc = float(roc_auc_score(y, dist_to_absent))

    return {
        "set": label,
        "n_features": int(len(keep)),
        "mahalanobis_centroids": mahalanobis,
        "fisher_trace_ratio": fisher_trace,
        "silhouette_by_label": sil,
        "dist_to_absent_auc": score_auc,
    }


def dist_to_absent_series(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    y = (df["murmur"].to_numpy() == "Present").astype(int)
    X_raw, keep = _clean_matrix(df, cols)
    X = StandardScaler().fit_transform(X_raw)
    absent = X[y == 0]
    mu0 = absent.mean(axis=0)
    cov_absent = LedoitWolf().fit(absent - mu0)
    diff = X - mu0
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, cov_absent.precision_, diff), 0.0))


def feature_sets(all_cols: list[str]) -> dict[str, list[str]]:
    tex = [c for c in all_cols if c.startswith("tex_")]
    v32_cols = [c for c in all_cols if not c.startswith("tex_")]
    return {
        "v32_only": v32_cols,
        "texture_only": tex,
        "combined": v32_cols + tex,
    }


def run_separability(df: pd.DataFrame, all_cols: list[str], level: str, output_dir: Path) -> list[dict]:
    sets = feature_sets(all_cols)
    summary_rows = []
    for set_name, cols in sets.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        row = multivariate_separation(df, cols, set_name)
        row["level"] = level
        summary_rows.append(row)
        per_feat = per_feature_separability(df, cols)
        per_feat.insert(0, "level", level)
        per_feat.insert(1, "feature_set", set_name)
        per_feat.to_csv(output_dir / f"separability_per_feature_{level}_{set_name}.csv", index=False)
    # score continuo no conjunto combinado
    combined = [c for c in sets["combined"] if c in df.columns]
    scores = dist_to_absent_series(df, combined)
    score_df = df[[c for c in ("patient_id", "recording_id", "murmur", "systolic_murmur_grading", "systolic_murmur_pitch") if c in df.columns]].copy()
    score_df["dist_to_absent_combined"] = scores
    score_df.sort_values("dist_to_absent_combined").to_csv(output_dir / f"dist_to_absent_{level}.csv", index=False)
    return summary_rows


def write_summary(output_path: Path, summary_df: pd.DataFrame, top_features: pd.DataFrame, n_rec: int, n_pat: int, n_tex: int) -> None:
    lines = [
        "# Grupo B v3.3 - textura banda baixa + separabilidade Present x Absent",
        "",
        "## Objetivo",
        "",
        "Adicionar (1) extracoes de TEXTURA na banda baixa (<=260 Hz) que faltavam no v3.2",
        "(flatness/entropia, tilt, sub-bandas finas, skew/kurtosis, Gini, flux, HNR-proxy) e",
        "(2) uma camada explicita de SEPARABILIDADE que mede o quanto um audio com sopro se",
        "afasta de um sem sopro (AUC por feature, Mahalanobis/Fisher/silhueta, score dist-ao-Absent).",
        "",
        "## Dados",
        "",
        f"- Gravacoes: {n_rec}",
        f"- Pacientes: {n_pat}",
        f"- Features de textura novas: {n_tex}",
        "",
        "## Leitura",
        "",
        "- `auc` por feature = P(Present > Absent). 0.5 = inutil; >=0.8 = forte.",
        "- `mahalanobis_centroids` = distancia normalizada entre os centroides Present/Absent.",
        "- `fisher_trace_ratio` = scatter entre-grupos / dentro-do-grupo (maior = mais separavel).",
        "- `silhouette_by_label` = coesao/separacao dos dois grupos no espaco de features (-1 a 1).",
        "- `dist_to_absent_auc` = AUC do score continuo de distancia ao centroide Absent.",
        "",
        "## Separabilidade multivariada por conjunto de features",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Top 20 features por AUC (combinado, nivel gravacao)",
        "",
        top_features.head(20).to_markdown(index=False),
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

    v32_df = pd.read_csv(args.v32_recording_csv)
    print(f"v3.2 recording features: {v32_df.shape}", flush=True)

    wav_paths = sorted(data_dir.glob("*.wav"))
    if args.max_recordings is not None:
        wav_paths = wav_paths[: args.max_recordings]

    tex_rows: list[dict[str, float | str]] = []
    skipped = 0
    for index, wav_path in enumerate(wav_paths, start=1):
        row = extract_texture(wav_path, metadata, args)
        if row is None:
            skipped += 1
            continue
        tex_rows.append(row)
        if index % 250 == 0:
            print(f"Processed {index}/{len(wav_paths)} recordings...", flush=True)

    if not tex_rows:
        raise RuntimeError("No texture rows extracted. Check dataset path and filters.")

    tex_df = pd.DataFrame(tex_rows)
    tex_cols = [c for c in tex_df.columns if c.startswith("tex_")]
    tex_df.to_csv(output_dir / "recording_texture_features.csv", index=False)

    # Junta textura nova com features v3.2 por recording_id (inner join).
    recording_df = v32_df.merge(tex_df, on="recording_id", how="inner")
    recording_df.to_csv(output_dir / "recording_combined_features.csv", index=False)
    print(f"Combined recording features: {recording_df.shape}", flush=True)

    all_cols = v32.numeric_feature_columns(recording_df)
    summary_rows: list[dict] = []
    summary_rows.extend(run_separability(recording_df, all_cols, "recording", output_dir))

    # Nivel paciente.
    patient_df = v31.aggregate_by_patient(recording_df, all_cols)
    patient_df.to_csv(output_dir / "patient_combined_features.csv", index=False)
    patient_cols = v32.numeric_feature_columns(patient_df)
    # reescreve feature_sets para os prefixos mean_/max_/p90_ do nivel paciente
    summary_rows.extend(run_separability_patient(patient_df, patient_cols, output_dir))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "separability_summary.csv", index=False)

    top_features = per_feature_separability(recording_df, feature_sets(all_cols)["combined"])
    top_features.to_csv(output_dir / "separability_per_feature_recording_combined_sorted.csv", index=False)

    write_summary(
        output_dir / "summary.md", summary_df, top_features,
        n_rec=len(recording_df), n_pat=len(patient_df), n_tex=len(tex_cols),
    )
    print(f"Texture rows: {len(tex_df)} (skipped {skipped}).", flush=True)
    print(f"Outputs: {output_dir}", flush=True)


def run_separability_patient(df: pd.DataFrame, all_cols: list[str], output_dir: Path) -> list[dict]:
    """No nivel paciente as features ganham prefixo mean_/max_/p90_; reclassifica em tex vs v32."""
    tex = [c for c in all_cols if "tex_" in c]
    v32_cols = [c for c in all_cols if "tex_" not in c]
    sets = {"v32_only": v32_cols, "texture_only": tex, "combined": v32_cols + tex}
    summary_rows = []
    for set_name, cols in sets.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        row = multivariate_separation(df, cols, set_name)
        row["level"] = "patient"
        summary_rows.append(row)
        per_feat = per_feature_separability(df, cols)
        per_feat.insert(0, "level", "patient")
        per_feat.insert(1, "feature_set", set_name)
        per_feat.to_csv(output_dir / f"separability_per_feature_patient_{set_name}.csv", index=False)
    combined = [c for c in sets["combined"] if c in df.columns]
    scores = dist_to_absent_series(df, combined)
    score_df = df[[c for c in ("patient_id", "murmur") if c in df.columns]].copy()
    score_df["dist_to_absent_combined"] = scores
    score_df.sort_values("dist_to_absent_combined").to_csv(output_dir / "dist_to_absent_patient.csv", index=False)
    return summary_rows


if __name__ == "__main__":
    main()
