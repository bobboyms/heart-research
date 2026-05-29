# /// script
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scipy>=1.12",
#   "tabulate>=0.9",
# ]
# ///
"""Analyze patient-level error cases from a nested Grupo H run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


CLINICAL_COLUMNS = [
    "Patient ID",
    "Age",
    "Sex",
    "Height",
    "Weight",
    "Pregnancy status",
    "Murmur",
    "Murmur locations",
    "Most audible location",
    "Systolic murmur timing",
    "Systolic murmur shape",
    "Systolic murmur grading",
    "Systolic murmur pitch",
    "Systolic murmur quality",
    "Outcome",
    "Campaign",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FN/FP cases from nested Grupo H predictions.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs_nested_weight-multiplier",
        help="Grupo H run directory containing patient_oof_predictions.csv.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "circor-heart-sound-1.0.3",
        help="CirCor dataset directory.",
    )
    parser.add_argument(
        "--probability",
        choices=["calibrated", "raw"],
        default="calibrated",
        help="Probability column used to define errors.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.20],
        help="Decision thresholds to analyze.",
    )
    parser.add_argument(
        "--skip-signal-quality",
        action="store_true",
        help="Skip WAV-derived quality proxy metrics.",
    )
    return parser.parse_args()


def threshold_slug(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def clean_value(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    text = str(value)
    if text.lower() == "nan" or text == "":
        return "Missing"
    return text


def read_patient_oof(output_dir: Path, probability: str) -> pd.DataFrame:
    path = output_dir / "patient_oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"patient_id": str})
    prob_col = f"prob_present_{probability}"
    if prob_col not in df.columns:
        raise ValueError(f"Missing probability column {prob_col} in {path}")
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def read_clinical_table(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "training_data.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    clinical = pd.read_csv(path, dtype={"Patient ID": str})
    clinical = clinical[[col for col in CLINICAL_COLUMNS if col in clinical.columns]].copy()
    clinical = clinical.rename(columns={"Patient ID": "patient_id"})
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    return clinical


def read_oof_recording_metadata(output_dir: Path, oof: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    oof_keys = oof[["patient_id", "fold"]].copy()
    oof_keys["fold"] = oof_keys["fold"].astype(int)

    for fold_dir in sorted(output_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except ValueError:
            continue
        path = fold_dir / "recording_metadata.csv"
        if not path.exists():
            continue
        meta = pd.read_csv(path, dtype={"patient_id": str, "recording_id": str})
        meta["fold"] = fold
        meta["patient_id"] = meta["patient_id"].astype(str)
        frames.append(meta)

    if not frames:
        return pd.DataFrame()

    metadata = pd.concat(frames, ignore_index=True)
    return metadata.merge(oof_keys, on=["patient_id", "fold"], how="inner")


def compute_signal_quality(dataset_dir: Path, recordings: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path, dtype={"recording_id": str})

    rows: list[dict[str, object]] = []
    for recording_id in sorted(recordings["recording_id"].dropna().astype(str).unique()):
        wav_path = dataset_dir / "training_data" / f"{recording_id}.wav"
        row: dict[str, object] = {"recording_id": recording_id}
        if not wav_path.exists():
            row["quality_error"] = "missing_wav"
            rows.append(row)
            continue

        try:
            sample_rate, data = wavfile.read(wav_path)
            signal = np.asarray(data)
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            if np.issubdtype(signal.dtype, np.integer):
                max_abs = float(np.iinfo(signal.dtype).max)
                signal = signal.astype(np.float32) / max_abs
            else:
                signal = signal.astype(np.float32)

            finite = np.isfinite(signal)
            valid = signal[finite]
            if valid.size == 0:
                row["quality_error"] = "empty_or_nonfinite"
            else:
                abs_signal = np.abs(valid)
                rms = float(np.sqrt(np.mean(valid**2)))
                peak = float(abs_signal.max())
                row.update(
                    {
                        "sample_rate": int(sample_rate),
                        "recording_seconds": float(valid.size / sample_rate),
                        "rms": rms,
                        "peak_abs": peak,
                        "crest_factor": float(peak / rms) if rms > 0 else np.nan,
                        "clipping_fraction": float(np.mean(abs_signal >= 0.999)),
                        "near_silence_fraction": float(np.mean(abs_signal < 1e-4)),
                        "zero_crossing_rate": float(np.mean(valid[1:] * valid[:-1] < 0)) if valid.size > 1 else np.nan,
                        "finite_fraction": float(finite.mean()),
                        "quality_error": "",
                    }
                )
        except Exception as exc:  # noqa: BLE001 - keep report generation robust.
            row["quality_error"] = type(exc).__name__
        rows.append(row)

    quality = pd.DataFrame(rows)
    quality.to_csv(cache_path, index=False)
    return quality


def summarize_recordings(recordings: pd.DataFrame) -> pd.DataFrame:
    if recordings.empty:
        return pd.DataFrame(columns=["patient_id"])

    agg_map = {
        "recording_id": "count",
        "systole_seconds": ["sum", "mean", "min", "max"],
        "systole_segments": ["sum", "mean", "min", "max"],
    }
    for col in ["recording_seconds", "rms", "peak_abs", "crest_factor", "clipping_fraction", "near_silence_fraction"]:
        if col in recordings.columns:
            agg_map[col] = ["mean", "min", "max"]

    patient = recordings.groupby("patient_id").agg(agg_map)
    patient.columns = ["_".join(col).strip("_") for col in patient.columns.to_flat_index()]
    patient = patient.rename(columns={"recording_id_count": "recording_count"}).reset_index()

    locations = recordings.pivot_table(
        index="patient_id",
        columns="location",
        values="systole_seconds",
        aggfunc="sum",
        fill_value=0,
    )
    locations.columns = [f"systole_seconds_{col}" for col in locations.columns]

    segment_locations = recordings.pivot_table(
        index="patient_id",
        columns="location",
        values="systole_segments",
        aggfunc="sum",
        fill_value=0,
    )
    segment_locations.columns = [f"systole_segments_{col}" for col in segment_locations.columns]

    return patient.merge(locations.reset_index(), on="patient_id", how="left").merge(
        segment_locations.reset_index(), on="patient_id", how="left"
    )


def classify_errors(df: pd.DataFrame, prob_col: str, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["threshold"] = threshold
    out["predicted_target"] = (out[prob_col] >= threshold).astype(int)
    out["error_type"] = "Correct"
    out.loc[(out["target"] == 1) & (out["predicted_target"] == 0), "error_type"] = "FN"
    out.loc[(out["target"] == 0) & (out["predicted_target"] == 1), "error_type"] = "FP"
    return out


def grouped_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    table = df.copy()
    table[group_col] = table[group_col].map(clean_value)
    grouped = (
        table.groupby(group_col)
        .agg(
            patients=("patient_id", "count"),
            present=("target", "sum"),
            predicted_present=("predicted_target", "sum"),
            false_negatives=("error_type", lambda s: int((s == "FN").sum())),
            false_positives=("error_type", lambda s: int((s == "FP").sum())),
        )
        .reset_index()
    )
    grouped["fn_rate_among_present"] = np.where(
        grouped["present"] > 0, grouped["false_negatives"] / grouped["present"], np.nan
    )
    grouped["fp_rate_among_absent"] = np.where(
        grouped["patients"] > grouped["present"],
        grouped["false_positives"] / (grouped["patients"] - grouped["present"]),
        np.nan,
    )
    return grouped.sort_values(["false_negatives", "false_positives", "patients"], ascending=False)


def numeric_error_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in columns:
        if col not in df.columns:
            continue
        for error_type, subset in df.groupby("error_type"):
            values = pd.to_numeric(subset[col], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "metric": col,
                    "error_type": error_type,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "error_type"])


def confusion_row(df: pd.DataFrame) -> dict[str, int | float]:
    y = df["target"].to_numpy()
    pred = df["predicted_target"].to_numpy()
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    return {
        "threshold": float(df["threshold"].iloc[0]),
        "patients": int(len(df)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "balanced_accuracy": (sensitivity + specificity) / 2,
    }


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_Sem dados._"
    return df.head(max_rows).to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    analysis_dir = output_dir / "error_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    prob_col = f"prob_present_{args.probability}"
    oof = read_patient_oof(output_dir, args.probability)
    clinical = read_clinical_table(dataset_dir)
    recordings = read_oof_recording_metadata(output_dir, oof)

    if not args.skip_signal_quality and not recordings.empty:
        quality = compute_signal_quality(dataset_dir, recordings, analysis_dir / "recording_signal_quality.csv")
        recordings = recordings.merge(quality, on="recording_id", how="left")

    patient_recording_summary = summarize_recordings(recordings)
    base = (
        oof.merge(clinical, on="patient_id", how="left", suffixes=("", "_clinical"))
        .merge(patient_recording_summary, on="patient_id", how="left")
        .sort_values(["fold", "patient_id"])
    )
    base.to_csv(analysis_dir / "patient_oof_with_clinical_and_recording_features.csv", index=False)

    if not recordings.empty:
        recordings.to_csv(analysis_dir / "oof_recording_metadata_with_quality.csv", index=False)

    overview_rows: list[dict[str, int | float]] = []
    report_parts: list[str] = [
        "# Analise de erros - Grupo H Nested TCN + CNN sistole",
        "",
        f"- Output analisado: `{output_dir}`",
        f"- Probabilidade usada: `{prob_col}`",
        f"- Pacientes OOF: `{len(base)}`",
        f"- Present: `{int(base['target'].sum())}`",
        f"- Absent: `{int((base['target'] == 0).sum())}`",
        "",
    ]

    categorical_cols = [
        "fold",
        "Age",
        "Sex",
        "Murmur locations",
        "Most audible location",
        "Systolic murmur timing",
        "Systolic murmur shape",
        "Systolic murmur grading",
        "Systolic murmur pitch",
        "Systolic murmur quality",
        "Outcome",
        "Campaign",
    ]
    numeric_cols = [
        "recording_count",
        "systole_seconds_sum",
        "systole_seconds_mean",
        "systole_seconds_min",
        "systole_segments_sum",
        "systole_segments_mean",
        "rms_mean",
        "rms_min",
        "clipping_fraction_max",
        "near_silence_fraction_mean",
        "recording_seconds_mean",
    ]

    for threshold in args.thresholds:
        slug = threshold_slug(threshold)
        cases = classify_errors(base, prob_col, threshold)
        overview_rows.append(confusion_row(cases))

        errors = cases.loc[cases["error_type"].isin(["FN", "FP"])].copy()
        false_negatives = cases.loc[cases["error_type"] == "FN"].copy()
        false_positives = cases.loc[cases["error_type"] == "FP"].copy()

        cases.to_csv(analysis_dir / f"patient_cases_threshold_{slug}.csv", index=False)
        errors.to_csv(analysis_dir / f"patient_errors_threshold_{slug}.csv", index=False)
        false_negatives.to_csv(analysis_dir / f"false_negatives_threshold_{slug}.csv", index=False)
        false_positives.to_csv(analysis_dir / f"false_positives_threshold_{slug}.csv", index=False)

        if not recordings.empty:
            recording_cases = recordings.merge(
                cases[["patient_id", "threshold", "predicted_target", "error_type", prob_col]],
                on="patient_id",
                how="inner",
            )
            recording_cases.to_csv(analysis_dir / f"recording_cases_threshold_{slug}.csv", index=False)

        for col in categorical_cols:
            if col not in cases.columns:
                continue
            grouped = grouped_rate(cases, col)
            safe_col = str(col).lower().replace(" ", "_").replace(":", "")
            grouped.to_csv(analysis_dir / f"group_by_{safe_col}_threshold_{slug}.csv", index=False)

        numeric_summary = numeric_error_summary(cases, numeric_cols)
        numeric_summary.to_csv(analysis_dir / f"numeric_summary_threshold_{slug}.csv", index=False)

        report_parts.extend(
            [
                f"## Threshold {threshold:.2f}",
                "",
                "### Casos mais importantes",
                "",
                f"- Falsos negativos: `{len(false_negatives)}`",
                f"- Falsos positivos: `{len(false_positives)}`",
                "",
                "#### Falsos negativos com menor probabilidade",
                "",
                md_table(
                    false_negatives.sort_values(prob_col)[
                        [
                            "patient_id",
                            "fold",
                            prob_col,
                            "Most audible location",
                            "Murmur locations",
                            "Systolic murmur grading",
                            "Systolic murmur timing",
                            "Systolic murmur quality",
                            "systole_seconds_sum",
                            "systole_segments_sum",
                            "rms_mean",
                            "near_silence_fraction_mean",
                        ]
                    ],
                    max_rows=15,
                ),
                "",
                "#### Falsos positivos com maior probabilidade",
                "",
                md_table(
                    false_positives.sort_values(prob_col, ascending=False)[
                        [
                            "patient_id",
                            "fold",
                            prob_col,
                            "Outcome",
                            "recording_count",
                            "systole_seconds_sum",
                            "systole_segments_sum",
                            "rms_mean",
                            "near_silence_fraction_mean",
                        ]
                    ],
                    max_rows=15,
                ),
                "",
                "### Agrupamento por local mais audivel",
                "",
                md_table(grouped_rate(cases, "Most audible location"), max_rows=20),
                "",
                "### Agrupamento por intensidade do sopro",
                "",
                md_table(grouped_rate(cases, "Systolic murmur grading"), max_rows=20),
                "",
                "### Resumo numerico por tipo de erro",
                "",
                md_table(numeric_summary, max_rows=80),
                "",
            ]
        )

    overview = pd.DataFrame(overview_rows)
    overview.to_csv(analysis_dir / "threshold_overview.csv", index=False)
    report_parts.insert(8, "## Visao geral por threshold")
    report_parts.insert(9, "")
    report_parts.insert(10, md_table(overview, max_rows=20))
    report_parts.insert(11, "")

    (analysis_dir / "error_analysis.md").write_text("\n".join(report_parts), encoding="utf-8")
    print(f"Wrote analysis to: {analysis_dir}")
    print(overview.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
