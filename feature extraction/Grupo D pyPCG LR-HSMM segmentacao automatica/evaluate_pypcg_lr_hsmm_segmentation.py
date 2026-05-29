# /// script
# dependencies = [
#   "hsmmlearn @ git+https://github.com/jvkersch/hsmmlearn@master",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "pyPCG-toolbox==0.1b5",
#   "scikit-learn>=1.4",
#   "scipy>=1.12",
#   "tabulate>=0.9",
#   "umap-learn>=0.5.7",
# ]
# ///
"""Evaluate pyPCG LR-HSMM automatic cardiac-phase segmentation on CirCor.

This experiment trains pyPCG's LR-HSMM segmenter on CirCor recordings with
provided TSV annotations, evaluates it on held-out patients, converts predicted
states to start/end intervals, compares them with local TSV intervals, and then
checks whether Grupo B v2 relative features keep murmur separation when they are
extracted from automatic segmentations.

Run from the repository root:

    uv run "feature extraction/Grupo D pyPCG LR-HSMM segmentacao automatica/evaluate_pypcg_lr_hsmm_segmentation.py" --skip-umap

Outputs:

    feature extraction/Grupo D pyPCG LR-HSMM segmentacao automatica/outputs/
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyPCG.lr_hsmm as lr_hsmm
from scipy.io import wavfile
from scipy.signal import resample_poly


PHASE_LABELS = {
    1: "s1",
    2: "systole",
    3: "s2",
    4: "diastole",
}

LOCATIONS = ["AV", "PV", "TV", "MV"]
META_COLUMNS = ["patient_id", "recording_id", "location", "wav_path", "murmur", "outcome", "age", "sex", "campaign"]


@dataclass(frozen=True)
class Recording:
    wav_path: Path
    tsv_path: Path
    patient_id: str
    location: str
    murmur: str


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="Train/evaluate pyPCG LR-HSMM segmentation and compare Grupo B v2 features."
    )
    parser.add_argument("--dataset-dir", type=Path, default=repo_root / "circor-heart-sound-1.0.3")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs")
    parser.add_argument(
        "--max-patients",
        type=int,
        default=80,
        help="Limit patients for a bounded experiment. Use 0 for all eligible patients.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fs", type=int, default=1000, help="pyPCG LR-HSMM signal sampling rate.")
    parser.add_argument("--feature-fs", type=int, default=50, help="pyPCG LR-HSMM internal feature sampling rate.")
    parser.add_argument("--min-heart-rate", type=float, default=30.0)
    parser.add_argument(
        "--max-heart-rate",
        type=float,
        default=120.0,
        help="Upper bound used by pyPCG timing estimation. Values above 120 can trigger pyPCG timing failures.",
    )
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument(
        "--reuse-model",
        action="store_true",
        help="Reuse outputs/lr_hsmm_circor_model.json instead of retraining when it exists.",
    )
    return parser.parse_args()


def load_group_b_v2_module(repo_root: Path) -> Any:
    module_path = repo_root / "feature extraction/Grupo B v2 features relativas por local/extract_relative_phase_features_by_location.py"
    spec = importlib.util.spec_from_file_location("group_b_v2_relative_features", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Grupo B v2 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_audio(path: Path, target_fs: int) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    original_dtype = audio.dtype
    audio = audio.astype(np.float64)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        audio = audio / max(abs(info.min), info.max)
    else:
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak
    audio = audio - np.mean(audio)
    if sample_rate != target_fs:
        gcd = math.gcd(sample_rate, target_fs)
        audio = resample_poly(audio, target_fs // gcd, sample_rate // gcd)
        sample_rate = target_fs
    return sample_rate, audio


def read_segments(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=["start_time", "end_time", "label"],
        dtype={"start_time": float, "end_time": float, "label": int},
    )


def parse_recording_id(path: Path) -> tuple[str, str]:
    match = re.match(r"(?P<patient>\d+)_(?P<location>[A-Za-z]+)(?:_\d+)?$", path.stem)
    if not match:
        raise ValueError(f"Unexpected recording name: {path.name}")
    return match.group("patient"), match.group("location")


def collect_recordings(dataset_dir: Path, seed: int, max_patients: int) -> list[Recording]:
    metadata = pd.read_csv(dataset_dir / "training_data.csv")
    metadata = metadata.loc[metadata["Murmur"].isin(["Present", "Absent"])].copy()
    murmur_by_patient = dict(zip(metadata["Patient ID"].astype(str), metadata["Murmur"].astype(str)))

    rows: list[Recording] = []
    for wav_path in sorted((dataset_dir / "training_data").glob("*.wav")):
        patient_id, location = parse_recording_id(wav_path)
        if location not in LOCATIONS:
            continue
        tsv_path = wav_path.with_suffix(".tsv")
        if not tsv_path.exists() or patient_id not in murmur_by_patient:
            continue
        rows.append(Recording(wav_path, tsv_path, patient_id, location, murmur_by_patient[patient_id]))

    patients = pd.DataFrame({"patient_id": sorted({row.patient_id for row in rows})})
    patients["murmur"] = patients["patient_id"].map(murmur_by_patient)
    selected: list[str] = []
    for _, group in patients.groupby("murmur"):
        ids = group["patient_id"].to_numpy()
        rng = np.random.default_rng(seed + (1 if group.iloc[0]["murmur"] == "Present" else 0))
        rng.shuffle(ids)
        selected.extend(ids.tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(selected)
    if max_patients and max_patients > 0:
        selected = selected[:max_patients]
    selected_set = set(selected)
    return [row for row in rows if row.patient_id in selected_set]


def split_by_patient(recordings: list[Recording], train_fraction: float, seed: int) -> tuple[list[Recording], list[Recording]]:
    patients = pd.DataFrame(
        [{"patient_id": row.patient_id, "murmur": row.murmur} for row in recordings]
    ).drop_duplicates()
    train_patients: set[str] = set()
    for _, group in patients.groupby("murmur"):
        ids = group["patient_id"].to_numpy()
        rng = np.random.default_rng(seed + (11 if group.iloc[0]["murmur"] == "Present" else 7))
        rng.shuffle(ids)
        n_train = max(1, int(round(len(ids) * train_fraction)))
        if len(ids) > 1:
            n_train = min(n_train, len(ids) - 1)
        train_patients.update(ids[:n_train].tolist())

    train = [row for row in recordings if row.patient_id in train_patients]
    test = [row for row in recordings if row.patient_id not in train_patients]
    return train, test


def estimate_sound_durations_ms(recordings: list[Recording]) -> tuple[float, float, float, float]:
    s1_durations: list[float] = []
    s2_durations: list[float] = []
    for recording in recordings:
        segments = read_segments(recording.tsv_path)
        s1_durations.extend((segments.loc[segments["label"] == 1, "end_time"] - segments.loc[segments["label"] == 1, "start_time"]).tolist())
        s2_durations.extend((segments.loc[segments["label"] == 3, "end_time"] - segments.loc[segments["label"] == 3, "start_time"]).tolist())
    if not s1_durations or not s2_durations:
        return 122.0, 99.0, 22.0, 22.0
    return (
        float(np.median(s1_durations) * 1000.0),
        float(np.median(s2_durations) * 1000.0),
        float(max(np.std(s1_durations) * 1000.0, 10.0)),
        float(max(np.std(s2_durations) * 1000.0, 10.0)),
    )


def train_or_load_model(
    train_recordings: list[Recording],
    model_path: Path,
    target_fs: int,
    feature_fs: int,
    min_heart_rate: float,
    max_heart_rate: float,
    reuse_model: bool,
) -> lr_hsmm.LR_HSMM:
    model = lr_hsmm.LR_HSMM()
    if reuse_model and model_path.exists():
        model.load_model(str(model_path))
        return model

    mean_s1, mean_s2, std_s1, std_s2 = estimate_sound_durations_ms(train_recordings)
    model.signal_fs = target_fs
    model.feature_fs = feature_fs
    model.expected_hr_range = (min_heart_rate, max_heart_rate)
    model.mean_s1_len = mean_s1
    model.mean_s2_len = mean_s2
    model.std_s1_len = std_s1
    model.std_s2_len = std_s2

    train_data: list[np.ndarray] = []
    train_s1: list[np.ndarray] = []
    train_s2: list[np.ndarray] = []
    for recording in train_recordings:
        _, audio = read_audio(recording.wav_path, target_fs)
        segments = read_segments(recording.tsv_path)
        train_data.append(audio)
        train_s1.append(segments.loc[segments["label"] == 1, "start_time"].to_numpy(dtype=np.float64))
        train_s2.append(segments.loc[segments["label"] == 3, "start_time"].to_numpy(dtype=np.float64))

    model.train_model(train_data, train_s1, train_s2)
    model.save_model(str(model_path))
    return model


def states_to_segments(states: np.ndarray, sample_rate: int) -> pd.DataFrame:
    if len(states) == 0:
        return pd.DataFrame(columns=["start_time", "end_time", "label"])
    states = states.astype(int)
    change_points = np.flatnonzero(np.diff(states) != 0) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, len(states)]
    rows = [
        {"start_time": start / sample_rate, "end_time": end / sample_rate, "label": int(label)}
        for start, end, label in zip(starts, ends, states[starts])
        if int(label) in PHASE_LABELS and end > start
    ]
    return pd.DataFrame(rows, columns=["start_time", "end_time", "label"])


def segments_to_state_vector(segments: pd.DataFrame, n_samples: int, sample_rate: int) -> np.ndarray:
    states = np.zeros(n_samples, dtype=np.int16)
    for row in segments.itertuples(index=False):
        label = int(row.label)
        if label not in PHASE_LABELS:
            continue
        start = max(0, min(n_samples, int(round(float(row.start_time) * sample_rate))))
        end = max(0, min(n_samples, int(round(float(row.end_time) * sample_rate))))
        if end > start:
            states[start:end] = label
    return states


def segmentation_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    n = min(len(reference), len(predicted))
    reference = reference[:n]
    predicted = predicted[:n]
    valid = np.isin(reference, list(PHASE_LABELS))
    out: dict[str, float | int] = {
        "samples_compared": int(valid.sum()),
        "overall_accuracy": float((reference[valid] == predicted[valid]).mean()) if valid.any() else float("nan"),
    }
    for label, phase in PHASE_LABELS.items():
        ref_pos = reference == label
        pred_pos = predicted == label
        tp = int(np.sum(ref_pos & pred_pos))
        fp = int(np.sum(~ref_pos & pred_pos))
        fn = int(np.sum(ref_pos & ~pred_pos))
        union = int(np.sum(ref_pos | pred_pos))
        out[f"{phase}_iou"] = float(tp / union) if union else float("nan")
        out[f"{phase}_recall"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        out[f"{phase}_precision"] = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    return out


def metadata_for_recording(recording: Recording, metadata: pd.DataFrame) -> dict[str, str]:
    row = metadata.loc[metadata["Patient ID"].astype(str) == recording.patient_id].iloc[0]
    return {
        "patient_id": recording.patient_id,
        "recording_id": recording.wav_path.stem,
        "location": recording.location,
        "wav_path": str(recording.wav_path),
        "murmur": str(row["Murmur"]),
        "outcome": str(row["Outcome"]),
        "age": str(row["Age"]),
        "sex": str(row["Sex"]),
        "campaign": str(row["Campaign"]),
    }


def extract_relative_feature_row(
    group_b: Any,
    recording: Recording,
    metadata: pd.DataFrame,
    segments: pd.DataFrame,
    target_fs: int,
) -> dict[str, float | str]:
    sample_rate, audio = group_b.read_audio(recording.wav_path)
    measurements: dict[str, dict[str, float]] = {}
    for label, phase in PHASE_LABELS.items():
        phase_audio = group_b.segment_audio(audio, sample_rate, segments, label)
        durations = group_b.phase_durations(segments, label)
        measurements[phase] = group_b.phase_measurements(phase_audio, sample_rate, durations)
    features: dict[str, float | str] = metadata_for_recording(recording, metadata)
    features.update(group_b.build_relative_features(measurements))
    return features


def project_feature_set(
    group_b: Any,
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    n_clusters: int,
    skip_umap: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = [column for column in group_b.numeric_feature_columns(df) if column not in {"samples_compared"}]
    projected_recordings, recording_metrics = group_b.project_cluster(df, feature_columns, n_clusters, skip_umap)
    projected_recordings.to_csv(output_dir / f"{prefix}_recording_relative_phase_features_with_projection.csv", index=False)
    group_b.scatter(
        projected_recordings,
        "pca_1",
        "pca_2",
        output_dir / f"{prefix}_recording_pca_murmur.png",
        f"{prefix}: PCA por gravacao",
    )
    if not skip_umap and "umap_1" in projected_recordings.columns:
        group_b.scatter(
            projected_recordings,
            "umap_1",
            "umap_2",
            output_dir / f"{prefix}_recording_umap_murmur.png",
            f"{prefix}: UMAP por gravacao",
        )

    patient_df = group_b.aggregate_by_patient(df, feature_columns)
    patient_feature_columns = group_b.numeric_feature_columns(patient_df)
    projected_patients, patient_metrics = group_b.project_cluster(patient_df, patient_feature_columns, n_clusters, skip_umap)
    projected_patients.to_csv(output_dir / f"{prefix}_patient_relative_phase_features_with_projection.csv", index=False)
    group_b.scatter(
        projected_patients,
        "pca_1",
        "pca_2",
        output_dir / f"{prefix}_patient_pca_murmur.png",
        f"{prefix}: PCA por paciente",
    )
    if not skip_umap and "umap_1" in projected_patients.columns:
        group_b.scatter(
            projected_patients,
            "umap_1",
            "umap_2",
            output_dir / f"{prefix}_patient_umap_murmur.png",
            f"{prefix}: UMAP por paciente",
        )

    metric_rows = [
        {
            "feature_source": prefix,
            "level": "recording",
            "rows": int(len(projected_recordings)),
            "patients": int(projected_recordings["patient_id"].nunique()),
            "present_rate": float((projected_recordings["murmur"] == "Present").mean()),
            **group_b.projection_diagnostic_metrics(projected_recordings),
            **recording_metrics,
        },
        {
            "feature_source": prefix,
            "level": "patient",
            "rows": int(len(projected_patients)),
            "patients": int(len(projected_patients)),
            "present_rate": float((projected_patients["murmur"] == "Present").mean()),
            **group_b.projection_diagnostic_metrics(projected_patients),
            **patient_metrics,
        },
    ]
    return projected_recordings, projected_patients, pd.DataFrame(metric_rows)


def confusion_matrix_rows(metrics_df: pd.DataFrame) -> list[str]:
    phase_cols = ["overall_accuracy"] + [f"{phase}_iou" for phase in PHASE_LABELS.values()]
    summary = metrics_df[phase_cols].mean(numeric_only=True).to_frame("mean").T
    return [summary.to_markdown(index=False)]


def write_summary(
    output_path: Path,
    train_recordings: list[Recording],
    test_recordings: list[Recording],
    segmentation_df: pd.DataFrame,
    projection_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Grupo D - pyPCG LR-HSMM segmentacao automatica",
        "",
        "## Objetivo",
        "",
        "Avaliar se o pyPCG LR-HSMM consegue substituir os `.tsv` do CirCor para segmentar fases cardiacas e manter o sinal de separacao de sopro visto no Grupo B v2.",
        "",
        "## Configuracao",
        "",
        f"- Pacientes limite: {args.max_patients if args.max_patients else 'todos'}",
        f"- Gravacoes de treino: {len(train_recordings)}",
        f"- Gravacoes de teste: {len(test_recordings)}",
        f"- Frequencia usada pelo LR-HSMM: {args.target_fs} Hz",
        f"- Frequencia interna de features do LR-HSMM: {args.feature_fs} Hz",
        f"- Faixa de frequencia cardiaca no pyPCG: {args.min_heart_rate:.0f}-{args.max_heart_rate:.0f} bpm",
        f"- UMAP gerado: {'nao' if args.skip_umap else 'sim'}",
        "",
        "## Concordancia de segmentacao",
        "",
        segmentation_df[
            ["overall_accuracy", "s1_iou", "systole_iou", "s2_iou", "diastole_iou"]
        ].mean(numeric_only=True).to_frame("media").T.to_markdown(index=False),
        "",
        "## Metricas de separacao por features",
        "",
        projection_metrics.to_markdown(index=False),
        "",
        "## Leitura inicial",
        "",
        "- Compare `manual_tsv` contra `auto_lr_hsmm`: se a melhor taxa de `Present` em cluster ou quintil cair muito no automatico, a segmentacao automatica esta degradando o sinal do Grupo B v2.",
        "- A acuracia global de segmentacao pode parecer aceitavel mesmo com baixa IoU em `S1`/`S2`, porque sistole e diastole ocupam mais tempo. Por isso, as IoUs por fase sao mais informativas.",
        "- Este experimento treina e testa por pacientes separados, entao mede generalizacao para pacientes nao vistos dentro da amostra selecionada.",
        "",
        "## Arquivos gerados",
        "",
        "- `lr_hsmm_circor_model.json`: modelo LR-HSMM treinado na divisao de treino.",
        "- `auto_segments/*.tsv`: segmentacoes automaticas por gravacao de teste.",
        "- `segmentation_agreement_by_recording.csv`: concordancia por gravacao.",
        "- `manual_tsv_recording_relative_phase_features.csv`: features do Grupo B v2 usando os `.tsv` locais.",
        "- `auto_lr_hsmm_recording_relative_phase_features.csv`: features do Grupo B v2 usando a segmentacao automatica.",
        "- `*_with_projection.csv` e `*.png`: PCA/UMAP e k-means diagnosticos.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    auto_segments_dir = output_dir / "auto_segments"
    output_dir.mkdir(parents=True, exist_ok=True)
    auto_segments_dir.mkdir(parents=True, exist_ok=True)

    group_b = load_group_b_v2_module(repo_root)
    metadata = pd.read_csv(dataset_dir / "training_data.csv")
    metadata = metadata.loc[metadata["Murmur"].isin(["Present", "Absent"])].copy()

    recordings = collect_recordings(dataset_dir, args.seed, args.max_patients)
    train_recordings, test_recordings = split_by_patient(recordings, args.train_fraction, args.seed)
    if not train_recordings or not test_recordings:
        raise RuntimeError("Need both train and test recordings. Increase --max-patients.")

    split_df = pd.DataFrame(
        [
            {
                "split": "train" if row in train_recordings else "test",
                "patient_id": row.patient_id,
                "recording_id": row.wav_path.stem,
                "location": row.location,
                "murmur": row.murmur,
                "wav_path": str(row.wav_path),
            }
            for row in train_recordings + test_recordings
        ]
    )
    split_df.to_csv(output_dir / "train_test_split.csv", index=False)

    model_path = output_dir / "lr_hsmm_circor_model.json"
    model = train_or_load_model(
        train_recordings,
        model_path,
        args.target_fs,
        args.feature_fs,
        args.min_heart_rate,
        args.max_heart_rate,
        args.reuse_model,
    )

    metric_rows: list[dict[str, float | int | str]] = []
    auto_feature_rows: list[dict[str, float | str]] = []
    manual_feature_rows: list[dict[str, float | str]] = []

    for index, recording in enumerate(test_recordings, start=1):
        sample_rate, audio = read_audio(recording.wav_path, args.target_fs)
        states, _ = model.segment_single(audio, recalc_timing=True)
        auto_segments = states_to_segments(states, sample_rate)
        auto_segments.to_csv(auto_segments_dir / f"{recording.wav_path.stem}.tsv", sep="\t", header=False, index=False)

        reference_segments = read_segments(recording.tsv_path)
        reference_vector = segments_to_state_vector(reference_segments, len(audio), sample_rate)
        predicted_vector = segments_to_state_vector(auto_segments, len(audio), sample_rate)
        row_metrics = {
            "patient_id": recording.patient_id,
            "recording_id": recording.wav_path.stem,
            "location": recording.location,
            "murmur": recording.murmur,
            **segmentation_metrics(reference_vector, predicted_vector),
        }
        metric_rows.append(row_metrics)

        manual_feature_rows.append(
            extract_relative_feature_row(group_b, recording, metadata, reference_segments, args.target_fs)
        )
        auto_feature_rows.append(
            extract_relative_feature_row(group_b, recording, metadata, auto_segments, args.target_fs)
        )
        if index % 25 == 0:
            print(f"Evaluated {index}/{len(test_recordings)} held-out recordings...")

    segmentation_df = pd.DataFrame(metric_rows)
    segmentation_df.to_csv(output_dir / "segmentation_agreement_by_recording.csv", index=False)

    manual_df = pd.DataFrame(manual_feature_rows)
    auto_df = pd.DataFrame(auto_feature_rows)
    manual_df.to_csv(output_dir / "manual_tsv_recording_relative_phase_features.csv", index=False)
    auto_df.to_csv(output_dir / "auto_lr_hsmm_recording_relative_phase_features.csv", index=False)

    _, _, manual_projection_metrics = project_feature_set(
        group_b, manual_df, output_dir, "manual_tsv", args.n_clusters, args.skip_umap
    )
    _, _, auto_projection_metrics = project_feature_set(
        group_b, auto_df, output_dir, "auto_lr_hsmm", args.n_clusters, args.skip_umap
    )
    projection_metrics = pd.concat([manual_projection_metrics, auto_projection_metrics], ignore_index=True)
    projection_metrics.to_csv(output_dir / "projection_metrics.csv", index=False)

    write_summary(output_dir / "summary.md", train_recordings, test_recordings, segmentation_df, projection_metrics, args)

    print(f"Train recordings: {len(train_recordings)}")
    print(f"Test recordings: {len(test_recordings)}")
    print(
        "Mean segmentation accuracy: "
        f"{segmentation_df['overall_accuracy'].mean():.3f}; "
        f"mean systole IoU: {segmentation_df['systole_iou'].mean():.3f}; "
        f"mean diastole IoU: {segmentation_df['diastole_iou'].mean():.3f}"
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
