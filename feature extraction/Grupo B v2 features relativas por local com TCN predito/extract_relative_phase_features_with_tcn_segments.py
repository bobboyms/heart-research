# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.4",
#   "scipy>=1.12",
#   "tabulate>=0.9",
#   "torch>=2.2",
#   "tqdm>=4.66",
#   "umap-learn>=0.5.7",
# ]
# ///
"""Grupo B v2 using TCN-predicted cardiac phase segmentations.

This experiment mirrors "Grupo B v2 features relativas por local", but it does
not read the CirCor ground-truth `.tsv` files for feature extraction. Instead,
it loads a trained Grupo E TCN frame segmenter, predicts S1/systole/S2/diastole
segments for each `.wav`, caches those predicted `.tsv` files, and then runs the
same relative systole-vs-other-phase feature extraction and PCA/UMAP diagnostics.

Run from the repository root:

    uv run "feature extraction/Grupo B v2 features relativas por local com TCN predito/extract_relative_phase_features_with_tcn_segments.py"

For a quick smoke test:

    uv run "feature extraction/Grupo B v2 features relativas por local com TCN predito/extract_relative_phase_features_with_tcn_segments.py" --max-recordings 40 --skip-umap
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


class LazyUmapModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("umap")
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            sys.modules.pop("umap", None)
            self._module = __import__("umap")
            sys.modules["umap"] = self
        return self._module

    def __getattr__(self, name: str) -> object:
        return getattr(self._load(), name)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
B2_SCRIPT = REPO_ROOT / "feature extraction" / "Grupo B v2 features relativas por local" / "extract_relative_phase_features_by_location.py"
TCN_SCRIPT = REPO_ROOT / "modeling" / "Grupo E TCN segmentacao frame a frame" / "train_tcn_frame_segmenter.py"

# The original Grupo B v2 script imports umap at module import time. Importing
# umap triggers numba compilation and can take long enough to look frozen before
# any progress message appears. Use a proxy so UMAP is imported only if the run
# actually generates UMAP plots.
sys.modules.setdefault("umap", LazyUmapModule())

b2 = load_module("grupo_b2_relative_features", B2_SCRIPT)
tcn = load_module("grupo_e_tcn_segmenter", TCN_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Grupo B v2 relative features using TCN-predicted cardiac phase segments."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "circor-heart-sound-1.0.3",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "modeling"
        / "Grupo E TCN segmentacao frame a frame"
        / "outputs_noncausal_overlap"
        / "best_model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
    )
    parser.add_argument(
        "--predicted-tsv-dir",
        type=Path,
        default=None,
        help="Directory used to cache predicted TSV files. Defaults to <output-dir>/predicted_tsvs.",
    )
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument(
        "--locations",
        nargs="+",
        choices=b2.LOCATIONS,
        default=b2.LOCATIONS,
        help="Auscultation locations to include. Default includes AV PV TV MV.",
    )
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    parser.add_argument("--allow-mps-predict", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--median-filter-frames", type=int, default=5)
    parser.add_argument("--min-segment-frames", type=int, default=3)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def choose_prediction_device(requested: str, allow_mps: bool) -> torch.device:
    device = tcn.choose_device(requested)
    if device.type == "mps" and not allow_mps:
        print("Prediction is using CPU by default; pass --allow-mps-predict to force MPS.")
        return torch.device("cpu")
    return device


def predicted_tsv_path(predicted_tsv_dir: Path, wav_path: Path) -> Path:
    return predicted_tsv_dir / f"{wav_path.stem}.predicted.tsv"


def read_predicted_segments(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["start_time", "end_time", "label"])
    return pd.read_csv(
        path,
        sep="\t",
        names=["start_time", "end_time", "label"],
        dtype={"start_time": float, "end_time": float, "label": int},
    )


def write_predicted_segments(path: Path, segments: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if segments.empty:
        path.write_text("", encoding="utf-8")
        return
    segments[["start_time", "end_time", "label"]].to_csv(
        path,
        sep="\t",
        header=False,
        index=False,
        float_format="%.6f",
    )


@torch.no_grad()
def predict_segments_for_audio(
    wav_path: Path,
    model: torch.nn.Module,
    normalizer: object,
    cfg: object,
    device: torch.device,
    postprocess: bool,
    median_filter_frames: int,
    min_segment_frames: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    sample_rate, audio = tcn.read_audio(wav_path)
    features, centers_s, starts_s, ends_s = tcn.extract_frame_features(audio, sample_rate, cfg)
    normalized = normalizer.apply(features)
    x = torch.from_numpy(normalized.T.copy()).unsqueeze(0).to(device)

    model.eval()
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).permute(1, 0).detach().cpu().numpy()
    pred = probs.argmax(axis=1).astype(np.int64)
    pred[ends_s <= starts_s] = tcn.IGNORE_INDEX
    if postprocess:
        valid_mask = pred != tcn.IGNORE_INDEX
        pred[valid_mask] = tcn.postprocess_prediction(
            pred[valid_mask],
            median_filter_frames,
            min_segment_frames,
        )

    segments_with_confidence = tcn.prediction_segments(pred, centers_s, starts_s, ends_s, probs)
    if segments_with_confidence.empty:
        segments = pd.DataFrame(columns=["start_time", "end_time", "label"])
    else:
        segments = segments_with_confidence[["start_time", "end_time", "label"]].copy()
    quality = {
        "recording_id": wav_path.stem,
        "sample_rate": int(sample_rate),
        "frames": int(len(pred)),
        "predicted_segments": int(len(segments_with_confidence)),
        "mean_segment_confidence": float(segments_with_confidence["confidence"].mean())
        if not segments_with_confidence.empty
        else 0.0,
    }
    for label, name in tcn.LABEL_NAMES.items():
        if label == 0:
            continue
        quality[f"{name}_segments"] = int((segments_with_confidence["label"] == label).sum()) if not segments_with_confidence.empty else 0
        quality[f"{name}_frames"] = int((pred == label).sum())
    return segments, quality


def load_or_predict_segments(
    wav_path: Path,
    predicted_tsv_dir: Path,
    overwrite_predictions: bool,
    model: torch.nn.Module,
    normalizer: object,
    cfg: object,
    device: torch.device,
    postprocess: bool,
    median_filter_frames: int,
    min_segment_frames: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    path = predicted_tsv_path(predicted_tsv_dir, wav_path)
    if path.exists() and not overwrite_predictions:
        segments = read_predicted_segments(path)
        quality: dict[str, float | int | str] = {
            "recording_id": wav_path.stem,
            "sample_rate": 0,
            "frames": 0,
            "predicted_segments": int(len(segments)),
            "mean_segment_confidence": float("nan"),
        }
        for label, name in tcn.LABEL_NAMES.items():
            if label == 0:
                continue
            quality[f"{name}_segments"] = int((segments["label"] == label).sum()) if not segments.empty else 0
            quality[f"{name}_frames"] = 0
        return segments, quality

    segments, quality = predict_segments_for_audio(
        wav_path,
        model,
        normalizer,
        cfg,
        device,
        postprocess,
        median_filter_frames,
        min_segment_frames,
    )
    write_predicted_segments(path, segments)
    return segments, quality


def extract_recording_with_predicted_segments(
    wav_path: Path,
    metadata: pd.DataFrame,
    segments: pd.DataFrame,
) -> dict[str, float | str] | None:
    patient_id, location = b2.parse_recording_id(wav_path)
    if location not in b2.LOCATIONS:
        return None
    row = metadata.loc[metadata["Patient ID"].astype(str) == patient_id]
    if row.empty:
        return None

    sample_rate, audio = b2.read_audio(wav_path)
    measurements: dict[str, dict[str, float]] = {}
    for label, phase in b2.PHASE_LABELS.items():
        phase_audio = b2.segment_audio(audio, sample_rate, segments, label)
        durations = b2.phase_durations(segments, label)
        measurements[phase] = b2.phase_measurements(phase_audio, sample_rate, durations)

    features: dict[str, float | str] = {
        "patient_id": patient_id,
        "recording_id": wav_path.stem,
        "location": location,
        "wav_path": str(wav_path),
        "segmentation_source": "tcn_predicted",
        "murmur": str(row.iloc[0]["Murmur"]),
        "outcome": str(row.iloc[0]["Outcome"]),
        "age": str(row.iloc[0]["Age"]),
        "sex": str(row.iloc[0]["Sex"]),
        "campaign": str(row.iloc[0]["Campaign"]),
    }
    features.update(b2.build_relative_features(measurements))
    return features


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "patient_id",
        "recording_id",
        "location",
        "wav_path",
        "segmentation_source",
        "murmur",
        "outcome",
        "age",
        "sex",
        "campaign",
    }
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    cfg: object,
    recording_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    feature_columns: list[str],
    patient_feature_columns: list[str],
) -> None:
    lines = [
        "# Grupo B v2 - features relativas com segmentacao TCN predita",
        "",
        "## Objetivo",
        "",
        "Este experimento repete o Grupo B v2, mas substitui os `.tsv` reais do CirCor por segmentacoes preditas pelo modelo TCN frame a frame.",
        "",
        "## Configuracao",
        "",
        f"- Dataset: `{args.dataset_dir.resolve()}`",
        f"- Checkpoint TCN: `{args.checkpoint.resolve()}`",
        f"- Predicted TSV cache: `{(args.predicted_tsv_dir or (args.output_dir / 'predicted_tsvs')).resolve()}`",
        f"- Locais incluidos: `{', '.join(args.locations)}`",
        f"- TCN feature config: `{json.dumps(asdict(cfg), ensure_ascii=False)}`",
        f"- Pos-processamento TCN: {not args.no_postprocess}, median_filter_frames={args.median_filter_frames}, min_segment_frames={args.min_segment_frames}",
        f"- UMAP gerado: {'nao' if args.skip_umap else 'sim'}",
        "",
        "## Resumo",
        "",
        f"- Gravacoes analisadas: {len(recording_df)}",
        f"- Pacientes analisados: {recording_df['patient_id'].nunique()}",
        f"- Features relativas por gravacao: {len(feature_columns)}",
        f"- Pacientes agregados: {len(patient_df)}",
        f"- Features agregadas por paciente: {len(patient_feature_columns)}",
        f"- Gravacoes sem segmentos TCN uteis: {int((quality_df['predicted_segments'] == 0).sum()) if not quality_df.empty and 'predicted_segments' in quality_df else 0}",
        "",
        "## Qualidade da segmentacao predita",
        "",
        quality_df.describe(include="all").transpose().to_markdown() if not quality_df.empty else "Sem metricas de qualidade.",
        "",
        "## Murmur por local",
        "",
        pd.crosstab(recording_df["location"], recording_df["murmur"], margins=True).to_markdown(),
        "",
        "## Metricas por visualizacao",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Arquivos gerados",
        "",
        "- `predicted_tsvs/*.predicted.tsv`: segmentacoes TCN usadas para extrair features.",
        "- `predicted_segmentation_quality.csv`: contagens e confianca media dos segmentos preditos.",
        "- `recording_relative_phase_features.csv`: features relativas por gravacao.",
        "- `recording_relative_phase_features_with_projection.csv`: PCA/k-means global por gravacao.",
        "- `patient_relative_phase_features.csv`: agregacao por paciente usando media e maximo entre locais.",
        "- `patient_relative_phase_features_with_projection.csv`: PCA/k-means global por paciente.",
        "- `by_location/*_pca_murmur.png`: PCA separado por local.",
        "- `by_location/*_umap_murmur.png`: UMAP separado por local, se habilitado.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    data_dir = dataset_dir / "training_data"
    output_dir = args.output_dir.resolve()
    predicted_tsv_dir = (args.predicted_tsv_dir or (output_dir / "predicted_tsvs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_tsv_dir.mkdir(parents=True, exist_ok=True)

    device = choose_prediction_device(args.device, args.allow_mps_predict)
    model, normalizer, cfg, _checkpoint = tcn.load_checkpoint_for_eval(args.checkpoint.resolve(), device)

    metadata = pd.read_csv(dataset_dir / "training_data.csv")
    if not args.include_unknown:
        metadata = metadata.loc[metadata["Murmur"].isin(["Present", "Absent"])].copy()

    wav_paths = sorted(data_dir.glob("*.wav"))
    if args.max_recordings is not None:
        wav_paths = wav_paths[: args.max_recordings]

    rows: list[dict[str, float | str]] = []
    quality_rows: list[dict[str, float | int | str]] = []
    skipped = 0

    iterator = tqdm(wav_paths, desc="Predicting/extracting", unit="rec", disable=not args.progress)
    for wav_path in iterator:
        try:
            patient_id, location = b2.parse_recording_id(wav_path)
        except ValueError:
            skipped += 1
            continue
        if location not in args.locations:
            skipped += 1
            continue
        if metadata.loc[metadata["Patient ID"].astype(str) == patient_id].empty:
            skipped += 1
            continue

        segments, quality = load_or_predict_segments(
            wav_path,
            predicted_tsv_dir,
            args.overwrite_predictions,
            model,
            normalizer,
            cfg,
            device,
            postprocess=not args.no_postprocess,
            median_filter_frames=args.median_filter_frames,
            min_segment_frames=args.min_segment_frames,
        )
        quality_rows.append(quality)

        row = extract_recording_with_predicted_segments(wav_path, metadata, segments)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows extracted. Check dataset path, checkpoint, and filters.")

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(output_dir / "predicted_segmentation_quality.csv", index=False)

    recording_df = pd.DataFrame(rows)
    feature_columns = numeric_feature_columns(recording_df)
    recording_df.to_csv(output_dir / "recording_relative_phase_features.csv", index=False)

    projected_recordings, global_metrics = b2.project_cluster(
        recording_df,
        feature_columns,
        args.n_clusters,
        args.skip_umap,
    )
    projected_recordings.to_csv(output_dir / "recording_relative_phase_features_with_projection.csv", index=False)
    b2.scatter(
        projected_recordings,
        "pca_1",
        "pca_2",
        output_dir / "recording_pca_murmur.png",
        "Todas as gravacoes: PCA de features relativas com TCN",
    )
    if not args.skip_umap and "umap_1" in projected_recordings.columns:
        b2.scatter(
            projected_recordings,
            "umap_1",
            "umap_2",
            output_dir / "recording_umap_murmur.png",
            "Todas as gravacoes: UMAP de features relativas com TCN",
        )

    metrics_rows = [
        {
            "level": "recording_global",
            "location": "all",
            "rows": int(len(recording_df)),
            "patients": int(recording_df["patient_id"].nunique()),
            "present_rate": float((recording_df["murmur"] == "Present").mean()),
            **b2.projection_diagnostic_metrics(projected_recordings),
            **global_metrics,
        }
    ]
    metrics_rows.extend(b2.plot_per_location(recording_df, feature_columns, output_dir, args.n_clusters, args.skip_umap))

    patient_df = b2.aggregate_by_patient(recording_df, feature_columns)
    patient_feature_columns = numeric_feature_columns(patient_df)
    patient_df.to_csv(output_dir / "patient_relative_phase_features.csv", index=False)
    projected_patients, patient_metrics = b2.project_cluster(
        patient_df,
        patient_feature_columns,
        args.n_clusters,
        args.skip_umap,
    )
    projected_patients.to_csv(output_dir / "patient_relative_phase_features_with_projection.csv", index=False)
    b2.scatter(
        projected_patients,
        "pca_1",
        "pca_2",
        output_dir / "patient_pca_murmur.png",
        "Pacientes agregados: PCA de features relativas com TCN",
    )
    if not args.skip_umap and "umap_1" in projected_patients.columns:
        b2.scatter(
            projected_patients,
            "umap_1",
            "umap_2",
            output_dir / "patient_umap_murmur.png",
            "Pacientes agregados: UMAP de features relativas com TCN",
        )

    metrics_rows.append(
        {
            "level": "patient_aggregated",
            "location": "mean_max",
            "rows": int(len(patient_df)),
            "patients": int(len(patient_df)),
            "present_rate": float((patient_df["murmur"] == "Present").mean()),
            **b2.projection_diagnostic_metrics(projected_patients),
            **patient_metrics,
        }
    )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "projection_metrics.csv", index=False)
    write_summary(
        output_dir / "summary.md",
        args,
        cfg,
        recording_df,
        patient_df,
        metrics_df,
        quality_df,
        feature_columns,
        patient_feature_columns,
    )

    print(f"Extracted {len(recording_df)} recordings with TCN-predicted segments. Skipped {skipped}.")
    print(f"Recording feature columns: {len(feature_columns)}")
    print(f"Patient feature columns: {len(patient_feature_columns)}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
