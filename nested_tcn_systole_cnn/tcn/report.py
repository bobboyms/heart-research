from __future__ import annotations


import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from .config import FeatureConfig, Normalizer, RecordingItem


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cached_normalizer(output_dir: Path) -> Normalizer | None:
    path = output_dir / "normalization.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "mean" not in payload or "std" not in payload:
        return None
    return Normalizer(mean=list(payload["mean"]), std=list(payload["std"]))


def load_cached_label_counts(output_dir: Path, label_names: dict[int, str]) -> np.ndarray | None:
    path = output_dir / "train_label_counts.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    try:
        return np.asarray([int(payload[label_names[i]]) for i in range(len(label_names))], dtype=np.int64)
    except (KeyError, TypeError, ValueError):
        return None


def maybe_discard_cached_label_counts(label_counts: np.ndarray | None, cfg: FeatureConfig) -> np.ndarray | None:
    if label_counts is not None and cfg.other_mode == "ignore":
        print("Ignoring cached train_label_counts.json because --other-mode ignore changes label counts.")
        return None
    return label_counts


def save_split_manifest(output_dir: Path, splits: dict[str, list[RecordingItem]]) -> None:
    manifest = {
        split_name: [asdict(item) for item in split_items]
        for split_name, split_items in splits.items()
    }
    save_json(output_dir / "split_manifest.json", manifest)


def save_confusion_matrix(
    output_dir: Path,
    split_name: str,
    confusion: list[list[int]],
    label_names: dict[int, str],
) -> None:
    labels = [label_names[i] for i in range(len(label_names))]
    matrix = np.asarray(confusion, dtype=np.int64)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output_dir / f"{split_name}_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{split_name} confusion matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_history_plot(output_dir: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="val accuracy")
    axes[1].plot(frame["epoch"], frame["val_macro_f1"], label="val macro F1")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    cfg: FeatureConfig,
    splits: dict[str, list[RecordingItem]],
    label_counts: np.ndarray,
    label_names: dict[int, str],
    metrics: dict[str, dict[str, object]],
    best_epoch: int | None,
) -> None:
    lines: list[str] = []
    lines.append("# Grupo E - TCN segmentacao frame a frame")
    lines.append("")
    lines.append("## Configuracao")
    lines.append("")
    lines.append(f"- Dataset: `{args.dataset_dir}`")
    lines.append(f"- Features: {cfg.n_mels} log-mel bins, frame={cfg.frame_ms} ms, hop={cfg.hop_ms} ms, deltas={cfg.add_deltas}")
    model_kind = "TCN causal" if args.causal else "TCN nao causal"
    lines.append(
        f"- Modelo: {model_kind}, hidden={args.hidden_channels}, levels={args.levels}, "
        f"kernel={args.kernel_size}, dropout={args.dropout}, pooling={args.pooling}"
    )
    lines.append(
        f"- Rotulagem: target_mode={cfg.target_mode}, label_mode={cfg.label_mode}, "
        f"boundary_ignore_ms={cfg.boundary_ignore_ms}, other_mode={cfg.other_mode}"
    )
    lines.append(f"- Treino por janelas: {args.train_window_seconds}s com hop {args.train_window_hop_seconds}s")
    lines.append(
        f"- Loss: {args.loss}, dice_weight={args.dice_weight}, focal_gamma={args.focal_gamma}, "
        f"label_smoothing={args.label_smoothing}, systole_weight_multiplier={args.systole_weight_multiplier}"
    )
    lines.append(
        f"- Pos-processamento: {args.postprocess}, median_filter_frames={args.median_filter_frames}, "
        f"min_segment_frames={args.min_segment_frames}"
    )
    lines.append(f"- Device solicitado/usado: `{args.device}`")
    lines.append(f"- Melhor epoca por macro F1 de validacao: {best_epoch if best_epoch is not None else 'n/a'}")
    lines.append("")
    lines.append("## Split por paciente")
    lines.append("")
    for split_name, split_items in splits.items():
        patients = {item.patient_id for item in split_items}
        lines.append(f"- {split_name}: {len(split_items)} gravacoes, {len(patients)} pacientes")
    lines.append("")
    lines.append("## Distribuicao dos rotulos no treino")
    lines.append("")
    lines.append("| Classe | Frames |")
    lines.append("|---|---:|")
    for label, count in enumerate(label_counts):
        lines.append(f"| {label} = {label_names[label]} | {int(count)} |")
    lines.append("")
    lines.append("## Metricas")
    lines.append("")
    lines.append("| Split | Loss | Accuracy | Macro F1 | Weighted F1 | Balanced accuracy | Mean IoU | Frames |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split_name in ["val", "test"]:
        metric = metrics[split_name]
        lines.append(
            f"| {split_name} | {metric['loss']:.4f} | {metric['accuracy']:.4f} | "
            f"{metric['macro_f1']:.4f} | {metric['weighted_f1']:.4f} | "
            f"{metric['balanced_accuracy']:.4f} | {metric['mean_iou']:.4f} | {metric['frames']} |"
        )
    lines.append("")
    lines.append("## Metricas por classe no teste")
    lines.append("")
    lines.append("| Classe | Precision | Recall | F1 | IoU | Support |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    per_class = metrics["test"]["per_class"]
    assert isinstance(per_class, dict)
    for label in label_names.values():
        row = per_class[label]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['iou']:.4f} | {row['support']} |"
        )
    lines.append("")
    lines.append("Arquivos principais: `best_model.pt`, `metrics.json`, `training_history.csv`, matrizes de confusao CSV/PNG.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
