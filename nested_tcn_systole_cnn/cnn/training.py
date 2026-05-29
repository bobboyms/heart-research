from __future__ import annotations


import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


from .aggregate import aggregate_patient_location_features, aggregate_patient_probs, sample_weights_for_indices
from .augment import apply_smote_minority_augmentation, augmentation_config_from_args, build_cnn_augmenter
from .calibration import apply_location_aware_calibrator, apply_platt_calibrator, fit_location_aware_calibrator, fit_platt_calibrator
from .config import LOCATION_ORDER, ModelConfig, StftConfig
from .dataset import SpectrogramDataset, build_train_loader, compute_freq_norm_stats, encode_pitch_targets, use_stratified_batches
from .losses import add_auc_loss, build_binary_loss
from .metrics import average_precision, choose_threshold, metrics, roc_auc
from .mil import train_one_fold_patient_mil
from .models import build_systole_model


@torch.no_grad()
def predict_recordings(model: nn.Module, dataset: SpectrogramDataset, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    has_temporal = getattr(dataset, "temporal_features", None) is not None
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            temporal = batch[3].to(device) if has_temporal else None
            logits = model(x, temporal) if has_temporal else model(x)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def train_one_fold(
    fold: int,
    specs: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    model_config: ModelConfig,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    tune_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[str, float | int], list[dict[str, float | int]], pd.DataFrame]:
    if bool(getattr(args, "patient_mil_attention", False)):
        return train_one_fold_patient_mil(
            fold,
            specs,
            labels,
            meta,
            train_indices,
            val_indices,
            model_config,
            args,
            device,
            output_dir,
            tune_indices=tune_indices,
        )

    selection_indices = val_indices if tune_indices is None else tune_indices
    calibration_indices = train_indices if tune_indices is None else tune_indices
    train_subset = specs[train_indices]
    train_mean, train_std = compute_freq_norm_stats(train_subset, str(getattr(args, "freq_norm", "perbin")))
    weak_murmur_weight = float(getattr(args, "weak_murmur_weight", 1.0))
    moderate_murmur_weight = float(getattr(args, "moderate_murmur_weight", 1.0))
    augmenter = build_cnn_augmenter(args)
    augmentation_config = augmentation_config_from_args(args)
    train_weights = sample_weights_for_indices(meta, train_indices, weak_murmur_weight, moderate_murmur_weight)
    train_specs = specs[train_indices]
    train_labels = labels[train_indices]

    n_temporal = int(getattr(model_config, "n_temporal_features", 0))
    temporal_all = None
    if n_temporal > 0:
        if bool(augmentation_config["smote_minority_augmentation"]):
            raise ValueError("Temporal features are incompatible with SMOTE augmentation.")
        tf_cols = [f"tf_{j}" for j in range(n_temporal)]
        missing = [c for c in tf_cols if c not in meta.columns]
        if missing:
            raise ValueError(f"Temporal features requested but meta is missing columns: {missing}")
        temporal_all = meta[tf_cols].to_numpy(dtype=np.float32)
        tf_mean = temporal_all[train_indices].mean(axis=0)
        tf_std = temporal_all[train_indices].std(axis=0) + 1e-6
        temporal_all = ((temporal_all - tf_mean) / tf_std).astype(np.float32)

    smote_info: dict[str, float | int] = {
        "enabled": int(bool(augmentation_config["smote_minority_augmentation"])),
        "minority_class": -1,
        "original_minority_count": 0,
        "original_majority_count": 0,
        "synthetic_count": 0,
        "target_ratio": float(augmentation_config["smote_target_ratio"]),
        "k_neighbors": int(augmentation_config["smote_k_neighbors"]),
    }
    if bool(augmentation_config["smote_minority_augmentation"]):
        train_specs, train_labels, train_weights, smote_info = apply_smote_minority_augmentation(
            train_specs,
            train_labels,
            train_weights,
            k_neighbors=int(augmentation_config["smote_k_neighbors"]),
            target_ratio=float(augmentation_config["smote_target_ratio"]),
        )
    aux_weight = float(getattr(args, "aux_pitch_loss_weight", 0.0))
    aux_active = aux_weight > 0.0
    train_aux = encode_pitch_targets(meta.iloc[train_indices], train_labels) if aux_active else None
    train_ds = SpectrogramDataset(
        train_specs,
        train_labels,
        train_mean,
        train_std,
        train_weights,
        augmenter=augmenter,
        augmenter_minority_only=bool(augmentation_config["ltsrr_minority_only"]),
        temporal_features=None if temporal_all is None else temporal_all[train_indices],
        aux_targets=train_aux,
    )
    val_ds = SpectrogramDataset(specs[val_indices], labels[val_indices], train_mean, train_std,
                                temporal_features=None if temporal_all is None else temporal_all[val_indices])
    selection_ds = SpectrogramDataset(specs[selection_indices], labels[selection_indices], train_mean, train_std,
                                      temporal_features=None if temporal_all is None else temporal_all[selection_indices])
    loader = build_train_loader(train_ds, train_labels, args)
    model = build_systole_model(model_config).to(device)
    pos = float((train_labels == 1).sum())
    neg = float((train_labels == 0).sum())
    loss_fn = build_binary_loss(args, pos, neg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_auprc = -math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    mixup_alpha = float(getattr(args, "mixup_alpha", 0.0))
    aux_loss_fn = nn.CrossEntropyLoss(ignore_index=-1) if aux_active else None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            x = batch[0]
            y = batch[1]
            sample_weight = batch[2] if len(batch) > 2 else torch.ones_like(y)
            temporal = batch[3].to(device) if (n_temporal > 0 and len(batch) > 3) else None
            x = x.to(device)
            y = y.to(device)
            sample_weight = sample_weight.to(device)
            if mixup_alpha > 0.0 and x.size(0) >= 2:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1.0 - lam) * x[perm]
                y = lam * y + (1.0 - lam) * y[perm]
                sample_weight = lam * sample_weight + (1.0 - lam) * sample_weight[perm]
                if temporal is not None:
                    temporal = lam * temporal + (1.0 - lam) * temporal[perm]
            optimizer.zero_grad(set_to_none=True)
            if aux_active:
                logits, aux_logits = (
                    model(x, temporal, return_aux=True) if temporal is not None else model(x, return_aux=True)
                )
            else:
                logits = model(x, temporal) if temporal is not None else model(x)
            loss = (loss_fn(logits, y) * sample_weight).mean()
            loss = add_auc_loss(loss, logits, y, args)
            if aux_active and aux_logits is not None:
                aux_y = batch[-1].to(device)
                if bool((aux_y >= 0).any()):
                    loss = loss + aux_weight * aux_loss_fn(aux_logits, aux_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        selection_record_probs = predict_recordings(model, selection_ds, args.batch_size, device)
        selection_patient = aggregate_patient_probs(meta, selection_record_probs, selection_indices, method="max")
        val_auprc = average_precision(selection_patient["target"].to_numpy(), selection_patient["prob"].to_numpy())
        val_auroc = roc_auc(selection_patient["target"].to_numpy(), selection_patient["prob"].to_numpy())
        history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_patient_auprc": val_auprc,
                "val_patient_auroc": val_auroc,
            }
        )
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold} did not produce a model.")
    model.load_state_dict(best_state)
    val_record_probs = predict_recordings(model, val_ds, args.batch_size, device)
    calibration_ds = SpectrogramDataset(specs[calibration_indices], labels[calibration_indices], train_mean, train_std,
                                        temporal_features=None if temporal_all is None else temporal_all[calibration_indices])
    calibration_record_probs = predict_recordings(model, calibration_ds, args.batch_size, device)
    calibration_patient = aggregate_patient_probs(meta, calibration_record_probs, calibration_indices, method="max")
    val_patient = aggregate_patient_probs(meta, val_record_probs, val_indices, method="max")
    threshold = choose_threshold(calibration_patient["target"].to_numpy(), calibration_patient["prob"].to_numpy())

    calibrator = {"scale": 1.0, "bias": 0.0}
    calibration_kind = "none"
    val_calibrated_by_patient: pd.Series | None = None
    if bool(getattr(args, "location_aware_calibration", False)):
        calibration_features, feature_columns = aggregate_patient_location_features(meta, calibration_record_probs, calibration_indices)
        val_features, _feature_columns = aggregate_patient_location_features(meta, val_record_probs, val_indices)
        calibrator = fit_location_aware_calibrator(calibration_features, feature_columns)
        calibration_calibrated_prob = apply_location_aware_calibrator(calibration_features, calibrator)
        val_calibrated_prob = apply_location_aware_calibrator(val_features, calibrator)
        val_calibrated_by_patient = pd.Series(
            val_calibrated_prob,
            index=val_features["patient_id"].astype(str),
        )
        calibration_kind = "location_aware"
    elif args.calibration == "platt":
        calibrator = fit_platt_calibrator(calibration_patient["target"].to_numpy(), calibration_patient["prob"].to_numpy())
        calibration_calibrated_prob = apply_platt_calibrator(
            calibration_patient["prob"].to_numpy(dtype=float),
            calibrator,
        )
        val_calibrated_prob = apply_platt_calibrator(
            val_patient["prob"].to_numpy(dtype=float),
            calibrator,
        )
        calibration_kind = "platt"
    else:
        calibration_calibrated_prob = calibration_patient["prob"].to_numpy(dtype=float)
        val_calibrated_prob = val_patient["prob"].to_numpy(dtype=float)
    calibrated_threshold = choose_threshold(
        calibration_patient["target"].to_numpy(),
        calibration_calibrated_prob,
    )
    val_patient_calibrated = val_patient.copy()
    val_patient_calibrated["fold"] = fold
    val_patient_calibrated["prob_present_raw"] = val_patient_calibrated["prob"].to_numpy(dtype=float)
    if val_calibrated_by_patient is not None:
        val_patient_calibrated["prob_present_calibrated"] = (
            val_patient_calibrated["patient_id"].astype(str).map(val_calibrated_by_patient).to_numpy(dtype=float)
        )
    else:
        val_patient_calibrated["prob_present_calibrated"] = val_calibrated_prob
    val_patient_calibrated["calibration_kind"] = calibration_kind
    if calibration_kind == "location_aware":
        val_patient_calibrated["calibration_scale"] = float(np.linalg.norm(np.asarray(calibrator["weights"], dtype=float)))
        val_patient_calibrated["calibration_bias"] = float(calibrator["bias"])
    else:
        val_patient_calibrated["calibration_scale"] = float(calibrator.get("scale", 1.0))
        val_patient_calibrated["calibration_bias"] = float(calibrator.get("bias", 0.0))
    val_patient_calibrated = val_patient_calibrated.drop(columns=["prob"])

    fold_metrics = metrics(val_patient["target"].to_numpy(), val_patient["prob"].to_numpy(), threshold)
    fold_metrics["fold"] = fold
    fold_metrics["calibrated_threshold"] = float(calibrated_threshold)
    fold_metrics["epochs_trained"] = len(history)
    fold_metrics["best_val_auprc"] = float(best_auprc)
    calibrated_fold_metrics = metrics(
        val_patient_calibrated["target"].to_numpy(),
        val_patient_calibrated["prob_present_calibrated"].to_numpy(),
        0.5,
    )
    fold_metrics["calibrated_ba_05"] = float(calibrated_fold_metrics["balanced_accuracy"])
    fold_metrics["calibrated_sensitivity_05"] = float(calibrated_fold_metrics["sensitivity"])
    fold_metrics["calibrated_specificity_05"] = float(calibrated_fold_metrics["specificity"])
    fold_metrics["calibrated_precision_05"] = float(calibrated_fold_metrics["precision"])
    fold_metrics["calibrated_f1_05"] = float(calibrated_fold_metrics["f1"])
    fold_metrics["weak_murmur_weight"] = weak_murmur_weight
    fold_metrics["moderate_murmur_weight"] = moderate_murmur_weight
    fold_metrics["loss"] = getattr(args, "loss", "bce")
    fold_metrics["focal_gamma"] = float(getattr(args, "focal_gamma", 2.0))
    fold_metrics["focal_alpha"] = -1.0 if getattr(args, "focal_alpha", None) is None else float(args.focal_alpha)
    fold_metrics["auc_loss_weight"] = float(getattr(args, "auc_loss_weight", 0.0))
    fold_metrics["auc_loss_margin"] = float(getattr(args, "auc_loss_margin", 1.0))
    fold_metrics["stratified_batches"] = int(use_stratified_batches(args, train_labels))
    fold_metrics["ltsrr_prob"] = float(augmentation_config["ltsrr_prob"])
    fold_metrics["ltsrr_k"] = int(augmentation_config["ltsrr_k"])
    fold_metrics["ltsrr_frequency_ratio"] = float(augmentation_config["ltsrr_frequency_ratio"])
    fold_metrics["ltsrr_minority_only"] = int(bool(augmentation_config["ltsrr_minority_only"]))
    fold_metrics["smote_minority_augmentation"] = int(bool(augmentation_config["smote_minority_augmentation"]))
    fold_metrics["smote_k_neighbors"] = int(augmentation_config["smote_k_neighbors"])
    fold_metrics["smote_target_ratio"] = float(augmentation_config["smote_target_ratio"])
    fold_metrics["smote_synthetic_count"] = int(smote_info["synthetic_count"])
    fold_metrics["smote_minority_class"] = int(smote_info["minority_class"])
    fold_metrics["calibration_kind"] = calibration_kind
    if calibration_kind == "location_aware":
        fold_metrics["calibration_scale"] = float(np.linalg.norm(np.asarray(calibrator["weights"], dtype=float)))
        fold_metrics["calibration_bias"] = float(calibrator["bias"])
    else:
        fold_metrics["calibration_scale"] = float(calibrator.get("scale", 1.0))
        fold_metrics["calibration_bias"] = float(calibrator.get("bias", 0.0))
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": asdict(model_config),
            "stft_config": None,
            "spectrogram_mean": train_mean,
            "spectrogram_std": train_std,
            "threshold": threshold,
            "calibrated_threshold": calibrated_threshold,
            "calibration": calibrator,
            "calibration_kind": calibration_kind,
            "weak_murmur_weight": weak_murmur_weight,
            "moderate_murmur_weight": moderate_murmur_weight,
            "loss": getattr(args, "loss", "bce"),
            "focal_gamma": float(getattr(args, "focal_gamma", 2.0)),
            "focal_alpha": getattr(args, "focal_alpha", None),
            "auc_loss_weight": float(getattr(args, "auc_loss_weight", 0.0)),
            "auc_loss_margin": float(getattr(args, "auc_loss_margin", 1.0)),
            "stratified_batches": bool(use_stratified_batches(args, train_labels)),
            "augmentation_config": augmentation_config,
            "smote_info": smote_info,
        },
        output_dir / f"fold_{fold}_best_model.pt",
    )
    return val_record_probs, threshold, fold_metrics, history, val_patient_calibrated


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    order = np.argsort(-y_prob)
    y_sorted = y_true[order].astype(int)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / max(int(y_sorted.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall / sensitivity")
    ax.set_ylabel("Precision")
    ax.set_title("Patient-level out-of-fold precision-recall")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    stft_cfg: StftConfig,
    model_config: ModelConfig,
    meta: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    oof_metrics_05: dict[str, float | int],
    oof_metrics_tuned: dict[str, float | int],
    oof_metrics_calibrated_05: dict[str, float | int],
) -> None:
    patients = meta.drop_duplicates("patient_id")
    lines = [
        "# Grupo G - CNN dilatada em STFT de fase cardiaca predita por TCN",
        "",
        "## Objetivo",
        "",
        "Treinar um modelo deep learning que recebe trechos de fase cardiaca preditos pelo TCN, gera STFT log-magnitude e usa convolucoes dilatadas para prever `Murmur = Present` vs `Absent`.",
        "",
        "## Dados",
        "",
        f"- Locais incluidos: `{', '.join(args.locations)}`",
        f"- Gravacoes usadas: {len(meta)}",
        f"- Pacientes usados: {patients['patient_id'].nunique()}",
        f"- Present por paciente: {int((patients['murmur'] == 'Present').sum())}",
        f"- Absent por paciente: {int((patients['murmur'] == 'Absent').sum())}",
        "",
        "## STFT",
        "",
        f"- Target sample rate: {stft_cfg.target_sample_rate}",
        f"- spectrogram_type: {stft_cfg.spectrogram_type}",
        f"- n_mels: {stft_cfg.n_mels}",
        f"- n_fft: {stft_cfg.n_fft}",
        f"- hop_length: {stft_cfg.hop_length}",
        f"- low_hz: {getattr(stft_cfg, 'low_hz', 0.0)}",
        f"- high_hz: {stft_cfg.high_hz}",
        f"- max_frames: {stft_cfg.max_frames}",
        f"- cnn_phase_mode: {stft_cfg.cnn_phase_mode}",
        f"- min_phase_seconds: {stft_cfg.min_systole_seconds}",
        f"- systole_threshold: {stft_cfg.systole_threshold if stft_cfg.systole_threshold is not None else 'argmax'}",
        f"- systole_margin_ms: {stft_cfg.systole_margin_ms}",
        "",
        "## Modelo",
        "",
        f"- Base channels: {model_config.base_channels}",
        f"- Dilations: `{model_config.dilations}`",
        f"- Encoder block: `{model_config.encoder_block}`",
        f"- Pooling: `{model_config.pooling}`",
        f"- Patient fixed-location fusion (`--patient-mil-attention`): `{args.patient_mil_attention}`",
        f"- Location slots: `{LOCATION_ORDER}`",
        f"- Auxiliary per-location loss weight: `{args.mil_instance_loss_weight}`",
        f"- Calibration: `{args.calibration}`",
        f"- Dropout: {model_config.dropout}",
        f"- Saida: linear logit + sigmoid para probabilidade de sopro",
        "",
        "## Augmentation",
        "",
        f"- LTSRR probability: {args.ltsrr_prob}",
        f"- LTSRR K: {args.ltsrr_k}",
        f"- LTSRR frequency ratio: {args.ltsrr_frequency_ratio}",
        f"- LTSRR minority only: {args.ltsrr_minority_only}",
        f"- SMOTE minority augmentation: {args.smote_minority_augmentation}",
        f"- SMOTE k neighbors: {args.smote_k_neighbors}",
        f"- SMOTE target ratio: {args.smote_target_ratio}",
        f"- Loss: {args.loss}",
        f"- Focal gamma: {args.focal_gamma}",
        f"- Focal alpha: {args.focal_alpha if args.focal_alpha is not None else 'none'}",
        f"- AUC loss weight: {args.auc_loss_weight}",
        f"- AUC loss margin: {args.auc_loss_margin}",
        f"- Stratified train batches when AUC loss is active: {args.auc_loss_weight > 0.0}",
        "",
        "## Metricas paciente-level out-of-fold",
        "",
        "### Threshold 0.5",
        "",
        pd.DataFrame([oof_metrics_05]).to_markdown(index=False),
        "",
        "### Threshold 0.5 calibrado",
        "",
        pd.DataFrame([oof_metrics_calibrated_05]).to_markdown(index=False),
        "",
        "### Threshold ajustado por fold",
        "",
        pd.DataFrame([oof_metrics_tuned]).to_markdown(index=False),
        "",
        "## Metricas por fold",
        "",
        fold_metrics.to_markdown(index=False),
        "",
        "## Arquivos gerados",
        "",
        "- `recording_metadata.csv`: gravacoes usadas e estatisticas dos trechos de fase cardiaca.",
        "- `patient_oof_predictions.csv`: predicoes paciente-level out-of-fold.",
        "- `patient_oof_predictions_calibrated.csv`: predicoes paciente-level calibradas por fold.",
        "- `threshold_metrics_by_fold.md`: metricas por fold em varios thresholds.",
        "- `threshold_metrics_by_fold_raw.csv`: metricas por fold em varios thresholds para probabilidade bruta.",
        "- `threshold_metrics_by_fold_calibrated.csv`: metricas por fold em varios thresholds para probabilidade calibrada.",
        "- `threshold_metrics_by_fold/`: tabelas CSV separadas por fold.",
        "- `mil_instance_attention_oof.csv`: diagnostico por gravacao/local da fusao quando `--patient-mil-attention` esta ativo.",
        "- `recording_oof_predictions.csv`: predicoes por gravacao out-of-fold.",
        "- `fold_metrics.csv`: metricas por fold.",
        "- `training_history.csv`: historico de treino.",
        "- `precision_recall_oof.png`: curva precision-recall paciente-level.",
        "- `precision_recall_oof_calibrated.png`: curva precision-recall paciente-level calibrada.",
        "- `fold_*_best_model.pt`: checkpoints por fold.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
