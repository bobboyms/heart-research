from __future__ import annotations


import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


from .aggregate import aggregate_patient_location_features
from .augment import augmentation_config_from_args, build_cnn_augmenter
from .calibration import apply_location_aware_calibrator, apply_platt_calibrator, fit_location_aware_calibrator, fit_platt_calibrator
from .config import LOCATION_ORDER, ModelConfig
from .dataset import _broadcast_stats, build_train_loader, compute_freq_norm_stats, use_stratified_batches
from .losses import add_auc_loss, build_binary_loss
from .metrics import average_precision, choose_threshold, metrics, roc_auc
from .models import SystoleDilatedCNN
from .segments import parse_murmur_locations


class PatientMILDataset(Dataset):
    def __init__(
        self,
        specs: np.ndarray,
        labels: np.ndarray,
        meta: pd.DataFrame,
        indices: np.ndarray,
        mean: float | np.ndarray,
        std: float | np.ndarray,
        max_instances: int,
        weak_murmur_weight: float = 1.0,
        moderate_murmur_weight: float = 1.0,
        train: bool = False,
        augmenter: object | None = None,
        augmenter_minority_only: bool = False,
    ) -> None:
        self.specs = specs
        self.labels_array = labels.astype(np.float32)
        self.meta = meta.reset_index(drop=True)
        self.indices = np.asarray(indices, dtype=int)
        freq_bins = int(specs.shape[1])
        self.mean = _broadcast_stats(mean, freq_bins)
        self.std = _broadcast_stats(std, freq_bins)
        self.max_instances = len(LOCATION_ORDER)
        self.train = bool(train)
        self.augmenter = augmenter
        self.augmenter_minority_only = bool(augmenter_minority_only)

        frame = self.meta.iloc[self.indices].copy()
        frame["_array_index"] = self.indices
        self.patients: list[dict[str, object]] = []
        for patient_id, group in frame.groupby("patient_id", sort=True):
            group = group.sort_values(["location", "recording_id"])
            target = int(group["target"].iloc[0])
            murmur_value = group["murmur_locations"].iloc[0] if "murmur_locations" in group.columns else None
            murmur_locations = parse_murmur_locations(murmur_value)
            grade = ""
            if "systolic_murmur_grading" in group.columns:
                grade = str(group["systolic_murmur_grading"].fillna("").iloc[0]).strip().upper()
            weight = 1.0
            if target == 1 and grade == "I/VI":
                weight = float(weak_murmur_weight)
            elif target == 1 and grade == "II/VI":
                weight = float(moderate_murmur_weight)
            indices_by_location: dict[str, list[int]] = {location: [] for location in LOCATION_ORDER}
            for location, location_group in group.groupby("location", sort=False):
                location_text = str(location)
                if location_text in indices_by_location:
                    indices_by_location[location_text] = location_group["_array_index"].to_numpy(dtype=int).tolist()
            self.patients.append(
                {
                    "patient_id": str(patient_id),
                    "murmur": str(group["murmur"].iloc[0]),
                    "target": target,
                    "indices_by_location": indices_by_location,
                    "murmur_locations": murmur_locations,
                    "weight": weight,
                }
            )
        self.labels = np.asarray([patient["target"] for patient in self.patients], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.patients)

    def _choose_location_index(self, indices: list[int]) -> int:
        if not indices:
            return -1
        if self.train and len(indices) > 1:
            return int(np.random.choice(np.asarray(indices, dtype=int)))
        return int(indices[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patient = self.patients[index]
        freq_bins = int(self.specs.shape[1])
        max_frames = int(self.specs.shape[2])

        padded_x = np.zeros((self.max_instances, freq_bins, max_frames), dtype=np.float32)
        mask = np.zeros(self.max_instances, dtype=np.float32)
        raw_indices = np.full(self.max_instances, -1, dtype=np.int64)
        instance_targets = np.zeros(self.max_instances, dtype=np.float32)
        instance_loss_mask = np.zeros(self.max_instances, dtype=np.float32)

        target = int(patient["target"])
        murmur_locations = set(patient["murmur_locations"])
        indices_by_location: dict[str, list[int]] = patient["indices_by_location"]  # type: ignore[assignment]
        for loc_idx, location in enumerate(LOCATION_ORDER):
            chosen_index = self._choose_location_index(indices_by_location.get(location, []))
            if chosen_index < 0:
                continue
            spec = self.specs[chosen_index]
            if self.train and self.augmenter is not None and (not self.augmenter_minority_only or target == 1):
                spec = self.augmenter(spec)  # type: ignore[operator]
            padded_x[loc_idx] = ((spec - self.mean) / self.std).astype(np.float32)
            mask[loc_idx] = 1.0
            raw_indices[loc_idx] = chosen_index
            if target == 0:
                instance_targets[loc_idx] = 0.0
                instance_loss_mask[loc_idx] = 1.0
            elif location in murmur_locations:
                instance_targets[loc_idx] = 1.0
                instance_loss_mask[loc_idx] = 1.0

        return (
            torch.from_numpy(padded_x),
            torch.from_numpy(mask),
            torch.tensor(float(patient["target"]), dtype=torch.float32),
            torch.tensor(float(patient["weight"]), dtype=torch.float32),
            torch.from_numpy(raw_indices),
            torch.from_numpy(instance_targets),
            torch.from_numpy(instance_loss_mask),
        )


class PatientMILAttentionClassifier(nn.Module):
    def __init__(self, config: ModelConfig, location_embedding_dim: int) -> None:
        super().__init__()
        self.recording_encoder = SystoleDilatedCNN(config)
        channels = int(config.base_channels)
        self.instance_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(channels, 1),
        )
        fusion_dim = channels * len(LOCATION_ORDER) + len(LOCATION_ORDER) * 2
        fusion_hidden = max(channels * 2, 16)
        self.patient_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(fusion_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, instances, freq_bins, frames = x.shape
        flat_x = x.reshape(batch_size * instances, freq_bins, frames)
        encoded = self.recording_encoder.encode(flat_x).reshape(batch_size, instances, -1)
        masked_encoded = encoded * mask.unsqueeze(-1)
        instance_logits = self.instance_head(encoded).squeeze(-1)
        instance_probs = torch.sigmoid(instance_logits) * mask
        fusion_features = torch.cat(
            [
                masked_encoded.reshape(batch_size, -1),
                instance_probs,
                mask,
            ],
            dim=1,
        )
        valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        slot_weights = mask / valid_counts
        return {
            "patient_logits": self.patient_head(fusion_features).squeeze(-1),
            "instance_logits": instance_logits,
            "slot_weights": slot_weights,
        }


@torch.no_grad()
def predict_patient_mil(
    model: PatientMILAttentionClassifier,
    dataset: PatientMILDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    patient_probs: list[np.ndarray] = []
    record_probs: list[np.ndarray] = []
    record_indices: list[np.ndarray] = []
    record_slot_weights: list[np.ndarray] = []
    record_instance_targets: list[np.ndarray] = []
    record_instance_loss_masks: list[np.ndarray] = []
    for batch in loader:
        x, mask, _y, _weight, raw_indices, instance_targets, instance_loss_mask = batch
        x = x.to(device)
        mask_device = mask.to(device)
        outputs = model(x, mask_device)
        patient_probs.append(torch.sigmoid(outputs["patient_logits"]).detach().cpu().numpy())
        instance_prob = torch.sigmoid(outputs["instance_logits"]).detach().cpu().numpy()
        raw_indices_np = raw_indices.numpy()
        mask_np = mask.numpy() > 0
        record_probs.append(instance_prob[mask_np])
        record_indices.append(raw_indices_np[mask_np])
        record_slot_weights.append(outputs["slot_weights"].detach().cpu().numpy()[mask_np])
        record_instance_targets.append(instance_targets.numpy()[mask_np])
        record_instance_loss_masks.append(instance_loss_mask.numpy()[mask_np])
    patient_frame = pd.DataFrame(
        [
            {
                "patient_id": patient["patient_id"],
                "murmur": patient["murmur"],
                "target": int(patient["target"]),
            }
            for patient in dataset.patients
        ]
    )
    patient_frame["prob"] = np.concatenate(patient_probs) if patient_probs else np.asarray([], dtype=float)
    if record_probs:
        probs = np.concatenate(record_probs)
        indices = np.concatenate(record_indices).astype(int)
        slot_weights = np.concatenate(record_slot_weights)
        instance_targets = np.concatenate(record_instance_targets)
        instance_loss_masks = np.concatenate(record_instance_loss_masks)
        record_frame = dataset.meta.iloc[indices][
            ["patient_id", "recording_id", "location", "murmur", "target"]
        ].copy()
        record_frame["instance_prob"] = probs.astype(float)
        record_frame["instance_target"] = instance_targets.astype(float)
        record_frame["instance_loss_mask"] = instance_loss_masks.astype(float)
        record_frame["fusion_slot_weight"] = slot_weights.astype(float)
        record_frame["attention_weight"] = slot_weights.astype(float)
        return patient_frame, record_frame, probs, indices
    empty_record_frame = pd.DataFrame(
        columns=[
            "patient_id",
            "recording_id",
            "location",
            "murmur",
            "target",
            "instance_prob",
            "instance_target",
            "instance_loss_mask",
            "fusion_slot_weight",
            "attention_weight",
        ]
    )
    return patient_frame, empty_record_frame, np.asarray([], dtype=float), np.asarray([], dtype=int)


def train_one_fold_patient_mil(
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
    if bool(getattr(args, "smote_minority_augmentation", False)):
        raise ValueError("--smote-minority-augmentation is not supported with --patient-mil-attention.")

    selection_indices = val_indices if tune_indices is None else tune_indices
    calibration_indices = train_indices if tune_indices is None else tune_indices
    train_subset = specs[train_indices]
    train_mean, train_std = compute_freq_norm_stats(train_subset, str(getattr(args, "freq_norm", "perbin")))
    weak_murmur_weight = float(getattr(args, "weak_murmur_weight", 1.0))
    moderate_murmur_weight = float(getattr(args, "moderate_murmur_weight", 1.0))
    max_instances = len(LOCATION_ORDER)
    location_embedding_dim = int(getattr(args, "mil_location_embedding_dim", 4))
    instance_loss_weight = float(getattr(args, "mil_instance_loss_weight", 0.25))
    augmenter = build_cnn_augmenter(args)
    augmentation_config = augmentation_config_from_args(args)

    train_ds = PatientMILDataset(
        specs,
        labels,
        meta,
        train_indices,
        train_mean,
        train_std,
        max_instances,
        weak_murmur_weight,
        moderate_murmur_weight,
        train=True,
        augmenter=augmenter,
        augmenter_minority_only=bool(augmentation_config["ltsrr_minority_only"]),
    )
    selection_ds = PatientMILDataset(specs, labels, meta, selection_indices, train_mean, train_std, max_instances)
    calibration_ds = PatientMILDataset(specs, labels, meta, calibration_indices, train_mean, train_std, max_instances)
    val_ds = PatientMILDataset(specs, labels, meta, val_indices, train_mean, train_std, max_instances)
    loader = build_train_loader(train_ds, train_ds.labels, args)

    model = PatientMILAttentionClassifier(model_config, location_embedding_dim=location_embedding_dim).to(device)
    pos = float((train_ds.labels == 1).sum())
    neg = float((train_ds.labels == 0).sum())
    loss_fn = build_binary_loss(args, pos, neg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_auprc = -math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            x, mask, y, sample_weight, _raw_indices, instance_targets, instance_loss_mask = batch
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            sample_weight = sample_weight.to(device)
            instance_targets = instance_targets.to(device)
            instance_loss_mask = instance_loss_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, mask)
            patient_loss = loss_fn(outputs["patient_logits"], y)
            instance_loss = loss_fn(outputs["instance_logits"], instance_targets)
            valid_instance_loss = mask * instance_loss_mask
            instance_loss = (instance_loss * valid_instance_loss).sum(dim=1) / valid_instance_loss.sum(dim=1).clamp_min(1.0)
            loss = ((patient_loss + instance_loss_weight * instance_loss) * sample_weight).mean()
            loss = add_auc_loss(loss, outputs["patient_logits"], y, args)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        selection_patient, _selection_record_frame, _selection_record_probs, _selection_record_indices = predict_patient_mil(
            model,
            selection_ds,
            args.batch_size,
            device,
        )
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
        raise RuntimeError(f"Fold {fold} did not produce a MIL model.")
    model.load_state_dict(best_state)

    calibration_patient, calibration_record_frame, calibration_record_probs, calibration_record_indices = predict_patient_mil(
        model,
        calibration_ds,
        args.batch_size,
        device,
    )
    val_patient, val_record_frame, val_record_probs_observed, val_record_indices_observed = predict_patient_mil(
        model,
        val_ds,
        args.batch_size,
        device,
    )
    threshold = choose_threshold(calibration_patient["target"].to_numpy(), calibration_patient["prob"].to_numpy())

    calibrator: dict[str, object] = {"scale": 1.0, "bias": 0.0}
    calibration_kind = "none"
    val_calibrated_by_patient: pd.Series | None = None
    if bool(getattr(args, "location_aware_calibration", False)):
        calibration_features, feature_columns = aggregate_patient_location_features(
            meta,
            calibration_record_probs,
            calibration_record_indices,
        )
        val_features, _feature_columns = aggregate_patient_location_features(meta, val_record_probs_observed, val_record_indices_observed)
        calibrator = fit_location_aware_calibrator(calibration_features, feature_columns)
        calibration_calibrated_prob = apply_location_aware_calibrator(calibration_features, calibrator)
        val_calibrated_prob = apply_location_aware_calibrator(val_features, calibrator)
        val_calibrated_by_patient = pd.Series(
            val_calibrated_prob,
            index=val_features["patient_id"].astype(str),
        )
        calibration_kind = "mil_location_aware"
    elif args.calibration == "platt":
        calibrator = fit_platt_calibrator(calibration_patient["target"].to_numpy(), calibration_patient["prob"].to_numpy())
        calibration_calibrated_prob = apply_platt_calibrator(
            calibration_patient["prob"].to_numpy(dtype=float),
            calibrator,  # type: ignore[arg-type]
        )
        val_calibrated_prob = apply_platt_calibrator(
            val_patient["prob"].to_numpy(dtype=float),
            calibrator,  # type: ignore[arg-type]
        )
        calibration_kind = "mil_platt"
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
    if calibration_kind == "mil_location_aware":
        val_patient_calibrated["calibration_scale"] = float(np.linalg.norm(np.asarray(calibrator["weights"], dtype=float)))
        val_patient_calibrated["calibration_bias"] = float(calibrator["bias"])
    else:
        val_patient_calibrated["calibration_scale"] = float(calibrator.get("scale", 1.0))
        val_patient_calibrated["calibration_bias"] = float(calibrator.get("bias", 0.0))
    val_patient_calibrated = val_patient_calibrated.drop(columns=["prob"])

    val_patient_export = val_patient_calibrated[
        ["patient_id", "prob_present_raw", "prob_present_calibrated", "calibration_kind"]
    ].copy()
    val_record_frame = val_record_frame.merge(val_patient_export, on="patient_id", how="left")
    val_record_frame.insert(0, "fold", fold)
    val_record_frame.to_csv(output_dir / f"fold_{fold}_mil_instance_attention_validation.csv", index=False)

    calibration_record_frame = calibration_record_frame.copy()
    calibration_record_frame.insert(0, "fold", fold)
    calibration_record_frame.to_csv(output_dir / f"fold_{fold}_mil_instance_attention_calibration.csv", index=False)

    val_record_probs = np.zeros(len(val_indices), dtype=np.float32)
    local_position = {int(global_idx): pos for pos, global_idx in enumerate(np.asarray(val_indices, dtype=int))}
    for global_idx, prob in zip(val_record_indices_observed, val_record_probs_observed, strict=False):
        if int(global_idx) in local_position:
            val_record_probs[local_position[int(global_idx)]] = float(prob)
    patient_prob_by_id = val_patient.set_index("patient_id")["prob"].astype(float).to_dict()
    missing_positions = val_record_probs == 0
    if missing_positions.any():
        for local_idx in np.flatnonzero(missing_positions):
            patient_id = str(meta.iloc[int(val_indices[local_idx])]["patient_id"])
            val_record_probs[local_idx] = float(patient_prob_by_id.get(patient_id, 0.0))

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
    fold_metrics["stratified_batches"] = int(use_stratified_batches(args, train_ds.labels))
    fold_metrics["patient_mil_attention"] = 1
    fold_metrics["mil_instance_loss_weight"] = instance_loss_weight
    fold_metrics["ltsrr_prob"] = float(augmentation_config["ltsrr_prob"])
    fold_metrics["ltsrr_k"] = int(augmentation_config["ltsrr_k"])
    fold_metrics["ltsrr_frequency_ratio"] = float(augmentation_config["ltsrr_frequency_ratio"])
    fold_metrics["ltsrr_minority_only"] = int(bool(augmentation_config["ltsrr_minority_only"]))
    fold_metrics["smote_minority_augmentation"] = int(bool(augmentation_config["smote_minority_augmentation"]))
    fold_metrics["smote_k_neighbors"] = int(augmentation_config["smote_k_neighbors"])
    fold_metrics["smote_target_ratio"] = float(augmentation_config["smote_target_ratio"])
    fold_metrics["smote_synthetic_count"] = 0
    fold_metrics["smote_minority_class"] = -1
    fold_metrics["calibration_kind"] = calibration_kind
    if calibration_kind == "mil_location_aware":
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
            "stratified_batches": bool(use_stratified_batches(args, train_ds.labels)),
            "augmentation_config": augmentation_config,
            "patient_mil_attention": True,
            "mil_max_instances": max_instances,
            "mil_location_embedding_dim": location_embedding_dim,
            "mil_instance_loss_weight": instance_loss_weight,
        },
        output_dir / f"fold_{fold}_best_model.pt",
    )
    return val_record_probs, threshold, fold_metrics, history, val_patient_calibrated
