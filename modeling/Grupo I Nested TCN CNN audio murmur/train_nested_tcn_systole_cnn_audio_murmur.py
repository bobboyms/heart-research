# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scipy>=1.12",
#   "scikit-learn>=1.4",
#   "tabulate>=0.9",
#   "torch>=2.2",
#   "tqdm>=4.66",
# ]
# ///
"""Nested patient-split validation for audio-level murmur classification.

This experiment keeps the Grupo H leakage control:

1. Split train/validation by patient.
2. Train each fold-specific TCN only on train patients.
3. Use that TCN to extract systole+diastole STFTs for train/tune/validation recordings.

The difference from Grupo H is the classifier target:

    recording target = 1 only when the recording auscultation location is listed
    in `Murmur locations` for a `Murmur = Present` patient.

All other recordings from Present/Absent patients are target 0. Patients with
`Murmur = Unknown` are excluded by the reused Grupo G item builder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
NESTED_PATIENT_SCRIPT = (
    REPO_ROOT / "modeling" / "Grupo H Nested TCN CNN systole" / "train_nested_tcn_systole_cnn.py"
)

nested: ModuleType
cnn: ModuleType
LOCATION_TO_ID = {"AV": 0, "PV": 1, "TV": 2, "MV": 3, "Phc": 4}
LABEL_SYSTOLE = 2
LABEL_DIASTOLE = 4


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_project_modules() -> None:
    global nested, cnn
    nested = load_module("grupo_i_nested_patient_base", NESTED_PATIENT_SCRIPT)
    nested.load_project_modules()
    cnn = nested.cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested TCN + systole+diastole CNN with audio-level murmur labels.")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "circor-heart-sound-1.0.3")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--locations", nargs="+", default=["AV", "PV", "TV", "MV"], choices=["AV", "PV", "TV", "MV", "Phc"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--force-retrain-tcn", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--keep-fold-training-data",
        action="store_true",
        help=(
            "Keep per-fold generated training data after each fold finishes. By default the script deletes "
            "the temporary TCN dataset, TCN feature cache, predicted TSVs, and spectrogram cache to save disk space."
        ),
    )

    parser.add_argument("--tcn-epochs", type=int, default=10)
    parser.add_argument("--tcn-batch-size", type=int, default=42)
    parser.add_argument("--tcn-device", choices=["auto", "cpu", "mps"], default="mps")
    parser.add_argument("--tcn-val-size", type=float, default=0.15)
    parser.add_argument("--tcn-test-size", type=float, default=0.15)
    parser.add_argument("--tcn-pooling", choices=["none", "attention"], default="none")
    parser.add_argument("--tcn-boundary-ignore-ms", type=float, default=0.0)
    parser.add_argument("--tcn-systole-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--tcn-target-mode", choices=["cardiac-phase", "systole-binary"], default="cardiac-phase")
    parser.add_argument("--other-mode", "--tcn-other-mode", dest="tcn_other_mode", choices=["keep", "ignore"], default="keep")

    parser.add_argument("--cnn-epochs", type=int, default=50)
    parser.add_argument("--cnn-patience", type=int, default=8)
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--cnn-inner-val-size", type=float, default=0.15)
    parser.add_argument("--cnn-device", choices=["auto", "cpu", "mps"], default="mps")
    parser.add_argument(
        "--location-embedding",
        action="store_true",
        help="Pass the auscultation location to the CNN as a trainable embedding added to the input spectrogram.",
    )
    parser.add_argument(
        "--location-lstm-after-embedding",
        action="store_true",
        help="After adding the location embedding to x, pass the temporal sequence through an LSTM before the CNN.",
    )
    parser.add_argument(
        "--location-lstm-layers",
        type=int,
        default=1,
        help="Number of LSTM layers used by --location-lstm-after-embedding.",
    )
    parser.add_argument(
        "--location-lstm-dropout",
        type=float,
        default=0.0,
        help="Dropout between LSTM layers. Only applies when --location-lstm-layers is greater than 1.",
    )
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use a WeightedRandomSampler for CNN training so positive audio-level samples appear more often.",
    )
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help=(
            "Apply SMOTE to positive CNN training STFTs only. This runs after TCN systole+diastole extraction "
            "and does not alter audio, tuning, or validation samples."
        ),
    )
    parser.add_argument(
        "--smote-target-ratio",
        type=float,
        default=1.0,
        help="Target positive/negative ratio after SMOTE in the CNN fit split. 1.0 balances positives and negatives.",
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="Maximum number of positive-neighbor candidates used by SMOTE.",
    )
    parser.add_argument(
        "--sampler-positive-fraction",
        type=float,
        default=0.5,
        help="Target fraction of positive samples drawn by --balanced-sampler.",
    )
    parser.add_argument(
        "--loss-pos-weight-mode",
        choices=["auto", "ratio", "sqrt", "none"],
        default="auto",
        help=(
            "Positive class weight for BCE. 'ratio' uses neg/pos, 'sqrt' uses sqrt(neg/pos), "
            "'none' uses 1.0. 'auto' uses ratio without sampler and 1.0 with --balanced-sampler."
        ),
    )
    parser.add_argument(
        "--cnn-loss",
        choices=["bce", "focal"],
        default="bce",
        help="Loss used to train the CNN. 'bce' preserves the current weighted BCE behavior; 'focal' applies focal modulation.",
    )
    parser.add_argument(
        "--cnn-focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma used when --cnn-loss focal. 0.0 is equivalent to weighted BCE.",
    )
    parser.add_argument("--pooling", choices=["avg", "attention"], default="attention")
    parser.add_argument("--calibration", choices=["none", "platt"], default="platt")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--weak-murmur-weight", type=float, default=1.0)
    parser.add_argument("--moderate-murmur-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--dilations", type=str, default="1,2,4,8")
    parser.add_argument("--encoder-block", choices=["residual", "multiscale"], default="residual")

    parser.add_argument("--target-sample-rate", type=int, default=4000)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--low-hz", type=float, default=0.0)
    parser.add_argument("--high-hz", type=float, default=1000.0)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--min-systole-seconds", type=float, default=0.10)
    parser.add_argument("--min-diastole-seconds", type=float, default=0.10)
    parser.add_argument(
        "--systole-threshold",
        type=float,
        default=None,
        help="Deprecated for this script. Systole+diastole extraction requires multiclass argmax.",
    )
    parser.add_argument(
        "--systole-margin-ms",
        type=float,
        default=0.0,
        help="Expand each predicted systole and diastole segment by this many milliseconds before STFT extraction.",
    )
    return parser.parse_args()


def format_threshold_key(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def cleanup_fold_training_data(fold_dir: Path, show_progress: bool) -> None:
    removable_dirs = [
        fold_dir / "tcn_dataset_train_patients",
        fold_dir / "tcn" / "cache",
        fold_dir / "predicted_tsvs",
        fold_dir / "spectrogram_cache",
    ]
    removed: list[str] = []
    for path in removable_dirs:
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(fold_dir)))
    if show_progress and removed:
        print(f"Removed fold training data from {fold_dir}: {', '.join(removed)}", flush=True)


def binary_focal_loss_from_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_loss: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pt = torch.where(target >= 0.5, probs, 1.0 - probs).clamp(min=1e-6, max=1.0)
    return ((1.0 - pt) ** gamma) * bce_loss


def recording_has_murmur(patient_murmur: object, murmur_locations: object, location: object) -> int:
    if str(patient_murmur) != "Present":
        return 0
    return int(str(location) in cnn.parse_murmur_locations(murmur_locations))


def apply_audio_level_targets(meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    meta = meta.copy()
    meta["patient_murmur"] = meta["murmur"].astype(str)
    meta["patient_target"] = (meta["patient_murmur"] == "Present").astype(int)
    meta["audio_target"] = [
        recording_has_murmur(row.patient_murmur, row.murmur_locations, row.location)
        for row in meta.itertuples(index=False)
    ]
    meta["target"] = meta["audio_target"].astype(int)
    labels = meta["audio_target"].to_numpy(dtype=np.float32)
    return meta, labels


def split_fit_tune_patients(
    meta: pd.DataFrame,
    train_patient_ids: set[str],
    tune_size: float,
    seed: int,
    fold: int,
) -> tuple[set[str], set[str]]:
    train_meta = meta.loc[meta["patient_id"].astype(str).isin(train_patient_ids)].copy()
    patient_table = (
        train_meta.groupby("patient_id", as_index=False)
        .agg(patient_target=("patient_target", "first"), positive_recordings=("audio_target", "sum"))
    )
    rng = np.random.default_rng(seed + fold * 1009)
    tune_ids: list[str] = []

    for patient_target in [1, 0]:
        ids = patient_table.loc[patient_table["patient_target"] == patient_target, "patient_id"].astype(str).to_numpy()
        if len(ids) < 2:
            raise RuntimeError(
                f"Fold {fold} does not have enough patient class {patient_target} patients "
                "to create an internal CNN tuning split."
            )
        rng.shuffle(ids)
        count = max(1, int(round(len(ids) * tune_size)))
        count = min(count, len(ids) - 1)
        tune_ids.extend(ids[:count].tolist())

    tune_patient_ids = set(tune_ids)
    fit_patient_ids = set(patient_table["patient_id"].astype(str)) - tune_patient_ids
    for name, ids in [("fit", fit_patient_ids), ("tune", tune_patient_ids)]:
        subset = train_meta.loc[train_meta["patient_id"].astype(str).isin(ids), "audio_target"].to_numpy(dtype=int)
        if len(np.unique(subset)) < 2:
            raise RuntimeError(f"Fold {fold} internal CNN {name} split has only one audio-level class.")
    return fit_patient_ids, tune_patient_ids


class LocationSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        specs: np.ndarray,
        labels: np.ndarray,
        locations: np.ndarray,
        mean: float,
        std: float,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.specs = ((specs - mean) / std).astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.location_ids = np.asarray([LOCATION_TO_ID[str(location)] for location in locations], dtype=np.int64)
        self.sample_weights = (
            np.ones(len(self.labels), dtype=np.float32)
            if sample_weights is None
            else sample_weights.astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.specs[index]),
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(self.sample_weights[index], dtype=torch.float32),
            torch.tensor(self.location_ids[index], dtype=torch.long),
        )


def apply_smote_to_training_stfts(
    specs: np.ndarray,
    labels: np.ndarray,
    locations: np.ndarray,
    sample_weights: np.ndarray,
    target_ratio: float,
    k_neighbors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    labels_int = labels.astype(int)
    positive_indices = np.flatnonzero(labels_int == 1)
    negative_count = int((labels_int == 0).sum())
    positive_count = int(len(positive_indices))
    desired_positive_count = max(positive_count, int(math.ceil(negative_count * float(target_ratio))))
    new_count = desired_positive_count - positive_count
    info: dict[str, int | float] = {
        "enabled": 1,
        "original_samples": int(len(labels_int)),
        "original_positive": positive_count,
        "original_negative": negative_count,
        "target_ratio": float(target_ratio),
        "k_neighbors_requested": int(k_neighbors),
        "k_neighbors_used": 0,
        "synthetic_positive": 0,
        "final_samples": int(len(labels_int)),
        "final_positive": positive_count,
        "final_negative": negative_count,
    }
    if new_count <= 0:
        return specs, labels, locations, sample_weights, info
    if positive_count < 2:
        print("SMOTE skipped: at least two positive training STFTs are required.")
        return specs, labels, locations, sample_weights, info

    n_neighbors = min(int(k_neighbors), positive_count - 1)
    if n_neighbors < 1:
        print("SMOTE skipped: --smote-k-neighbors must allow at least one neighbor.")
        return specs, labels, locations, sample_weights, info

    original_shape = specs.shape[1:]
    flat_specs = specs.reshape(len(specs), -1).astype(np.float32)
    positive_flat = flat_specs[positive_indices]
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1)
    neighbors.fit(positive_flat)
    neighbor_positions = neighbors.kneighbors(positive_flat, return_distance=False)[:, 1:]

    rng = np.random.default_rng(seed)
    anchor_positions = rng.integers(0, positive_count, size=new_count)
    neighbor_choices = rng.integers(0, n_neighbors, size=new_count)
    chosen_neighbor_positions = neighbor_positions[anchor_positions, neighbor_choices]
    gaps = rng.random((new_count, 1), dtype=np.float32)
    synthetic_flat = positive_flat[anchor_positions] + gaps * (
        positive_flat[chosen_neighbor_positions] - positive_flat[anchor_positions]
    )
    synthetic_specs = synthetic_flat.reshape((new_count, *original_shape)).astype(np.float32)
    anchor_indices = positive_indices[anchor_positions]
    synthetic_labels = np.ones(new_count, dtype=labels.dtype)
    synthetic_locations = locations[anchor_indices]
    synthetic_weights = sample_weights[anchor_indices].astype(np.float32)

    augmented_specs = np.concatenate([specs.astype(np.float32), synthetic_specs], axis=0)
    augmented_labels = np.concatenate([labels, synthetic_labels], axis=0)
    augmented_locations = np.concatenate([locations, synthetic_locations], axis=0)
    augmented_weights = np.concatenate([sample_weights.astype(np.float32), synthetic_weights], axis=0)

    info.update(
        {
            "k_neighbors_used": int(n_neighbors),
            "synthetic_positive": int(new_count),
            "final_samples": int(len(augmented_labels)),
            "final_positive": int((augmented_labels.astype(int) == 1).sum()),
            "final_negative": negative_count,
        }
    )
    return augmented_specs, augmented_labels, augmented_locations, augmented_weights, info


class LocationAwareSystoleDiastoleDilatedCNN(nn.Module):
    def __init__(
        self,
        config: object,
        num_locations: int,
        use_lstm_after_embedding: bool = False,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        freq_bins = int(config.freq_bins)
        self.location_embedding = nn.Embedding(num_locations, int(config.freq_bins))
        nn.init.zeros_(self.location_embedding.weight)
        self.use_lstm_after_embedding = bool(use_lstm_after_embedding)
        self.pre_cnn_lstm = (
            nn.LSTM(
                input_size=freq_bins,
                hidden_size=freq_bins,
                num_layers=int(lstm_layers),
                batch_first=True,
                dropout=float(lstm_dropout) if int(lstm_layers) > 1 else 0.0,
            )
            if self.use_lstm_after_embedding
            else None
        )
        self.cnn = cnn.SystoleDilatedCNN(config)

    def forward(self, x: torch.Tensor, location_ids: torch.Tensor) -> torch.Tensor:
        location_bias = self.location_embedding(location_ids).unsqueeze(-1)
        conditioned = x + location_bias
        if self.pre_cnn_lstm is not None:
            sequence = conditioned.transpose(1, 2)
            sequence, _hidden = self.pre_cnn_lstm(sequence)
            conditioned = sequence.transpose(1, 2)
        return self.cnn(conditioned)


@torch.no_grad()
def predict_audio_model(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: torch.device,
    use_location_embedding: bool,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    probs: list[np.ndarray] = []
    for batch in loader:
        x = batch[0].to(device)
        if use_location_embedding:
            location_ids = batch[3].to(device)
            logits = model(x, location_ids)
        else:
            logits = model(x)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def train_one_fold_audio(
    fold: int,
    specs: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    train_indices: np.ndarray,
    tune_indices: np.ndarray,
    val_indices: np.ndarray,
    model_config: object,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[np.ndarray, float, dict[str, float | int], list[dict[str, float | int]], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_mean = float(specs[train_indices].mean())
    train_std = float(specs[train_indices].std() + 1e-6)
    train_specs = specs[train_indices]
    train_labels = labels[train_indices]
    train_locations = meta.iloc[train_indices]["location"].to_numpy()
    train_weights = cnn.sample_weights_for_indices(
        meta,
        train_indices,
        float(args.weak_murmur_weight),
        float(args.moderate_murmur_weight),
    )
    original_train_positive = int((train_labels.astype(int) == 1).sum())
    original_train_negative = int((train_labels.astype(int) == 0).sum())
    train_dataset_mean = train_mean
    train_dataset_std = train_std
    smote_info: dict[str, int | float] = {
        "enabled": int(bool(getattr(args, "use_smote", False))),
        "original_samples": int(len(train_labels)),
        "original_positive": original_train_positive,
        "original_negative": original_train_negative,
        "target_ratio": float(getattr(args, "smote_target_ratio", 1.0)),
        "k_neighbors_requested": int(getattr(args, "smote_k_neighbors", 5)),
        "k_neighbors_used": 0,
        "synthetic_positive": 0,
        "final_samples": int(len(train_labels)),
        "final_positive": original_train_positive,
        "final_negative": original_train_negative,
    }
    if bool(getattr(args, "use_smote", False)):
        normalized_train_specs = ((train_specs - train_mean) / train_std).astype(np.float32)
        train_specs, train_labels, train_locations, train_weights, smote_info = apply_smote_to_training_stfts(
            normalized_train_specs,
            train_labels,
            train_locations,
            train_weights,
            float(args.smote_target_ratio),
            int(args.smote_k_neighbors),
            int(args.seed + fold * 10007),
        )
        train_dataset_mean = 0.0
        train_dataset_std = 1.0

    train_ds = LocationSpectrogramDataset(
        train_specs,
        train_labels,
        train_locations,
        train_dataset_mean,
        train_dataset_std,
        train_weights,
    )
    tune_ds = LocationSpectrogramDataset(
        specs[tune_indices],
        labels[tune_indices],
        meta.iloc[tune_indices]["location"].to_numpy(),
        train_mean,
        train_std,
    )
    val_ds = LocationSpectrogramDataset(
        specs[val_indices],
        labels[val_indices],
        meta.iloc[val_indices]["location"].to_numpy(),
        train_mean,
        train_std,
    )

    pos = float((train_labels == 1).sum())
    neg = float((train_labels == 0).sum())
    if pos <= 0 or neg <= 0:
        raise RuntimeError(f"Fold {fold} training split must contain both audio-level classes.")
    sampler = None
    if bool(getattr(args, "balanced_sampler", False)):
        sampler_labels = train_labels.astype(int)
        positive_fraction = float(getattr(args, "sampler_positive_fraction", 0.5))
        sample_prob = np.where(sampler_labels == 1, positive_fraction / pos, (1.0 - positive_fraction) / neg)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_prob, dtype=torch.double),
            num_samples=len(sampler_labels),
            replacement=True,
        )
    loader = DataLoader(train_ds, batch_size=args.cnn_batch_size, shuffle=sampler is None, sampler=sampler)
    use_location_embedding = bool(getattr(args, "location_embedding", False))
    model = (
        LocationAwareSystoleDiastoleDilatedCNN(
            model_config,
            len(LOCATION_TO_ID),
            use_lstm_after_embedding=bool(getattr(args, "location_lstm_after_embedding", False)),
            lstm_layers=int(getattr(args, "location_lstm_layers", 1)),
            lstm_dropout=float(getattr(args, "location_lstm_dropout", 0.0)),
        )
        if use_location_embedding
        else cnn.SystoleDilatedCNN(model_config)
    ).to(device)
    pos_weight_mode = str(getattr(args, "loss_pos_weight_mode", "auto"))
    if pos_weight_mode == "auto":
        pos_weight_value = 1.0 if bool(getattr(args, "balanced_sampler", False)) else neg / pos
    elif pos_weight_mode == "ratio":
        pos_weight_value = neg / pos
    elif pos_weight_mode == "sqrt":
        pos_weight_value = math.sqrt(neg / pos)
    elif pos_weight_mode == "none":
        pos_weight_value = 1.0
    else:
        raise ValueError(f"Unsupported --loss-pos-weight-mode: {pos_weight_mode}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device), reduction="none")
    cnn_loss = str(getattr(args, "cnn_loss", "bce"))
    focal_gamma = float(getattr(args, "cnn_focal_gamma", 2.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_auprc = -math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    tune_y = labels[tune_indices].astype(int)
    if args.progress:
        print(
            f"Fold {fold}: training audio-level CNN on {len(train_ds)} STFTs "
            f"({int(pos)} positive, {int(neg)} negative); "
            f"tuning on {len(tune_ds)} STFTs; validating on {len(val_ds)} STFTs; "
            f"loss={cnn_loss}" + (f"(gamma={focal_gamma:g})" if cnn_loss == "focal" else ""),
            flush=True,
        )
    for epoch in range(1, args.cnn_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            sample_weight = batch[2].to(device)
            location_ids = batch[3].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, location_ids) if use_location_embedding else model(x)
            per_sample_loss = loss_fn(logits, y)
            if cnn_loss == "focal":
                per_sample_loss = binary_focal_loss_from_bce(logits, y, per_sample_loss, focal_gamma)
            loss = (per_sample_loss * sample_weight).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        tune_probs = predict_audio_model(model, tune_ds, args.cnn_batch_size, device, use_location_embedding)
        tune_auprc = cnn.average_precision(tune_y, tune_probs)
        tune_auroc = cnn.roc_auc(tune_y, tune_probs)
        history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "tune_recording_auprc": tune_auprc,
                "tune_recording_auroc": tune_auroc,
                "loss_pos_weight": float(pos_weight_value),
                "cnn_focal_loss": int(cnn_loss == "focal"),
                "cnn_focal_gamma": float(focal_gamma),
                "balanced_sampler": int(bool(getattr(args, "balanced_sampler", False))),
                "smote": int(bool(getattr(args, "use_smote", False))),
                "smote_synthetic_positive": int(smote_info["synthetic_positive"]),
                "train_positive_original": original_train_positive,
                "train_negative_original": original_train_negative,
                "train_positive_after_smote": int(smote_info["final_positive"]),
                "train_negative_after_smote": int(smote_info["final_negative"]),
                "location_embedding": int(use_location_embedding),
                "location_lstm_after_embedding": int(bool(getattr(args, "location_lstm_after_embedding", False))),
            }
        )
        if tune_auprc > best_auprc:
            best_auprc = tune_auprc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
            improved = True
        else:
            stale_epochs += 1
            improved = False
        if args.progress:
            print(
                f"Fold {fold} CNN epoch {epoch}/{args.cnn_epochs}: "
                f"loss={float(np.mean(losses)):.4f} "
                f"tune_AUPRC={tune_auprc:.3f} "
                f"tune_AUROC={tune_auroc:.3f} "
                f"best_AUPRC={best_auprc:.3f} "
                f"stale={stale_epochs}/{args.cnn_patience}"
                f"{' *' if improved else ''}",
                flush=True,
            )
        if stale_epochs >= args.cnn_patience:
            if args.progress:
                print(
                    f"Fold {fold}: stopping CNN early after {epoch} epochs "
                    f"(patience={args.cnn_patience}).",
                    flush=True,
                )
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold} did not produce a model.")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": asdict(model_config),
            "train_mean": train_mean,
            "train_std": train_std,
            "smote_info": smote_info,
            "cnn_loss": cnn_loss,
            "cnn_focal_gamma": focal_gamma,
            "target": "audio_murmur_from_murmur_locations",
            "input_phases": ["systole", "diastole"],
            "location_embedding": use_location_embedding,
            "location_lstm_after_embedding": bool(getattr(args, "location_lstm_after_embedding", False)),
            "location_lstm_layers": int(getattr(args, "location_lstm_layers", 1)),
            "location_lstm_dropout": float(getattr(args, "location_lstm_dropout", 0.0)),
            "location_to_id": LOCATION_TO_ID,
            "args": vars(args),
        },
        output_dir / f"fold_{fold}_best_model.pt",
    )

    tune_probs = predict_audio_model(model, tune_ds, args.cnn_batch_size, device, use_location_embedding)
    val_probs = predict_audio_model(model, val_ds, args.cnn_batch_size, device, use_location_embedding)
    threshold = cnn.choose_threshold(tune_y, tune_probs)

    calibrator = {"scale": 1.0, "bias": 0.0}
    calibration_kind = "none"
    if args.calibration == "platt":
        calibrator = cnn.fit_platt_calibrator(tune_y, tune_probs)
        calibration_kind = "platt"
    tune_calibrated = cnn.apply_platt_calibrator(tune_probs, calibrator) if calibration_kind == "platt" else tune_probs
    val_calibrated = cnn.apply_platt_calibrator(val_probs, calibrator) if calibration_kind == "platt" else val_probs
    calibrated_threshold = cnn.choose_threshold(tune_y, tune_calibrated)

    val_frame = meta.iloc[val_indices][
        [
            "recording_id",
            "patient_id",
            "location",
            "patient_murmur",
            "patient_target",
            "murmur_locations",
            "most_audible_location",
            "audio_target",
            "target",
            "systole_seconds",
            "systole_segments",
            "diastole_seconds",
            "diastole_segments",
            "phase_seconds",
            "phase_segments",
        ]
    ].copy()
    val_frame["fold"] = fold
    val_frame["prob_murmur_raw"] = val_probs.astype(float)
    val_frame["prob_murmur_calibrated"] = val_calibrated.astype(float)
    val_frame["calibration_kind"] = calibration_kind
    val_frame["calibration_scale"] = float(calibrator.get("scale", 1.0))
    val_frame["calibration_bias"] = float(calibrator.get("bias", 0.0))

    val_y = labels[val_indices].astype(int)
    fold_metrics = cnn.metrics(val_y, val_probs, threshold)
    fold_metrics["fold"] = fold
    fold_metrics["calibrated_threshold"] = float(calibrated_threshold)
    fold_metrics["epochs_trained"] = len(history)
    fold_metrics["best_tune_auprc"] = float(best_auprc)
    fold_metrics["loss_pos_weight"] = float(pos_weight_value)
    fold_metrics["cnn_focal_loss"] = int(cnn_loss == "focal")
    fold_metrics["cnn_focal_gamma"] = float(focal_gamma)
    fold_metrics["balanced_sampler"] = int(bool(getattr(args, "balanced_sampler", False)))
    fold_metrics["smote"] = int(bool(getattr(args, "use_smote", False)))
    fold_metrics["smote_synthetic_positive"] = int(smote_info["synthetic_positive"])
    fold_metrics["train_positive_original"] = original_train_positive
    fold_metrics["train_negative_original"] = original_train_negative
    fold_metrics["train_positive_after_smote"] = int(smote_info["final_positive"])
    fold_metrics["train_negative_after_smote"] = int(smote_info["final_negative"])
    fold_metrics["location_embedding"] = int(use_location_embedding)
    fold_metrics["location_lstm_after_embedding"] = int(bool(getattr(args, "location_lstm_after_embedding", False)))
    calibrated_metrics_05 = cnn.metrics(val_y, val_calibrated, 0.5)
    fold_metrics["calibrated_ba_05"] = float(calibrated_metrics_05["balanced_accuracy"])
    fold_metrics["calibrated_sensitivity_05"] = float(calibrated_metrics_05["sensitivity"])
    fold_metrics["calibrated_specificity_05"] = float(calibrated_metrics_05["specificity"])
    fold_metrics["brier_score"] = cnn.brier_score(val_y, val_calibrated)
    return val_probs, threshold, fold_metrics, history, val_frame


def make_cnn_model_config(args: argparse.Namespace, specs: np.ndarray) -> object:
    dilations = tuple(int(part.strip()) for part in args.dilations.split(",") if part.strip())
    return cnn.ModelConfig(
        freq_bins=int(specs.shape[1]),
        max_frames=int(specs.shape[2]),
        base_channels=args.base_channels,
        dropout=args.dropout,
        dilations=dilations,
        pooling=args.pooling,
        encoder_block=args.encoder_block,
    )


def systole_diastole_cache_path(cache_dir: Path, item: object, cfg: object, min_diastole_seconds: float) -> Path:
    low_hz = float(getattr(cfg, "low_hz", 0.0))
    band_key = f"hi{cfg.high_hz:g}" if low_hz == 0.0 else f"lo{low_hz:g}_hi{cfg.high_hz:g}"
    key = (
        f"systole_diastole_sr{cfg.target_sample_rate}_fft{cfg.n_fft}_hop{cfg.hop_length}_"
        f"{band_key}_frames{cfg.max_frames}_minsyst{format_threshold_key(cfg.min_systole_seconds)}_"
        f"mindias{format_threshold_key(min_diastole_seconds)}"
    )
    if cfg.systole_threshold is not None or cfg.systole_margin_ms != 0:
        threshold_key = "argmax" if cfg.systole_threshold is None else f"thr{format_threshold_key(cfg.systole_threshold)}"
        margin_key = f"margin{format_threshold_key(cfg.systole_margin_ms)}ms"
        key = f"{key}_{threshold_key}_{margin_key}"
    return cache_dir / key / f"{item.recording_id}.npz"


def extract_phase_audio(
    audio: np.ndarray,
    sample_rate: int,
    segments: pd.DataFrame,
    labels: set[int],
    margin_ms: float,
) -> tuple[np.ndarray, dict[int, int], dict[int, float]]:
    chunks: list[np.ndarray] = []
    sample_counts = {label: 0 for label in labels}
    segment_counts = {label: 0 for label in labels}
    n_samples = len(audio)
    margin_seconds = max(0.0, margin_ms) / 1000.0
    if segments.empty:
        return np.array([], dtype=np.float32), segment_counts, {label: 0.0 for label in labels}
    for row in segments.loc[segments["label"].isin(labels)].sort_values("start_time").itertuples(index=False):
        label = int(row.label)
        start_time = float(row.start_time) - margin_seconds
        end_time = float(row.end_time) + margin_seconds
        start = max(0, min(n_samples, int(round(start_time * sample_rate))))
        end = max(0, min(n_samples, int(round(end_time * sample_rate))))
        if end > start:
            chunk = audio[start:end]
            chunks.append(chunk)
            segment_counts[label] += 1
            sample_counts[label] += int(len(chunk))
    seconds = {
        label: (sample_counts[label] / float(sample_rate) if sample_rate else 0.0)
        for label in labels
    }
    if not chunks:
        return np.array([], dtype=np.float32), segment_counts, seconds
    return np.concatenate(chunks).astype(np.float32), segment_counts, seconds


def prepare_systole_diastole_spectrograms(
    items: list[object],
    stft_cfg: object,
    min_diastole_seconds: float,
    cache_dir: Path,
    overwrite_cache: bool,
    predicted_tsv_dir: Path,
    overwrite_predictions: bool,
    tcn_model: nn.Module,
    tcn_normalizer: object,
    tcn_cfg: object,
    tcn_device: torch.device,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    specs: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, object]] = []
    iterator = cnn.tqdm(items, desc="Preparing systole+diastole STFTs", unit="rec", disable=not show_progress)
    for item in iterator:
        path = systole_diastole_cache_path(cache_dir, item, stft_cfg, min_diastole_seconds)
        if path.exists() and not overwrite_cache:
            with np.load(path) as data:
                spec = data["spec"].astype(np.float32)
                systole_seconds = float(data["systole_seconds"])
                systole_segments = int(data["systole_segments"])
                diastole_seconds = float(data["diastole_seconds"])
                diastole_segments = int(data["diastole_segments"])
                phase_seconds = float(data["phase_seconds"])
                phase_segments = int(data["phase_segments"])
        else:
            sample_rate, audio = cnn.read_audio(item.wav_path)
            segments = cnn.get_segments(
                item.wav_path,
                predicted_tsv_dir,
                overwrite_predictions,
                stft_cfg,
                tcn_model,
                tcn_normalizer,
                tcn_cfg,
                tcn_device,
            )
            phase_audio, segment_counts, seconds = extract_phase_audio(
                audio,
                sample_rate,
                segments,
                {LABEL_SYSTOLE, LABEL_DIASTOLE},
                stft_cfg.systole_margin_ms,
            )
            systole_seconds = float(seconds[LABEL_SYSTOLE])
            diastole_seconds = float(seconds[LABEL_DIASTOLE])
            systole_segments = int(segment_counts[LABEL_SYSTOLE])
            diastole_segments = int(segment_counts[LABEL_DIASTOLE])
            phase_seconds = len(phase_audio) / float(sample_rate) if sample_rate else 0.0
            phase_segments = systole_segments + diastole_segments
            if systole_seconds < stft_cfg.min_systole_seconds or diastole_seconds < min_diastole_seconds:
                spec = np.zeros((0, 0), dtype=np.float32)
            else:
                spec = cnn.systole_stft(phase_audio, sample_rate, stft_cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                spec=spec.astype(np.float32),
                systole_seconds=np.asarray(systole_seconds, dtype=np.float32),
                systole_segments=np.asarray(systole_segments, dtype=np.int32),
                diastole_seconds=np.asarray(diastole_seconds, dtype=np.float32),
                diastole_segments=np.asarray(diastole_segments, dtype=np.int32),
                phase_seconds=np.asarray(phase_seconds, dtype=np.float32),
                phase_segments=np.asarray(phase_segments, dtype=np.int32),
            )
        if spec.size == 0:
            continue
        specs.append(spec)
        labels.append(1 if item.murmur == "Present" else 0)
        rows.append(
            {
                "recording_id": item.recording_id,
                "patient_id": item.patient_id,
                "location": item.location,
                "murmur": item.murmur,
                "target": 1 if item.murmur == "Present" else 0,
                "systole_seconds": systole_seconds,
                "systole_segments": systole_segments,
                "diastole_seconds": diastole_seconds,
                "diastole_segments": diastole_segments,
                "phase_seconds": phase_seconds,
                "phase_segments": phase_segments,
            }
        )
    if not specs:
        raise RuntimeError("No systole+diastole spectrograms were prepared.")
    return np.stack(specs), np.asarray(labels, dtype=np.float32), pd.DataFrame(rows)


def prepare_fold_spectrograms(
    args: argparse.Namespace,
    fold: int,
    fold_dir: Path,
    tcn_checkpoint: Path,
    all_items: list[object],
    all_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, object]:
    predicted_tsv_dir = fold_dir / "predicted_tsvs"
    cache_dir = fold_dir / "spectrogram_cache"
    tcn_device = torch.device("cpu")
    tcn_model, tcn_normalizer, tcn_cfg, _checkpoint = nested.tcn.load_checkpoint_for_eval(tcn_checkpoint, tcn_device)
    stft_cfg = cnn.StftConfig(
        target_sample_rate=args.target_sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        high_hz=args.high_hz,
        max_frames=args.max_frames,
        min_systole_seconds=args.min_systole_seconds,
        systole_threshold=args.systole_threshold,
        systole_margin_ms=args.systole_margin_ms,
        low_hz=args.low_hz,
    )
    specs, _patient_labels, meta = prepare_systole_diastole_spectrograms(
        all_items,
        stft_cfg,
        float(args.min_diastole_seconds),
        cache_dir,
        args.overwrite_cache,
        predicted_tsv_dir,
        False,
        tcn_model,
        tcn_normalizer,
        tcn_cfg,
        tcn_device,
        args.progress,
    )
    patient_context_columns = [
        "patient_id",
        "murmur_locations",
        "most_audible_location",
        "systolic_murmur_grading",
        "outcome",
    ]
    context = all_meta[[col for col in patient_context_columns if col in all_meta.columns]].drop_duplicates("patient_id")
    meta = meta.merge(context, on="patient_id", how="left")
    meta, labels = apply_audio_level_targets(meta)
    meta.to_csv(fold_dir / "recording_metadata_audio_targets.csv", index=False)
    return specs, labels, meta, stft_cfg


def write_recording_threshold_tables(
    recording_oof: pd.DataFrame,
    output_dir: Path,
    prob_column: str,
    label: str,
) -> pd.DataFrame:
    thresholds = cnn.threshold_grid()
    rows: list[pd.DataFrame] = []
    table_dir = output_dir / "threshold_metrics_by_fold"
    table_dir.mkdir(parents=True, exist_ok=True)
    for fold in sorted(recording_oof["fold"].unique()):
        fold_df = recording_oof[recording_oof["fold"] == fold]
        table = cnn.threshold_sweep(
            fold_df["target"].to_numpy(dtype=int),
            fold_df[prob_column].to_numpy(dtype=float),
            thresholds,
        )
        table.insert(0, "fold", int(fold))
        table.insert(1, "probability", label)
        table.insert(2, "recordings", int(len(fold_df)))
        table.insert(3, "positive_audio", int((fold_df["target"] == 1).sum()))
        table.insert(4, "negative_audio", int((fold_df["target"] == 0).sum()))
        table.to_csv(table_dir / f"fold_{int(fold)}_{label}_recording_threshold_metrics.csv", index=False)
        rows.append(table)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_dir / f"threshold_metrics_by_fold_{label}.csv", index=False)
    return combined


def write_threshold_report(path: Path, raw_tables: pd.DataFrame, calibrated_tables: pd.DataFrame) -> None:
    lines = [
        "# Metricas por fold em varios thresholds",
        "",
        "As tabelas abaixo avaliam predicoes audio-level out-of-fold.",
        "",
    ]
    metric_columns = [
        "threshold",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    for fold in sorted(raw_tables["fold"].unique()):
        raw_fold = raw_tables[raw_tables["fold"] == fold]
        calibrated_fold = calibrated_tables[calibrated_tables["fold"] == fold]
        lines.extend(
            [
                f"## Fold {int(fold)}",
                "",
                f"Gravacoes: `{int(raw_fold['recordings'].iloc[0])}` | "
                f"Positivas: `{int(raw_fold['positive_audio'].iloc[0])}` | "
                f"Negativas: `{int(raw_fold['negative_audio'].iloc[0])}`",
                "",
                "### Probabilidade bruta",
                "",
                raw_fold[metric_columns].to_markdown(index=False),
                "",
                "### Probabilidade calibrada",
                "",
                calibrated_fold[metric_columns].to_markdown(index=False),
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    recording_oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    raw_metrics_05: dict[str, float | int],
    calibrated_metrics_05: dict[str, float | int],
    decision_metrics: dict[str, float | int],
) -> None:
    lines = [
        "# Grupo I - Nested TCN + CNN audio-level murmur",
        "",
        "## Objetivo",
        "",
        "Prever se uma gravacao/local especifico contem sopro, mantendo separacao por paciente entre treino e validacao.",
        "",
        "## Definicao do alvo",
        "",
        "- `target = 1`: `Murmur = Present` e o local da gravacao aparece em `Murmur locations`.",
        "- `target = 0`: paciente `Absent` ou gravacao em local nao listado em `Murmur locations`.",
        "- Pacientes `Unknown` nao entram no experimento.",
        "",
        "## Dados",
        "",
        "- Entrada da CNN: STFT dos trechos preditos como `systole` e `diastole`, concatenados no tempo.",
        f"- Locais: `{', '.join(args.locations)}`",
        f"- Gravacoes OOF: {len(recording_oof)}",
        f"- Pacientes OOF: {recording_oof['patient_id'].nunique()}",
        f"- Audios positivos: {int((recording_oof['target'] == 1).sum())}",
        f"- Audios negativos: {int((recording_oof['target'] == 0).sum())}",
        f"- Folds por paciente: {args.folds}",
        f"- TCN target mode: {args.tcn_target_mode}",
        f"- CNN inner validation size: {args.cnn_inner_val_size}",
        f"- STFT low Hz: {args.low_hz}",
        f"- STFT high Hz: {args.high_hz}",
        f"- Min systole seconds: {args.min_systole_seconds}",
        f"- Min diastole seconds: {args.min_diastole_seconds}",
        f"- Location embedding: {args.location_embedding}",
        f"- LSTM after location embedding: {args.location_lstm_after_embedding}",
        f"- Location LSTM layers: {args.location_lstm_layers}",
        f"- Location LSTM dropout: {args.location_lstm_dropout}",
        f"- Decision threshold: {args.decision_threshold}",
        f"- Balanced sampler: {args.balanced_sampler}",
        f"- Sampler positive fraction: {args.sampler_positive_fraction}",
        f"- CNN loss: {args.cnn_loss}",
        f"- CNN focal gamma: {args.cnn_focal_gamma}",
        f"- Loss positive weight mode: {args.loss_pos_weight_mode}",
        f"- SMOTE: {args.use_smote}",
        f"- SMOTE target positive/negative ratio: {args.smote_target_ratio}",
        f"- SMOTE k-neighbors: {args.smote_k_neighbors}",
        "",
        "## Metricas audio-level OOF",
        "",
        "### Probabilidade bruta @0.5",
        "",
        pd.DataFrame([raw_metrics_05]).to_markdown(index=False),
        "",
        "### Probabilidade calibrada @0.5",
        "",
        pd.DataFrame([calibrated_metrics_05]).to_markdown(index=False),
        "",
        f"### Probabilidade calibrada @{args.decision_threshold:g}",
        "",
        pd.DataFrame([decision_metrics]).to_markdown(index=False),
        "",
        "## Metricas por fold",
        "",
        fold_metrics.to_markdown(index=False),
        "",
        "## Arquivos gerados",
        "",
        "- `recording_oof_predictions.csv`",
        "- `fold_metrics.csv`",
        "- `training_history.csv`",
        "- `threshold_metrics_by_fold.md`",
        "- `fold_*/recording_metadata_audio_targets.csv`",
        "- `fold_*/tcn/best_model.pt`",
        "- `fold_*/cnn/fold_*_best_model.pt`",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.cnn_inner_val_size < 0.5:
        raise ValueError("--cnn-inner-val-size must be greater than 0.0 and less than 0.5.")
    if args.systole_threshold is not None and not 0.0 <= args.systole_threshold <= 1.0:
        raise ValueError("--systole-threshold must be between 0.0 and 1.0.")
    if args.systole_threshold is not None:
        raise ValueError("--systole-threshold is not supported for systole+diastole extraction; use argmax.")
    if args.systole_margin_ms < 0.0:
        raise ValueError("--systole-margin-ms must be non-negative.")
    if args.min_systole_seconds < 0.0:
        raise ValueError("--min-systole-seconds must be non-negative.")
    if args.min_diastole_seconds < 0.0:
        raise ValueError("--min-diastole-seconds must be non-negative.")
    if args.low_hz < 0.0:
        raise ValueError("--low-hz must be non-negative.")
    if args.high_hz <= args.low_hz:
        raise ValueError("--high-hz must be greater than --low-hz.")
    if args.tcn_boundary_ignore_ms < 0.0:
        raise ValueError("--tcn-boundary-ignore-ms must be non-negative.")
    if args.tcn_systole_weight_multiplier <= 0.0:
        raise ValueError("--tcn-systole-weight-multiplier must be greater than 0.")
    if args.tcn_target_mode != "cardiac-phase":
        raise ValueError("--tcn-target-mode must be cardiac-phase to extract both systole and diastole.")
    if not 0.0 <= args.decision_threshold <= 1.0:
        raise ValueError("--decision-threshold must be between 0.0 and 1.0.")
    if args.weak_murmur_weight <= 0.0:
        raise ValueError("--weak-murmur-weight must be greater than 0.")
    if args.moderate_murmur_weight <= 0.0:
        raise ValueError("--moderate-murmur-weight must be greater than 0.")
    if not 0.0 < args.sampler_positive_fraction < 1.0:
        raise ValueError("--sampler-positive-fraction must be greater than 0.0 and less than 1.0.")
    if args.smote_target_ratio <= 0.0:
        raise ValueError("--smote-target-ratio must be greater than 0.")
    if args.smote_k_neighbors <= 0:
        raise ValueError("--smote-k-neighbors must be greater than 0.")
    if args.cnn_focal_gamma < 0.0:
        raise ValueError("--cnn-focal-gamma must be non-negative.")
    if args.location_lstm_after_embedding and not args.location_embedding:
        raise ValueError("--location-lstm-after-embedding requires --location-embedding.")
    if args.location_lstm_layers <= 0:
        raise ValueError("--location-lstm-layers must be greater than 0.")
    if not 0.0 <= args.location_lstm_dropout < 1.0:
        raise ValueError("--location-lstm-dropout must be greater than or equal to 0.0 and less than 1.0.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    load_project_modules()
    cnn.set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_items = cnn.build_items(args.dataset_dir.resolve(), args.locations, None)
    all_meta = pd.DataFrame(
        {
            "recording_id": [item.recording_id for item in all_items],
            "patient_id": [item.patient_id for item in all_items],
            "location": [item.location for item in all_items],
            "murmur": [item.murmur for item in all_items],
            "target": [1 if item.murmur == "Present" else 0 for item in all_items],
        }
    )
    all_meta = all_meta.merge(nested.load_patient_context(args.dataset_dir.resolve()), on="patient_id", how="left")
    all_meta = nested.select_patient_subset(all_meta, args.max_patients, args.seed)
    allowed_patients = set(all_meta["patient_id"].astype(str))
    all_items = [item for item in all_items if item.patient_id in allowed_patients]

    patient_table = all_meta.drop_duplicates("patient_id")[["patient_id", "target"]].copy()
    patient_ids = patient_table["patient_id"].astype(str).to_numpy()
    y_patient = patient_table["target"].to_numpy(dtype=int)
    fold_patient_ids = cnn.stratified_patient_folds(patient_ids, y_patient, args.folds, args.seed)

    recording_oof_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | int]] = []
    history_rows: list[dict[str, float | int]] = []

    for fold, val_patient_ids_array in enumerate(fold_patient_ids, start=1):
        val_patient_ids = set(str(pid) for pid in val_patient_ids_array)
        train_patient_ids = set(str(pid) for pid in patient_ids if str(pid) not in val_patient_ids)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "train_patient_ids.txt").write_text("\n".join(sorted(train_patient_ids)) + "\n", encoding="utf-8")
        (fold_dir / "val_patient_ids.txt").write_text("\n".join(sorted(val_patient_ids)) + "\n", encoding="utf-8")

        print(
            f"Fold {fold}/{args.folds}: training TCN on {len(train_patient_ids)} patients; "
            f"validating audio-level CNN on {len(val_patient_ids)} patients"
        )
        tcn_checkpoint = nested.train_tcn_for_fold(args, fold_dir, train_patient_ids)
        specs, labels, meta, stft_cfg = prepare_fold_spectrograms(
            args,
            fold,
            fold_dir,
            tcn_checkpoint,
            all_items,
            all_meta,
        )

        fit_patient_ids, tune_patient_ids = split_fit_tune_patients(
            meta,
            train_patient_ids,
            args.cnn_inner_val_size,
            args.seed,
            fold,
        )
        train_idx = np.flatnonzero(meta["patient_id"].astype(str).isin(fit_patient_ids).to_numpy())
        tune_idx = np.flatnonzero(meta["patient_id"].astype(str).isin(tune_patient_ids).to_numpy())
        val_idx = np.flatnonzero(meta["patient_id"].astype(str).isin(val_patient_ids).to_numpy())
        if len(train_idx) == 0 or len(tune_idx) == 0 or len(val_idx) == 0:
            raise RuntimeError(f"Fold {fold} has empty fit, tune, or validation recordings after systole+diastole extraction.")

        (fold_dir / "cnn_fit_patient_ids.txt").write_text("\n".join(sorted(fit_patient_ids)) + "\n", encoding="utf-8")
        (fold_dir / "cnn_tune_patient_ids.txt").write_text("\n".join(sorted(tune_patient_ids)) + "\n", encoding="utf-8")

        model_cfg = make_cnn_model_config(args, specs)
        device = cnn.choose_device(args.cnn_device)
        val_probs, threshold, fold_metrics, history, val_recordings = train_one_fold_audio(
            fold,
            specs,
            labels,
            meta,
            train_idx,
            tune_idx,
            val_idx,
            model_cfg,
            args,
            device,
            fold_dir / "cnn",
        )
        fold_config = {
            "stft_config": asdict(stft_cfg),
            "model_config": asdict(model_cfg),
            "target": "audio_murmur_from_murmur_locations",
            "input_phases": ["systole", "diastole"],
            "min_diastole_seconds": float(args.min_diastole_seconds),
            "tcn_train_patients": sorted(train_patient_ids),
            "cnn_fit_patients": sorted(fit_patient_ids),
            "cnn_tune_patients": sorted(tune_patient_ids),
            "outer_val_patients": sorted(val_patient_ids),
            "tcn_checkpoint": str(tcn_checkpoint),
            "location_embedding": bool(args.location_embedding),
            "location_lstm_after_embedding": bool(args.location_lstm_after_embedding),
            "location_lstm_layers": int(args.location_lstm_layers),
            "location_lstm_dropout": float(args.location_lstm_dropout),
            "location_to_id": LOCATION_TO_ID,
            "cnn_loss": str(args.cnn_loss),
            "cnn_focal_gamma": float(args.cnn_focal_gamma),
            "smote": bool(args.use_smote),
            "smote_target_ratio": float(args.smote_target_ratio),
            "smote_k_neighbors": int(args.smote_k_neighbors),
            "smote_synthetic_positive": int(fold_metrics["smote_synthetic_positive"]),
            "train_positive_original": int(fold_metrics["train_positive_original"]),
            "train_negative_original": int(fold_metrics["train_negative_original"]),
            "train_positive_after_smote": int(fold_metrics["train_positive_after_smote"]),
            "train_negative_after_smote": int(fold_metrics["train_negative_after_smote"]),
            "threshold": float(threshold),
            "validation_recordings": int(len(val_recordings)),
            "validation_positive_audio": int((val_recordings["target"] == 1).sum()),
        }
        (fold_dir / "fold_config.json").write_text(json.dumps(fold_config, indent=2), encoding="utf-8")

        recording_oof_rows.append(val_recordings)
        fold_rows.append(fold_metrics)
        history_rows.extend(history)
        print(
            f"Fold {fold}/{args.folds}: "
            f"audio AUPRC={fold_metrics['auprc']:.3f} AUROC={fold_metrics['auroc']:.3f} "
            f"BA={fold_metrics['balanced_accuracy']:.3f}"
        )
        if not args.keep_fold_training_data:
            cleanup_fold_training_data(fold_dir, args.progress)

    recording_oof = pd.concat(recording_oof_rows, ignore_index=True)
    recording_oof["pred_murmur_raw_threshold_05"] = (
        recording_oof["prob_murmur_raw"].to_numpy(dtype=float) >= 0.5
    ).astype(int)
    recording_oof["pred_murmur_calibrated_threshold_05"] = (
        recording_oof["prob_murmur_calibrated"].to_numpy(dtype=float) >= 0.5
    ).astype(int)
    decision_key = format_threshold_key(args.decision_threshold)
    recording_oof[f"pred_murmur_calibrated_threshold_{decision_key}"] = (
        recording_oof["prob_murmur_calibrated"].to_numpy(dtype=float) >= args.decision_threshold
    ).astype(int)
    recording_oof.to_csv(output_dir / "recording_oof_predictions.csv", index=False)

    fold_metrics = pd.DataFrame(fold_rows)
    history = pd.DataFrame(history_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)

    raw_tables = write_recording_threshold_tables(recording_oof, output_dir, "prob_murmur_raw", "raw")
    calibrated_tables = write_recording_threshold_tables(
        recording_oof,
        output_dir,
        "prob_murmur_calibrated",
        "calibrated",
    )
    write_threshold_report(output_dir / "threshold_metrics_by_fold.md", raw_tables, calibrated_tables)

    y_true = recording_oof["target"].to_numpy(dtype=int)
    raw_prob = recording_oof["prob_murmur_raw"].to_numpy(dtype=float)
    calibrated_prob = recording_oof["prob_murmur_calibrated"].to_numpy(dtype=float)
    raw_metrics_05 = cnn.metrics(y_true, raw_prob, 0.5)
    calibrated_metrics_05 = cnn.metrics(y_true, calibrated_prob, 0.5)
    calibrated_metrics_05["brier_score"] = cnn.brier_score(y_true, calibrated_prob)
    calibrated_metrics_05["raw_brier_score"] = cnn.brier_score(y_true, raw_prob)
    decision_metrics = cnn.metrics(y_true, calibrated_prob, args.decision_threshold)

    cnn.plot_pr(y_true, raw_prob, output_dir / "precision_recall_oof_raw.png")
    cnn.plot_pr(y_true, calibrated_prob, output_dir / "precision_recall_oof_calibrated.png")

    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    write_summary(
        output_dir,
        args,
        recording_oof,
        fold_metrics,
        raw_metrics_05,
        calibrated_metrics_05,
        decision_metrics,
    )
    print(f"Done. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
