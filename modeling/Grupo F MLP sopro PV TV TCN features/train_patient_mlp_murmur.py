# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.4",
#   "tabulate>=0.9",
#   "torch>=2.2",
# ]
# ///
"""Train a patient-level MLP murmur classifier on Grupo B v2 TCN PV+TV features.

Input features come from:

    feature extraction/Grupo B v2 features relativas por local com TCN predito/outputs_pv_tv/patient_relative_phase_features.csv

The dataset is already patient-level and contains only PV/TV recordings
aggregated with mean/max features. The target is:

    Murmur Present vs Absent

Run from the repository root:

    uv run "modeling/Grupo F MLP sopro PV TV TCN features/train_patient_mlp_murmur.py"
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "feature extraction"
    / "Grupo B v2 features relativas por local com TCN predito"
    / "outputs_pv_tv"
    / "patient_relative_phase_features.csv"
)


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_layers: tuple[int, ...]
    dropout: float


class PatientMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for hidden_dim in config.hidden_layers:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a deep-learning MLP on patient-level PV/TV TCN Grupo B v2 features."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--hidden-layers", type=str, default="256,128,64")
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--feature-prefixes",
        nargs="+",
        default=["mean_", "max_"],
        help="Feature column prefixes to include. Default uses patient-level mean/max aggregated features.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "mps":
        print("MPS requested but not available; using CPU.")
    return torch.device("cpu")


def parse_hidden_layers(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("--hidden-layers must contain at least one integer")
    return values


def load_feature_table(path: Path, feature_prefixes: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    df = df.loc[df["murmur"].isin(["Present", "Absent"])].copy()
    y = (df["murmur"] == "Present").astype(np.float32).to_numpy()
    feature_columns = [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in feature_prefixes)
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not feature_columns:
        raise RuntimeError(f"No numeric feature columns found with prefixes: {feature_prefixes}")
    return df, y, feature_columns


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        score = sensitivity + specificity - 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    dummy_y = np.zeros(len(x), dtype=np.float32)
    loader = make_loader(x, dummy_y, batch_size=batch_size, shuffle=False)
    for batch_x, _batch_y in loader:
        logits = model(batch_x.to(device))
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def train_fold(
    fold: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    model_config: ModelConfig,
    device: torch.device,
    output_dir: Path,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> tuple[np.ndarray, dict[str, float | int], list[dict[str, float | int]]]:
    model = PatientMLP(model_config).to(device)
    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_auprc = -math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_prob = predict_proba(model, x_val, device, args.batch_size)
        val_auprc = float(average_precision_score(y_val, val_prob))
        val_auroc = float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else float("nan")
        val_loss = float(loss_fn(torch.tensor(np.log(np.clip(val_prob, 1e-6, 1 - 1e-6) / np.clip(1 - val_prob, 1e-6, 1)), dtype=torch.float32, device=device), torch.tensor(y_val, dtype=torch.float32, device=device)).detach().cpu())
        history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "val_loss": val_loss,
                "val_auprc": val_auprc,
                "val_auroc": val_auroc,
            }
        )

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold} did not produce a model state")
    model.load_state_dict(best_state)
    val_prob = predict_proba(model, x_val, device, args.batch_size)
    train_prob = predict_proba(model, x_train, device, args.batch_size)
    threshold = choose_threshold_youden(y_train.astype(int), train_prob)
    metrics = binary_metrics(y_val.astype(int), val_prob, threshold)
    metrics["fold"] = fold
    metrics["epochs_trained"] = len(history)
    metrics["best_val_auprc"] = float(best_auprc)

    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": asdict(model_config),
            "threshold": threshold,
            "feature_columns": feature_columns,
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
        },
        output_dir / f"fold_{fold}_best_model.pt",
    )
    return val_prob, metrics, history


def train_final_model(
    x: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    model_config: ModelConfig,
    device: torch.device,
    epochs: int,
) -> tuple[PatientMLP, StandardScaler, list[dict[str, float | int]]]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)
    model = PatientMLP(model_config).to(device)
    neg = float((y == 0).sum())
    pos = float((y == 1).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(x_scaled, y, args.batch_size, shuffle=True)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"fold": 0, "epoch": epoch, "train_loss": float(np.mean(losses))})
    return model, scaler, history


def save_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    precision, recall, _thresholds = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall / sensitivity")
    ax.set_ylabel("Precision")
    ax.set_title("Out-of-fold precision-recall curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    df: pd.DataFrame,
    feature_columns: list[str],
    fold_metrics: pd.DataFrame,
    oof_metrics_05: dict[str, float | int],
    oof_metrics_tuned: dict[str, float | int],
) -> None:
    lines = [
        "# Grupo F - MLP para sopro com features PV+TV TCN",
        "",
        "## Objetivo",
        "",
        "Treinar um modelo deep learning tabular para prever `Murmur = Present` vs `Absent` usando features Grupo B v2 extraidas de segmentacao TCN predita, filtradas para `PV` e `TV` e agregadas por paciente com `mean`/`max`.",
        "",
        "## Dados",
        "",
        f"- Input CSV: `{args.input_csv}`",
        f"- Pacientes: {len(df)}",
        f"- Present: {int((df['murmur'] == 'Present').sum())}",
        f"- Absent: {int((df['murmur'] == 'Absent').sum())}",
        f"- Features usadas: {len(feature_columns)}",
        f"- Prefixos de features: `{', '.join(args.feature_prefixes)}`",
        "",
        "## Modelo",
        "",
        f"- Arquitetura: MLP `{args.hidden_layers}`",
        f"- Dropout: {args.dropout}",
        f"- Loss: BCEWithLogitsLoss com `pos_weight` por fold",
        f"- Folds: {args.folds}",
        f"- Epochs max: {args.epochs}",
        f"- Patience: {args.patience}",
        "",
        "## Metricas out-of-fold",
        "",
        "### Threshold 0.5",
        "",
        pd.DataFrame([oof_metrics_05]).to_markdown(index=False),
        "",
        "### Threshold ajustado por Youden no treino de cada fold",
        "",
        pd.DataFrame([oof_metrics_tuned]).to_markdown(index=False),
        "",
        "## Metricas por fold",
        "",
        fold_metrics.to_markdown(index=False),
        "",
        "## Arquivos gerados",
        "",
        "- `features_used.json`: lista das features usadas.",
        "- `feature_matrix.csv`: matriz final `patient_id` + target + features usada pelo MLP.",
        "- `oof_predictions.csv`: predicoes out-of-fold por paciente.",
        "- `fold_metrics.csv`: metricas por fold.",
        "- `training_history.csv`: historico de treino.",
        "- `precision_recall_oof.png`: curva precision-recall out-of-fold.",
        "- `fold_*_best_model.pt`: checkpoints dos modelos por fold.",
        "- `final_model.pt`: modelo final treinado em todos os pacientes, com scaler, threshold e lista de features.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    df, y, feature_columns = load_feature_table(args.input_csv.resolve(), args.feature_prefixes)
    x_raw = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    model_config = ModelConfig(input_dim=len(feature_columns), hidden_layers=hidden_layers, dropout=args.dropout)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_prob = np.zeros(len(df), dtype=np.float32)
    oof_threshold = np.zeros(len(df), dtype=np.float32)
    fold_metric_rows: list[dict[str, float | int]] = []
    history_rows: list[dict[str, float | int]] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_raw, y), start=1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_raw[train_idx]).astype(np.float32)
        x_val = scaler.transform(x_raw[val_idx]).astype(np.float32)
        y_train = y[train_idx]
        y_val = y[val_idx]

        val_prob, metrics, history = train_fold(
            fold,
            x_train,
            y_train,
            x_val,
            y_val,
            args,
            model_config,
            device,
            output_dir,
            scaler,
            feature_columns,
        )
        oof_prob[val_idx] = val_prob
        oof_threshold[val_idx] = float(metrics["threshold"])
        fold_metric_rows.append(metrics)
        history_rows.extend(history)
        print(
            f"Fold {fold}/{args.folds}: "
            f"AUPRC={metrics['auprc']:.3f} AUROC={metrics['auroc']:.3f} "
            f"BA={metrics['balanced_accuracy']:.3f} sens={metrics['sensitivity']:.3f} spec={metrics['specificity']:.3f}"
        )

    y_int = y.astype(int)
    oof_pred_tuned = (oof_prob >= oof_threshold).astype(int)
    oof_metrics_05 = binary_metrics(y_int, oof_prob, 0.5)
    tn, fp, fn, tp = confusion_matrix(y_int, oof_pred_tuned, labels=[0, 1]).ravel()
    oof_metrics_tuned = {
        "threshold": "per_fold_youden",
        "auroc": float(roc_auc_score(y_int, oof_prob)),
        "auprc": float(average_precision_score(y_int, oof_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_int, oof_pred_tuned)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y_int, oof_pred_tuned, zero_division=0)),
        "f1": float(f1_score(y_int, oof_pred_tuned, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    oof_default_threshold = choose_threshold_youden(y_int, oof_prob)
    final_epochs = int(np.median([int(row["epochs_trained"]) for row in fold_metric_rows]))
    final_model, final_scaler, final_history = train_final_model(
        x_raw,
        y,
        args,
        model_config,
        device,
        epochs=max(1, final_epochs),
    )
    history_rows.extend(final_history)

    fold_metrics = pd.DataFrame(fold_metric_rows)
    history_df = pd.DataFrame(history_rows)
    oof_df = df[["patient_id", "murmur", "outcome", "age", "sex", "campaign", "location_count", "recording_count"]].copy()
    oof_df["target"] = y_int
    oof_df["prob_present"] = oof_prob
    oof_df["threshold"] = oof_threshold
    oof_df["pred_present_threshold_05"] = (oof_prob >= 0.5).astype(int)
    oof_df["pred_present_threshold_tuned"] = oof_pred_tuned
    feature_matrix = df[["patient_id", "murmur"] + feature_columns].copy()

    save_pr_curve(y_int, oof_prob, output_dir / "precision_recall_oof.png")
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    feature_matrix.to_csv(output_dir / "feature_matrix.csv", index=False)
    (output_dir / "features_used.json").write_text(
        json.dumps({"feature_columns": feature_columns}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
            "model_config": asdict(model_config),
            "feature_columns": feature_columns,
            "scaler_mean": final_scaler.mean_.astype(float).tolist(),
            "scaler_scale": final_scaler.scale_.astype(float).tolist(),
            "threshold_oof_youden": float(oof_default_threshold),
            "threshold_default": 0.5,
            "epochs_trained": final_epochs,
            "input_csv": str(args.input_csv.resolve()),
            "target": "murmur_present",
        },
        output_dir / "final_model.pt",
    )
    write_summary(
        output_dir / "summary.md",
        args,
        df,
        feature_columns,
        fold_metrics,
        oof_metrics_05,
        oof_metrics_tuned,
    )

    print(f"OOF AUPRC={oof_metrics_05['auprc']:.3f} AUROC={oof_metrics_05['auroc']:.3f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
