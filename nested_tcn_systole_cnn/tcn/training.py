from __future__ import annotations


import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


from .config import IGNORE_INDEX, confusion_matrix
from .dataset import progress_iter
from .postprocess import postprocess_prediction


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float,
    show_progress: bool,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_frames = 0

    iterator = progress_iter(
        loader,
        show_progress,
        desc=f"Epoch {epoch:03d} train",
        unit="batch",
        leave=False,
    )
    for batch in iterator:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        valid_frames = int((y != IGNORE_INDEX).sum().item())
        total_loss += float(loss.item()) * valid_frames
        total_frames += valid_frames
        running_loss = total_loss / max(total_frames, 1)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{running_loss:.4f}", frames=total_frames)

    return total_loss / max(total_frames, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    label_names: dict[int, str],
    show_progress: bool = True,
    desc: str = "Evaluating",
    postprocess: bool = True,
    median_filter_frames: int = 5,
    min_segment_frames: int = 3,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_frames = 0
    label_ids = list(label_names.keys())
    confusion = np.zeros((len(label_names), len(label_names)), dtype=np.int64)

    iterator = progress_iter(loader, show_progress, desc=desc, unit="batch", leave=False)
    for batch in iterator:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        pred = logits.argmax(dim=1)
        mask = y != IGNORE_INDEX

        valid_frames = int(mask.sum().item())
        total_loss += float(loss.item()) * valid_frames
        total_frames += valid_frames

        y_cpu = y.detach().cpu().numpy()
        pred_cpu = pred.detach().cpu().numpy()
        lengths = batch["lengths"].detach().cpu().numpy()
        for sample_index, length in enumerate(lengths):
            y_np = y_cpu[sample_index, :length]
            pred_np = pred_cpu[sample_index, :length]
            if postprocess:
                pred_np = postprocess_prediction(pred_np, median_filter_frames, min_segment_frames)
            confusion += confusion_matrix(y_np, pred_np, labels=label_ids)
        if show_progress and hasattr(iterator, "set_postfix"):
            running_loss = total_loss / max(total_frames, 1)
            iterator.set_postfix(loss=f"{running_loss:.4f}", frames=total_frames)

    metrics = metrics_from_confusion(confusion, label_names)
    metrics["loss"] = total_loss / max(total_frames, 1)
    metrics["frames"] = int(total_frames)
    metrics["confusion"] = confusion.tolist()
    return metrics


def metrics_from_confusion(confusion: np.ndarray, label_names: dict[int, str]) -> dict[str, object]:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    iou_values: list[float] = []
    weighted_f1_sum = 0.0

    for label, name in label_names.items():
        tp = float(confusion[label, label])
        fp = float(confusion[:, label].sum() - tp)
        fn = float(confusion[label, :].sum() - tp)
        support = int(confusion[label, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        f1_values.append(f1)
        recall_values.append(recall)
        iou_values.append(iou)
        weighted_f1_sum += f1 * support
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "support": support,
        }

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": weighted_f1_sum / total if total else 0.0,
        "balanced_accuracy": float(np.mean(recall_values)),
        "mean_iou": float(np.mean(iou_values)),
        "per_class": per_class,
    }
