from __future__ import annotations


from dataclasses import asdict, dataclass
import numpy as np


LABEL_NAMES = {
    0: "other",
    1: "s1",
    2: "systole",
    3: "s2",
    4: "diastole",
}


BINARY_SYSTOLE_LABEL_NAMES = {
    0: "non_systole",
    1: "systole",
}


IGNORE_INDEX = -100


TARGET_MODES = ("cardiac-phase", "systole-binary")


OTHER_MODES = ("keep", "ignore")


SYSTOLE_LABEL = 2


def label_names_for_target_mode(target_mode: str) -> dict[int, str]:
    if target_mode == "systole-binary":
        return BINARY_SYSTOLE_LABEL_NAMES
    return LABEL_NAMES


def label_names_for_cfg(cfg: "FeatureConfig") -> dict[int, str]:
    return label_names_for_target_mode(cfg.target_mode)


def prediction_output_label(model_label: int, target_mode: str) -> int:
    if target_mode == "systole-binary":
        return SYSTOLE_LABEL if model_label == 1 else 0
    return model_label


def systole_probability_index(target_mode: str) -> int:
    return 1 if target_mode == "systole-binary" else SYSTOLE_LABEL


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        if int(true_label) in label_to_index and int(pred_label) in label_to_index:
            matrix[label_to_index[int(true_label)], label_to_index[int(pred_label)]] += 1
    return matrix


@dataclass(frozen=True)
class FeatureConfig:
    frame_ms: float
    hop_ms: float
    n_mels: int
    low_hz: float
    high_hz: float
    add_deltas: bool
    label_mode: str = "overlap"
    boundary_ignore_ms: float = 0.0
    target_mode: str = "cardiac-phase"
    other_mode: str = "keep"


@dataclass(frozen=True)
class RecordingItem:
    recording_id: str
    patient_id: str
    location: str
    wav_path: str
    tsv_path: str
    murmur: str
    outcome: str


@dataclass
class Normalizer:
    mean: list[float]
    std: list[float]

    def apply(self, features: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return (features - mean) / std
