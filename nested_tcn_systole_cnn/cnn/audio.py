from __future__ import annotations


import math
import random
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly, stft


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
        print("MPS requested but unavailable; using CPU.")
    return torch.device("cpu")


def parse_recording_id(path: Path) -> tuple[str, str]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected recording name: {path.name}")
    return parts[0], parts[1]


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    original_dtype = audio.dtype
    audio = audio.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        audio = audio / float(max(abs(info.min), info.max))
    else:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
    audio = audio - float(np.mean(audio)) if audio.size else audio
    return int(sample_rate), audio.astype(np.float32)


def resample_audio(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return audio.astype(np.float32)
    gcd = math.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)
