from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.signal import resample_poly, stft


from .audio import resample_audio
from .config import LABEL_SYSTOLE, N_TEMPORAL_FEATURES, StftConfig


def systole_stft(audio: np.ndarray, sample_rate: int, cfg: StftConfig) -> np.ndarray:
    audio = resample_audio(audio, sample_rate, cfg.target_sample_rate)
    if len(audio) < cfg.n_fft:
        padded = np.zeros(cfg.n_fft, dtype=np.float32)
        padded[: len(audio)] = audio
        audio = padded
    freqs, _times, zxx = stft(
        audio,
        fs=cfg.target_sample_rate,
        window="hann",
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_length,
        nfft=cfg.n_fft,
        boundary=None,
        padded=False,
    )
    spec = np.log1p(np.abs(zxx).astype(np.float32))
    low_hz = float(getattr(cfg, "low_hz", 0.0))
    freq_mask = (freqs >= low_hz) & (freqs <= cfg.high_hz)
    spec = spec[freq_mask]
    if spec.shape[1] >= cfg.max_frames:
        start = (spec.shape[1] - cfg.max_frames) // 2
        spec = spec[:, start : start + cfg.max_frames]
    else:
        padded = np.zeros((spec.shape[0], cfg.max_frames), dtype=np.float32)
        padded[:, : spec.shape[1]] = spec
        spec = padded
    return spec.astype(np.float32)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.power(10.0, np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    freqs: np.ndarray,
    n_mels: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    mel_points = np.linspace(float(hz_to_mel(low_hz)), float(hz_to_mel(high_hz)), n_mels + 2)
    hz_points = np.asarray(mel_to_hz(mel_points), dtype=np.float32)
    filters = np.zeros((n_mels, len(freqs)), dtype=np.float32)

    for mel_idx in range(n_mels):
        left = hz_points[mel_idx]
        center = hz_points[mel_idx + 1]
        right = hz_points[mel_idx + 2]
        if center <= left or right <= center:
            continue
        rising = (freqs - left) / (center - left)
        falling = (right - freqs) / (right - center)
        filters[mel_idx] = np.maximum(0.0, np.minimum(rising, falling))

    filter_sums = filters.sum(axis=1, keepdims=True)
    return filters / np.maximum(filter_sums, 1e-8)


def phase_spectrogram(audio: np.ndarray, sample_rate: int, cfg: StftConfig) -> np.ndarray:
    if getattr(cfg, "spectrogram_type", "stft") == "stft":
        return systole_stft(audio, sample_rate, cfg)

    audio = resample_audio(audio, sample_rate, cfg.target_sample_rate)
    if len(audio) < cfg.n_fft:
        padded = np.zeros(cfg.n_fft, dtype=np.float32)
        padded[: len(audio)] = audio
        audio = padded
    freqs, _times, zxx = stft(
        audio,
        fs=cfg.target_sample_rate,
        window="hann",
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_length,
        nfft=cfg.n_fft,
        boundary=None,
        padded=False,
    )
    low_hz = float(getattr(cfg, "low_hz", 0.0))
    nyquist = float(cfg.target_sample_rate) / 2.0
    high_hz = min(float(cfg.high_hz), nyquist)
    if high_hz <= low_hz:
        raise ValueError("--high-hz must be greater than --low-hz after Nyquist clipping.")
    power = np.abs(zxx).astype(np.float32) ** 2
    filters = mel_filterbank(freqs.astype(np.float32), int(cfg.n_mels), low_hz, high_hz)
    spec = np.log1p(filters @ power).astype(np.float32)
    if spec.shape[1] >= cfg.max_frames:
        start = (spec.shape[1] - cfg.max_frames) // 2
        spec = spec[:, start : start + cfg.max_frames]
    else:
        padded = np.zeros((spec.shape[0], cfg.max_frames), dtype=np.float32)
        padded[:, : spec.shape[1]] = spec
        spec = padded
    return spec.astype(np.float32)


def _segment_audio_chunks(
    audio: np.ndarray,
    sample_rate: int,
    segments: pd.DataFrame,
    phase_labels: tuple[int, ...],
    margin_ms: float,
) -> list[np.ndarray]:
    n_samples = len(audio)
    margin_seconds = max(0.0, margin_ms) / 1000.0
    selected = segments.loc[segments["label"].isin(phase_labels)].sort_values("start_time")
    chunks: list[np.ndarray] = []
    for row in selected.itertuples(index=False):
        start_time = float(row.start_time) - margin_seconds
        end_time = float(row.end_time) + margin_seconds
        start = max(0, min(n_samples, int(round(start_time * sample_rate))))
        end = max(0, min(n_samples, int(round(end_time * sample_rate))))
        if end > start:
            chunks.append(audio[start:end].astype(np.float32))
    return chunks


def _crop_or_pad(spec: np.ndarray, max_frames: int) -> np.ndarray:
    if spec.shape[1] >= max_frames:
        start = (spec.shape[1] - max_frames) // 2
        return spec[:, start : start + max_frames].astype(np.float32)
    padded = np.zeros((spec.shape[0], max_frames), dtype=np.float32)
    padded[:, : spec.shape[1]] = spec
    return padded


def phase_spectrogram_per_segment(
    audio: np.ndarray,
    sample_rate: int,
    segments: pd.DataFrame,
    phase_labels: tuple[int, ...],
    cfg: StftConfig,
) -> np.ndarray:
    """STFT computed per phase segment and then frame-concatenated.

    Avoids spectral leakage that occurs when disjoint segments are concatenated in the
    time domain before a single STFT.
    """
    resampled = resample_audio(audio, sample_rate, cfg.target_sample_rate)
    chunks = _segment_audio_chunks(resampled, cfg.target_sample_rate, segments, phase_labels, cfg.systole_margin_ms)
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)

    spec_type = getattr(cfg, "spectrogram_type", "stft")
    nyquist = float(cfg.target_sample_rate) / 2.0
    low_hz = float(getattr(cfg, "low_hz", 0.0))
    high_hz = min(float(cfg.high_hz), nyquist)
    if spec_type == "log-mel" and high_hz <= low_hz:
        raise ValueError("--high-hz must be greater than --low-hz after Nyquist clipping.")

    filters: np.ndarray | None = None
    freq_mask: np.ndarray | None = None
    frame_blocks: list[np.ndarray] = []

    for seg_audio in chunks:
        if len(seg_audio) < cfg.n_fft:
            padded = np.zeros(cfg.n_fft, dtype=np.float32)
            padded[: len(seg_audio)] = seg_audio
            seg_audio = padded
        freqs, _times, zxx = stft(
            seg_audio,
            fs=cfg.target_sample_rate,
            window="hann",
            nperseg=cfg.n_fft,
            noverlap=cfg.n_fft - cfg.hop_length,
            nfft=cfg.n_fft,
            boundary=None,
            padded=False,
        )
        if spec_type == "log-mel":
            if filters is None:
                filters = mel_filterbank(freqs.astype(np.float32), int(cfg.n_mels), low_hz, high_hz)
            power = np.abs(zxx).astype(np.float32) ** 2
            seg_spec = np.log1p(filters @ power).astype(np.float32)
        else:
            seg_spec = np.log1p(np.abs(zxx).astype(np.float32))
            if freq_mask is None:
                freq_mask = (freqs >= low_hz) & (freqs <= cfg.high_hz)
            seg_spec = seg_spec[freq_mask]
        if seg_spec.shape[1] > 0:
            frame_blocks.append(seg_spec)

    if not frame_blocks:
        return np.zeros((0, 0), dtype=np.float32)

    spec = np.concatenate(frame_blocks, axis=1)
    return _crop_or_pad(spec, cfg.max_frames)


def compute_temporal_features(
    audio: np.ndarray,
    sample_rate: int,
    segments: pd.DataFrame,
    phase_labels: tuple[int, ...],
    cfg: StftConfig,
) -> np.ndarray:
    """Per-beat temporal-dynamics descriptor (length N_TEMPORAL_FEATURES), averaged over beats.

    Captures *how* energy evolves through the phase (fill fraction, envelope shape, spectral
    flux, sustained high band, temporal coefficient of variation) — signal the static spectral
    envelope discards, which separates soft (I/VI) murmurs from normal sounds.
    """
    resampled = resample_audio(audio, sample_rate, cfg.target_sample_rate)
    chunks = _segment_audio_chunks(resampled, cfg.target_sample_rate, segments, phase_labels, cfg.systole_margin_ms)
    low_hz = float(getattr(cfg, "low_hz", 0.0))
    blocks: list[np.ndarray] = []
    freqs_kept: np.ndarray | None = None
    for seg_audio in chunks:
        if len(seg_audio) < cfg.n_fft:
            continue
        freqs, _t, zxx = stft(
            seg_audio, fs=cfg.target_sample_rate, window="hann", nperseg=cfg.n_fft,
            noverlap=cfg.n_fft - cfg.hop_length, nfft=cfg.n_fft, boundary=None, padded=False,
        )
        spec = np.log1p(np.abs(zxx).astype(np.float32))
        if freqs_kept is None:
            mask = (freqs >= low_hz) & (freqs <= cfg.high_hz)
            freqs_kept = freqs[mask]
        spec = spec[(freqs >= low_hz) & (freqs <= cfg.high_hz)]
        if spec.shape[1] >= 2:
            blocks.append(spec)
    if not blocks or freqs_kept is None:
        return np.zeros(N_TEMPORAL_FEATURES, dtype=np.float32)

    murmur_band = (freqs_kept >= 100) & (freqs_kept <= 500)
    high_band = (freqs_kept >= 250) & (freqs_kept <= 700)
    per_beat: list[list[float]] = []
    for spec in blocks:
        t = spec.shape[1]
        e = spec.sum(axis=0)
        e_n = (e - e.min()) / (e.max() - e.min() + 1e-6)
        tt = np.linspace(0.0, 1.0, t)
        fill_fraction = float((e_n > 0.5).mean())
        peak_pos = float(np.argmax(e) / max(1, t - 1))
        slope = float(np.polyfit(tt, e_n, 1)[0]) if t >= 2 else 0.0
        curv = float(np.polyfit(tt, e_n, 2)[0]) if t >= 3 else 0.0
        diffs = np.linalg.norm(np.diff(spec, axis=1), axis=0)
        flux_mean = float(diffs.mean()) if diffs.size else 0.0
        flux_std = float(diffs.std()) if diffs.size else 0.0
        hb = spec[high_band].sum(axis=0) if high_band.any() else np.zeros(t)
        hb_fill = float((hb > np.median(hb)).mean()) if t else 0.0
        hb_mean = float(hb.mean())
        mb = spec[murmur_band]
        cv = float((mb.std(axis=1) / (mb.mean(axis=1) + 1e-6)).mean()) if murmur_band.any() else 0.0
        ac1 = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if (t >= 2 and e.std() > 1e-6) else 0.0
        per_beat.append([fill_fraction, peak_pos, slope, curv, flux_mean,
                         flux_std, hb_fill, hb_mean, cv, ac1])
    arr = np.asarray(per_beat, dtype=np.float32)
    mean_feats = arr.mean(axis=0)
    extras = np.array([float(len(blocks)), float(arr[:, 0].std())], dtype=np.float32)
    out = np.concatenate([mean_feats, extras]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def peak_window_specs(
    audio: np.ndarray,
    sample_rate: int,
    segments: pd.DataFrame,
    cfg: StftConfig,
) -> np.ndarray:
    """One STFT per fixed-length window centered on each S1 onset (cardiac-cycle peak).

    Mirrors the paper's onset-peak windowing: each ~1s window keeps a full cardiac cycle
    (S1 + systole + S2 + diastole) intact and yields many overlapping samples per recording.
    Returns a stacked array (n_windows, freq_bins, frames); empty (0,...) if no onsets.
    """
    resampled = resample_audio(audio, sample_rate, cfg.target_sample_rate)
    sr = int(cfg.target_sample_rate)
    win = max(cfg.n_fft, int(round(float(cfg.peak_window_seconds) * sr)))
    half = win // 2
    target_frames = 1 + (win - cfg.n_fft) // cfg.hop_length
    onsets = segments.loc[segments["label"] == 1, "start_time"].to_numpy(dtype=float)
    if onsets.size == 0:
        onsets = segments.loc[segments["label"] == LABEL_SYSTOLE, "start_time"].to_numpy(dtype=float)
    low_hz = float(getattr(cfg, "low_hz", 0.0))
    freq_mask: np.ndarray | None = None
    specs: list[np.ndarray] = []
    for t in onsets:
        center = int(round(float(t) * sr))
        start = center - half
        seg = np.zeros(win, dtype=np.float32)
        s0 = max(0, start)
        e0 = min(len(resampled), start + win)
        if e0 <= s0:
            continue
        seg[s0 - start : e0 - start] = resampled[s0:e0]
        freqs, _times, zxx = stft(
            seg, fs=sr, window="hann", nperseg=cfg.n_fft,
            noverlap=cfg.n_fft - cfg.hop_length, nfft=cfg.n_fft, boundary=None, padded=False,
        )
        spec = np.log1p(np.abs(zxx).astype(np.float32))
        if freq_mask is None:
            freq_mask = (freqs >= low_hz) & (freqs <= cfg.high_hz)
        spec = _crop_or_pad(spec[freq_mask], target_frames)
        specs.append(spec)
    if not specs:
        return np.zeros((0, 0, 0), dtype=np.float32)
    return np.stack(specs).astype(np.float32)
