from __future__ import annotations


import numpy as np
import torch
from torch import nn


def bin_center_freqs(n_bins: int, fmax: float, mel_scale: bool = False) -> np.ndarray:
    """Center frequency (Hz) of each spectrogram bin: linear over [0, fmax], or mel-spaced."""
    if mel_scale:
        mel = np.linspace(0.0, 2595.0 * np.log10(1.0 + float(fmax) / 700.0), int(n_bins))
        return (700.0 * (np.power(10.0, mel / 2595.0) - 1.0)).astype(np.float32)
    return np.linspace(0.0, float(fmax), int(n_bins), dtype=np.float32)


def band_bin_mask(n_bins: int, fmax: float, f_low: float, f_high: float,
                  mel_scale: bool = False) -> np.ndarray:
    """Boolean mask (length n_bins) of bins whose center frequency falls in [f_low, f_high]."""
    freqs = bin_center_freqs(n_bins, fmax, mel_scale)
    mask = (freqs >= float(f_low)) & (freqs <= float(f_high))
    if not mask.any():  # fall back to the single closest bin so the branch is never empty
        mask[int(np.argmin(np.abs(freqs - 0.5 * (f_low + f_high))))] = True
    return mask


def build_freq_prior(sr: int, n_bins: int, fmax: float, f_low: float, f_high: float,
                     mel_scale: bool = False) -> np.ndarray:
    """Soft band-emphasis prior over the frequency-bin axis (length n_bins, values in [0, 1]).

    Bin center frequencies are linear over [0, fmax] for STFT, or mel-spaced when `mel_scale`
    (so the emphasized band lands on the correct log-mel bins). The prior is a smooth band-pass
    bump that is ~1 inside [f_low, f_high] and tapers to ~0 outside, so a learnable gain can
    emphasize the murmur band without a hard cut.
    """
    freqs = bin_center_freqs(n_bins, fmax, mel_scale)
    width = max(1e-6, 0.15 * (f_high - f_low))
    prior = 0.5 * (np.tanh((freqs - f_low) / width) - np.tanh((freqs - f_high) / width))
    peak = float(prior.max())
    if peak > 1e-6:
        prior = prior / peak
    return prior.astype(np.float32)


class FrequencyEmphasis(nn.Module):
    """Learnable soft emphasis of a frequency band on the input spectrogram.

    Input/output: (B, C, F, T). `alpha=0` at init -> identity (neutral, matches the baseline).
    Only the scalar gain `alpha` is trained; the band prior itself is a fixed buffer.
    """

    def __init__(self, sr: int, n_bins: int, fmax: float, f_low: float = 100.0,
                 f_high: float = 600.0, alpha_init: float = 0.0, mel_scale: bool = False) -> None:
        super().__init__()
        prior = build_freq_prior(sr, n_bins, fmax, f_low, f_high, mel_scale=mel_scale)
        self.register_buffer("freq_prior", torch.from_numpy(prior))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        gain = 1.0 + self.alpha * self.freq_prior.view(1, 1, -1, 1)
        return spec * gain


class FrequencyAttention(nn.Module):
    """Content-based per-band reweighting that keeps the frequency axis (does not collapse it).

    Input/output: (B, C, F, T). Each frequency band is described by its time-averaged channel
    statistics, then assigned a gate in [0, 1] that rescales that band across all time steps.
    """

    def __init__(self, channels: int, dropout: float, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = x.mean(dim=-1)               # (B, C, F)
        scores = self.score(desc)           # (B, 1, F)
        weights = torch.sigmoid(scores)     # (B, 1, F)
        return x * weights.unsqueeze(-1)    # (B, C, F, T)


def _band_token_features(band: torch.Tensor) -> torch.Tensor:
    """Per-band-bin descriptor over time: mean, std, max -> (B, n_band, 3)."""
    return torch.stack([band.mean(dim=-1), band.std(dim=-1), band.amax(dim=-1)], dim=-1)


class FrequencyBandMLP(nn.Module):
    """Simple MLP over the flattened (mean, std) descriptor of the murmur band. Input: (B, n_band, T)."""

    def __init__(self, n_band: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.out_dim = hidden
        self.net = nn.Sequential(
            nn.Linear(2 * n_band, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, band: torch.Tensor) -> torch.Tensor:
        desc = torch.cat([band.mean(dim=-1), band.std(dim=-1)], dim=1)  # (B, 2*n_band)
        return self.net(desc)


class FrequencyBandTransformer(nn.Module):
    """Transformer encoder over frequency-band tokens. Input: (B, n_band, T) -> (B, d_model).

    Each band bin becomes a token (described by its mean/std/max over time), plus a learnable
    positional embedding for the bin index. Stacked pre-norm self-attention blocks (multi-head
    attention + LayerNorm + residual + GELU FFN) let bands attend to each other; tokens are then
    mean-pooled into a single band representation.
    """

    def __init__(self, n_band: int, d_model: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.out_dim = d_model
        heads = max(1, min(heads, d_model))
        while d_model % heads != 0 and heads > 1:
            heads -= 1
        self.embed = nn.Linear(3, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_band, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=max(1, layers), enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, band: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(_band_token_features(band)) + self.pos  # (B, n_band, d_model)
        tokens = self.encoder(tokens)
        return self.norm(tokens.mean(dim=1))  # (B, d_model)


def build_freq_band_branch(arch: str, n_band: int, hidden: int, heads: int,
                           layers: int, dropout: float) -> nn.Module:
    if str(arch).lower() == "mlp":
        return FrequencyBandMLP(n_band, hidden, dropout)
    return FrequencyBandTransformer(n_band, hidden, heads, layers, dropout)


class Conv2dBlock(nn.Module):
    """Residual 2D conv block over (B, C, F, T); keeps F and T resolution.

    `dilation` widens the receptive field in both frequency and time while preserving size.
    """

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MultiScaleConv2dBlock(nn.Module):
    """2D analogue of MultiScaleDilatedBlock: parallel kernels over frequency x time + SE gating.

    Captures murmur energy "blobs" at multiple time-frequency scales, then recalibrates channels
    with a squeeze-excite gate. Residual; preserves (B, C, F, T) size.
    """

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        branch_channels = max(4, channels // 2)
        se_channels = max(1, channels // 8)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(1, branch_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, kernel_size=5, padding=2 * dilation, dilation=dilation),
                    nn.GroupNorm(1, branch_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, kernel_size=3, padding=2 * dilation, dilation=2 * dilation),
                    nn.GroupNorm(1, branch_channels),
                    nn.GELU(),
                ),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * len(self.branches), channels, kernel_size=1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(4, channels),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, se_channels),
            nn.GELU(),
            nn.Linear(se_channels, channels),
            nn.Sigmoid(),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multiscale = torch.cat([branch(x) for branch in self.branches], dim=1)
        projected = self.project(multiscale)
        gate = self.se(projected)[:, :, None, None]
        projected = projected * gate
        return self.activation(x + projected)


class DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MultiScaleDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        branch_channels = max(4, channels // 2)
        se_channels = max(1, channels // 8)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, branch_channels, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(1, branch_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv1d(channels, branch_channels, kernel_size=5, padding=2 * dilation, dilation=dilation),
                    nn.GroupNorm(1, branch_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv1d(channels, branch_channels, kernel_size=9, padding=4 * dilation, dilation=dilation),
                    nn.GroupNorm(1, branch_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv1d(channels, branch_channels, kernel_size=3, padding=2 * dilation, dilation=2 * dilation),
                    nn.GroupNorm(1, branch_channels),
                    nn.GELU(),
                ),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv1d(branch_channels * len(self.branches), channels, kernel_size=1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(4, channels),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, se_channels),
            nn.GELU(),
            nn.Linear(se_channels, channels),
            nn.Sigmoid(),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multiscale = torch.cat([branch(x) for branch in self.branches], dim=1)
        projected = self.project(multiscale)
        projected = projected * self.se(projected).unsqueeze(-1)
        return self.activation(x + projected)


class TemporalAttentionPool(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return (x * weights).sum(dim=-1)
