from __future__ import annotations


import torch
from torch import nn


from .blocks import (
    Conv2dBlock,
    DilatedBlock,
    FrequencyAttention,
    FrequencyEmphasis,
    MultiScaleConv2dBlock,
    MultiScaleDilatedBlock,
    TemporalAttentionPool,
    band_bin_mask,
    build_freq_band_branch,
)
from .config import ModelConfig


class SystoleDilatedCNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c = config.base_channels
        layers: list[nn.Module] = [
            nn.Conv1d(config.freq_bins, c, kernel_size=3, padding=1),
            nn.GroupNorm(4, c),
            nn.GELU(),
        ]
        block_class: type[nn.Module]
        if config.encoder_block == "multiscale":
            block_class = MultiScaleDilatedBlock
        elif config.encoder_block == "residual":
            block_class = DilatedBlock
        else:
            raise ValueError(f"Unsupported encoder_block: {config.encoder_block}")
        for dilation in config.dilations:
            layers.append(block_class(c, dilation=dilation, dropout=config.dropout))
        self.encoder = nn.Sequential(*layers)
        self.pool = (
            TemporalAttentionPool(c, config.dropout)
            if config.pooling == "attention"
            else nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        )
        head_in = c
        self.n_temporal_features = int(getattr(config, "n_temporal_features", 0))
        if self.n_temporal_features > 0:
            temporal_hidden = max(16, c)
            self.temporal_branch = nn.Sequential(
                nn.Linear(self.n_temporal_features, temporal_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(temporal_hidden, temporal_hidden),
                nn.GELU(),
            )
            head_in += temporal_hidden
        else:
            self.temporal_branch = None

        # Parallel linear branch over the murmur frequency band. It sees only the band bins,
        # summarized as per-bin mean+std over time, complementing the full-spectrogram CNN.
        self.freq_linear = None
        if bool(getattr(config, "freq_linear_branch", False)):
            mask = band_bin_mask(
                config.freq_bins,
                float(getattr(config, "freq_fmax", 1000.0)),
                float(getattr(config, "freq_low_hz", 100.0)),
                float(getattr(config, "freq_high_hz", 600.0)),
                bool(getattr(config, "freq_mel_scale", False)),
            )
            self.register_buffer("freq_band_mask", torch.from_numpy(mask))
            n_band = int(mask.sum())
            self.freq_linear = build_freq_band_branch(
                arch=str(getattr(config, "freq_linear_arch", "transformer")),
                n_band=n_band,
                hidden=max(8, int(getattr(config, "freq_linear_hidden", 32))),
                heads=int(getattr(config, "freq_linear_heads", 4)),
                layers=int(getattr(config, "freq_linear_layers", 2)),
                dropout=config.dropout,
            )
            head_in += self.freq_linear.out_dim

        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(head_in, 1),
        )

        # Auxiliary multi-task head over the pooled encoder features (dim c, before any branch
        # concatenation) so the gradient pushes the *encoder* to represent murmur pitch.
        aux_classes = int(getattr(config, "aux_pitch_classes", 0))
        self.aux_pitch_head = nn.Linear(c, aux_classes) if aux_classes > 0 else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.pool(encoded)

    def forward(self, x: torch.Tensor, temporal: torch.Tensor | None = None, return_aux: bool = False):
        pooled = self.encode(x)
        aux_logits = self.aux_pitch_head(pooled) if self.aux_pitch_head is not None else None
        feat = pooled
        if self.freq_linear is not None:
            band = x[:, self.freq_band_mask, :]                       # (B, n_band, T)
            feat = torch.cat([feat, self.freq_linear(band)], dim=1)
        if self.temporal_branch is not None:
            if temporal is None:
                raise ValueError("Model expects temporal features but none were provided.")
            feat = torch.cat([feat, self.temporal_branch(temporal)], dim=1)
        logits = self.head(feat).squeeze(-1)
        if return_aux:
            return logits, aux_logits
        return logits


class SystoleRNN(nn.Module):
    """Recurrent classifier over the systole spectrogram.

    The spectrogram (freq x time) is read as a sequence of `time` frames, each a `freq_bins`
    vector, by a bidirectional GRU/LSTM. Pooling is attention or mean over the recurrent states.
    Same head/temporal-branch contract as SystoleDilatedCNN so the training loop is unchanged.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = int(getattr(config, "rnn_hidden", 64))
        layers = int(getattr(config, "rnn_layers", 2))
        rnn_type = str(getattr(config, "rnn_type", "gru")).lower()
        dropout = config.dropout
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=config.freq_bins,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        out_dim = hidden * 2
        self.pooling = config.pooling
        self.pool = TemporalAttentionPool(out_dim, dropout) if config.pooling == "attention" else None
        self.n_temporal_features = int(getattr(config, "n_temporal_features", 0))
        head_in = out_dim
        if self.n_temporal_features > 0:
            th = max(16, out_dim)
            self.temporal_branch = nn.Sequential(
                nn.Linear(self.n_temporal_features, th), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(th, th), nn.GELU(),
            )
            head_in = out_dim + th
        else:
            self.temporal_branch = None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, 1))

    def forward(self, x: torch.Tensor, temporal: torch.Tensor | None = None) -> torch.Tensor:
        seq = x.transpose(1, 2)  # (B, freq, time) -> (B, time, freq)
        out, _ = self.rnn(seq)   # (B, time, 2*hidden)
        if self.pool is not None:
            pooled = self.pool(out.transpose(1, 2))  # (B, 2*hidden)
        else:
            pooled = out.mean(dim=1)
        if self.temporal_branch is not None:
            if temporal is None:
                raise ValueError("Model expects temporal features but none were provided.")
            pooled = torch.cat([pooled, self.temporal_branch(temporal)], dim=1)
        return self.head(pooled).squeeze(-1)


class SystoleFreq2dCNN(nn.Module):
    """2D-conv classifier that keeps the frequency axis through the encoder.

    Pipeline: (B, F, T) -> (B, 1, F, T) -> [optional FrequencyEmphasis] -> stack of Conv2dBlocks
    with [optional FrequencyAttention] between them -> mean over F -> (B, C, T) -> temporal pool
    -> head. FrequencyEmphasis is neutral at init (alpha=0) and FrequencyAttention is identity-ish
    early in training, so this degrades gracefully to a plain 2D CNN when both are off.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c = config.base_channels
        dropout = config.dropout
        self.use_emphasis = bool(getattr(config, "freq_emphasis", False))
        self.use_attention = bool(getattr(config, "freq_attention", False))

        self.emphasis = (
            FrequencyEmphasis(
                sr=int(getattr(config, "freq_sample_rate", 4000)),
                n_bins=config.freq_bins,
                fmax=float(getattr(config, "freq_fmax", 1000.0)),
                f_low=float(getattr(config, "freq_low_hz", 100.0)),
                f_high=float(getattr(config, "freq_high_hz", 600.0)),
                alpha_init=float(getattr(config, "freq_emphasis_alpha_init", 0.0)),
                mel_scale=bool(getattr(config, "freq_mel_scale", False)),
            )
            if self.use_emphasis else None
        )

        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1),
            nn.GroupNorm(4, c),
            nn.GELU(),
        )
        if config.encoder_block == "multiscale":
            block_class: type[nn.Module] = MultiScaleConv2dBlock
        elif config.encoder_block == "residual":
            block_class = Conv2dBlock
        else:
            raise ValueError(f"Unsupported encoder_block for freq2d: {config.encoder_block}")
        dilations = config.dilations if config.dilations else (1,)
        self.blocks = nn.ModuleList(
            [block_class(c, dilation=d, dropout=dropout) for d in dilations]
        )
        self.freq_attentions = nn.ModuleList(
            [FrequencyAttention(c, dropout) if self.use_attention else nn.Identity()
             for _ in dilations]
        )

        self.pool = (
            TemporalAttentionPool(c, dropout)
            if config.pooling == "attention"
            else nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        )
        self.n_temporal_features = int(getattr(config, "n_temporal_features", 0))
        head_in = c
        if self.n_temporal_features > 0:
            th = max(16, c)
            self.temporal_branch = nn.Sequential(
                nn.Linear(self.n_temporal_features, th), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(th, th), nn.GELU(),
            )
            head_in = c + th
        else:
            self.temporal_branch = None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, 1))

    def forward(self, x: torch.Tensor, temporal: torch.Tensor | None = None) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, F, T) -> (B, 1, F, T)
        if self.emphasis is not None:
            x = self.emphasis(x)
        x = self.stem(x)
        for block, attn in zip(self.blocks, self.freq_attentions):
            x = block(x)
            x = attn(x)
        x = x.mean(dim=2)  # collapse frequency -> (B, C, T)
        pooled = self.pool(x)
        if self.temporal_branch is not None:
            if temporal is None:
                raise ValueError("Model expects temporal features but none were provided.")
            pooled = torch.cat([pooled, self.temporal_branch(temporal)], dim=1)
        return self.head(pooled).squeeze(-1)


def build_systole_model(config: ModelConfig) -> nn.Module:
    arch = str(getattr(config, "arch", "cnn")).lower()
    if arch == "rnn":
        return SystoleRNN(config)
    if arch == "freq2d":
        return SystoleFreq2dCNN(config)
    return SystoleDilatedCNN(config)
