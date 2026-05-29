from __future__ import annotations


import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        total_padding = (kernel_size - 1) * dilation
        if causal:
            padding = total_padding
            crop1: nn.Module = Chomp1d(padding)
            crop2: nn.Module = Chomp1d(padding)
        else:
            if total_padding % 2 != 0:
                raise ValueError("Non-causal TCN requires an odd kernel size when dilation is a power of two.")
            padding = total_padding // 2
            crop1 = nn.Identity()
            crop2 = nn.Identity()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            crop1,
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            crop2,
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.downsample(x)


class TemporalAttentionPoolingContext(nn.Module):
    """Attention pooling context that preserves frame-level predictions.

    The module pools a global context vector over time using learned attention
    weights, then broadcasts that context back to every frame. This gives the
    frame classifier access to whole-recording context without collapsing the
    temporal axis needed for segmentation.
    """

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        inner_channels = max(8, channels // 2)
        self.score = nn.Sequential(
            nn.Conv1d(channels, inner_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(inner_channels, 1, kernel_size=1),
        )
        self.context = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        pooled = (x * weights).sum(dim=-1, keepdim=True)
        context = self.context(pooled).expand_as(x)
        gate = self.gate(torch.cat([x, context], dim=1))
        return x + gate * context


class TCNFrameSegmenter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        levels: int,
        kernel_size: int,
        dropout: float,
        num_classes: int = 5,
        causal: bool = True,
        pooling: str = "none",
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for level in range(levels):
            dilation = 2**level
            block_in = in_channels if level == 0 else hidden_channels
            blocks.append(TemporalBlock(block_in, hidden_channels, kernel_size, dilation, dropout, causal))
        self.tcn = nn.Sequential(*blocks)
        self.pooling = pooling
        self.attention_pool = (
            TemporalAttentionPoolingContext(hidden_channels, dropout)
            if pooling == "attention"
            else nn.Identity()
        )
        self.classifier = nn.Conv1d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        features = self.attention_pool(features)
        return self.classifier(features)
