"""Shared building blocks for pixel-space conditional U-Net models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    """Return a small stable GroupNorm configuration for one channel width."""
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class FiLMResBlock(nn.Module):
    """Residual convolution block conditioned by per-sample scale and shift."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.film1 = nn.Linear(condition_dim, 2 * in_channels)
        self.film2 = nn.Linear(condition_dim, 2 * out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale1, shift1 = self.film1(condition).chunk(2, dim=-1)
        scale2, shift2 = self.film2(condition).chunk(2, dim=-1)
        scale1 = scale1[:, :, None, None]
        shift1 = shift1[:, :, None, None]
        scale2 = scale2[:, :, None, None]
        shift2 = shift2[:, :, None, None]

        residual = hidden
        hidden = self.norm1(hidden)
        hidden = hidden * (1.0 + scale1) + shift1
        hidden = F.silu(hidden)
        hidden = self.conv1(hidden)

        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale2) + shift2
        hidden = F.silu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)
        return hidden + self.skip(residual)
