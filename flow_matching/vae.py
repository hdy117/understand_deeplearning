"""A convolutional VAE whose 8x8 latent grid can feed a token Transformer.

Image tensors use ``float32`` values in ``[0, 1]`` and NCHW layout.  The
default image size is 64x64, so the latent tensor is ``[B, 4, 8, 8]``.  The
spatial latent positions become 64 tokens after moving the channel axis last.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VAEConfig
import os
from utils import get_logger

logger = get_logger()


class VAE(nn.Module):
    """Encode 64x64 images into a Gaussian latent and decode them back."""

    def __init__(
        self,
        config: VAEConfig | None = None,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else VAEConfig()
        self.image_size = image_size

        if image_size <= 0 or image_size % self.config.downsample_factor != 0:
            raise ValueError(
                "image_size must be positive and divisible by "
                f"downsample_factor={self.config.downsample_factor}"
            )
        channels = self.config.hidden_channels
        if len(channels) != 4 or any(channel <= 0 for channel in channels):
            raise ValueError("hidden_channels must contain four positive integers")

        self.latent_size = image_size // self.config.downsample_factor
        self.encoder = nn.Sequential(
            self._block(self.config.in_channels, channels[0], stride=1),
            self._block(channels[0], channels[1], stride=2),
            self._block(channels[1], channels[2], stride=2),
            self._block(channels[2], channels[3], stride=2),
            nn.Conv2d(channels[3], self.config.latent_channels * 2, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(
                self.config.latent_channels,
                channels[3],
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
            self._block(channels[3], channels[2], stride=1, upsample=True),
            self._block(channels[2], channels[1], stride=1, upsample=True),
            self._block(channels[1], channels[0], stride=1, upsample=True),
            nn.Conv2d(channels[0], self.config.in_channels, kernel_size=3, padding=1),
        )

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        upsample: bool = False,
    ) -> nn.Module:
        """Build a small residual-free convolutional block."""
        layers: list[nn.Module] = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.extend(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                nn.GroupNorm(
                    num_groups=min(8, out_channels), num_channels=out_channels
                ),
                nn.SiLU(),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GroupNorm(
                    num_groups=min(8, out_channels), num_channels=out_channels
                ),
                nn.SiLU(),
            ]
        )
        return nn.Sequential(*layers)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the posterior mean and log variance for a batch of images."""
        self._validate_images(images)
        parameters = self.encoder(images)
        mean, logvar = parameters.chunk(2, dim=1)
        return mean, logvar.clamp(min=-30.0, max=20.0)

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the Gaussian posterior with the reparameterization trick."""
        if mean.shape != logvar.shape:
            raise ValueError("mean and logvar must have the same shape")
        standard_deviation = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(mean) * standard_deviation

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a latent batch and return pixel values in ``[0, 1]``."""
        if latents.ndim != 4:
            raise ValueError(f"latents must be rank 4, got {latents.shape}")
        if latents.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"latents must have {self.config.latent_channels} channels, "
                f"got {latents.shape[1]}"
            )
        if latents.shape[-2:] != (self.latent_size, self.latent_size):
            raise ValueError(
                f"latents must be {self.latent_size}x{self.latent_size}, "
                f"got {tuple(latents.shape[-2:])}"
            )
        if latents.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"latents must be floating point, got {latents.dtype}")
        return torch.sigmoid(self.decoder(latents))

    def forward(
        self,
        images: torch.Tensor,
        *,
        sample_latent: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct images and return reconstruction, mean, logvar, latent."""
        mean, logvar = self.encode(images)
        latent = self.reparameterize(mean, logvar) if sample_latent else mean
        reconstruction = self.decode(latent)
        return reconstruction, mean, logvar, latent

    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return deterministic latent tokens as ``[B, 64, 4]``."""
        mean, _ = self.encode(images)
        return mean.flatten(start_dim=2).transpose(1, 2)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode ``[B, 64, 4]`` flow-matching tokens into images."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.latent_channels:
            raise ValueError(
                f"tokens must be [B, positions, {self.config.latent_channels}], "
                f"got {tuple(tokens.shape)}"
            )
        expected_positions = self.latent_size * self.latent_size
        if tokens.shape[1] != expected_positions:
            raise ValueError(
                f"tokens must contain {expected_positions} positions, "
                f"got {tokens.shape[1]}"
            )
        latents = tokens.transpose(1, 2).reshape(
            tokens.shape[0],
            self.config.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        return self.decode(latents)

    def _validate_images(self, images: torch.Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(f"images must be rank 4 NCHW, got {images.shape}")
        expected_shape = (self.config.in_channels, self.image_size, self.image_size)
        if tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                f"images must have shape [B, {expected_shape}], got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise ValueError(f"images must be floating point, got {images.dtype}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "VAE":
        """Restore a VAE and its image-size contract from a training checkpoint."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        model = cls(
            config=VAEConfig(**checkpoint["vae_config"]),
            image_size=checkpoint["image_size"],
        )
        model.load_state_dict(checkpoint["model"])
        return model


def vae_loss(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    *,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction, and mean KL terms for one batch."""
    if kl_weight < 0:
        raise ValueError("kl_weight must be non-negative")
    reconstruction_loss = F.mse_loss(reconstruction, target)
    kl_per_batch = -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).sum(
        dim=(1, 2, 3)
    )
    kl_loss = kl_per_batch.mean()
    total_loss = reconstruction_loss + kl_weight * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def load_vae_from_ckpt(ckpt_path: str = None) -> VAE:
    if not os.path.exists(ckpt_path):
        err = f"{ckpt_path} does not exist"
        raise ValueError(err)
    ckpt = torch.load(ckpt_path, weights_only=False)
    vae = VAE()
    vae.load_state_dict(ckpt["model"])
    return vae
