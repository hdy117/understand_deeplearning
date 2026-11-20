"""Pixel-space U-Net DDPM."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import DDPMConfig, DEVICE, PixelFlowModelConfig
from pixel_flow_unet import PixelUNetFlowMatching


class PixelDDPMUNet(nn.Module):
    """Epsilon-prediction DDPM using the same U-Net as Pixel Flow."""

    def __init__(
        self,
        *,
        pixel_config: PixelFlowModelConfig | None = None,
        ddpm_config: DDPMConfig | None = None,
        device=DEVICE,
    ) -> None:
        super().__init__()
        self.pixel_config = pixel_config or PixelFlowModelConfig()
        self.ddpm_config = ddpm_config or DDPMConfig()
        schedule = self.ddpm_config
        if schedule.timesteps <= 1:
            raise ValueError("timesteps must be greater than one")
        if not 0.0 < schedule.beta_start < schedule.beta_end < 1.0:
            raise ValueError("beta_start and beta_end must satisfy 0 < start < end < 1")

        self.backbone = PixelUNetFlowMatching(self.pixel_config, device=device)
        betas = torch.linspace(
            schedule.beta_start,
            schedule.beta_end,
            schedule.timesteps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat(
            (torch.ones(1, dtype=torch.float32), alpha_bars[:-1]),
            dim=0,
        )
        posterior_variance = (
            betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance)
        self.to(device)

    @property
    def timesteps(self) -> int:
        return self.ddpm_config.timesteps

    def _validate_timesteps(self, timesteps: torch.Tensor, batch_size: int) -> None:
        if timesteps.shape != (batch_size,):
            raise ValueError(
                f"timesteps must have shape [{batch_size}], got {tuple(timesteps.shape)}"
            )
        if timesteps.is_floating_point():
            raise ValueError("timesteps must use an integer dtype")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.timesteps):
            raise ValueError(f"timesteps must be in [0,{self.timesteps - 1}]")

    @staticmethod
    def _extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return values[timesteps].reshape(
            target.shape[0],
            *((1,) * (target.ndim - 1)),
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if x0.shape != noise.shape:
            raise ValueError(
                f"x0 and noise shapes differ: {x0.shape} and {noise.shape}"
            )
        self._validate_timesteps(timesteps, x0.shape[0])
        if timesteps.device != x0.device:
            raise ValueError("timesteps and x0 must be on the same device")
        alpha_bar = self._extract(self.alpha_bars, timesteps, x0)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def predict_noise(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        label_batch: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_timesteps(timesteps, xt.shape[0])
        if timesteps.device != xt.device:
            raise ValueError("timesteps and xt must be on the same device")
        continuous_time = (
            timesteps.float() / (self.timesteps - 1)
        ).reshape(-1, 1)
        return self.backbone(xt, continuous_time, label_batch)

    forward = predict_noise

    @torch.inference_mode()
    def sample(
        self,
        label_batch: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if label_batch.ndim != 2:
            raise ValueError("label_batch must have rank 2")
        batch_size = label_batch.shape[0]
        images = torch.randn(
            batch_size,
            self.pixel_config.image_channels,
            self.pixel_config.image_size,
            self.pixel_config.image_size,
            device=label_batch.device,
            generator=generator,
        )
        for timestep_value in range(self.timesteps - 1, -1, -1):
            timesteps = torch.full(
                (batch_size,),
                timestep_value,
                dtype=torch.long,
                device=label_batch.device,
            )
            predicted_noise = self.predict_noise(images, timesteps, label_batch)
            beta = self._extract(self.betas, timesteps, images)
            alpha = self._extract(self.alphas, timesteps, images)
            alpha_bar = self._extract(self.alpha_bars, timesteps, images)
            mean = (
                images - beta / (1.0 - alpha_bar).sqrt() * predicted_noise
            ) / alpha.sqrt()
            if timestep_value > 0:
                variance = self._extract(
                    self.posterior_variance,
                    timesteps,
                    images,
                )
                images = mean + variance.sqrt() * torch.randn(
                    images.shape,
                    device=images.device,
                    generator=generator,
                )
            else:
                images = mean
            if not torch.isfinite(images).all():
                raise RuntimeError(
                    f"non-finite image encountered at timestep {timestep_value}"
                )
        return images


if __name__ == "__main__":
    model = PixelDDPMUNet()
    print("parameters:", sum(parameter.numel() for parameter in model.parameters()))
