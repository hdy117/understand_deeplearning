import unittest

import torch

from config import DDPMConfig, PixelFlowModelConfig
from pixel_ddpm import PixelDDPMUNet


def small_configs(timesteps: int = 5) -> tuple[PixelFlowModelConfig, DDPMConfig]:
    return (
        PixelFlowModelConfig(
            image_size=16,
            base_channels=8,
            channel_multipliers=(1, 2, 4, 8),
            blocks_per_level=1,
            condition_dim=32,
        ),
        DDPMConfig(timesteps=timesteps, beta_start=1e-4, beta_end=0.02),
    )


class PixelDDPMTests(unittest.TestCase):
    def test_schedule_is_monotonic(self) -> None:
        pixel_config, ddpm_config = small_configs(timesteps=20)
        model = PixelDDPMUNet(
            pixel_config=pixel_config,
            ddpm_config=ddpm_config,
            device="cpu",
        )
        self.assertEqual(tuple(model.betas.shape), (20,))
        self.assertTrue(torch.all(model.betas > 0))
        self.assertTrue(torch.all(model.betas < 1))
        self.assertTrue(torch.all(model.alpha_bars[:-1] > model.alpha_bars[1:]))

    def test_q_sample_has_closed_form(self) -> None:
        pixel_config, ddpm_config = small_configs()
        model = PixelDDPMUNet(
            pixel_config=pixel_config,
            ddpm_config=ddpm_config,
            device="cpu",
        )
        x0 = torch.ones(2, 3, 16, 16)
        noise = torch.zeros_like(x0)
        timesteps = torch.tensor([1, 3], dtype=torch.long)
        sampled = model.q_sample(x0, timesteps, noise)
        expected = model.alpha_bars[timesteps].sqrt().reshape(2, 1, 1, 1)
        torch.testing.assert_close(sampled, expected.expand_as(x0))

    def test_noise_prediction_and_gradients(self) -> None:
        pixel_config, ddpm_config = small_configs()
        model = PixelDDPMUNet(
            pixel_config=pixel_config,
            ddpm_config=ddpm_config,
            device="cpu",
        )
        images = torch.randn(2, 3, 16, 16, requires_grad=True)
        timesteps = torch.tensor([1, 4], dtype=torch.long)
        labels = torch.eye(10)[:2]
        prediction = model.predict_noise(images, timesteps, labels)
        prediction.square().mean().backward()
        self.assertEqual(tuple(prediction.shape), tuple(images.shape))
        self.assertIsNotNone(images.grad)
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )

    def test_sampling_is_reproducible_with_seed(self) -> None:
        pixel_config, ddpm_config = small_configs(timesteps=3)
        torch.manual_seed(0)
        model = PixelDDPMUNet(
            pixel_config=pixel_config,
            ddpm_config=ddpm_config,
            device="cpu",
        ).eval()
        labels = torch.eye(10)[:2]
        first = model.sample(labels, generator=torch.Generator().manual_seed(123))
        second = model.sample(labels, generator=torch.Generator().manual_seed(123))
        torch.testing.assert_close(first, second)

    def test_rejects_invalid_timesteps(self) -> None:
        pixel_config, ddpm_config = small_configs()
        model = PixelDDPMUNet(
            pixel_config=pixel_config,
            ddpm_config=ddpm_config,
            device="cpu",
        )
        x0 = torch.randn(2, 3, 16, 16)
        noise = torch.randn_like(x0)
        with self.assertRaises(ValueError):
            model.q_sample(
                x0,
                torch.tensor([0, 5], dtype=torch.long),
                noise,
            )
        with self.assertRaises(ValueError):
            model.q_sample(x0, torch.tensor([0.0, 1.0]), noise)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_bfloat16_noise_prediction(self) -> None:
        pixel_config, ddpm_config = small_configs()
        model = PixelDDPMUNet(
            pixel_config=pixel_config,
            ddpm_config=ddpm_config,
            device="cuda",
        ).eval()
        images = torch.randn(1, 3, 16, 16, device="cuda")
        timesteps = torch.tensor([2], dtype=torch.long, device="cuda")
        labels = torch.eye(10, device="cuda")[:1]
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            prediction = model.predict_noise(images, timesteps, labels)
        self.assertEqual(tuple(prediction.shape), (1, 3, 16, 16))
        self.assertTrue(torch.isfinite(prediction.float()).all())


if __name__ == "__main__":
    unittest.main()
