import unittest

import torch

from config import PixelFlowModelConfig
from pixel_flow_unet import PixelUNetFlowMatching


class PixelFlowUNetTests(unittest.TestCase):
    @staticmethod
    def _small_config() -> PixelFlowModelConfig:
        return PixelFlowModelConfig(
            image_size=16,
            base_channels=8,
            channel_multipliers=(1, 2, 4, 8),
            blocks_per_level=1,
            condition_dim=32,
        )

    def test_default_model_accepts_32x32_rgb(self) -> None:
        model = PixelUNetFlowMatching(
            PixelFlowModelConfig(),
            device="cpu",
        ).eval()
        images = torch.randn(1, 3, 32, 32)
        labels = torch.eye(10)[:1]
        with torch.inference_mode():
            velocity = model(images, torch.rand(1, 1), labels)
        self.assertEqual(tuple(velocity.shape), (1, 3, 32, 32))

    def test_small_model_has_finite_gradients(self) -> None:
        model = PixelUNetFlowMatching(self._small_config(), device="cpu")
        images = torch.randn(2, 3, 16, 16, requires_grad=True)
        labels = torch.eye(10)[:2]
        velocity = model(images, torch.rand(2, 1), labels)
        velocity.square().mean().backward()
        self.assertIsNotNone(images.grad)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))

    def test_image_coordinate_conversion_round_trip(self) -> None:
        self.assertEqual(
            PixelUNetFlowMatching.flow_to_images(
                PixelUNetFlowMatching.images_to_flow(torch.rand(2, 3, 32, 32))
            ).shape,
            (2, 3, 32, 32),
        )

    def test_rejects_bad_inputs(self) -> None:
        model = PixelUNetFlowMatching(self._small_config(), device="cpu")
        labels = torch.eye(10)[:1]
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 32, 32), torch.rand(1, 1), labels)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 16, 16), torch.full((1, 1), 1.1), labels)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 16, 16), torch.rand(1, 1), torch.zeros(1, 10))
        with self.assertRaises(ValueError):
            PixelUNetFlowMatching.images_to_flow(torch.full((1, 3, 16, 16), 1.1))

    def test_rejects_incompatible_image_size(self) -> None:
        with self.assertRaises(ValueError):
            PixelUNetFlowMatching(
                PixelFlowModelConfig(
                    image_size=12,
                    base_channels=8,
                    channel_multipliers=(1, 2, 4, 8),
                    blocks_per_level=1,
                    condition_dim=32,
                ),
                device="cpu",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_bfloat16_forward(self) -> None:
        model = PixelUNetFlowMatching(self._small_config(), device="cuda").eval()
        images = torch.randn(1, 3, 16, 16, device="cuda")
        labels = torch.eye(10, device="cuda")[:1]
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            velocity = model(images, torch.rand(1, 1, device="cuda"), labels)
        self.assertEqual(tuple(velocity.shape), (1, 3, 16, 16))
        self.assertTrue(torch.isfinite(velocity.float()).all())


if __name__ == "__main__":
    unittest.main()
