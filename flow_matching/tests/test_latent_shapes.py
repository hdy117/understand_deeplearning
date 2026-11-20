import tempfile
import unittest
from dataclasses import asdict

import torch

from config import (
    FlowMatchingModelConfig,
    IMG_SIZE,
    LATENT_SIZE,
    LATENT_TOKEN_COUNT,
    VAEConfig,
)
from flow_matching import FlowMatching
from vae import (
    LPIPSLoss,
    VAE,
    load_vae_from_ckpt,
    structural_similarity,
    vae_loss,
)


class LatentShapeTests(unittest.TestCase):
    def test_default_vae_uses_native_cifar10_size(self) -> None:
        config = VAEConfig()
        model = VAE(config, image_size=IMG_SIZE).eval()
        images = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

        with torch.inference_mode():
            mean, _ = model.encode(images)
            encoded_parameters = model.encoder(images)
            tokens = model.encode_tokens(images)

        self.assertEqual(IMG_SIZE, 32)
        self.assertEqual(LATENT_SIZE, 8)
        self.assertEqual(LATENT_TOKEN_COUNT, 64)
        self.assertEqual(model.latent_channels, 32)
        self.assertEqual(tuple(encoded_parameters.shape), (1, 64, 8, 8))
        self.assertEqual(tuple(mean.shape), (1, 32, 8, 8))
        self.assertEqual(
            tuple(tokens.shape),
            (1, FlowMatchingModelConfig().num_tokens, model.latent_channels),
        )

    def test_vae_supports_configured_downsampling_factors(self) -> None:
        expected_sizes = {
            8: (8, 64),
            4: (16, 256),
        }
        images = torch.randn(2, 3, 64, 64)

        for factor, (latent_size, token_count) in expected_sizes.items():
            with self.subTest(factor=factor):
                model = VAE(
                    VAEConfig(latent_channels=32, downsample_factor=factor),
                    image_size=64,
                ).eval()
                with torch.inference_mode():
                    mean, _ = model.encode(images)
                    tokens = model.encode_tokens(images)
                    reconstruction = model.decode_tokens(tokens)

                self.assertEqual(
                    tuple(mean.shape),
                    (2, 32, latent_size, latent_size),
                )
                self.assertEqual(tuple(tokens.shape), (2, token_count, 32))
                self.assertEqual(tuple(reconstruction.shape), (2, 3, 64, 64))

    def test_vae_can_use_a_configured_latent_width(self) -> None:
        model = VAE(
            VAEConfig(latent_channels=32),
            image_size=IMG_SIZE,
        ).eval()
        images = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

        with torch.inference_mode():
            mean, logvar = model.encode(images)

        self.assertEqual(tuple(mean.shape), (2, 32, 8, 8))
        self.assertEqual(tuple(logvar.shape), (2, 32, 8, 8))

    def test_vae_can_preserve_every_final_encoder_channel(self) -> None:
        model = VAE(
            VAEConfig(latent_channels=None),
            image_size=IMG_SIZE,
        ).eval()
        images = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

        with torch.inference_mode():
            encoded_parameters = model.encoder(images)
            mean, logvar = model.encode(images)

        self.assertEqual(model.latent_channels, 128)
        self.assertEqual(tuple(encoded_parameters.shape), (2, 256, 8, 8))
        self.assertEqual(tuple(mean.shape), (2, 128, 8, 8))
        self.assertEqual(tuple(logvar.shape), (2, 128, 8, 8))

    def test_token_and_tensor_vae_paths_match(self) -> None:
        model = VAE(
            VAEConfig(latent_channels=32, downsample_factor=4),
            image_size=64,
        ).eval()
        images = torch.randn(2, 3, 64, 64)

        with torch.inference_mode():
            mean, _ = model.encode(images)
            tensor_reconstruction = model.decode(mean)
            token_reconstruction = model.decode_tokens(model.encode_tokens(images))

        torch.testing.assert_close(
            token_reconstruction,
            tensor_reconstruction,
        )

    def test_flow_matching_accepts_256_latent_tokens(self) -> None:
        config = FlowMatchingModelConfig(num_tokens=256)
        model = FlowMatching(config, device="cpu").eval()
        xt = torch.randn(2, config.num_tokens, config.latent_channels)
        time = torch.rand(2, 1)
        labels = torch.eye(config.category_num)[:2]

        with torch.inference_mode():
            velocity = model(xt, time, labels)

        self.assertEqual(tuple(velocity.shape), tuple(xt.shape))

    def test_structural_loss_is_zero_for_identical_images(self) -> None:
        images = torch.rand(2, 3, 64, 64)
        self.assertAlmostEqual(structural_similarity(images, images).item(), 1.0)

        result = vae_loss(
            images,
            images,
            torch.zeros(2, 32, 8, 8),
            torch.zeros(2, 32, 8, 8),
            kl_weight=0.0,
            l1_weight=0.5,
            ssim_weight=0.2,
        )
        self.assertAlmostEqual(result.total.item(), 0.0)
        self.assertAlmostEqual(result.reconstruction.item(), 0.0)
        self.assertAlmostEqual(result.kl.item(), 0.0)

    def test_lpips_is_zero_for_identical_images_and_differentiable(self) -> None:
        model = LPIPSLoss(device="cpu")
        target = torch.rand(2, 3, 32, 32)
        prediction = target.clone().requires_grad_(True)
        loss = model(prediction, target)
        loss.backward()

        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_vae_loader_restores_saved_architecture(self) -> None:
        config = VAEConfig(downsample_factor=4)
        model = VAE(config, image_size=64)
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save(
                {
                    "model": model.state_dict(),
                    "vae_config": asdict(config),
                    "image_size": 64,
                },
                checkpoint.name,
            )
            restored = load_vae_from_ckpt(checkpoint.name)

        self.assertEqual(restored.config.downsample_factor, 4)
        self.assertEqual(restored.latent_size, 16)


if __name__ == "__main__":
    unittest.main()
