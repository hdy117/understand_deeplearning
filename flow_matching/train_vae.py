"""Train the VAE that supplies latent tokens to flow matching."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image

from config import FMConfig
from dataset import ImgDataset
from utils import get_logger, set_seed
from vae import VAE, vae_loss

logger = get_logger()


@dataclass(frozen=True)
class ForwardResult:
    """Tensor outputs shared by the training and validation paths."""

    # 训练和验证都要跨函数使用这四个张量；命名比四元素 tuple 更不容易错位。
    reconstruction: torch.Tensor  # decoder 输出，像素值在 [0, 1]
    mean: torch.Tensor  # Gaussian posterior 的均值
    logvar: torch.Tensor  # posterior log-variance，与 mean 同形状
    latent: torch.Tensor  # 训练时为采样值，验证时可能直接使用 mean


@dataclass(frozen=True)
class LossMetrics:
    """Average loss components produced by one pass over a loader."""

    # 一个 epoch 结束后只创建一次，避免在每条样本的热路径中反复分配对象。
    loss: float  # reconstruction + kl_weight * kl
    reconstruction: float  # MSE(reconstruction, target)
    kl: float  # Gaussian posterior 与标准正态先验之间的 KL


class VAETrainer:
    """Train a VAE with one explicit owner for data, metrics, and checkpoints."""

    def __init__(
        self,
        cfg: FMConfig,
        *,
        train_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        # 运行级配置：固定随机种子，并把逻辑设备与输出路径保存到实例中。
        self.cfg: FMConfig = cfg
        set_seed(cfg.vae_train_cfg.seed)
        self.device: torch.device = torch.device(cfg.vae_train_cfg.device)
        self.output_dir: Path = Path(cfg.vae_train_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # checkpoint 和重建图使用固定的输出位置，便于训练后稳定找到产物。
        self.best_path: Path = self.output_dir / "vae_best.pt"
        self.latest_path: Path = self.output_dir / "vae_latest.pt"
        self.best_image_path: Path = self.output_dir / "best_reconstruction.png"
        self.latest_image_path: Path = self.output_dir / "latest_reconstruction.png"

        # 默认使用项目自己的 ImgDataset；测试时可注入 Subset 等轻量 Dataset。
        self.train_dataset: Dataset = train_dataset or ImgDataset(
            data_dir=cfg.data_cfg.train_dir, data_config=self.cfg.data_cfg
        )
        self.test_dataset: Dataset = test_dataset or ImgDataset(
            data_dir=cfg.data_cfg.test_dir, data_config=self.cfg.data_cfg
        )
        # DataLoader 的参数集中构造，避免训练集和验证集出现无关的设置漂移。
        loader_args: dict = dict(
            batch_size=cfg.vae_train_cfg.batch_size,
            num_workers=cfg.vae_train_cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=cfg.vae_train_cfg.num_workers > 0,
        )
        # 训练集打乱顺序并丢弃不完整尾批；验证集保持顺序且不丢样本。
        self.train_loader: DataLoader = DataLoader(
            self.train_dataset, shuffle=True, drop_last=True, **loader_args
        )
        self.test_loader: DataLoader = DataLoader(
            self.test_dataset, shuffle=False, **loader_args
        )

        # 模型、优化器和学习率调度器共同构成一次训练运行的可变核心状态。
        self.model: VAE = VAE(config=cfg.vae_cfg, image_size=cfg.train_cfg.img_size).to(
            self.device
        )
        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.vae_train_cfg.lr,
            weight_decay=cfg.vae_train_cfg.weight_decay,
        )
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.vae_train_cfg.epochs
            )
        )
        # 只有 CUDA 且启用 AMP 时才缩放梯度；CPU 路径保持普通 float32 训练。
        self.scaler: torch.amp.GradScaler = torch.amp.GradScaler(
            "cuda",
            enabled=cfg.vae_train_cfg.amp and self.device.type == "cuda",
        )
        # resume 会覆盖这两个值；从头训练时从第 0 个 epoch 和无穷大 best loss 开始。
        self.start_epoch: int = 0
        self.best_validation_loss: float = math.inf

    def run(self, resume_path: Path | None = None) -> int:
        """Train all remaining epochs and return a process exit code."""
        # resume() 返回“下一轮要开始的索引”和已观察到的最佳验证 loss。
        start_epoch, best_loss = (
            self.resume(resume_path) if resume_path else (0, math.inf)
        )
        self.start_epoch, self.best_validation_loss = start_epoch, best_loss
        self.log_run_information()

        for epoch in range(self.start_epoch, self.cfg.vae_train_cfg.epochs):
            # 每个 epoch 先完整训练一轮，再在验证集上计算指标并保留首批可视化样本。
            train_metrics = self.train_one_epoch()
            validation_metrics, images, reconstructions = self.evaluate()

            # 先记录本轮实际使用的学习率，再推进 scheduler 到下一轮时间表。
            learning_rate = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            self.log_metrics(epoch, learning_rate, train_metrics, validation_metrics)

            # latest 产物每轮覆盖；best 产物只在验证 loss 严格下降时覆盖。
            self.save_images(images, reconstructions, self.latest_image_path)
            if validation_metrics.loss < self.best_validation_loss:
                self.best_validation_loss = validation_metrics.loss
                self.save_checkpoint(self.best_path, epoch, self.best_validation_loss)
                self.save_images(images, reconstructions, self.best_image_path)
            self.save_checkpoint(self.latest_path, epoch, self.best_validation_loss)
        logger.info(f"training complete, best checkpoint:{self.best_path}")
        return 0

    def train_one_epoch(self) -> LossMetrics:
        """Run one training pass and return its average loss components."""
        # seperator
        logger.info("=========================")

        # train() 会让需要不同训练行为的层（如以后加入的 Dropout）进入训练模式。
        self.model.train()

        # 逐 batch 先累加 Python float，epoch 结束后再统一构造一次 LossMetrics。
        loss_sum = reconstruction_sum = kl_sum = 0.0
        batches: int = 0

        # ImgDataset 返回 (image, one-hot label)；VAE 重建是无监督任务，label 被忽略。
        for images, _ in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            # 训练时从 posterior 采样 latent，以保留 VAE 的随机性。
            output = self.forward(images, sample_latent=True)
            loss, reconstruction_loss, kl_loss = vae_loss(
                images,
                output.reconstruction,
                output.mean,
                output.logvar,
                kl_weight=self.cfg.vae_train_cfg.kl_weight,
            )

            # AMP 顺序不能随意调换：先 scale/backward，再 unscale，之后才能安全裁剪梯度。
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_sum += loss.detach().item()
            reconstruction_sum += reconstruction_loss.detach().item()
            kl_sum += kl_loss.detach().item()
            batches += 1

        # 空 loader 通常意味着数据路径或数据集配置有误，直接失败比返回 0 loss 更安全。
        if not batches:
            raise RuntimeError("training loader produced no batches")
        return LossMetrics(
            loss_sum / batches,
            reconstruction_sum / batches,
            kl_sum / batches,
        )

    @torch.inference_mode()
    def evaluate(
        self,
    ) -> tuple[LossMetrics, torch.Tensor, torch.Tensor]:
        """Evaluate the validation loader and retain its first batch."""
        # inference_mode 关闭 autograd，并禁止参数更新；上方装饰器负责整个验证过程。
        self.model.eval()
        loss_sum = reconstruction_sum = kl_sum = 0.0
        batches: int = 0
        images: torch.Tensor | None = None
        reconstructions: torch.Tensor | None = None
        for batch, _ in self.test_loader:
            batch = batch.to(self.device, non_blocking=True)

            # 验证使用 posterior mean，避免每次采样导致 loss 和重建图随机抖动。
            output = self.forward(batch, sample_latent=False)
            loss, reconstruction_loss, kl_loss = vae_loss(
                batch,
                output.reconstruction,
                output.mean,
                output.logvar,
                kl_weight=self.cfg.vae_train_cfg.kl_weight,
            )

            # 只保留第一个验证 batch 的 Tensor 到 CPU，作为图片对比的固定样本。
            if images is None:
                images = batch.detach().cpu()
                reconstructions = output.reconstruction.detach().float().cpu()
            loss_sum += loss.detach().item()
            reconstruction_sum += reconstruction_loss.detach().item()
            kl_sum += kl_loss.detach().item()
            batches += 1
        if not batches or images is None or reconstructions is None:
            raise RuntimeError("validation loader produced no batches")

        # 这里是 batch 均值的均值；验证尾批较小时，不等价于按样本数加权平均。
        metrics = LossMetrics(
            loss_sum / batches,
            reconstruction_sum / batches,
            kl_sum / batches,
        )
        return metrics, images, reconstructions

    def forward(self, images: torch.Tensor, *, sample_latent: bool) -> ForwardResult:
        """Run one autocast-wrapped VAE forward pass."""
        # autocast 只在“配置启用 AMP”且设备为 CUDA 时生效，CPU 会自动走普通精度。
        with torch.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.cfg.vae_train_cfg.amp and self.device.type == "cuda",
        ):
            reconstruction, mean, logvar, latent = self.model(
                images,
                sample_latent=sample_latent,
            )

        # 转换成具名结果，让调用者不必记住四元素 tuple 的字段顺序。
        return ForwardResult(reconstruction, mean, logvar, latent)

    def resume(self, path: Path) -> tuple[int, float]:
        """Restore model, optimizer, epoch, and best validation loss."""
        # checkpoint 含有普通 Python 配置字典，因此这里不能使用 weights_only=True。
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # checkpoint 保存的是最后完成的 epoch，下一次训练要从它的下一个索引开始。
        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info(f"resumed from {path} at epoch {start_epoch}")

        # 刻意不恢复 scheduler state，以保持旧 checkpoint 格式的兼容行为。
        return start_epoch, float(checkpoint["best_validation_loss"])

    def save_checkpoint(self, path: Path, epoch: int, best_loss: float) -> None:
        """Atomically save state needed to resume or restore the VAE."""
        # 先写同目录临时文件，再用 replace 提交，避免中断时留下半写 checkpoint。
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "epoch": epoch,
                "best_validation_loss": best_loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                # 模型配置和 image_size 都保存，VAE.from_checkpoint 才能重建正确结构。
                "vae_config": asdict(self.cfg.vae_cfg),
                "image_size": self.cfg.train_cfg.img_size,
                # 保留数据路径快照，便于追溯该 checkpoint 来自哪组训练数据。
                "data_config": {
                    "data_dir": str(self.cfg.data_cfg.data_dir),
                    "train_dir": str(self.cfg.data_cfg.train_dir),
                    "test_dir": str(self.cfg.data_cfg.test_dir),
                },
            },
            temporary_path,
        )
        temporary_path.replace(path)

    @torch.inference_mode()
    def save_images(
        self, images: torch.Tensor, reconstructions: torch.Tensor, path: Path
    ) -> None:
        """Save originals above their reconstructions."""
        # 上半排是原图，下半排是重建图；最多每行放 8 张，便于人工对比细节。
        count = min(len(images), len(reconstructions))
        if count:
            grid = make_grid(
                torch.cat((images[:count], reconstructions[:count])),
                nrow=max(1, min(count, 8)),
                padding=2,
            )
            save_image(grid, path)

    def log_run_information(self) -> None:
        """Log the fixed facts shared by every epoch."""
        # 参数量和 latent 形状在运行期间固定，只在训练开始时打印一次。
        parameters = sum(parameter.numel() for parameter in self.model.parameters())
        logger.info(f"device:{self.device}")
        logger.info(f"image_size:{self.cfg.train_cfg.img_size}")
        logger.info(f"latent_channels:{self.cfg.vae_cfg.latent_channels}")
        logger.info(f"latent_size:{self.model.latent_size}")
        logger.info(f"parameters:{parameters}")
        logger.info(f"train:{len(self.train_dataset)}, test:{len(self.test_dataset)}")

    @staticmethod
    def log_metrics(
        epoch: int,
        learning_rate: float,
        train: LossMetrics,
        validation: LossMetrics,
    ) -> None:
        """Log one epoch's loss components."""
        # 训练和验证使用相同前缀格式，便于日志检索与并排比较。
        logger.info(f"epoch:{epoch + 1}")
        logger.info(f"learning_rate:{learning_rate:.2e}")
        for prefix, metrics in (("train", train), ("validation", validation)):
            logger.info(f"{prefix} loss:{metrics.loss:.6f}")
            logger.info(f"{prefix} reconstruction:{metrics.reconstruction:.6f}")
            logger.info(f"{prefix} kl:{metrics.kl:.4f}")


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI."""
    # CLI 只负责选择训练入口；其余超参数由 config.py 的 dataclass 集中管理。
    parser = argparse.ArgumentParser(
        description="Train the image-to-token VAE using config.py settings."
    )
    parser.add_argument(
        "--resume", type=Path, help="optional checkpoint from which to resume"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the configured trainer."""
    # 显式返回 0，让 SystemExit 的退出码与训练成功保持一致。
    args = build_parser().parse_args(argv)
    return VAETrainer(FMConfig()).run(args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
