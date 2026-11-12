from utils import get_logger, Timer
from torch.utils.data import DataLoader, Dataset
import os
import config
import torch
from PIL import Image
import numpy as np
import torchvision

logger = get_logger()


class ImgDataset(Dataset):
    def __init__(self, data_dir: str = ""):
        super().__init__()

        # check if dir exists
        if not os.path.exists(data_dir):
            err: str = f"{data_dir} does not exist."
            logger.error(err)
            raise FileNotFoundError(err)

        # data dir
        self.data_dir: str = data_dir
        logger.info(f"data_dir:{self.data_dir}")

        # data list, [{"category":"", "path":""}]
        self.data_list: list = []
        self.category_set: set[str] = []

        # list all dir under self.data_dir
        category_dir_list: list[str] = os.listdir(self.data_dir)
        for category_dir in category_dir_list:
            # update category set
            self.category_set.append(category_dir)

            category_full_path: str = os.path.join(self.data_dir, category_dir)
            for img_file in os.listdir(category_full_path):
                one_pair_data: dict = {}  # {"category":"", "path":""}
                one_pair_data["path"] = os.path.join(category_full_path, img_file)
                one_pair_data["category"] = category_dir
                self.data_list.append(one_pair_data)

        # check
        if self.data_list.__len__() <= 0:
            err: str = f"no category dir in {self.data_dir}"
            logger.error(err)
            raise FileNotFoundError(err)

        self.category_num: int = len(self.category_set)
        logger.info(
            f"category list:{self.category_set}\ntotal category num:{self.category_num}"
        )
        self.category_to_index: dict = {
            name: index for index, name in enumerate(self.category_set)
        }
        logger.info(f"{self.category_to_index}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        # {"category":"", "path":""}
        one_pair_data = self.data_list[index]
        image_tensor = None
        # load image
        try:
            image_tensor = torchvision.io.read_image(one_pair_data["path"]) / 255.5
        except Exception as e:
            logger.error(f"error:{str(e)}")

        if image_tensor is None:
            raise RuntimeError(f"fail to load image {one_pair_data["path"]}")
        # from HWC to CHW, and normalize
        label_one_hot: torch.Tensor = self.label_2_one_hot(one_pair_data["category"])
        return image_tensor, label_one_hot

    def label_2_one_hot(self, label: str = "airplane") -> torch.Tensor:
        one_hot_tensor: torch.Tensor = torch.zeros(size=[self.category_num])

        if label in self.category_set:
            one_hot_tensor[self.category_to_index[label]] = 1.0
        return one_hot_tensor


if __name__ == "__main__":
    fm_data_cfg = config.DataConfig()
    img_train_dataset = ImgDataset(fm_data_cfg.train_dir)
    img_test_dataset = ImgDataset(fm_data_cfg.test_dir)

    img_tensor, label_one_hot_tensor = img_train_dataset.__getitem__(20)
    logger.info(f"{img_tensor}")
    logger.info(f"{label_one_hot_tensor}")
    logger.info(f"{img_tensor.shape}")
    logger.info(f"{label_one_hot_tensor.shape}")
    logger.info(f"total len:{img_train_dataset.__len__()}")
    logger.info(f"total len:{img_test_dataset.__len__()}")

    train_data_loader = DataLoader(
        dataset=img_train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        prefetch_factor=8,
        persistent_workers=16,
    )

    timer = Timer()
    timer.start()
    for index, (img_batch, label_batch) in enumerate(train_data_loader):
        # logger.info(f"img_batch:{img_batch.shape}")
        img_batch
        label_batch

    elapsed = timer.stop()
    logger.info(f"elapsed {elapsed} while loop all train dataset")
