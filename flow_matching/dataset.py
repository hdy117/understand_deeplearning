from dataclasses import dataclass, field

from utils import get_logger, Timer
from torch.utils.data import DataLoader, Dataset
import os
import config
import torch
import torchvision
from config import DataConfig

logger = get_logger()


@dataclass
class DataHolder:
    category: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    img_tensor: list[torch.Tensor] = field(default_factory=list)
    label_tensor: list[torch.Tensor] = field(default_factory=list)
    category_set: set = field(default_factory=set)

    def get_category_set(self) -> set:
        return self.category_set

    def load_data(self, data_dir: str, data_config: DataConfig = None) -> None:
        # pre check
        if data_config is None:
            err = "data_config is None"
            logger.error(err)
            raise ValueError(err)

        # list all dir under self.data_dir
        category_dir_list: list[str] = os.listdir(data_dir)

        # one hot tensor, [10,10]
        one_hot_tensor: torch.Tensor = torch.eye(data_config.category_num).float()

        # load all
        for category in category_dir_list:
            # update category set
            self.category_set.add(category)

            # load from dir
            category_full_path: str = os.path.join(data_dir, category)
            for img_file in os.listdir(category_full_path):
                img_full_path = os.path.join(category_full_path, img_file)
                self.category.append(category)
                self.path.append(img_full_path)

                image_tensor, label_one_hot = None, None
                # load image
                try:
                    image_tensor = (
                        torchvision.io.read_image(img_full_path).float() / 255.5
                    )
                    # one hot label
                    category_index = list(self.category_set).index(category)
                    label_one_hot: torch.Tensor = one_hot_tensor[category_index].clone()

                    # save tensors
                    self.img_tensor.append(image_tensor)
                    self.label_tensor.append(label_one_hot)
                except Exception as e:
                    logger.error(f"error:{str(e)}")

        # check
        if len(self.img_tensor) <= 0:
            err: str = f"load nothing from {self.data_dir}"
            logger.error(err)
            raise ValueError(err)

        # check
        if len(self.img_tensor) != len(self.label_tensor):
            err: str = f"img_tensor and label_tensor size not match."
            logger.error(err)
            raise ValueError(err)

    def __len__(self) -> int:
        return len(self.img_tensor)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        if index >= self.__len__():
            err: str = f"index out of range."
            logger.error(err)
            raise IndexError(err)
        return self.img_tensor[index], self.label_tensor[index]


class ImgDataset(Dataset):
    def __init__(self, data_dir: str = "", data_config: DataConfig = None):
        super().__init__()

        # check if dir exists
        if not os.path.exists(data_dir):
            err: str = f"{data_dir} does not exist."
            logger.error(err)
            raise FileNotFoundError(err)

        # data dir
        self.data_dir: str = data_dir
        logger.info(f"data_dir:{self.data_dir}")

        # data holder
        self.data_holder: DataHolder = DataHolder()
        timer_loader = Timer()

        # load data into data_list
        timer_loader.start()
        self.data_holder.load_data(data_dir=self.data_dir, data_config=data_config)
        elapsed = timer_loader.stop()
        logger.info(f"load data time:{elapsed}")

        # print something
        logger.info(
            f"category_set:{self.data_holder.category_set}, category_num:{len(self.data_holder.category_set)}"
        )
        logger.info(
            f"num img_tensor:{len(self.data_holder.img_tensor)}, num label_tensor:{len(self.data_holder.label_tensor)}"
        )

    def __len__(self) -> int:
        return len(self.data_holder)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data_holder.__getitem__(index)

    def get_category_set(self):
        return self.data_holder.get_category_set()


if __name__ == "__main__":
    fm_data_cfg = config.DataConfig()
    img_train_dataset = ImgDataset(
        data_dir=fm_data_cfg.train_dir, data_config=fm_data_cfg
    )
    img_test_dataset = ImgDataset(
        data_dir=fm_data_cfg.test_dir, data_config=fm_data_cfg
    )

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
