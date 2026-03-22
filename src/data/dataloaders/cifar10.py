from typing import Literal

import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .base import BaseRealDataset


class CIFAR10Dataset(BaseRealDataset):

    def __init__(
        self,
        split: str = "train",
        res: int = 256,
        crop_res: int = 256,
        crop_mode: Literal["center", "random"] = "center",
        data_root: str = "data/datasets",
    ):
        super().__init__()

        self.num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        self.transform = transforms.Compose(
            [
                transforms.Resize(res),
                (
                    transforms.CenterCrop(crop_res)
                    if crop_mode == "center"
                    else transforms.RandomCrop(crop_res)
                ),
                transforms.ToTensor(),
            ]
        )

        self.mean = torch.tensor(mean).reshape(1, 3, 1, 1)
        self.std = torch.tensor(std).reshape(1, 3, 1, 1)

        self.ds = torchvision.datasets.CIFAR10(
            root=f"{data_root}/cifar10",
            train=(split == "train"),
            download=True,
            transform=self.transform,
        )
        self.targets = list(self.ds.targets)
        self.class_names = self.ds.classes

    def __getitem__(self, index):
        image, label = self.ds.__getitem__(index)
        return image, label

    def __len__(self):
        return len(self.ds)
