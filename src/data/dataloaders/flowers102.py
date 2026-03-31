from copy import deepcopy
from typing import Literal

import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseRealDataset


class Flowers102(BaseRealDataset):

    def __init__(
        self,
        split: str = "train",
        res=256,
        crop_res: int = 256,
        crop_mode: Literal["center", "random"] = "center",
        data_root: str = "datasets",
    ):

        super().__init__()

        self.num_classes = 102

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

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

        self.ds = torchvision.datasets.Flowers102(
            root=f"{data_root}/flowers102",
            split=split,
            download=True,
            transform=self.transform,
        )

        self.full_image_files = deepcopy(self.ds._image_files)
        self.full_labels = deepcopy(self.ds._labels)
        self.targets = [int(v) for v in self.full_labels]

        self.class_names = self.ds.classes
        self.class_names = [s.replace("?", "") for s in self.class_names]

    def __getitem__(self, index):

        image, label = self.ds.__getitem__(index)
        return image, label

    def __len__(self):
        return len(self.ds)
