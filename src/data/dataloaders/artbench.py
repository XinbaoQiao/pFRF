import copy
import os
import shutil
import tarfile
import urllib.request
from copy import deepcopy
from typing import Literal

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseRealDataset


class ArtBench(BaseRealDataset):

    def __init__(
        self,
        split: str = "train",
        res=256,
        crop_res: int = 256,
        crop_mode: Literal["center", "random"] = "center",
        data_root: str = "datasets",
    ):

        super().__init__()
        self.data_root = data_root

        self.class_names = [
            "Art Nouveau",
            "Baroque",
            "Expressionism",
            "Impressionism",
            "Post-Impressionism",
            "Realism",
            "Renaissance",
            "Romanticism",
            "Surrealism",
            "Ukiyo-e",
        ]

        self.verify_files()

        self.num_classes = 10

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose(
            [
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

        self.full_ds = torchvision.datasets.ImageFolder(
            root="{}/artbench/{}".format(data_root, split), transform=self.transform
        )
        self.ds = copy.deepcopy(self.full_ds)
        self.targets = self.ds.targets

    def __getitem__(self, index):

        image, label = self.ds.__getitem__(index)
        return image, label

    def __len__(self):
        return len(self.ds)

    def verify_files(self):

        artbench_root = os.path.join(self.data_root, "artbench")
        train_dir = os.path.join(artbench_root, "train")
        test_dir = os.path.join(artbench_root, "test")

        if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
            os.makedirs(artbench_root, exist_ok=True)

            print("Downloading ArtBench (this may take some time)...")
            urllib.request.urlretrieve(
                "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar",
                os.path.join(artbench_root, "artbench-10-imagefolder-split.tar"),
            )

            print("Extracting ArtBench (this may take some time)...")
            with tarfile.open(os.path.join(artbench_root, "artbench-10-imagefolder-split.tar")) as tar:
                tar.extractall(path=artbench_root)

            os.rename(os.path.join(artbench_root, "artbench-10-imagefolder-split", "train"), train_dir)
            os.rename(os.path.join(artbench_root, "artbench-10-imagefolder-split", "test"), test_dir)
            shutil.rmtree(os.path.join(artbench_root, "artbench-10-imagefolder-split"))
            os.remove(os.path.join(artbench_root, "artbench-10-imagefolder-split.tar"))

            print("ArtBench download complete!")
