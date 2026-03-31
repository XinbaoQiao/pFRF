import copy
import ssl
from typing import Literal

import torch
import torchvision.transforms as transforms
import tqdm
import wilds
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseRealDataset

ssl._create_default_https_context = ssl._create_unverified_context


class Waterbirds(BaseRealDataset):

    def __init__(
        self,
        split: str = "train",
        res=256,
        crop_res: int = 256,
        crop_mode: Literal["center", "random"] = "center",
        data_root: str = "datasets",
    ):

        super().__init__()

        self.class_names = [
            "Land Bird",
            "Water Bird",
        ]

        self.num_classes = 2

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

        self.full_ds = wilds.get_dataset(
            dataset="waterbirds", root_dir=f"{data_root}/waterbirds", download=True
        ).get_subset(split=split, transform=self.transform)

        self.ds = copy.deepcopy(self.full_ds)
        self.targets = self._extract_targets_fast()
        if self.targets is None:
            self.targets = self.get_targets()
        self.full_ds.targets = self.targets

    def __getitem__(self, index):

        image, label, background = self.ds.__getitem__(index)
        return image, label

    def __len__(self):
        return len(self.full_ds)

    def _extract_targets_fast(self):
        if hasattr(self.ds, "y_array"):
            arr = self.ds.y_array
            if torch.is_tensor(arr):
                return [int(v) for v in arr.reshape(-1).tolist()]
            return [int(v) for v in list(arr)]
        if hasattr(self.ds, "dataset") and hasattr(self.ds, "indices") and hasattr(self.ds.dataset, "y_array"):
            y_all = self.ds.dataset.y_array
            idx = self.ds.indices
            if torch.is_tensor(idx):
                idx = [int(v) for v in idx.reshape(-1).tolist()]
            else:
                idx = [int(v) for v in list(idx)]
            if torch.is_tensor(y_all):
                return [int(y_all[i].item()) for i in idx]
            return [int(y_all[i]) for i in idx]
        return None

    def get_targets(self) -> Tensor:
        targets = []
        loader = DataLoader(self.ds, num_workers=16, batch_size=16)
        for image, label, background in tqdm.tqdm(loader, desc="Getting targets..."):
            targets.append(label)

        targets = torch.cat(targets)
        return targets
