import copy
from typing import Literal

import spawrious.torch as sp
import torch
import torchvision.transforms as transforms
import tqdm
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader

from .base import BaseRealDataset


class Spawrious(BaseRealDataset):

    def __init__(
        self,
        split: str = "train",
        res=256,
        crop_res: int = 256,
        crop_mode: Literal["center", "random"] = "center",
        data_root: str = "data/datasets",
    ):

        super().__init__()

        self.class_names = [
            "Bulldog",
            "Corgi",
            "Dachshund",
            "Labrador",
        ]

        def _prepare_data_lists(
            self, train_combinations, test_combinations, root_dir, augment
        ):
            test_transforms = transforms.Compose(
                [
                    transforms.Resize(res),
                    transforms.CenterCrop(crop_res),
                    transforms.transforms.ToTensor(),
                ]
            )

            train_transforms = test_transforms

            train_data_list = self._create_data_list(
                train_combinations, root_dir, train_transforms
            )

            test_data_list = self._create_data_list(
                test_combinations, root_dir, test_transforms
            )

            return train_data_list, test_data_list

        sp.SpawriousBenchmark._prepare_data_lists = _prepare_data_lists
        sp.SpawriousBenchmark.input_shape = (3, res, res)

        spawrious_benchmark = sp.SpawriousBenchmark(
            "o2o_hard", f"{data_root}/spawrious", augment=False
        )

        self.num_classes = 4

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.mean = torch.tensor(mean).reshape(1, 3, 1, 1)
        self.std = torch.tensor(std).reshape(1, 3, 1, 1)

        if split == "train":
            self.full_ds = spawrious_benchmark.get_train_dataset()
        else:
            self.full_ds = spawrious_benchmark.get_test_dataset()

        self.ds = copy.deepcopy(self.full_ds)
        self.targets = self._extract_targets_fast()
        if self.targets is not None:
            self.full_ds.targets = self.targets

    def __getitem__(self, index):

        image, label, background = self.ds.__getitem__(index)
        return image, label

    def __len__(self):
        return len(self.ds)

    def _extract_targets_fast(self):
        def _labels_from_concat(node):
            if isinstance(node, ConcatDataset):
                labels = []
                for sub in node.datasets:
                    sub_labels = _labels_from_concat(sub)
                    if sub_labels is None:
                        return None
                    labels.extend(sub_labels)
                return labels
            if hasattr(node, "class_index") and hasattr(node, "image_paths"):
                return [int(node.class_index)] * int(len(node.image_paths))
            if hasattr(node, "targets"):
                vals = node.targets
                if vals is not None and len(vals) == len(node):
                    if torch.is_tensor(vals):
                        return [int(v) for v in vals.reshape(-1).tolist()]
                    return [int(v) for v in list(vals)]
            return None

        if hasattr(self.ds, "y_array"):
            arr = self.ds.y_array
            if torch.is_tensor(arr):
                return [int(v) for v in arr.reshape(-1).tolist()]
            return [int(v) for v in list(arr)]
        if hasattr(self.ds, "targets"):
            vals = self.ds.targets
            if vals is not None and len(vals) == len(self.ds):
                if torch.is_tensor(vals):
                    return [int(v) for v in vals.reshape(-1).tolist()]
                return [int(v) for v in list(vals)]
        if hasattr(self.ds, "dataset") and hasattr(self.ds, "indices") and hasattr(self.ds.dataset, "targets"):
            base_targets = self.ds.dataset.targets
            idx = self.ds.indices
            if torch.is_tensor(idx):
                idx = [int(v) for v in idx.reshape(-1).tolist()]
            else:
                idx = [int(v) for v in list(idx)]
            return [int(base_targets[i]) for i in idx]
        concat_labels = _labels_from_concat(self.ds)
        if concat_labels is not None and len(concat_labels) == len(self.ds):
            return concat_labels
        return None

    def get_targets(self) -> Tensor:
        targets = []
        loader = DataLoader(self.ds, num_workers=16, batch_size=16)
        for image, label, background in tqdm.tqdm(loader, desc="Getting targets..."):
            targets.append(label)

        targets = torch.cat(targets)
        return targets
