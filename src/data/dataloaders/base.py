from typing import List

import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset


class BaseRealDataset(Dataset):

    class_names: List[str]
    mean: torch.Tensor
    std: torch.Tensor
    num_classes: int = 0
    res: int

    def __init__(self):
        pass

    def __len__(self) -> int:
        raise NotImplementedError("__len__ is not implemented for this dataset.")

    def normalize(self, x) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def denormalize(self, x) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x * std) + mean

    def get_random_reals(self, ipc=1):

        images = [[] for c in range(self.num_classes)]
        loader_iter = iter(DataLoader(self, num_workers=8, batch_size=1, shuffle=True))

        while any([len(images[c]) < ipc for c in range(self.num_classes)]):
            x, y = next(loader_iter)
            y_idx = int(y.item()) if torch.is_tensor(y) else int(y)
            if len(images[y_idx]) < ipc:
                images[y_idx].append(x)

        images = [torch.cat(l) for l in images]
        images = torch.cat(images).cuda()

        return images
