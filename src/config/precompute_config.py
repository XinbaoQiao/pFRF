from typing import Literal

import torch
from tap import Tap


class PrecomputeCfg(Tap):
    dataset: str
    model: str

    data_root: str = "datasets"
    job_tag: str = "distillation"

    num_workers: int = 16
    batch_size: int = 256

    real_res: int = 256
    crop_res: int = 224
    train_crop_mode: Literal["center", "random"] = "center"

    train_epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0

    eval_it: int = 1

    device_count: int = torch.cuda.device_count()
