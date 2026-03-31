from typing import Literal

import torch
from tap import Tap


class ProjectorCfg(Tap):
    dataset: str
    distill_model: str
    eval_model: str

    job_tag: str = "distillation"
    data_root: str = "datasets"
    num_workers: int = 16

    real_res: int = 256
    crop_res: int = 224
    train_crop_mode: Literal["center", "random"] = "random"

    real_batch_size: int = 100
    syn_batch_size: int = 100

    projector_lr: float = 0.01
    projector_steps: int = 1000
    classifier_lr: float = 0.001
    classifier_steps: int = 1000
    use_projector: bool = True

    syn_data_path: str | None = None
    golden_fc_path: str | None = None

    skip_if_exists: bool = True

    device_count: int = torch.cuda.device_count()
