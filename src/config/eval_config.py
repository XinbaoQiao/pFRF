from typing import Literal

import torch
from tap import Tap


class EvalCfg(Tap):
    dataset: str
    model: str
    eval_model: str
    job_tag: str = "distillation"
    syn_data_path: str | None = None
    barycenter_path: str | None = None
    run_dir_override: str | None = None
    data_root: str = "datasets"
    num_workers: int = 16
    real_batch_size: int = 100
    syn_batch_size: int = 100
    real_res: int = 256
    crop_res: int = 224
    num_eval: int = 5
    eval_epochs: int = 1000
    train_crop_mode: Literal["center", "random"] = "random"
    train_mode: Literal["linear_probe", "finetune", "cosineprototype"] = "linear_probe"
    prototype_source: Literal["fl_mean", "fl_barycenter", "full_train"] = "fl_barycenter"
    prototype_temperature: float = 0.07
    auto_prototype_temperature: bool = False
    prototype_temperature_min: float = 0.01
    prototype_temperature_max: float = 1.0
    prototype_temperature_steps: int = 41
    head_lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 0.0

    device_count: int = torch.cuda.device_count()

    skip_if_exists: bool = True

    patience: int = 5
    eval_it: int = -1

    checkpoint_it: int = 100

    job_id: str | None = None
