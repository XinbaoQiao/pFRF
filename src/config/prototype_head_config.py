from typing import Literal

import torch
from tap import Tap


class PrototypeHeadCfg(Tap):
    dataset: str
    distill_model: str
    eval_model: str

    job_tag: str = "distillation"
    syn_data_path: str | None = None
    barycenter_path: str | None = None
    data_root: str = "datasets"
    num_workers: int = 16
    real_batch_size: int = 100
    syn_batch_size: int = 100
    real_res: int = 256
    crop_res: int = 224
    train_crop_mode: Literal["center", "random"] = "random"

    projector_lr: float = 1e-3
    projector_steps: int = 1000
    projector_weight_decay: float = 0.0
    align_weight: float = 1.0
    ce_weight: float = 1.0
    temperature: float = 0.07
    use_multi_prototypes: bool = True
    projector_type: Literal["linear", "mlp"] = "linear"
    projector_hidden_dim: int = 2048
    projector_init: Literal["random", "ridge"] = "random"
    projector_init_ridge: float = 1e-3
    projector_init_max_samples: int = 10000
    auto_temperature: bool = False
    temperature_min: float = 0.03
    temperature_max: float = 0.3
    temperature_steps: int = 10
    use_logit_prior: bool = False
    prior_tau: float = 1.0
    auto_prior_tau: bool = True
    prior_tau_min: float = 0.0
    prior_tau_max: float = 2.0
    prior_tau_steps: int = 9
    save_readable_json: bool = True

    skip_if_exists: bool = True

    device_count: int = torch.cuda.device_count()
