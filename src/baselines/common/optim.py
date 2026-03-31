from __future__ import annotations

import torch


def scaled_linear_head_lr(base_lr: float, batch_size: int, ref_batch_size: int = 256) -> float:
    del batch_size, ref_batch_size
    return float(base_lr)


def build_linear_head_optimizer(
    parameters,
    base_lr: float,
    batch_size: int,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
) -> tuple[torch.optim.Optimizer, float]:
    effective_lr = float(base_lr)
    optimizer = torch.optim.Adam(
        list(parameters),
        lr=effective_lr,
        weight_decay=float(weight_decay),
    )
    return optimizer, float(effective_lr)


def build_sgd_linear_head_optimizer(
    parameters,
    base_lr: float,
    batch_size: int,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
) -> tuple[torch.optim.Optimizer, float]:
    effective_lr = float(base_lr)
    optimizer = torch.optim.SGD(
        list(parameters),
        lr=effective_lr,
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    return optimizer, float(effective_lr)


def build_cosine_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(total_epochs), 1),
        eta_min=0.0,
    )


def build_constant_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
