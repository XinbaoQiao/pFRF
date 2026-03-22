from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_backbone(backbone: torch.nn.Module):
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()


def assert_head_only_trainable(backbone: torch.nn.Module, head: torch.nn.Module):
    if any(bool(p.requires_grad) for p in backbone.parameters()):
        raise RuntimeError("Backbone parameters must be frozen.")
    if not any(bool(p.requires_grad) for p in head.parameters()):
        raise RuntimeError("Linear head has no trainable parameters.")

