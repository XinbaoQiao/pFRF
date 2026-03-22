from __future__ import annotations

import copy

import torch


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def state_dict_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    total = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            total += tensor_bytes(v)
    return int(total)


def linear_head_step_flops(batch_size: int, in_dim: int, out_dim: int) -> int:
    forward = 2 * batch_size * in_dim * out_dim
    backward = 2 * batch_size * in_dim * out_dim
    return int(forward + backward)


class ProfileMeter:
    def __init__(self):
        self.local_flops = 0
        self.bytes_up = 0
        self.bytes_down = 0
        self.flops_breakdown = {
            "base": 0,
            "extra": 0,
            "setup": 0,
        }

    def add_local_flops(self, value: int, bucket: str = "base"):
        v = int(value)
        self.local_flops += v
        if bucket not in self.flops_breakdown:
            self.flops_breakdown[bucket] = 0
        self.flops_breakdown[bucket] += v

    def add_up(self, value: int):
        self.bytes_up += int(value)

    def add_down(self, value: int):
        self.bytes_down += int(value)

    @property
    def bytes_total(self) -> int:
        return int(self.bytes_up + self.bytes_down)

    def merge_flops_breakdown(self, other: dict[str, int]):
        for k, v in other.items():
            iv = int(v)
            if k not in self.flops_breakdown:
                self.flops_breakdown[k] = 0
            self.flops_breakdown[k] += iv
            self.local_flops += iv

    def flops_breakdown_dict(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.flops_breakdown.items()}

    def as_dict(self) -> dict:
        return {
            "local_flops": int(self.local_flops),
            "flops_breakdown": copy.deepcopy(self.flops_breakdown_dict()),
            "bytes_up": int(self.bytes_up),
            "bytes_down": int(self.bytes_down),
            "bytes_total": int(self.bytes_total),
        }
