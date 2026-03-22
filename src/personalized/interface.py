from __future__ import annotations

import torch
import torch.nn as nn


class IdentitySemanticTranslator(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearSemanticTranslator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim))
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ResidualSemanticTranslator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.delta = nn.Linear(dim, dim)
        with torch.no_grad():
            self.delta.weight.zero_()
            self.delta.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta(x)


def build_semantic_translator(translator_type: str, dim: int) -> nn.Module:
    translator_type = str(translator_type).lower()
    if translator_type == "identity":
        return IdentitySemanticTranslator()
    if translator_type == "linear":
        return LinearSemanticTranslator(dim=dim)
    if translator_type == "residual":
        return ResidualSemanticTranslator(dim=dim)
    raise ValueError(f"Unknown translator_type: {translator_type}")


def build_semantic_interface(interface_type: str, dim: int) -> nn.Module:
    return build_semantic_translator(translator_type=interface_type, dim=dim)
