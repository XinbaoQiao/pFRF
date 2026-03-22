from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AdapterInjectionSummary:
    kind: str
    num_wrapped_modules: int
    wrapped_module_names: tuple[str, ...]
    bottleneck_dim: int


class ResidualTokenAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.down = nn.Linear(int(hidden_dim), int(bottleneck_dim))
        self.act = nn.GELU()
        self.up = nn.Linear(int(bottleneck_dim), int(hidden_dim))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


class ResidualSpatialAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, int(channels))
        self.down = nn.Conv2d(int(channels), int(bottleneck_dim), kernel_size=1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(int(bottleneck_dim), int(channels), kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


class AdapterWrappedModule(nn.Module):
    def __init__(self, module: nn.Module, adapter: nn.Module):
        super().__init__()
        self.module = module
        self.adapter = adapter
        self.adapter_enabled = True

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        if not self.adapter_enabled:
            return out
        return self.adapter(out)


def _infer_module_device_dtype(module: nn.Module) -> tuple[torch.device | None, torch.dtype | None]:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device, tensor.dtype
    return None, None


def _move_adapter_like_module(adapter: nn.Module, module: nn.Module) -> nn.Module:
    device, dtype = _infer_module_device_dtype(module)
    if device is None:
        return adapter
    return adapter.to(device=device, dtype=dtype)


def _resolve_bottleneck_dim(hidden_dim: int, reduction: int, min_dim: int) -> int:
    reduction = max(int(reduction), 1)
    min_dim = max(int(min_dim), 1)
    return max(int(hidden_dim) // reduction, min_dim)


def _selected_indices(length: int, last_n: int) -> list[int]:
    if int(last_n) <= 0 or int(last_n) >= int(length):
        return list(range(int(length)))
    start = int(length) - int(last_n)
    return list(range(start, int(length)))


def _selected_indices_from_scope(length: int, scope: str, last_n: int) -> list[int]:
    length = int(length)
    scope = str(scope).lower().strip()
    if scope in {"all", "full"}:
        return list(range(length))
    if scope in {"last_n", "tail"}:
        return _selected_indices(length, last_n=last_n)
    if scope in {"last_half", "half"}:
        keep = max(length // 2, 1)
        return list(range(length - keep, length))
    if scope in {"last_quarter", "quarter"}:
        keep = max(length // 4, 1)
        return list(range(length - keep, length))
    raise ValueError(f"Unknown adapter scope: {scope}")


def _freeze_backbone_keep_adapters(backbone: nn.Module) -> None:
    for param in backbone.parameters():
        param.requires_grad_(False)
    for module in backbone.modules():
        if isinstance(module, AdapterWrappedModule):
            for param in module.adapter.parameters():
                param.requires_grad_(True)


def trainable_adapter_parameters(backbone: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in backbone.modules():
        if isinstance(module, AdapterWrappedModule):
            params.extend([p for p in module.adapter.parameters() if p.requires_grad])
    return params


def set_adapters_enabled(backbone: nn.Module, enabled: bool) -> None:
    for module in backbone.modules():
        if isinstance(module, AdapterWrappedModule):
            module.adapter_enabled = bool(enabled)


def adapter_state_dict(backbone: nn.Module) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {}
    for name, module in backbone.named_modules():
        if isinstance(module, AdapterWrappedModule):
            for key, value in module.adapter.state_dict().items():
                payload[f"{name}.adapter.{key}"] = value.detach().cpu()
    return payload


def inject_internal_adapters(
    backbone: nn.Module,
    *,
    model_name: str,
    reduction: int = 16,
    min_dim: int = 8,
    scope: str = "all",
    last_n: int = 0,
) -> AdapterInjectionSummary:
    wrapped_names: list[str] = []
    model_name = str(model_name).lower().strip()

    if hasattr(backbone, "blocks") and isinstance(backbone.blocks, nn.ModuleList):
        hidden_dim = int(getattr(backbone, "embed_dim"))
        bottleneck_dim = _resolve_bottleneck_dim(hidden_dim, reduction=reduction, min_dim=min_dim)
        for idx in _selected_indices_from_scope(len(backbone.blocks), scope=scope, last_n=last_n):
            wrapped_block = backbone.blocks[idx]
            backbone.blocks[idx] = AdapterWrappedModule(
                wrapped_block,
                _move_adapter_like_module(
                    ResidualTokenAdapter(hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim),
                    wrapped_block,
                ),
            )
            wrapped_names.append(f"blocks.{idx}")
        kind = "vit_blocks"
    elif hasattr(backbone, "transformer") and hasattr(backbone.transformer, "resblocks"):
        resblocks = backbone.transformer.resblocks
        if not isinstance(resblocks, (nn.Sequential, nn.ModuleList)):
            raise TypeError(f"Unsupported CLIP resblocks container: {type(resblocks)!r}")
        hidden_dim = int(getattr(backbone, "transformer").width)
        bottleneck_dim = _resolve_bottleneck_dim(hidden_dim, reduction=reduction, min_dim=min_dim)
        for idx in _selected_indices_from_scope(len(resblocks), scope=scope, last_n=last_n):
            wrapped_block = resblocks[idx]
            resblocks[idx] = AdapterWrappedModule(
                wrapped_block,
                _move_adapter_like_module(
                    ResidualTokenAdapter(hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim),
                    wrapped_block,
                ),
            )
            wrapped_names.append(f"transformer.resblocks.{idx}")
        kind = "clip_resblocks"
    elif all(hasattr(backbone, stage_name) for stage_name in ("layer1", "layer2", "layer3", "layer4")):
        stage_names = ["layer1", "layer2", "layer3", "layer4"]
        selected_indices = _selected_indices_from_scope(len(stage_names), scope=scope, last_n=last_n)
        selected_names = [stage_names[idx] for idx in selected_indices]
        last_block = getattr(backbone, selected_names[-1])[-1]
        channels = int(last_block.bn3.num_features if hasattr(last_block, "bn3") else last_block.bn2.num_features)
        bottleneck_dim = _resolve_bottleneck_dim(channels, reduction=reduction, min_dim=min_dim)
        for stage_name in selected_names:
            stage = getattr(backbone, stage_name)
            stage_last_block = stage[-1]
            stage_channels = int(
                stage_last_block.bn3.num_features if hasattr(stage_last_block, "bn3") else stage_last_block.bn2.num_features
            )
            stage_bottleneck = _resolve_bottleneck_dim(stage_channels, reduction=reduction, min_dim=min_dim)
            setattr(
                backbone,
                stage_name,
                AdapterWrappedModule(
                    stage,
                    _move_adapter_like_module(
                        ResidualSpatialAdapter(channels=stage_channels, bottleneck_dim=stage_bottleneck),
                        stage,
                    ),
                ),
            )
            wrapped_names.append(stage_name)
        kind = "resnet_stages"
        bottleneck_dim = _resolve_bottleneck_dim(channels, reduction=reduction, min_dim=min_dim)
    else:
        raise NotImplementedError(f"Internal adapter injection is not implemented for model={model_name}")

    _freeze_backbone_keep_adapters(backbone)
    return AdapterInjectionSummary(
        kind=kind,
        num_wrapped_modules=len(wrapped_names),
        wrapped_module_names=tuple(wrapped_names),
        bottleneck_dim=int(bottleneck_dim),
    )
