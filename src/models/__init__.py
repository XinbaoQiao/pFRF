from typing import Tuple

import torch
import torch.nn as nn
import os
from torch.utils.checkpoint import checkpoint

try:
    import clip
except Exception:
    clip = None

try:
    import timm
except Exception:
    timm = None

try:
    from transformers import AutoModel, pipeline
except Exception:
    AutoModel = None
    pipeline = None

try:
    from torchvision.models import resnet50
except Exception:
    resnet50 = None

from .dino import vision_transformer
from .dino.utils import load_pretrained_weights
from .lambda_layer import LambdaLayer
from .linear_classifier import LinearClassifier
from .moco_vision_tansformer import VisionTransformerMoCoV3


class ActivationCheckpointBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        if torch.is_grad_enabled() and isinstance(x, torch.Tensor) and x.requires_grad:
            return checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


def enable_activation_checkpointing(model: nn.Module, name: str) -> tuple[nn.Module, bool, str]:
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        return model, True, "set_grad_checkpointing(True)"

    if hasattr(model, "transformer") and hasattr(model.transformer, "resblocks"):
        blocks = model.transformer.resblocks
        if isinstance(blocks, nn.Sequential):
            model.transformer.resblocks = nn.Sequential(
                *(ActivationCheckpointBlock(block) for block in blocks)
            )
            return model, True, "transformer.resblocks"
        if isinstance(blocks, nn.ModuleList):
            model.transformer.resblocks = nn.ModuleList(
                [ActivationCheckpointBlock(block) for block in blocks]
            )
            return model, True, "transformer.resblocks"

    if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList):
        model.blocks = nn.ModuleList([ActivationCheckpointBlock(block) for block in model.blocks])
        return model, True, "blocks"

    resnet_stage_names = ["layer1", "layer2", "layer3", "layer4"]
    if all(hasattr(model, stage_name) for stage_name in resnet_stage_names):
        for stage_name in resnet_stage_names:
            stage = getattr(model, stage_name)
            if isinstance(stage, nn.Module):
                setattr(model, stage_name, ActivationCheckpointBlock(stage))
        return model, True, "layer1-4"

    return model, False, "unsupported"


def _load_mocov3_resnet50() -> nn.Module:
    if resnet50 is None:
        raise ModuleNotFoundError(
            "torchvision is not installed. Install torchvision to use mocov3_resnet50."
        )
    ckpt = torch.hub.load_state_dict_from_url(
        "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar",
        map_location="cpu",
        progress=False,
        check_hash=False,
        file_name="mocov3_r50_1000ep.pth.tar",
    )
    state = ckpt.get("state_dict", ckpt)
    clean_state = {}
    prefix = "module.base_encoder."
    for k, v in state.items():
        if k.startswith(prefix) and not k.startswith(prefix + "fc."):
            clean_state[k[len(prefix):]] = v
    model = resnet50(weights=None)
    model.load_state_dict(clean_state, strict=False)
    model.fc = nn.Identity()
    return model


def get_model(name: str, distributed: bool) -> Tuple[nn.Module, int]:
    if name == "dino_vits8":
        model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        num_feat = 384
    elif name == "dino_vits16":
        model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
        num_feat = 384
    elif name == "dino_vitb8":
        model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        num_feat = 768
    elif name == "dino_vitb16":
        model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        num_feat = 768
    elif name == "dinov2_vits":
        hub_dir = torch.hub.get_dir()
        local_repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
        if os.path.isfile(os.path.join(local_repo, "hubconf.py")):
            model = torch.hub.load(local_repo, "dinov2_vits14", source="local")
        else:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        num_feat = 384
    elif name == "dinov2_vitb":
        hub_dir = torch.hub.get_dir()
        local_repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
        if os.path.isfile(os.path.join(local_repo, "hubconf.py")):
            model = torch.hub.load(local_repo, "dinov2_vitb14", source="local")
        else:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        num_feat = 768
    elif name == "dinov2_vitl":
        hub_dir = torch.hub.get_dir()
        local_repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
        if os.path.isfile(os.path.join(local_repo, "hubconf.py")):
            model = torch.hub.load(local_repo, "dinov2_vitl14", source="local")
        else:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        num_feat = 1024
    elif name == "dinov3_vitb16":
        if pipeline is None:
            raise ModuleNotFoundError(
                "transformers is not installed. Install transformers to use dinov3_vitb16."
            )
        model = pipeline(
            model="facebook/dinov3-vitb16-pretrain-lvd1689m",
            task="image-feature-extraction",
        ).model
        model = WrappedModel(model)
        num_feat = 768
    elif name == "clip_resnet50":
        model = _clip_load("RN50")[0].visual.float().cuda()
        num_feat = 1024
    elif name == "clip_resnet101":
        model = _clip_load("RN101")[0].visual.float().cuda()
        num_feat = 512
    elif name == "clip_vitb32":
        model = _clip_load("ViT-B/32")[0].visual.float().cuda()
        num_feat = 512
    elif name == "clip_vitb":
        model = _clip_load("ViT-B/16")[0].visual.float().cuda()
        num_feat = 512
    elif name == "clip_vitl":
        model = _clip_load("ViT-L/14")[0].visual.float().cuda()
        num_feat = 768
    elif name == "eva02_vitl":
        if timm is None:
            raise ModuleNotFoundError(
                "timm is not installed. Install timm to use eva02 models."
            )
        model = timm.create_model("eva02_large_patch14_224.mim_in22k", pretrained=True)
        num_feat = 1024
    elif name == "eva02_vitb":
        if timm is None:
            raise ModuleNotFoundError(
                "timm is not installed. Install timm to use eva02 models."
            )
        model = timm.create_model("eva02_base_patch14_224.mim_in22k", pretrained=True)
        num_feat = 768
    elif name == "eva02_vits":
        if timm is None:
            raise ModuleNotFoundError(
                "timm is not installed. Install timm to use eva02 models."
            )
        model = timm.create_model("eva02_small_patch14_224.mim_in22k", pretrained=True)
        num_feat = 384
    elif name == "eva02_vitt":
        if timm is None:
            raise ModuleNotFoundError(
                "timm is not installed. Install timm to use eva02 models."
            )
        model = timm.create_model("eva02_tiny_patch14_224.mim_in22k", pretrained=True)
        num_feat = 192
    elif name == "mocov3_vitb":
        model = VisionTransformerMoCoV3.from_pretrained(
            "nyu-visionx/moco-v3-vit-b", num_classes=0
        )
        num_feat = 768
    elif name == "mocov3_vitl":
        model = VisionTransformerMoCoV3.from_pretrained(
            "nyu-visionx/moco-v3-vit-l", num_classes=0
        )
        num_feat = 1024
    elif name == "mocov3_resnet50":
        model = _load_mocov3_resnet50()
        num_feat = 2048
    else:
        raise NotImplementedError("Model {} not implemented".format(name))

    if distributed:
        model = nn.DataParallel(model)
    model = model.cuda()
    # this is to disable running stats in batchnorm, dropout, etc
    model.eval()

    return model, num_feat


def get_fc(num_feats: int, num_classes: int, distributed: bool):

    fc = LinearClassifier(dim=num_feats, num_labels=num_classes).cuda()

    if distributed:
        fc = nn.DataParallel(fc)
        fc.linear = fc.module.linear

    fc.eval()

    return fc


class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.vision_model(*args, **kwargs)[1]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CLIP_ROOT = os.path.join(PROJECT_ROOT, "pretrained_models", "clip")


def _clip_load(name: str):
    if clip is None:
        raise ModuleNotFoundError(
            "clip is not installed. Install openai/CLIP to use clip models."
        )
    os.makedirs(CLIP_ROOT, exist_ok=True)
    return clip.load(name, download_root=CLIP_ROOT)
