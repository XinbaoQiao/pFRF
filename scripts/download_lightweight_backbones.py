#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


MODELS = {
    "mobilenetv3_large": ("mobilenetv3_large_100.ra_in1k", 1280),
    "mobileone_s4": ("mobileone_s4.apple_in1k", 2048),
    "repvit_m1_5": ("repvit_m1_5.dist_300e_in1k", 512),
    "efficientformer_l1": ("efficientformer_l1.snap_dist_in1k", 448),
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_root = project_root / "pretrained_models"
    parser = argparse.ArgumentParser(
        description="Pre-download ImageNet pretrained lightweight backbones into a local cache."
    )
    parser.add_argument("--pretrained_root", type=Path, default=default_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pretrained_root = args.pretrained_root.resolve()
    hf_root = pretrained_root / "hf"
    hub_root = pretrained_root / "hub"
    for path in (pretrained_root, hf_root, hub_root):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_HOME", str(pretrained_root))
    os.environ.setdefault("HF_HOME", str(hf_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_root))

    import timm

    print(f"pretrained_root={pretrained_root}")
    for alias, (timm_name, feat_dim) in MODELS.items():
        print(f"downloading {alias} <- {timm_name}")
        model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        actual_dim = getattr(model, "num_features", None)
        print(f"ready {alias}: expected_output_dim={feat_dim}, model_num_features={actual_dim}")


if __name__ == "__main__":
    main()
