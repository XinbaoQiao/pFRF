import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

from main_fed import train_adapter_probe  # noqa: E402
from data.dataloaders import get_dataset, resolve_dataset_resolution  # noqa: E402
from models import get_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--data_root",
        default=os.path.join(os.path.dirname(PROJECT_ROOT), "datasets"),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval_epochs", type=int, default=1000)
    parser.add_argument("--eval_lr", type=float, default=1e-3)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_num_workers", type=int, default=2)
    parser.add_argument("--adapter_weight_decay", type=float, default=0.0)
    parser.add_argument("--adapter_reduction", type=int, default=16)
    parser.add_argument("--adapter_min_dim", type=int, default=8)
    parser.add_argument("--adapter_scope", default="all")
    parser.add_argument("--adapter_last_n", type=int, default=0)
    parser.add_argument("--adapter_feature_anchor_weight", type=float, default=0.0)
    parser.add_argument("--adapter_view_feature_weight", type=float, default=0.0)
    parser.add_argument("--adapter_view_logit_weight", type=float, default=0.0)
    parser.add_argument("--adapter_view_kl_temperature", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    effective_real_res, effective_crop_res = resolve_dataset_resolution(
        name=args.dataset,
        res=256,
        crop_res=224,
    )
    _, test_dataset = get_dataset(
        name=args.dataset,
        res=effective_real_res,
        crop_res=effective_crop_res,
        train_crop_mode="random",
        data_root=args.data_root,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        pin_memory=True,
        drop_last=False,
    )

    payload_path = os.path.join(args.source_dir, "artifacts", "data.pth")
    if not os.path.exists(payload_path):
        payload_path = os.path.join(args.source_dir, "data.pth")
    payload = torch.load(payload_path, map_location="cpu")
    syn_images = payload["images"]
    syn_labels = payload["labels"]
    num_classes = int(torch.unique(syn_labels).numel())

    backbone, num_feats = get_model(name=args.model, distributed=False)

    out = train_adapter_probe(
        syn_images=syn_images,
        syn_labels=syn_labels,
        test_loader=test_loader,
        backbone=backbone,
        model_name=args.model,
        normalize=test_dataset.normalize,
        num_feats=num_feats,
        num_classes=num_classes,
        crop_res=effective_crop_res,
        lr=args.eval_lr,
        epochs=args.eval_epochs,
        batch_size=args.eval_batch_size,
        weight_decay=args.adapter_weight_decay,
        reduction=args.adapter_reduction,
        min_dim=args.adapter_min_dim,
        scope=args.adapter_scope,
        last_n=args.adapter_last_n,
        feature_anchor_weight=args.adapter_feature_anchor_weight,
        view_feature_weight=args.adapter_view_feature_weight,
        view_logit_weight=args.adapter_view_logit_weight,
        view_kl_temperature=args.adapter_view_kl_temperature,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "tag": args.tag,
        "dataset": args.dataset,
        "model": args.model,
        "top1": float(out["top1"]),
        "top5": float(out["top5"]),
        "adapter_scope": str(args.adapter_scope),
        "adapter_last_n": int(args.adapter_last_n),
        "adapter_reduction": int(args.adapter_reduction),
        "adapter_min_dim": int(args.adapter_min_dim),
        "eval_lr": float(args.eval_lr),
        "eval_epochs": int(args.eval_epochs),
        "adapter_weight_decay": float(args.adapter_weight_decay),
        "adapter_feature_anchor_weight": float(args.adapter_feature_anchor_weight),
        "adapter_view_feature_weight": float(args.adapter_view_feature_weight),
        "adapter_view_logit_weight": float(args.adapter_view_logit_weight),
        "adapter_view_kl_temperature": float(args.adapter_view_kl_temperature),
        "adapter_summary": out["adapter_summary"],
    }
    with open(os.path.join(args.output_dir, f"{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save(
        {
            "fc_state_dict": out["fc_state_dict"],
            "adapter_state_dict": out["adapter_state_dict"],
            "adapter_summary": out["adapter_summary"],
            "result": result,
        },
        os.path.join(args.output_dir, f"{args.tag}.pth"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
