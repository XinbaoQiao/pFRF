from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from baselines.centralized import run_full_dataset, run_lgm
from baselines.common import ensure_dir, write_json
from baselines.fedncm import build_runner as build_fedncm_runner
from baselines.federated.base_runner import BaseFederatedRunner, FederatedRunArgs
from model_resolution import align_model_resolution_inplace


FED_METHODS = ["fedavg", "fedprox", "scaffold", "fedmd", "fedntd", "fedpcl", "ccvr", "afl", "fedncm"]
CENTRAL_METHODS = ["lgm", "full_dataset"]


def _parse_seeds(raw: str) -> list[int]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return [int(v) for v in vals]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "datasets"))
    parser.add_argument("--pretrained_root", type=str, default=os.path.join(PROJECT_ROOT, "pretrained_models"))
    parser.add_argument("--output_root", type=str, default="output/baselines")
    parser.add_argument("--real_res", type=int, default=256)
    parser.add_argument("--crop_res", type=int, default=224)
    parser.add_argument("--train_crop_mode", type=str, default="random")
    parser.add_argument("--seeds", type=str, default="3407")
    parser.add_argument("--max_rounds", type=int, default=500)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--local_batch_size", type=int, default=64)
    parser.add_argument("--feature_batch_size", type=int, default=1024)
    parser.add_argument("--local_lr", type=float, default=None)
    parser.add_argument("--local_momentum", type=float, default=0.9)
    parser.add_argument("--local_weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--partition", type=str, default="dirichlet")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.01)
    parser.add_argument("--dirichlet_balance", dest="dirichlet_balance", action="store_true")
    parser.add_argument("--no_dirichlet_balance", dest="dirichlet_balance", action="store_false")
    parser.set_defaults(dirichlet_balance=True)
    parser.add_argument("--dirichlet_min_size", type=int, default=1)
    parser.add_argument("--shard_per_client", type=int, default=2)
    parser.add_argument("--classes_per_client", type=int, default=2)
    parser.add_argument("--patience_rounds", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--warmup_rounds", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_num_workers", type=int, default=16)
    parser.add_argument("--local_num_workers", type=int, default=16)
    parser.add_argument("--prox_mu", type=float, default=1e-3)
    parser.add_argument("--afl_ri_reg", type=float, default=1.0)
    parser.add_argument("--afl_clean_reg", dest="afl_clean_reg", action="store_true")
    parser.add_argument("--no_afl_clean_reg", dest="afl_clean_reg", action="store_false")
    parser.set_defaults(afl_clean_reg=True)
    parser.add_argument("--kd_weight", type=float, default=1.0)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--pcl_weight", type=float, default=0.5)
    parser.add_argument("--pcl_temp", type=float, default=0.07)
    parser.add_argument("--pcl_momentum", type=float, default=0.0)
    parser.add_argument("--ntd_weight", type=float, default=1.0)
    parser.add_argument("--ntd_temp", type=float, default=1.0)
    parser.add_argument("--scaffold_eta", type=float, default=1.0)
    parser.add_argument("--ccvr_calib_epochs", type=int, default=5)
    parser.add_argument("--ccvr_calib_samples_per_class", type=int, default=100)
    parser.add_argument("--ccvr_calib_lr", type=float, default=None)
    parser.add_argument("--ipc", type=int, default=1)
    parser.add_argument("--augs_per_batch", type=int, default=3)
    parser.add_argument("--lgm_iterations", type=int, default=5000)
    parser.add_argument("--run_lgm_train", action="store_true")
    parser.add_argument("--smoke_max_train_batches", type=int, default=0)
    parser.add_argument("--smoke_max_eval_batches", type=int, default=0)
    parser.add_argument("--cache_features", dest="cache_features", action="store_true")
    parser.add_argument("--no_cache_features", dest="cache_features", action="store_false")
    parser.set_defaults(cache_features=False)
    return parser


def _normalize_dataset_name(name: str) -> str:
    low = name.lower().strip()
    alias = {
        "imagenet1k": "imagenet-1k",
        "imagenet100": "imagenet-100",
    }
    return alias.get(low, low)


def _normalize_model_name(name: str) -> str:
    low = name.lower().strip()
    alias = {
        "dino-v2": "dinov2_vitb",
        "dinov2": "dinov2_vitb",
        "dinov2_vitb14": "dinov2_vitb",
    }
    return alias.get(low, low)


def _validate_supported_dataset(name: str):
    allowed = {
        "artbench",
        "cifar10",
        "cifar100",
        "cub2011",
        "flowers102",
        "food101",
        "imagenette",
        "imagenet-100",
        "imagenet-1k",
        "spawrious",
        "stanforddogs",
        "waterbirds",
    }
    if name not in allowed:
        raise ValueError(f"Unsupported dataset for FL baselines: {name}. Allowed: {sorted(allowed)}")


def _build_fed_args(args, method: str) -> FederatedRunArgs:
    return FederatedRunArgs(
        method=method,
        dataset=args.dataset,
        model=args.model,
        data_root=args.data_root,
        output_root=args.output_root,
        real_res=args.real_res,
        crop_res=args.crop_res,
        train_crop_mode=args.train_crop_mode,
        seeds=_parse_seeds(args.seeds),
        max_rounds=args.max_rounds,
        local_epochs=args.local_epochs,
        local_batch_size=args.local_batch_size,
        feature_batch_size=args.feature_batch_size,
        local_lr=args.local_lr,
        local_momentum=args.local_momentum,
        local_weight_decay=args.local_weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=args.use_amp,
        num_clients=args.num_clients,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_balance=args.dirichlet_balance,
        dirichlet_min_size=args.dirichlet_min_size,
        shard_per_client=args.shard_per_client,
        classes_per_client=args.classes_per_client,
        patience_rounds=args.patience_rounds,
        min_delta=args.min_delta,
        warmup_rounds=args.warmup_rounds,
        eval_batch_size=args.eval_batch_size,
        eval_num_workers=args.eval_num_workers,
        local_num_workers=args.local_num_workers,
        prox_mu=args.prox_mu,
        afl_ri_reg=args.afl_ri_reg,
        afl_clean_reg=args.afl_clean_reg,
        kd_weight=args.kd_weight,
        kd_temp=args.kd_temp,
        pcl_weight=args.pcl_weight,
        pcl_temp=args.pcl_temp,
        pcl_momentum=args.pcl_momentum,
        ntd_weight=args.ntd_weight,
        ntd_temp=args.ntd_temp,
        scaffold_eta=args.scaffold_eta,
        ccvr_calib_epochs=args.ccvr_calib_epochs,
        ccvr_calib_samples_per_class=args.ccvr_calib_samples_per_class,
        ccvr_calib_lr=args.ccvr_calib_lr,
        smoke_max_train_batches=args.smoke_max_train_batches,
        smoke_max_eval_batches=args.smoke_max_eval_batches,
        cache_features=args.cache_features,
    )


def _expand_methods(name: str) -> list[str]:
    if name == "all":
        return FED_METHODS + CENTRAL_METHODS
    if name == "all_federated":
        return list(FED_METHODS)
    if name == "all_centralized":
        return list(CENTRAL_METHODS)
    return [name]


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.dataset = _normalize_dataset_name(args.dataset)
    args.model = _normalize_model_name(args.model)
    align_model_resolution_inplace(args)
    _validate_supported_dataset(args.dataset)
    pretrained_root = os.path.abspath(args.pretrained_root)
    ensure_dir(pretrained_root)
    ensure_dir(os.path.join(pretrained_root, "hub"))
    ensure_dir(os.path.join(pretrained_root, "hf"))
    os.environ.setdefault("TORCH_HOME", pretrained_root)
    os.environ.setdefault("HF_HOME", os.path.join(pretrained_root, "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(pretrained_root, "hf"))
    methods = _expand_methods(args.method.lower())
    if args.local_lr is None:
        if all(method in FED_METHODS for method in methods):
            args.local_lr = 0.05
        else:
            args.local_lr = 1e-3
    if args.ccvr_calib_lr is None:
        args.ccvr_calib_lr = 0.05
    summaries = []
    for method in methods:
        if method in FED_METHODS:
            fed_args = _build_fed_args(args, method=method)
            runner = build_fedncm_runner(fed_args) if method == "fedncm" else BaseFederatedRunner(fed_args)
            summaries.append(runner.run())
            continue
        if method == "full_dataset":
            summaries.append(run_full_dataset(args))
            continue
        if method == "lgm":
            summaries.append(run_lgm(args))
            continue
        raise NotImplementedError(method)
    summary_path = os.path.join(args.output_root, "summary", f"{args.dataset}_{args.model}.json")
    ensure_dir(os.path.dirname(summary_path))
    write_json(summary_path, {"methods": methods, "summaries": summaries})
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
