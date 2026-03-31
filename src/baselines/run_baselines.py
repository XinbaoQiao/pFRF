from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from baselines.centralized import run_full_dataset, run_lgm
from baselines.common import ensure_dir, sanitize_path_for_log, write_json
from baselines.fedncm import build_runner as build_fedncm_runner
from baselines.federated.base_runner import BaseFederatedRunner, FederatedRunArgs
from model_resolution import align_model_resolution_inplace


FED_METHODS = ["fedavg", "fedprox", "ccvr", "fedpcl", "fedproto", "fedncm", "afl"]
CENTRAL_METHODS = ["lgm", "full_dataset"]
LIGHTWEIGHT_CACHE_MODELS = {
    "mobilenetv3_large",
    "mobileone_s4",
    "repvit_m1_5",
    "efficientformer_l1",
}


def _parse_seeds(raw: str) -> list[int]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return [int(v) for v in vals]


def _parse_optional_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    low = str(raw).strip().lower()
    if low in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if low in {"0", "false", "f", "no", "n", "off"}:
        return False
    if low in {"auto", "default", "none"}:
        return None
    raise argparse.ArgumentTypeError(
        "Expected one of: true/false/auto"
    )


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
    parser.add_argument("--fl_optimizer", type=str, default="sgd")
    parser.add_argument("--fl_scheduler", type=str, default="constant")
    parser.add_argument("--local_momentum", type=float, default=0.0)
    parser.add_argument("--local_weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--partition", type=str, default="dirichlet")
    parser.add_argument("--non_iid", dest="non_iid", action="store_true")
    parser.add_argument("--iid", dest="non_iid", action="store_false")
    parser.set_defaults(non_iid=True)
    parser.add_argument("--partition_train_ratio", type=float, default=0.75)
    parser.add_argument("--partition_alpha", type=float, default=None)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.01)
    parser.add_argument("--dirichlet_balance", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dirichlet_min_size", type=int, default=1)
    parser.add_argument("--shard_per_client", type=int, default=2)
    parser.add_argument("--classes_per_client", type=int, default=2)
    parser.add_argument("--patience_rounds", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--warmup_rounds", type=int, default=10)
    parser.add_argument("--join_ratio", type=float, default=1.0)
    parser.add_argument("--random_join_ratio", dest="random_join_ratio", action="store_true")
    parser.add_argument("--fixed_join_ratio", dest="random_join_ratio", action="store_false")
    parser.set_defaults(random_join_ratio=False)
    parser.add_argument("--client_drop_rate", type=float, default=0.0)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_num_workers", type=int, default=16)
    parser.add_argument("--eval_on_global_test", dest="eval_on_global_test", action="store_true")
    parser.add_argument("--eval_on_client_split", dest="eval_on_global_test", action="store_false")
    parser.set_defaults(eval_on_global_test=True)
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
    parser.add_argument("--proto_lamda", type=float, default=1.0)
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
    parser.add_argument(
        "--cache_features",
        type=_parse_optional_bool,
        nargs="?",
        const=True,
        default=None,
        help="Enable feature caching with true/false, or omit to use model-specific defaults.",
    )
    parser.add_argument("--train_mode", type=str, default="linear_probe")
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--full_dataset_output_layout", type=str, default="legacy")
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
        "mobilenetv3-large": "mobilenetv3_large",
        "mobilenetv3_large": "mobilenetv3_large",
        "mobileone-s4": "mobileone_s4",
        "mobileone_s4": "mobileone_s4",
        "repvit-m1.5": "repvit_m1_5",
        "repvit_m1.5": "repvit_m1_5",
        "repvit-m1_5": "repvit_m1_5",
        "repvit_m1_5": "repvit_m1_5",
        "efficientformer-l1": "efficientformer_l1",
        "efficientformer_l1": "efficientformer_l1",
    }
    return alias.get(low, low)


def _normalize_train_mode(name: str) -> str:
    low = str(name).lower().strip()
    alias = {
        "linearprobe": "linear_probe",
        "linear_probe": "linear_probe",
        "linearprobing": "linear_probe",
        "lp": "linear_probe",
        "finetune": "finetune",
        "fine_tune": "finetune",
        "finetuning": "finetune",
        "ft": "finetune",
    }
    return alias.get(low, low)


def _resolve_cache_features_default(model: str, cache_features: bool | None) -> bool:
    if cache_features is not None:
        return bool(cache_features)
    return model in LIGHTWEIGHT_CACHE_MODELS


def _full_dataset_mode_dir(train_mode: str) -> str:
    if train_mode == "linear_probe":
        return "linearprobing"
    if train_mode == "finetune":
        return "finetuning"
    raise ValueError(f"Unsupported train_mode for full_dataset: {train_mode}")


def _summary_output_path(args, methods: list[str]) -> str:
    if (
        len(methods) == 1
        and methods[0] == "full_dataset"
        and str(getattr(args, "full_dataset_output_layout", "legacy")).lower() == "by_mode"
    ):
        mode_dir = _full_dataset_mode_dir(str(getattr(args, "train_mode", "linear_probe")))
        return os.path.join(
            args.output_root,
            "full_dataset",
            f"{mode_dir}_meta",
            "summary",
            f"{args.dataset}_{args.model}.json",
        )
    return os.path.join(args.output_root, "summary", f"{args.dataset}_{args.model}.json")


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
        fl_optimizer=args.fl_optimizer,
        fl_scheduler=args.fl_scheduler,
        local_momentum=args.local_momentum,
        local_weight_decay=args.local_weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=args.use_amp,
        num_clients=args.num_clients,
        partition=args.partition,
        non_iid=args.non_iid,
        partition_train_ratio=args.partition_train_ratio,
        partition_alpha=args.partition_alpha,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_balance=args.dirichlet_balance,
        dirichlet_min_size=args.dirichlet_min_size,
        shard_per_client=args.shard_per_client,
        classes_per_client=args.classes_per_client,
        join_ratio=args.join_ratio,
        random_join_ratio=args.random_join_ratio,
        client_drop_rate=args.client_drop_rate,
        patience_rounds=args.patience_rounds,
        min_delta=args.min_delta,
        warmup_rounds=args.warmup_rounds,
        eval_batch_size=args.eval_batch_size,
        eval_num_workers=args.eval_num_workers,
        eval_on_global_test=args.eval_on_global_test,
        local_num_workers=args.local_num_workers,
        prox_mu=args.prox_mu,
        afl_ri_reg=args.afl_ri_reg,
        afl_clean_reg=args.afl_clean_reg,
        kd_weight=args.kd_weight,
        kd_temp=args.kd_temp,
        pcl_weight=args.pcl_weight,
        pcl_temp=args.pcl_temp,
        pcl_momentum=args.pcl_momentum,
        proto_lamda=args.proto_lamda,
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
    args.train_mode = _normalize_train_mode(args.train_mode)
    args.full_dataset_output_layout = str(args.full_dataset_output_layout).lower().strip()
    args.dirichlet_balance = bool(int(args.dirichlet_balance))
    if args.partition_alpha is None:
        args.partition_alpha = float(args.dirichlet_alpha)
    args.cache_features = _resolve_cache_features_default(args.model, args.cache_features)
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
            args.local_lr = 5e-2
        else:
            args.local_lr = 1e-3
    if args.ccvr_calib_lr is None:
        args.ccvr_calib_lr = 1e-3
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
    summary_path = _summary_output_path(args, methods)
    ensure_dir(os.path.dirname(summary_path))
    write_json(summary_path, {"methods": methods, "summaries": summaries})
    print(f"Saved summary to {sanitize_path_for_log(summary_path, project_root=PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
