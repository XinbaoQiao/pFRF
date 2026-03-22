import os
import random
import re
import sys
import gc
import json
import time
from dataclasses import dataclass
from typing import Literal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PROJECT_ROOT = os.path.dirname(__file__)
# Preferred env var name is PFRF_SHARED_ROOT; keep PFEDDD_SHARED_ROOT for backward compatibility.
SHARED_FRF_ROOT = os.environ.get(
    "PFRF_SHARED_ROOT",
    os.environ.get("PFEDDD_SHARED_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "frf_project")),
)
os.environ.setdefault("TORCH_HOME", os.path.join(SHARED_FRF_ROOT, "pretrained_models"))
os.environ.setdefault("HF_HOME", os.path.join(SHARED_FRF_ROOT, "pretrained_models"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tap import Tap
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from augmentation import AugBasic
from data.feature_cache import (
    FederatedFeatureCacheSpec,
    build_feature_loader,
    discover_readable_federated_feature_cache_dirs,
    federated_feature_cache_scope_dir,
    forward_features,
    load_cached_federated_client_features,
    save_federated_client_features,
    shared_feature_cache_dir,
)
from data.dataloaders import get_dataset, resolve_dataset_resolution
from federated.server import AggregatedStats, FederatedServer
from model_resolution import align_model_resolution_inplace
from models import enable_activation_checkpointing, get_fc, get_model
from models.adapters import (
    adapter_state_dict,
    inject_internal_adapters,
    set_adapters_enabled,
    trainable_adapter_parameters,
)
from personalized import train_personalized_clients


@dataclass
class DistillArgs:
    ipc: int
    lr: float
    iterations: int
    augs_per_batch: int
    forward_batch_size: int
    distill_simultaneous_classes: int
    distill_mode: Literal["pixel", "pyramid"]
    aug_mode: Literal["standard", "none"]
    decorrelate_color: bool
    init_mode: str
    pyramid_extent_it: int
    pyramid_start_res: int
    syn_res: int
    crop_res: int
    force_save_on_cpu_acts: bool
    distill_gpu_ids: tuple[int, ...]


class FedCfg(Tap):
    experiment_name: str
    dataset: str
    model: str

    data_root: str = os.path.join(SHARED_FRF_ROOT, "datasets")
    output_root: str = "output"

    num_clients: int = 100
    partition: Literal["iid", "dirichlet"] = "dirichlet"
    dirichlet_alpha: float = 0.01
    dirichlet_balance: bool = True
    dirichlet_min_size: int = 1
    shard_per_client: int = 2
    classes_per_client: int = 2
    seed: int = 3407

    stats_batch_size: int = 1024
    stats_num_workers: int = 2

    real_res: int = 256
    crop_res: int = 224
    train_crop_mode: Literal["center", "random"] = "random"

    ipc: int = 1
    syn_res: int = 256
    distill_mode: Literal["pixel", "pyramid"] = "pyramid"
    aug_mode: Literal["standard", "none"] = "standard"
    decorrelate_color: bool = True
    init_mode: str = "noise"
    lr: float = 2e-3
    iterations: int = 5000
    augs_per_batch: int = 10
    # <= 0 means auto:
    # use all synthetic images in one forward chunk, capped at 120
    forward_batch_size: int = 0
    # Only used when ipc > 1:
    # -1 means distill all classes simultaneously (default, preserves current behavior).
    # A positive value k means distill k classes at a time, then move to the next group.
    distill_simultaneous_classes: int = -1
    pyramid_extent_it: int = 200
    pyramid_start_res: int = 1
    loss_max_samples_per_class: int = 0
    loss_interp_rounds: int = 5
    loss_interp_t: float = 0.5
    loss_init_mode: Literal["mean", "gaussian", "proxy"] = "mean"
    loss_gaussian_std: float = 1.0
    loss_update_support_weights: bool = True
    # Looser barycenter convergence defaults (easier to mark stable):
    # larger tolerances + fewer consecutive stable rounds required.
    loss_stop_tol_xi: float = 1e-3
    loss_stop_tol_a: float = 5e-3
    loss_stop_patience: int = 2
    loss_ot_solver: Literal["emd", "sinkhorn"] = "emd"
    loss_sinkhorn_reg: float = 1e-2
    loss_ot_warmstart: bool = True
    dp_enable: bool = False
    dp_epsilon: float = 1.0
    dp_delta: str = "auto"
    force_save_on_cpu_acts: bool = False
    distill_gpu_ids: str = ""
    only_build_feature_cache: bool = False
    barycenter_cache_root: str = ""
    activation_checkpointing: bool = True

    eval_epochs: int = 1000
    eval_lr: float = 0.001
    eval_batch_size: int = 256
    eval_num_workers: int = 2
    enable_adapter_eval: bool = False
    adapter_epochs: int = 0
    adapter_lr: float = 0.0
    adapter_weight_decay: float = 0.0
    adapter_reduction: int = 16
    adapter_min_dim: int = 8
    adapter_scope: Literal["all", "last_half", "last_quarter", "last_n"] = "all"
    adapter_last_n: int = 0
    adapter_feature_anchor_weight: float = 0.0
    adapter_view_feature_weight: float = 0.0
    adapter_view_logit_weight: float = 0.0
    adapter_view_kl_temperature: float = 1.0

    personalize_enable: bool = False
    personalize_eval_partition_seed_offset: int = 1
    personalize_head_epochs: int = 200
    personalize_head_lr: float = 0.001
    personalize_adapter_type: Literal["identity", "linear", "residual"] = "linear"
    personalize_adapter_epochs: int = 300
    personalize_adapter_lr: float = 0.01
    personalize_batch_size: int = 256
    personalize_identity_lambda: float = 0.1

    verify_centers: bool = False
    verify_tol: float = 1e-3


def _resolve_experiment_name(cfg: FedCfg) -> str:
    experiment_name = str(cfg.experiment_name).strip()
    if not experiment_name:
        raise ValueError("experiment_name must not be empty")
    if os.path.sep in experiment_name:
        return experiment_name

    dataset = str(cfg.dataset)
    level1 = (
        f"{dataset}_ipc{cfg.ipc}_dp{cfg.dp_enable}_"
        f"noniid_{cfg.partition}{cfg.dirichlet_alpha}_clients{cfg.num_clients}"
    )
    output_base = os.path.basename(os.path.normpath(str(cfg.output_root)))
    is_dataset_scope_root = output_base == dataset or output_base.startswith(f"{dataset}_ipc")
    if is_dataset_scope_root:
        return os.path.join(level1, experiment_name)
    return os.path.join(dataset, level1, experiment_name)


def _resolve_effective_dp_delta(dp_delta: str, count_all: int) -> float:
    return FederatedServer.resolve_dp_delta(dp_delta, count_all)


def _parse_distill_gpu_ids(raw: str) -> tuple[int, ...]:
    vals = [v.strip() for v in str(raw).split(",") if v.strip()]
    if len(vals) == 0:
        return tuple()
    if len(vals) == 1 and vals[0] == "-1":
        if not torch.cuda.is_available():
            return tuple()
        return tuple(range(torch.cuda.device_count()))
    return tuple(int(v) for v in vals)


def _sanitize_token(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "", str(x))


def _warn_if_seed_looks_like_date(seed: int) -> None:
    seed = int(seed)
    if 19000101 <= seed <= 29991231:
        print(
            f"[warning] seed={seed} looks like a YYYYMMDD date. "
            "Feature-cache suffix `_s...` uses the random seed, not a timestamp.",
            flush=True,
        )


def _preprocess_backbone_input(x: torch.Tensor, normalize, crop_res: int) -> torch.Tensor:
    target_res = int(crop_res)
    if int(x.shape[-1]) != target_res or int(x.shape[-2]) != target_res:
        x = torch.nn.functional.interpolate(
            x,
            size=(target_res, target_res),
            mode="bilinear",
            align_corners=False,
        )
    return normalize(x)




def _subset_indices(x) -> list[int]:
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.reshape(-1).tolist()]
    return [int(v) for v in list(x)]


def get_labels_for_partition(ds) -> list[int]:
    def _as_int_list(x):
        if torch.is_tensor(x):
            return [int(v) for v in x.reshape(-1).tolist()]
        return [int(v) for v in list(x)]

    if hasattr(ds, "targets") and ds.targets is not None and len(ds.targets) == len(ds):
        return _as_int_list(ds.targets)

    if hasattr(ds, "full_labels") and ds.full_labels is not None and len(ds.full_labels) == len(ds):
        return _as_int_list(ds.full_labels)

    if hasattr(ds, "data") and hasattr(ds.data, "columns") and "target" in ds.data.columns and len(ds.data) == len(ds):
        return [int(v) - 1 for v in ds.data["target"].tolist()]

    if hasattr(ds, "ds") and hasattr(ds.ds, "_labels") and len(ds.ds._labels) == len(ds):
        return _as_int_list(ds.ds._labels)

    if hasattr(ds, "ds") and hasattr(ds.ds, "y_array") and len(ds.ds.y_array) == len(ds):
        return _as_int_list(ds.ds.y_array)

    if hasattr(ds, "ds") and isinstance(ds.ds, Subset):
        base = ds.ds.dataset
        idx = _subset_indices(ds.ds.indices)
        if hasattr(base, "targets"):
            raw = [int(base.targets[i]) for i in idx]
            if hasattr(ds, "convert_label"):
                return [int(ds.convert_label(v)) for v in raw]
            return raw
        if hasattr(base, "y_array"):
            y_all = base.y_array
            if torch.is_tensor(y_all):
                return [int(y_all[i].item()) for i in idx]
            return [int(y_all[i]) for i in idx]

    if hasattr(ds, "_flat_breed_images"):
        pairs = list(ds._flat_breed_images)
        if len(pairs) == len(ds):
            return [int(v[1]) for v in pairs]

    if hasattr(ds, "full_dataset"):
        pairs = list(ds.full_dataset)
        if len(pairs) == len(ds) and len(pairs) > 0 and isinstance(pairs[0], (tuple, list)) and len(pairs[0]) >= 2:
            return [int(v[1]) for v in pairs]

    labels = []
    for i in range(len(ds)):
        item = ds[i]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            labels.append(int(item[1]))
        else:
            break
    if len(labels) == len(ds):
        return labels

    raise RuntimeError("Unable to extract labels for partitioning.")


def split_iid(n: int, k: int, rng: np.random.Generator) -> list[list[int]]:
    idx = np.arange(n)
    rng.shuffle(idx)
    splits = np.array_split(idx, k)
    return [s.tolist() for s in splits]


def split_dirichlet(labels: list[int], k: int, alpha: float, rng: np.random.Generator) -> list[list[int]]:
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels.max()) + 1
    per_client = [[] for _ in range(k)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        if len(idx_c) == 0:
            continue
        props = rng.dirichlet(alpha * np.ones(k))
        counts = (props * len(idx_c)).astype(np.int64)
        diff = len(idx_c) - int(counts.sum())
        if diff > 0:
            for i in rng.choice(k, size=diff, replace=True):
                counts[i] += 1
        if diff < 0:
            for i in rng.choice(k, size=-diff, replace=True):
                if counts[i] > 0:
                    counts[i] -= 1

        start = 0
        for i in range(k):
            take = int(counts[i])
            if take <= 0:
                continue
            per_client[i].extend(idx_c[start : start + take].tolist())
            start += take

    for i in range(k):
        rng.shuffle(per_client[i])
    return per_client


def build_partition_splits(
    dataset,
    partition: str,
    num_clients: int,
    dirichlet_alpha: float,
    seed: int,
) -> list[list[int]]:
    labels = get_labels_for_partition(dataset)
    rng = np.random.default_rng(int(seed))
    if partition == "iid":
        return split_iid(len(dataset), num_clients, rng=rng)
    if partition == "dirichlet":
        return split_dirichlet(labels, num_clients, dirichlet_alpha, rng=rng)
    raise NotImplementedError(partition)


def build_client_test_loaders(
    test_dataset,
    test_splits: list[list[int]],
    batch_size: int,
    num_workers: int,
) -> list[DataLoader]:
    loaders = []
    for idx in test_splits:
        subset = Subset(test_dataset, idx)
        loaders.append(
            DataLoader(
                subset,
                batch_size=min(int(batch_size), max(len(subset), 1)),
                shuffle=False,
                num_workers=int(num_workers),
                drop_last=False,
                pin_memory=True,
            )
        )
    return loaders


@torch.no_grad()
def compute_centers_centralized(
    train_dataset,
    backbone: nn.Module,
    num_feats: int,
    crop_res: int,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    ds_len = None
    try:
        ds_len = int(len(train_dataset))
    except Exception:
        ds_len = None
    use_pin_memory = True if (ds_len is None or ds_len <= 200000) else False
    use_persistent_workers = True if (num_workers > 0 and (ds_len is not None and ds_len <= 50000)) else False
    prefetch_factor = 2 if (num_workers > 0 and (ds_len is None or ds_len <= 50000)) else 1
    dl_kwargs = dict(
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=use_pin_memory,
        drop_last=False,
        persistent_workers=use_persistent_workers,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(
        train_dataset,
        **dl_kwargs,
    )
    num_classes = train_dataset.num_classes
    sum_per_class = torch.zeros((num_classes, num_feats), device="cuda", dtype=torch.float64)
    cnt_per_class = torch.zeros((num_classes,), device="cuda", dtype=torch.long)

    for x, y in tqdm(loader, desc="Centralized Centers", leave=False):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        x = _preprocess_backbone_input(x, train_dataset.normalize, crop_res=crop_res)
        with autocast(enabled=True):
            z = backbone(x)
        z = z.float()
        sum_per_class.index_add_(0, y, z.to(dtype=torch.float64))
        cnt_per_class += torch.bincount(y, minlength=num_classes)

    mu = (sum_per_class / cnt_per_class.clamp_min(1)[:, None]).to(dtype=torch.float32)
    return mu.detach().cpu()


def train_linear_probe(
    syn_images: torch.Tensor,
    syn_labels: torch.Tensor,
    test_loader: DataLoader,
    backbone: nn.Module,
    normalize,
    num_feats: int,
    num_classes: int,
    crop_res: int,
    lr: float,
    epochs: int,
    batch_size: int,
) -> dict:
    ds = TensorDataset(syn_images.detach().cpu(), syn_labels.detach().cpu())
    train_loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    augmentor = AugBasic(crop_res=crop_res).cuda()
    fc = get_fc(num_feats=num_feats, num_classes=num_classes, distributed=False)
    fc.train()

    optimizer = torch.optim.Adam(fc.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    scaler = GradScaler()

    for _ in tqdm(range(epochs), desc="Linear Probe Train", leave=True):
        for x, y in tqdm(train_loader, desc="Probe Epoch", leave=False):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            with autocast(enabled=True):
                with torch.no_grad():
                    x = augmentor(x)
                    x = _preprocess_backbone_input(x, normalize, crop_res=crop_res)
                    z = backbone(x)
                out = fc(z)
                loss = nn.functional.cross_entropy(out, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

    fc.eval()
    top1_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=1).cuda()
    if num_classes >= 5:
        top5_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=5).cuda()

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Probe Eval", leave=False):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            x = _preprocess_backbone_input(x, normalize, crop_res=crop_res)
            z = backbone(x)
            out = fc(z)
            top1_metric.update(out, y)
            if num_classes >= 5:
                top5_metric.update(out, y)

    top1 = float(top1_metric.compute().item())
    top5 = float(top5_metric.compute().item()) if num_classes >= 5 else 0.0

    return {
        "top1": top1,
        "top5": top5,
        "fc_state_dict": {k: v.detach().cpu() for k, v in fc.state_dict().items()},
    }


def train_adapter_probe(
    syn_images: torch.Tensor,
    syn_labels: torch.Tensor,
    test_loader: DataLoader,
    backbone: nn.Module,
    model_name: str,
    normalize,
    num_feats: int,
    num_classes: int,
    crop_res: int,
    lr: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    reduction: int,
    min_dim: int,
    scope: str,
    last_n: int,
    feature_anchor_weight: float,
    view_feature_weight: float,
    view_logit_weight: float,
    view_kl_temperature: float,
) -> dict:
    ds = TensorDataset(syn_images.detach().cpu(), syn_labels.detach().cpu())
    train_loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    augmentor = AugBasic(crop_res=crop_res).cuda()
    summary = inject_internal_adapters(
        backbone,
        model_name=model_name,
        reduction=reduction,
        min_dim=min_dim,
        scope=scope,
        last_n=last_n,
    )
    fc = get_fc(num_feats=num_feats, num_classes=num_classes, distributed=False)
    fc.train()
    trainable_params = trainable_adapter_parameters(backbone)
    if len(trainable_params) == 0:
        raise RuntimeError("Adapter injection produced no trainable parameters.")

    optimizer = torch.optim.Adam(
        list(trainable_params) + list(fc.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    scaler = GradScaler()
    use_view_consistency = float(view_feature_weight) > 0 or float(view_logit_weight) > 0
    kl_temperature = max(float(view_kl_temperature), 1e-6)

    for _ in tqdm(range(epochs), desc="Adapter Probe Train", leave=True):
        for x, y in tqdm(train_loader, desc="Adapter Epoch", leave=False):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            with autocast(enabled=True):
                x_primary = augmentor(x)
                x_primary = _preprocess_backbone_input(x_primary, normalize, crop_res=crop_res)
                x_secondary = None
                if use_view_consistency:
                    x_secondary = augmentor(x)
                    x_secondary = _preprocess_backbone_input(x_secondary, normalize, crop_res=crop_res)
                z_frozen = None
                if float(feature_anchor_weight) > 0:
                    with torch.no_grad():
                        set_adapters_enabled(backbone, False)
                        z_frozen = backbone(x_primary).detach()
                        set_adapters_enabled(backbone, True)
                z = backbone(x_primary)
                out = fc(z)
                loss = nn.functional.cross_entropy(out, y)
                if z_frozen is not None:
                    loss = loss + (float(feature_anchor_weight) * nn.functional.mse_loss(z, z_frozen))
                if x_secondary is not None:
                    z_view = backbone(x_secondary)
                    out_view = fc(z_view)
                    loss = loss + nn.functional.cross_entropy(out_view, y)
                    if float(view_feature_weight) > 0:
                        z_norm = F.normalize(z.float(), dim=-1)
                        z_view_norm = F.normalize(z_view.float(), dim=-1)
                        loss = loss + (
                            float(view_feature_weight)
                            * (1.0 - (z_norm * z_view_norm).sum(dim=-1).mean())
                        )
                    if float(view_logit_weight) > 0:
                        log_p = F.log_softmax(out.float() / kl_temperature, dim=-1)
                        log_q = F.log_softmax(out_view.float() / kl_temperature, dim=-1)
                        p = log_p.exp()
                        q = log_q.exp()
                        sym_kl = 0.5 * (
                            F.kl_div(log_p, q, reduction="batchmean")
                            + F.kl_div(log_q, p, reduction="batchmean")
                        )
                        loss = loss + (
                            float(view_logit_weight) * (kl_temperature ** 2) * sym_kl
                        )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

    fc.eval()
    top1_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=1).cuda()
    if num_classes >= 5:
        top5_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=5).cuda()

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Adapter Eval", leave=False):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            x = _preprocess_backbone_input(x, normalize, crop_res=crop_res)
            z = backbone(x)
            out = fc(z)
            top1_metric.update(out, y)
            if num_classes >= 5:
                top5_metric.update(out, y)

    top1 = float(top1_metric.compute().item())
    top5 = float(top5_metric.compute().item()) if num_classes >= 5 else 0.0

    return {
        "top1": top1,
        "top5": top5,
        "fc_state_dict": {k: v.detach().cpu() for k, v in fc.state_dict().items()},
        "adapter_state_dict": adapter_state_dict(backbone),
        "adapter_summary": {
            "kind": summary.kind,
            "num_wrapped_modules": int(summary.num_wrapped_modules),
            "wrapped_module_names": list(summary.wrapped_module_names),
            "bottleneck_dim": int(summary.bottleneck_dim),
            "scope": str(scope),
        },
    }


def main(cfg: FedCfg):
    if torch.cuda.is_available():
        cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    align_model_resolution_inplace(cfg)
    _warn_if_seed_looks_like_date(cfg.seed)
    distill_gpu_ids = _parse_distill_gpu_ids(cfg.distill_gpu_ids)
    if torch.cuda.is_available() and len(distill_gpu_ids) > 0:
        torch.cuda.set_device(int(distill_gpu_ids[0]))

    effective_real_res, effective_crop_res = resolve_dataset_resolution(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
    )
    if int(effective_real_res) != int(cfg.real_res) or int(effective_crop_res) != int(cfg.crop_res):
        print(
            f"[override] set real/crop res: {cfg.real_res}/{cfg.crop_res} -> "
            f"{effective_real_res}/{effective_crop_res} (dataset={cfg.dataset})"
        )
    train_dataset, test_dataset = get_dataset(
        name=cfg.dataset,
        res=effective_real_res,
        crop_res=effective_crop_res,
        train_crop_mode=cfg.train_crop_mode,
        data_root=cfg.data_root,
    )
    num_classes = train_dataset.num_classes
    dataset_name = str(cfg.dataset).lower()
    effective_stats_batch_size = int(cfg.stats_batch_size)
    effective_stats_num_workers = int(cfg.stats_num_workers)
    if effective_stats_batch_size != int(cfg.stats_batch_size):
        print(
            f"[override] set stats_batch_size: {cfg.stats_batch_size} -> {effective_stats_batch_size} "
            f"(dataset={cfg.dataset}, model={cfg.model}, classes={num_classes})"
        )
    if effective_stats_num_workers != int(cfg.stats_num_workers):
        print(
            f"[override] set stats_num_workers: {cfg.stats_num_workers} -> {effective_stats_num_workers} "
            f"(dataset={cfg.dataset})"
        )

    test_dl_kwargs = dict(
        shuffle=False,
        num_workers=cfg.eval_num_workers,
        batch_size=cfg.eval_batch_size,
        drop_last=False,
        pin_memory=True if (cfg.eval_num_workers > 0) else False,
        persistent_workers=True if (cfg.eval_num_workers > 0 and len(test_dataset) <= 10000) else False,
    )
    if cfg.eval_num_workers > 0:
        test_dl_kwargs["prefetch_factor"] = 2
    test_loader = DataLoader(test_dataset, **test_dl_kwargs)

    effective_activation_checkpointing = bool(cfg.activation_checkpointing)

    backbone, num_feats = get_model(name=cfg.model, distributed=False)
    if effective_activation_checkpointing:
        backbone, ckpt_enabled, ckpt_path = enable_activation_checkpointing(backbone, str(cfg.model))
        if ckpt_enabled:
            print(
                f"[distill] activation checkpointing enabled for model={cfg.model} via {ckpt_path}"
            )
        else:
            print(
                f"[distill] activation checkpointing not supported for model={cfg.model} "
                f"(path={ckpt_path})"
            )
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    split_t0 = time.perf_counter()
    labels_t0 = time.perf_counter()
    labels = get_labels_for_partition(train_dataset)
    labels_dt = time.perf_counter() - labels_t0
    print(
        f"[split_timing] labels_ready dataset={cfg.dataset} samples={len(train_dataset)} "
        f"seconds={labels_dt:.3f}",
        flush=True,
    )
    rng = np.random.default_rng(cfg.seed)

    split_build_t0 = time.perf_counter()
    if cfg.partition == "iid":
        splits = split_iid(len(train_dataset), cfg.num_clients, rng=rng)
    elif cfg.partition == "dirichlet":
        splits = split_dirichlet(labels, cfg.num_clients, cfg.dirichlet_alpha, rng=rng)
    else:
        raise NotImplementedError(cfg.partition)
    split_build_dt = time.perf_counter() - split_build_t0
    split_total_dt = time.perf_counter() - split_t0
    split_sizes = [int(len(v)) for v in splits]
    print(
        f"[split_timing] partition_ready mode={cfg.partition} clients={cfg.num_clients} "
        f"seconds_build={split_build_dt:.3f} seconds_total={split_total_dt:.3f} "
        f"size_min={min(split_sizes) if len(split_sizes) > 0 else 0} "
        f"size_max={max(split_sizes) if len(split_sizes) > 0 else 0}",
        flush=True,
    )

    cache_spec = FederatedFeatureCacheSpec(
        dataset=str(cfg.dataset),
        model=str(cfg.model),
        real_res=int(effective_real_res),
        crop_res=int(effective_crop_res),
        train_crop_mode=str(cfg.train_crop_mode),
        num_clients=int(cfg.num_clients),
        partition=str(cfg.partition),
        dirichlet_alpha=float(cfg.dirichlet_alpha),
        dirichlet_balance=bool(cfg.dirichlet_balance),
        dirichlet_min_size=int(cfg.dirichlet_min_size),
        shard_per_client=int(cfg.shard_per_client),
        classes_per_client=int(cfg.classes_per_client),
        seed=int(cfg.seed),
    )
    feature_cache_root = shared_feature_cache_dir(cfg.output_root)
    feature_cache_dir = federated_feature_cache_scope_dir(feature_cache_root, cache_spec)
    os.makedirs(feature_cache_dir, exist_ok=True)
    if str(getattr(cfg, "barycenter_cache_root", "")).strip():
        barycenter_cache_root = os.path.abspath(str(cfg.barycenter_cache_root))
    else:
        barycenter_cache_root = os.path.join(os.path.dirname(os.path.abspath(feature_cache_root)), "barycenter_cache")
    os.makedirs(barycenter_cache_root, exist_ok=True)
    client_sizes = [int(len(v)) for v in splits]
    readable_feature_cache_dirs = discover_readable_federated_feature_cache_dirs(
        output_root=str(cfg.output_root),
        current_cache_dir=feature_cache_dir,
        spec=cache_spec,
        expected_client_sizes=client_sizes,
        expected_client_indices=[[int(i) for i in client] for client in splits],
    )
    reused_dirs = [p for p in readable_feature_cache_dirs if os.path.abspath(p) != os.path.abspath(feature_cache_dir)]
    if len(reused_dirs) > 0:
        print(f"[feature_cache] reuse candidates: {len(reused_dirs)}")
        print(f"[feature_cache] prefer newest: {reused_dirs[0]}")
    else:
        print("[feature_cache] no reusable cache found, will forward and cache under shared output dir")

    def load_or_build_client_features(client_id: int, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        cached = load_cached_federated_client_features(
            readable_cache_dirs=readable_feature_cache_dirs,
            spec=cache_spec,
            client_id=client_id,
            expected_num_samples=int(len(idx)),
            expected_num_features=int(num_feats),
        )
        if cached is not None:
            cache_path, feats, y_all = cached
            print(f"[feature_cache] hit client={client_id} path={cache_path}", flush=True)
            return feats, y_all
        local_ds = Subset(train_dataset, idx)
        loader = build_feature_loader(
            local_ds,
            batch_size=effective_stats_batch_size,
            num_workers=effective_stats_num_workers,
            is_heavy_dataset=dataset_name.startswith("imagenet"),
        )
        z_all, y_all = forward_features(
            loader=loader,
            backbone=backbone,
            preprocess_fn=lambda x: _preprocess_backbone_input(
                x, train_dataset.normalize, crop_res=effective_crop_res
            ),
            feature_dim=int(num_feats),
            progress_desc=f"Client {client_id} Feature Forward",
        )
        save_federated_client_features(
            cache_dir=feature_cache_dir,
            spec=cache_spec,
            client_id=client_id,
            features=z_all,
            labels=y_all,
        )
        return z_all, y_all

    if bool(cfg.only_build_feature_cache):
        total_samples = 0
        for client_id, idx in enumerate(splits):
            z_all, y_all = load_or_build_client_features(client_id=client_id, idx=idx)
            total_samples += int(y_all.shape[0])
            del z_all, y_all
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            "[feature_cache] warmup complete: "
            f"dataset={cfg.dataset} model={cfg.model} clients={len(splits)} samples={total_samples} "
            f"cache_dir={feature_cache_dir}",
            flush=True,
        )
        return

    resolved_experiment_name = _resolve_experiment_name(cfg)
    server = FederatedServer(experiment_name=resolved_experiment_name, output_root=cfg.output_root)
    client_feature_bank = []

    sum_per_class = None
    count_per_class = None
    sum_all = None
    count_all = 0
    for client_id, idx in enumerate(splits):
        z_all, y_all = load_or_build_client_features(client_id=client_id, idx=idx)
        client_feature_bank.append((client_id, z_all.detach().cpu(), y_all.detach().cpu()))
        z64 = z_all.to(dtype=torch.float64, copy=False)
        s_pc = torch.zeros((num_classes, num_feats), dtype=torch.float64)
        c_pc = torch.bincount(y_all.to(dtype=torch.long), minlength=num_classes).to(dtype=torch.long)
        if int(y_all.shape[0]) > 0:
            s_pc.index_add_(0, y_all.to(dtype=torch.long), z64)
        s_all = z64.sum(dim=0, dtype=torch.float64) if int(z64.shape[0]) > 0 else torch.zeros((num_feats,), dtype=torch.float64)
        if sum_per_class is None:
            sum_per_class = torch.zeros_like(s_pc)
            count_per_class = torch.zeros_like(c_pc)
            sum_all = torch.zeros_like(s_all)
        sum_per_class += s_pc
        count_per_class += c_pc
        sum_all += s_all
        count_all += int(y_all.shape[0])
        del s_pc, c_pc, s_all, z_all, y_all, z64
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if sum_per_class is None or count_per_class is None or sum_all is None:
        raise RuntimeError("No client stats aggregated.")
    mu_target = (sum_per_class / count_per_class.clamp_min(1)[:, None]).to(dtype=torch.float64)
    mu_all = (sum_all / max(count_all, 1)).to(dtype=torch.float64)
    agg = AggregatedStats(
        mu_target=mu_target,
        mu_all=mu_all,
        count_per_class=count_per_class,
        count_all=count_all,
    )
    effective_dp_delta = _resolve_effective_dp_delta(cfg.dp_delta, agg.count_all)
    client_loss_list = []
    loss_cache_dir = None
    if int(cfg.ipc) > 1:
        # Keep barycenter/interpolation cache separate from feature cache.
        scope_rel = os.path.relpath(feature_cache_dir, feature_cache_root)
        loss_cache_dir = os.path.join(barycenter_cache_root, scope_rel, f"ipc{int(cfg.ipc)}")
        os.makedirs(loss_cache_dir, exist_ok=True)
        print(f"[barycenter_cache] dir={loss_cache_dir}", flush=True)
        for client_id, idx in enumerate(splits):
            cache_file = f"client_{client_id:04d}.pt"
            cache_path = os.path.join(loss_cache_dir, cache_file)
            if os.path.exists(cache_path):
                client_loss_list.append(cache_path)
                continue
            z_all, y_all = load_or_build_client_features(client_id=client_id, idx=idx)
            nu_local = []
            nu_weights = []
            cpc = torch.zeros((num_classes,), dtype=torch.long)
            for c in range(num_classes):
                sel = (y_all == int(c))
                z_c = z_all[sel]
                n_c_full = int(z_c.shape[0])
                cpc[c] = n_c_full
                if int(cfg.loss_max_samples_per_class) > 0 and n_c_full > int(cfg.loss_max_samples_per_class):
                    rng_c = np.random.default_rng(int(cfg.seed) + int(client_id) * 10007 + int(c))
                    pick = rng_c.choice(np.arange(n_c_full), size=int(cfg.loss_max_samples_per_class), replace=False)
                    pick_t = torch.as_tensor(pick, dtype=torch.long)
                    z_c = z_c.index_select(0, pick_t)
                z_c64 = z_c.to(dtype=torch.float64)
                if int(z_c64.shape[0]) > 0:
                    z_c64 = z_c64 / torch.linalg.norm(z_c64, dim=1, keepdim=True).clamp_min(1e-12)
                    w_c = torch.full((int(z_c64.shape[0]),), 1.0 / float(z_c64.shape[0]), dtype=torch.float64)
                else:
                    z_c64 = torch.empty((0, num_feats), dtype=torch.float64)
                    w_c = torch.empty((0,), dtype=torch.float64)
                nu_local.append(z_c64.detach().cpu())
                nu_weights.append(w_c.detach().cpu())
            torch.save(
                {
                    "nu_local": nu_local,
                    "nu_weights": nu_weights,
                    "count_per_class": cpc,
                },
                cache_path,
            )
            client_loss_list.append(cache_path)
            del z_all, y_all, nu_local, nu_weights, cpc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    loss_targets = server.aggregate_loss_targets(
        client_loss_list=client_loss_list,
        aggregated_stats=agg,
        ipc=cfg.ipc,
        interpolation_rounds=cfg.loss_interp_rounds,
        interpolation_t=cfg.loss_interp_t,
        init_mode=str(cfg.loss_init_mode),
        gaussian_std=float(cfg.loss_gaussian_std),
        random_seed=int(cfg.seed),
        update_support_weights=bool(cfg.loss_update_support_weights),
        stop_tol_xi=(None if float(cfg.loss_stop_tol_xi) <= 0 else float(cfg.loss_stop_tol_xi)),
        stop_tol_a=(None if float(cfg.loss_stop_tol_a) <= 0 else float(cfg.loss_stop_tol_a)),
        stop_patience=int(cfg.loss_stop_patience),
        dp_enable=bool(cfg.dp_enable),
        dp_epsilon=float(cfg.dp_epsilon),
        dp_delta=cfg.dp_delta,
        ot_solver=str(cfg.loss_ot_solver),
        sinkhorn_reg=float(cfg.loss_sinkhorn_reg),
        ot_warmstart=bool(cfg.loss_ot_warmstart),
    )
    # Persist global barycenter targets (including support weights) for reproducibility/restart analysis.
    barycenter_targets_payload = {
        "b_star": loss_targets.b_star.detach().cpu(),
        "support_weights": loss_targets.support_weights.detach().cpu(),
        "class_weights": loss_targets.class_weights.detach().cpu(),
        "g_star": loss_targets.g_star.detach().cpu(),
        "mu_all": loss_targets.mu_all.detach().cpu(),
        "count_per_class": loss_targets.count_per_class.detach().cpu(),
        "count_all": int(loss_targets.count_all),
        "ipc": int(loss_targets.ipc),
    }
    torch.save(barycenter_targets_payload, os.path.join(server.output_dir, "barycenter_targets.pth"))
    if loss_cache_dir is not None:
        torch.save(barycenter_targets_payload, os.path.join(loss_cache_dir, "global_barycenter_targets.pth"))
    if cfg.verify_centers:
        mu_central = compute_centers_centralized(
            train_dataset=train_dataset,
            backbone=backbone,
            num_feats=num_feats,
            crop_res=effective_crop_res,
            batch_size=effective_stats_batch_size,
            num_workers=cfg.stats_num_workers,
        )
        mu_fed = agg.mu_target.detach().cpu()
        max_abs = float((mu_fed - mu_central).abs().max().item())
        server.save_json("verify_centers.json", {"max_abs_diff": max_abs, "tol": cfg.verify_tol})
        if max_abs > cfg.verify_tol:
            raise RuntimeError(f"Center verification failed: {max_abs} > {cfg.verify_tol}")

    effective_forward_batch_size = int(cfg.forward_batch_size)
    if effective_forward_batch_size <= 0:
        effective_forward_batch_size = min(int(num_classes) * int(cfg.ipc), 120)
        print(
            "[auto] set forward_batch_size: "
            f"{cfg.forward_batch_size} -> {effective_forward_batch_size} (ipc={cfg.ipc})"
        )

    effective_augs_per_batch = int(cfg.augs_per_batch)
    if dataset_name == "imagenet-1k":
        if effective_augs_per_batch != 3:
            print(
                f"[override] set augs_per_batch: {effective_augs_per_batch} -> 3 "
                f"(dataset={cfg.dataset})"
            )
        effective_augs_per_batch = 3
    force_save_on_cpu_acts = bool(cfg.force_save_on_cpu_acts)
    if dataset_name == "imagenet-1k" and not force_save_on_cpu_acts:
        print(
            f"[override] set force_save_on_cpu_acts: {cfg.force_save_on_cpu_acts} -> True "
            f"(dataset={cfg.dataset})"
        )
        force_save_on_cpu_acts = True
    distill_cfg = DistillArgs(
        ipc=cfg.ipc,
        lr=cfg.lr,
        iterations=cfg.iterations,
        augs_per_batch=effective_augs_per_batch,
        forward_batch_size=effective_forward_batch_size,
        distill_simultaneous_classes=cfg.distill_simultaneous_classes,
        distill_mode=cfg.distill_mode,
        aug_mode=cfg.aug_mode,
        decorrelate_color=cfg.decorrelate_color,
        init_mode=cfg.init_mode,
        pyramid_extent_it=cfg.pyramid_extent_it,
        pyramid_start_res=cfg.pyramid_start_res,
        syn_res=cfg.syn_res,
        crop_res=effective_crop_res,
        force_save_on_cpu_acts=force_save_on_cpu_acts,
        distill_gpu_ids=distill_gpu_ids,
    )

    gc.collect()
    torch.cuda.empty_cache()
    distill_backbone = backbone
    if torch.cuda.is_available() and len(distill_gpu_ids) > 1:
        primary = int(distill_gpu_ids[0])
        backbone = backbone.to(device=f"cuda:{primary}")
        distill_backbone = nn.DataParallel(
            backbone,
            device_ids=list(distill_gpu_ids),
            output_device=primary,
        )
        print(
            f"[distill] use multi-gpu frozen backbone on devices={list(distill_gpu_ids)} "
            f"(primary={primary})"
        )
    distill_out = server.distill(
        cfg=distill_cfg,
        backbone=distill_backbone,
        loss_targets=loss_targets,
        normalize=train_dataset.normalize,
        num_classes=num_classes,
        num_feats=num_feats,
        log_it=200,
    )

    gc.collect()
    torch.cuda.empty_cache()
    eval_out = train_linear_probe(
        syn_images=distill_out.images,
        syn_labels=distill_out.labels,
        test_loader=test_loader,
        backbone=backbone,
        normalize=train_dataset.normalize,
        num_feats=num_feats,
        num_classes=num_classes,
        crop_res=effective_crop_res,
        lr=cfg.eval_lr,
        epochs=cfg.eval_epochs,
        batch_size=cfg.eval_batch_size,
    )

    personalized_summary = None
    if bool(cfg.personalize_enable):
        test_splits = build_partition_splits(
            dataset=test_dataset,
            partition=cfg.partition,
            num_clients=cfg.num_clients,
            dirichlet_alpha=cfg.dirichlet_alpha,
            seed=int(cfg.seed) + int(cfg.personalize_eval_partition_seed_offset),
        )
        client_test_loaders = build_client_test_loaders(
            test_dataset=test_dataset,
            test_splits=test_splits,
            batch_size=cfg.eval_batch_size,
            num_workers=cfg.eval_num_workers,
        )
        personalized_results = train_personalized_clients(
            output_dir=server.output_dir,
            client_features=client_feature_bank,
            client_test_loaders=client_test_loaders,
            syn_images=distill_out.images,
            syn_labels=distill_out.labels,
            backbone=backbone,
            normalize=train_dataset.normalize,
            num_feats=num_feats,
            num_classes=num_classes,
            crop_res=effective_crop_res,
            classifier_lr=cfg.personalize_head_lr,
            classifier_epochs=cfg.personalize_head_epochs,
            interface_type=cfg.personalize_adapter_type,
            interface_lr=cfg.personalize_adapter_lr,
            interface_epochs=cfg.personalize_adapter_epochs,
            batch_size=cfg.personalize_batch_size,
            identity_lambda=cfg.personalize_identity_lambda,
        )
        personalized_summary = {
            "num_clients": len(personalized_results),
            "mean_local_head_train_top1": float(
                np.mean([item.train_top1 for item in personalized_results]) if personalized_results else 0.0
            ),
            "mean_personalized_test_top1": float(
                np.mean([item.test_top1 for item in personalized_results]) if personalized_results else 0.0
            ),
            "mean_personalized_test_top5": float(
                np.mean([item.test_top5 for item in personalized_results]) if personalized_results else 0.0
            ),
            "adapter_type": cfg.personalize_adapter_type,
        }
        server.save_json("personalized_metrics.json", personalized_summary)

    server.save_json(
        "metrics.json",
        {
            "top1": eval_out["top1"],
            "top5": eval_out["top5"],
            "dataset": cfg.dataset,
            "model": cfg.model,
            "num_clients": cfg.num_clients,
            "partition": cfg.partition,
            "personalized": personalized_summary,
        },
    )
    server.save_json(
        "run_config.json",
        {
            "experiment_name": resolved_experiment_name,
            "requested_experiment_name": cfg.experiment_name,
            "dataset": cfg.dataset,
            "model": cfg.model,
            "data_root": cfg.data_root,
            "output_root": cfg.output_root,
            "requested_real_res": cfg.real_res,
            "requested_crop_res": cfg.crop_res,
            "effective_real_res": effective_real_res,
            "effective_crop_res": effective_crop_res,
            "ipc": cfg.ipc,
            "num_clients": cfg.num_clients,
            "partition": cfg.partition,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "dirichlet_balance": bool(cfg.dirichlet_balance),
            "dirichlet_min_size": int(cfg.dirichlet_min_size),
            "shard_per_client": int(cfg.shard_per_client),
            "classes_per_client": int(cfg.classes_per_client),
            "dp_enable": cfg.dp_enable,
            "dp_epsilon": cfg.dp_epsilon,
            "dp_delta": effective_dp_delta,
            "requested_dp_delta": cfg.dp_delta,
            "iterations": cfg.iterations,
            "augs_per_batch": cfg.augs_per_batch,
            "forward_batch_size": effective_forward_batch_size,
            "distill_gpu_ids": list(distill_gpu_ids),
            "seed": cfg.seed,
            "personalize_enable": bool(cfg.personalize_enable),
            "personalize_head_epochs": int(cfg.personalize_head_epochs),
            "personalize_head_lr": float(cfg.personalize_head_lr),
            "personalize_adapter_type": str(cfg.personalize_adapter_type),
            "personalize_adapter_epochs": int(cfg.personalize_adapter_epochs),
            "personalize_adapter_lr": float(cfg.personalize_adapter_lr),
            "personalize_batch_size": int(cfg.personalize_batch_size),
            "personalize_identity_lambda": float(cfg.personalize_identity_lambda),
            **(
                {
                    "enable_adapter_eval": bool(cfg.enable_adapter_eval),
                    "adapter_epochs": int(cfg.adapter_epochs if int(cfg.adapter_epochs) > 0 else cfg.eval_epochs),
                    "adapter_lr": float(cfg.adapter_lr if float(cfg.adapter_lr) > 0 else cfg.eval_lr),
                    "adapter_weight_decay": float(cfg.adapter_weight_decay),
                    "adapter_reduction": int(cfg.adapter_reduction),
                    "adapter_min_dim": int(cfg.adapter_min_dim),
                    "adapter_scope": str(cfg.adapter_scope),
                    "adapter_last_n": int(cfg.adapter_last_n),
                    "adapter_feature_anchor_weight": float(cfg.adapter_feature_anchor_weight),
                    "adapter_view_feature_weight": float(cfg.adapter_view_feature_weight),
                    "adapter_view_logit_weight": float(cfg.adapter_view_logit_weight),
                    "adapter_view_kl_temperature": float(cfg.adapter_view_kl_temperature),
                }
                if bool(cfg.enable_adapter_eval)
                else {}
            ),
        },
    )
    torch.save({"fc_state_dict": eval_out["fc_state_dict"]}, os.path.join(server.output_dir, "linear_probe.pth"))

    if bool(cfg.enable_adapter_eval):
        adapter_epochs = int(cfg.adapter_epochs) if int(cfg.adapter_epochs) > 0 else int(cfg.eval_epochs)
        adapter_lr = float(cfg.adapter_lr) if float(cfg.adapter_lr) > 0 else float(cfg.eval_lr)
        gc.collect()
        torch.cuda.empty_cache()
        adapter_out = train_adapter_probe(
            syn_images=distill_out.images,
            syn_labels=distill_out.labels,
            test_loader=test_loader,
            backbone=backbone,
            model_name=cfg.model,
            normalize=train_dataset.normalize,
            num_feats=num_feats,
            num_classes=num_classes,
            crop_res=effective_crop_res,
            lr=adapter_lr,
            epochs=adapter_epochs,
            batch_size=cfg.eval_batch_size,
            weight_decay=float(cfg.adapter_weight_decay),
            reduction=int(cfg.adapter_reduction),
            min_dim=int(cfg.adapter_min_dim),
            scope=str(cfg.adapter_scope),
            last_n=int(cfg.adapter_last_n),
            feature_anchor_weight=float(cfg.adapter_feature_anchor_weight),
            view_feature_weight=float(cfg.adapter_view_feature_weight),
            view_logit_weight=float(cfg.adapter_view_logit_weight),
            view_kl_temperature=float(cfg.adapter_view_kl_temperature),
        )
        server.save_json(
            "adapter_metrics.json",
            {
                "top1": adapter_out["top1"],
                "top5": adapter_out["top5"],
                "dataset": cfg.dataset,
                "model": cfg.model,
                "num_clients": cfg.num_clients,
                "partition": cfg.partition,
                "adapter_summary": adapter_out["adapter_summary"],
            },
        )
        torch.save(
            {
                "fc_state_dict": adapter_out["fc_state_dict"],
                "adapter_state_dict": adapter_out["adapter_state_dict"],
                "adapter_summary": adapter_out["adapter_summary"],
            },
            os.path.join(server.output_dir, "adapter_probe.pth"),
        )


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = FedCfg(explicit_bool=True).parse_args()
    main(args)
