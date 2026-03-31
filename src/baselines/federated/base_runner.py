from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from baselines.common import (
    EarlyStopper,
    ProfileMeter,
    assert_head_only_trainable,
    build_or_load_partitions,
    build_constant_scheduler,
    build_cosine_scheduler,
    build_linear_head_optimizer,
    build_sgd_linear_head_optimizer,
    ensure_dir,
    freeze_backbone,
    get_labels_for_partition,
    linear_head_step_flops,
    mean_std,
    set_global_seed,
    state_dict_bytes,
    validate_partition_payload,
    write_curve_csv,
    write_json,
    write_jsonl,
)
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
from data.dataloaders import get_dataset
from models import get_fc, get_model


@dataclass
class FederatedRunArgs:
    method: str
    dataset: str
    model: str
    data_root: str
    output_root: str
    real_res: int
    crop_res: int
    train_crop_mode: str
    seeds: list[int]
    max_rounds: int
    local_epochs: int
    local_batch_size: int
    feature_batch_size: int
    local_lr: float
    fl_optimizer: str
    fl_scheduler: str
    local_momentum: float
    local_weight_decay: float
    grad_clip_norm: float
    use_amp: bool
    num_clients: int
    partition: str
    non_iid: bool
    partition_train_ratio: float
    partition_alpha: float
    dirichlet_alpha: float
    dirichlet_balance: bool
    dirichlet_min_size: int
    shard_per_client: int
    classes_per_client: int
    join_ratio: float
    random_join_ratio: bool
    client_drop_rate: float
    patience_rounds: int
    min_delta: float
    warmup_rounds: int
    eval_batch_size: int
    eval_num_workers: int
    eval_on_global_test: bool
    local_num_workers: int
    prox_mu: float
    afl_ri_reg: float
    afl_clean_reg: bool
    kd_weight: float
    kd_temp: float
    pcl_weight: float
    pcl_temp: float
    pcl_momentum: float
    proto_lamda: float
    ntd_weight: float
    ntd_temp: float
    scaffold_eta: float
    ccvr_calib_epochs: int
    ccvr_calib_samples_per_class: int
    ccvr_calib_lr: float
    smoke_max_train_batches: int
    smoke_max_eval_batches: int
    cache_features: bool


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _weighted_average(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    out = {}
    for k in states[0].keys():
        acc = None
        for s, w in zip(states, weights):
            v = s[k].to(dtype=torch.float32)
            cur = v * float(w)
            acc = cur if acc is None else (acc + cur)
        out[k] = acc
    return out


def _zeros_like_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state.items()}


def _state_numel(state: dict[str, torch.Tensor]) -> int:
    total = 0
    for v in state.values():
        if torch.is_tensor(v):
            total += int(v.numel())
    return int(total)


def _pfllib_partition_indices(
    labels: list[int],
    num_clients: int,
    num_classes: int,
    *,
    niid: bool,
    balance: bool,
    partition: str | None,
    class_per_client: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(int(seed))
    labels_np = np.asarray(labels, dtype=np.int64)
    idxs = np.arange(len(labels_np), dtype=np.int64)
    least_samples = int(min(10 / (1 - 0.75), len(labels_np) / max(int(num_clients), 1) / 2))
    if not niid:
        partition = "pat"
        class_per_client = int(num_classes)
    if partition == "pat":
        idx_for_each_class = [idxs[labels_np == i] for i in range(num_classes)]
        class_num_per_client = [int(class_per_client) for _ in range(num_clients)]
        dataidx_map: dict[int, np.ndarray] = {}
        for cls in range(num_classes):
            selected_clients = [cid for cid in range(num_clients) if class_num_per_client[cid] > 0]
            if len(selected_clients) == 0:
                break
            selected_clients = selected_clients[: int(np.ceil((num_clients / num_classes) * class_per_client))]
            num_all_samples = len(idx_for_each_class[cls])
            num_selected_clients = max(len(selected_clients), 1)
            num_per = num_all_samples / float(num_selected_clients)
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                low = max(int(num_per / 10), max(least_samples // max(num_classes, 1), 1))
                high = max(int(num_per), low + 1)
                num_samples = rng.integers(low=low, high=high, size=num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - int(sum(num_samples)))
            start = 0
            for client_id, n_take in zip(selected_clients, num_samples):
                end = start + int(n_take)
                chunk = idx_for_each_class[cls][start:end]
                if client_id not in dataidx_map:
                    dataidx_map[client_id] = chunk.copy()
                else:
                    dataidx_map[client_id] = np.append(dataidx_map[client_id], chunk, axis=0)
                start = end
                class_num_per_client[client_id] -= 1
        return [list(map(int, dataidx_map.get(j, np.asarray([], dtype=np.int64)).tolist())) for j in range(num_clients)]
    if partition == "dir":
        min_size = 0
        n_total = len(labels_np)
        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for cls in range(num_classes):
                idx_c = np.where(labels_np == cls)[0]
                rng.shuffle(idx_c)
                props = rng.dirichlet(np.repeat(float(alpha), num_clients))
                props = np.asarray(
                    [p * (len(idx_j) < n_total / max(num_clients, 1)) for p, idx_j in zip(props, idx_batch)],
                    dtype=np.float64,
                )
                if float(props.sum()) <= 0:
                    props = np.ones((num_clients,), dtype=np.float64)
                props = props / props.sum()
                cut = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_c, cut))]
            min_size = min((len(idx_j) for idx_j in idx_batch), default=0)
        return [[int(v) for v in client] for client in idx_batch]
    if partition == "exdir":
        cpc = int(class_per_client)
        min_size_per_label = 0
        min_require_size_per_label = max(cpc * num_clients // max(num_classes, 1) // 2, 1)
        clientidx_map: dict[int, list[int]] = {}
        while min_size_per_label < min_require_size_per_label:
            for k in range(num_classes):
                clientidx_map[k] = []
            for i in range(num_clients):
                labelidx = rng.choice(np.arange(num_classes), size=cpc, replace=False)
                for k in labelidx.tolist():
                    clientidx_map[int(k)].append(i)
            min_size_per_label = min((len(clientidx_map[k]) for k in range(num_classes)), default=0)
        min_size = 0
        while min_size < 10:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = np.where(labels_np == k)[0]
                rng.shuffle(idx_k)
                proportions = rng.dirichlet(np.repeat(float(alpha), num_clients))
                proportions = np.asarray(
                    [
                        p * (len(idx_j) < len(labels_np) / max(num_clients, 1) and j in clientidx_map[k])
                        for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))
                    ],
                    dtype=np.float64,
                )
                if float(proportions.sum()) <= 0:
                    mask = np.asarray([j in clientidx_map[k] for j in range(num_clients)], dtype=np.float64)
                    proportions = mask if float(mask.sum()) > 0 else np.ones((num_clients,), dtype=np.float64)
                proportions = proportions / proportions.sum()
                cut = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, cut))]
            min_size = min((len(idx_j) for idx_j in idx_batch), default=0)
        return [[int(v) for v in client] for client in idx_batch]
    raise NotImplementedError(partition)


class _AFLLinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.size(0), -1))


class BaseFederatedRunner:
    def __init__(self, args: FederatedRunArgs):
        self.args = args
        ensure_dir(self.args.output_root)
        self.train_dataset, self.test_dataset = get_dataset(
            name=self.args.dataset,
            res=self.args.real_res,
            crop_res=self.args.crop_res,
            train_crop_mode=self.args.train_crop_mode,
            data_root=self.args.data_root,
        )
        self.backbone, self.num_feats = get_model(name=self.args.model, distributed=False)
        freeze_backbone(self.backbone)
        self.num_classes = int(self.train_dataset.num_classes)
        self.partition_payload = self._prepare_partitions()
        if not bool(self.partition_payload.get("pfllib_partition", False)):
            validate_partition_payload(self.partition_payload, expected_num_samples=len(self.train_dataset))
        self.client_indices = self.partition_payload["client_indices"]
        self.client_sizes = [len(v) for v in self.client_indices]
        self.client_test_indices = self.partition_payload.get("client_test_indices", [])
        self.cache_spec = FederatedFeatureCacheSpec(
            dataset=str(self.args.dataset),
            model=str(self.args.model),
            real_res=int(self.args.real_res),
            crop_res=int(self.args.crop_res),
            train_crop_mode=str(self.args.train_crop_mode),
            num_clients=int(self.args.num_clients),
            partition=str(self.args.partition),
            dirichlet_alpha=float(self.args.dirichlet_alpha),
            dirichlet_balance=bool(self.args.dirichlet_balance),
            dirichlet_min_size=int(self.args.dirichlet_min_size),
            shard_per_client=int(self.args.shard_per_client),
            classes_per_client=int(self.args.classes_per_client),
            seed=int(self.args.seeds[0]) if len(self.args.seeds) > 0 else 0,
        )
        self.use_feature_cache = bool(self.args.cache_features)
        self.feature_cache_root = shared_feature_cache_dir(self.args.output_root)
        self.feature_cache_dir = federated_feature_cache_scope_dir(self.feature_cache_root, self.cache_spec)
        self.feature_cache_read_dirs = []
        if self.use_feature_cache:
            ensure_dir(self.feature_cache_dir)
            self.feature_cache_read_dirs = discover_readable_federated_feature_cache_dirs(
                output_root=str(self.args.output_root),
                current_cache_dir=self.feature_cache_dir,
                spec=self.cache_spec,
                expected_client_sizes=[int(v) for v in self.partition_payload.get("client_sizes", [])],
                expected_client_indices=[
                    [int(i) for i in client] for client in self.partition_payload.get("client_indices", [])
                ],
            )
            reused_dir = next(
                (d for d in self.feature_cache_read_dirs if os.path.abspath(d) != os.path.abspath(self.feature_cache_dir)),
                None,
            )
            if reused_dir is not None:
                print(f"[feature_cache] reuse from {reused_dir}")
            else:
                print("[feature_cache] no reusable cache found, will build under shared output dir")
        is_heavy = str(self.args.dataset).lower().startswith("imagenet")
        eval_workers = min(int(self.args.eval_num_workers), 4) if is_heavy else int(self.args.eval_num_workers)
        test_kwargs = dict(
            shuffle=False,
            num_workers=eval_workers,
            batch_size=self.args.eval_batch_size,
            drop_last=False,
            pin_memory=eval_workers > 0,
            persistent_workers=eval_workers > 0 and (not is_heavy) and len(self.test_dataset) <= 10000,
        )
        if eval_workers > 0:
            test_kwargs["prefetch_factor"] = 1 if is_heavy else 2
        self.test_loader = DataLoader(self.test_dataset, **test_kwargs)

    def _prepare_partitions(self) -> dict:
        labels = get_labels_for_partition(self.train_dataset)
        cache_dir = os.path.join(self.args.output_root, "partition_cache")
        ensure_dir(cache_dir)
        raw_partition = str(self.args.partition).lower().strip()
        use_pfllib = raw_partition in {"pat", "dir", "exdir"}
        cache_name = (
            f"{self.args.dataset}_{self.args.num_clients}_{self.args.partition}_"
            f"{self.args.dirichlet_alpha:.4f}_"
            f"bal{1 if bool(self.args.dirichlet_balance) else 0}_"
            f"min{int(self.args.dirichlet_min_size)}_"
            f"sh{int(self.args.shard_per_client)}_"
            f"cpc{int(self.args.classes_per_client)}_"
            f"palpha{float(self.args.partition_alpha):.4f}_"
            f"ptr{float(self.args.partition_train_ratio):.2f}_"
            f"niid{1 if bool(self.args.non_iid) else 0}_"
            f"seed{int(self.args.seeds[0])}.json"
        )
        cache_path = os.path.join(cache_dir, cache_name)
        if use_pfllib:
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = copy.deepcopy(json.load(f))
                if "client_test_indices" in cached:
                    return cached
            splits = _pfllib_partition_indices(
                labels=labels,
                num_clients=int(self.args.num_clients),
                num_classes=int(self.num_classes),
                niid=bool(self.args.non_iid),
                balance=bool(self.args.dirichlet_balance),
                partition=str(raw_partition),
                class_per_client=int(self.args.classes_per_client),
                alpha=float(self.args.partition_alpha),
                seed=int(self.args.seeds[0]),
            )
            rng = np.random.default_rng(int(self.args.seeds[0]) + 97)
            train_splits = []
            test_splits = []
            ratio = float(getattr(self.args, "partition_train_ratio", 0.75))
            ratio = min(max(ratio, 0.1), 0.99)
            for client in splits:
                arr = np.asarray(client, dtype=np.int64)
                if int(arr.shape[0]) > 0:
                    rng.shuffle(arr)
                cut = int(round(float(arr.shape[0]) * ratio))
                cut = min(max(cut, 1), int(arr.shape[0])) if int(arr.shape[0]) > 0 else 0
                train_splits.append([int(v) for v in arr[:cut].tolist()])
                test_splits.append([int(v) for v in arr[cut:].tolist()])
            payload = {
                "dataset": self.args.dataset,
                "num_clients": int(self.args.num_clients),
                "partition": str(raw_partition),
                "dirichlet_alpha": float(self.args.partition_alpha),
                "dirichlet_balance": bool(self.args.dirichlet_balance),
                "dirichlet_min_size": int(self.args.dirichlet_min_size),
                "shard_per_client": int(self.args.shard_per_client),
                "classes_per_client": int(self.args.classes_per_client),
                "seed": int(self.args.seeds[0]),
                "pfllib_partition": True,
                "client_indices": train_splits,
                "client_test_indices": test_splits,
                "client_sizes": [int(len(v)) for v in train_splits],
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return payload
        return build_or_load_partitions(
            cache_path=cache_path,
            dataset_name=self.args.dataset,
            num_clients=self.args.num_clients,
            partition=self.args.partition,
            dirichlet_alpha=self.args.dirichlet_alpha,
            dirichlet_balance=self.args.dirichlet_balance,
            dirichlet_min_size=self.args.dirichlet_min_size,
            shard_per_client=self.args.shard_per_client,
            classes_per_client=self.args.classes_per_client,
            labels=labels,
            seed=self.args.seeds[0],
        )

    def _find_readable_cache_path(self, client_id: int) -> str | None:
        cached = load_cached_federated_client_features(
            readable_cache_dirs=self.feature_cache_read_dirs,
            spec=self.cache_spec,
            client_id=client_id,
            expected_num_samples=int(self.client_sizes[client_id]),
            expected_num_features=int(self.num_feats),
            expected_client_indices=[int(v) for v in self.client_indices[client_id]],
        )
        return None if cached is None else cached[0]

    def _build_client_loader(self, client_id: int, shuffle: bool = True, batch_size: int | None = None) -> DataLoader:
        ds = Subset(self.train_dataset, self.client_indices[client_id])
        if not shuffle:
            return build_feature_loader(
                ds,
                batch_size=int(batch_size or self.args.feature_batch_size),
                num_workers=int(self.args.local_num_workers),
                is_heavy_dataset=str(self.args.dataset).lower().startswith("imagenet"),
            )
        is_heavy = str(self.args.dataset).lower().startswith("imagenet")
        local_workers = min(int(self.args.local_num_workers), 4) if is_heavy else int(self.args.local_num_workers)
        loader_kwargs = dict(
            shuffle=True,
            num_workers=local_workers,
            batch_size=int(batch_size or self.args.local_batch_size),
            drop_last=True,
            pin_memory=local_workers > 0,
            persistent_workers=local_workers > 0 and (not is_heavy),
        )
        if local_workers > 0:
            loader_kwargs["prefetch_factor"] = 1 if is_heavy else 2
        return DataLoader(ds, **loader_kwargs)

    def _train_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[-1]) != int(self.args.crop_res) or int(x.shape[-2]) != int(self.args.crop_res):
            x = F.interpolate(
                x,
                size=(int(self.args.crop_res), int(self.args.crop_res)),
                mode="bilinear",
                align_corners=False,
            )
        x = self.train_dataset.normalize(x)
        return x

    def _load_or_build_client_features(self, client_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = load_cached_federated_client_features(
            readable_cache_dirs=self.feature_cache_read_dirs,
            spec=self.cache_spec,
            client_id=client_id,
            expected_num_samples=int(self.client_sizes[client_id]),
            expected_num_features=int(self.num_feats),
            expected_client_indices=[int(v) for v in self.client_indices[client_id]],
        )
        if cached is not None:
            path, feats, labels = cached
            print(f"[feature_cache] hit client={client_id} path={path}", flush=True)
            return feats, labels
        loader = self._build_client_loader(client_id, shuffle=False, batch_size=int(self.args.feature_batch_size))
        z_all, y_all = forward_features(
            loader=loader,
            backbone=self.backbone,
            preprocess_fn=self._train_preprocess,
            feature_dim=int(self.num_feats),
        )
        save_federated_client_features(
            cache_dir=self.feature_cache_dir,
            spec=self.cache_spec,
            client_id=client_id,
            features=z_all,
            labels=y_all,
            client_indices=[int(v) for v in self.client_indices[client_id]],
        )
        return z_all, y_all

    def _evaluate(self, head: nn.Module) -> tuple[float, float]:
        head.eval()
        if (not bool(self.args.eval_on_global_test)) and bool(self.partition_payload.get("pfllib_partition", False)) and len(self.client_test_indices) == int(self.args.num_clients):
            correct1 = 0
            correct5 = 0
            total = 0
            max_eval_batches = max(int(self.args.smoke_max_eval_batches), 0)
            with torch.no_grad():
                for client_id in range(int(self.args.num_clients)):
                    idx = self.client_test_indices[client_id]
                    if len(idx) == 0:
                        continue
                    ds = Subset(self.train_dataset, idx)
                    loader = DataLoader(
                        ds,
                        batch_size=int(self.args.eval_batch_size),
                        shuffle=True,
                        num_workers=int(self.args.eval_num_workers),
                        drop_last=False,
                        pin_memory=int(self.args.eval_num_workers) > 0,
                        persistent_workers=int(self.args.eval_num_workers) > 0,
                    )
                    seen = 0
                    for x, y in loader:
                        x = x.cuda(non_blocking=True)
                        y = y.cuda(non_blocking=True)
                        x = self.train_dataset.normalize(x)
                        z = self.backbone(x)
                        logits = head(z)
                        pred1 = logits.argmax(dim=1)
                        correct1 += int((pred1 == y).sum().item())
                        if self.num_classes >= 5:
                            top5 = logits.topk(k=5, dim=1).indices
                            correct5 += int((top5 == y.unsqueeze(1)).any(dim=1).sum().item())
                        total += int(y.shape[0])
                        seen += 1
                        if max_eval_batches > 0 and seen >= max_eval_batches:
                            break
            top1 = float(correct1 / max(total, 1))
            top5 = float(correct5 / max(total, 1)) if self.num_classes >= 5 else 0.0
            return top1, top5
        correct1 = 0
        correct5 = 0
        total = 0
        with torch.no_grad():
            max_eval_batches = max(int(self.args.smoke_max_eval_batches), 0)
            seen_batches = 0
            for x, y in self.test_loader:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                x = self.test_dataset.normalize(x)
                z = self.backbone(x)
                logits = head(z)
                pred1 = logits.argmax(dim=1)
                correct1 += int((pred1 == y).sum().item())
                if self.num_classes >= 5:
                    top5 = logits.topk(k=5, dim=1).indices
                    match5 = (top5 == y.unsqueeze(1)).any(dim=1)
                    correct5 += int(match5.sum().item())
                total += int(y.shape[0])
                seen_batches += 1
                if max_eval_batches > 0 and seen_batches >= max_eval_batches:
                    break
        top1 = float(correct1 / max(total, 1))
        top5 = float(correct5 / max(total, 1)) if self.num_classes >= 5 else 0.0
        return top1, top5

    def _init_method_state(self, global_state: dict[str, torch.Tensor]) -> dict:
        if self.args.method == "scaffold":
            return {
                "c_global": _zeros_like_state(global_state),
                "c_local": [_zeros_like_state(global_state) for _ in range(self.args.num_clients)],
            }
        if self.args.method in {"fedpcl", "fedproto"}:
            return {
                "prototypes": torch.zeros((self.num_classes, self.num_feats), dtype=torch.float32),
                "proto_counts": torch.zeros((self.num_classes,), dtype=torch.float32),
            }
        if self.args.method == "ccvr":
            return {"class_counts": torch.zeros((self.num_classes,), dtype=torch.float32)}
        return {}

    def _extra_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        z: torch.Tensor,
        local_head: nn.Module,
        global_head: nn.Module,
        method_state: dict,
        prof: ProfileMeter,
    ) -> torch.Tensor:
        zero = logits.new_tensor(0.0)
        batch_size = int(labels.shape[0])
        approx_params = int(self.num_feats * self.num_classes + self.num_classes)
        per_batch_base = linear_head_step_flops(
            batch_size=batch_size,
            in_dim=int(self.num_feats),
            out_dim=int(self.num_classes),
        )
        if self.args.method == "fedprox":
            prox = zero
            for p_local, p_global in zip(local_head.parameters(), global_head.parameters()):
                prox = prox + (p_local - p_global.detach()).pow(2).sum()
            prof.add_local_flops(2 * approx_params, bucket="extra")
            return 0.5 * float(self.args.prox_mu) * prox
        if self.args.method == "fedmd":
            t = float(self.args.kd_temp)
            with torch.no_grad():
                teacher = global_head(z)
            student_logp = torch.log_softmax(logits / t, dim=1)
            teacher_p = torch.softmax(teacher / t, dim=1)
            kd = nn.functional.kl_div(student_logp, teacher_p, reduction="batchmean")
            kd_flops = per_batch_base // 2 + 8 * batch_size * int(self.num_classes)
            prof.add_local_flops(kd_flops, bucket="extra")
            return float(self.args.kd_weight) * (t * t) * kd
        if self.args.method == "fedntd":
            t = float(self.args.ntd_temp)
            with torch.no_grad():
                teacher = global_head(z)
            true_mask = torch.zeros_like(logits).scatter_(1, labels[:, None], 1.0)
            s_prob = torch.softmax(logits / t, dim=1) * (1.0 - true_mask)
            t_prob = torch.softmax(teacher / t, dim=1) * (1.0 - true_mask)
            s_prob = s_prob / s_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
            t_prob = t_prob / t_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
            kd = nn.functional.kl_div(torch.log(s_prob.clamp_min(1e-12)), t_prob, reduction="batchmean")
            ntd_flops = per_batch_base // 2 + 12 * batch_size * int(self.num_classes)
            prof.add_local_flops(ntd_flops, bucket="extra")
            return float(self.args.ntd_weight) * (t * t) * kd
        if self.args.method == "fedpcl":
            protos = method_state.get("prototypes", None)
            counts = method_state.get("proto_counts", None)
            if protos is None or counts is None:
                return zero
            if float(counts.sum().item()) <= 0:
                return zero
            protos = protos.to(device=z.device, dtype=z.dtype)
            valid = counts.to(device=z.device, dtype=z.dtype) > 0
            label_valid = valid[labels]
            if not bool(label_valid.any()):
                return zero
            p = protos[labels]
            p = nn.functional.normalize(p, dim=1)
            z_n = nn.functional.normalize(z, dim=1)
            align = 1.0 - (z_n[label_valid] * p[label_valid]).sum(dim=1).mean()
            t = max(float(self.args.pcl_temp), 1e-6)
            logits_proto = (z_n @ protos.T) / t
            logits_proto = logits_proto.masked_fill(~valid[None, :], -1e4)
            contra = nn.functional.cross_entropy(logits_proto, labels)
            pcl_flops = 6 * batch_size * int(self.num_feats) + 4 * batch_size * int(self.num_classes)
            prof.add_local_flops(pcl_flops, bucket="extra")
            return float(self.args.pcl_weight) * (align + contra)
        if self.args.method == "fedproto":
            protos = method_state.get("prototypes", None)
            counts = method_state.get("proto_counts", None)
            if protos is None or counts is None:
                return zero
            if float(counts.sum().item()) <= 0:
                return zero
            protos = protos.to(device=z.device, dtype=z.dtype)
            valid = counts.to(device=z.device, dtype=z.dtype) > 0
            label_valid = valid[labels]
            if not bool(label_valid.any()):
                return zero
            target = z.detach().clone()
            target[label_valid] = protos[labels[label_valid]]
            mse = nn.functional.mse_loss(z, target)
            proto_flops = 3 * batch_size * int(self.num_feats)
            prof.add_local_flops(proto_flops, bucket="extra")
            return float(self.args.proto_lamda) * mse
        return zero

    def _apply_scaffold_correction(self, local_head: nn.Module, method_state: dict, client_id: int, prof: ProfileMeter):
        if self.args.method != "scaffold":
            return
        c_g = method_state["c_global"]
        c_i = method_state["c_local"][client_id]
        corr_flops = 0
        for name, p in local_head.named_parameters():
            if p.grad is None:
                continue
            correction = c_g[name].to(device=p.grad.device, dtype=p.grad.dtype) - c_i[name].to(
                device=p.grad.device, dtype=p.grad.dtype
            )
            p.grad.add_(float(self.args.scaffold_eta) * correction)
            corr_flops += int(correction.numel() * 2)
        if corr_flops > 0:
            prof.add_local_flops(corr_flops, bucket="extra")

    def _collect_client_aux(self, z: torch.Tensor, y: torch.Tensor, aux: dict):
        if self.args.method in {"fedpcl", "fedproto"}:
            if "proto_sum" not in aux:
                aux["proto_sum"] = torch.zeros((self.num_classes, self.num_feats), device=z.device, dtype=torch.float32)
                aux["proto_cnt"] = torch.zeros((self.num_classes,), device=z.device, dtype=torch.float32)
            aux["proto_sum"].index_add_(0, y, z.detach().to(dtype=torch.float32))
            aux["proto_cnt"] += torch.bincount(y, minlength=self.num_classes).to(dtype=torch.float32)
        if self.args.method == "ccvr":
            if "class_cnt" not in aux:
                aux["class_cnt"] = torch.zeros((self.num_classes,), device=y.device, dtype=torch.float32)
            aux["class_cnt"] += torch.bincount(y, minlength=self.num_classes).to(dtype=torch.float32)

    def _train_one_client(
        self,
        client_id: int,
        global_state: dict[str, torch.Tensor],
        method_state: dict,
    ) -> dict:
        loader = None
        cached_z = None
        cached_y = None
        client_num_samples = int(len(self.client_indices[client_id]))
        if self.use_feature_cache:
            cached_z, cached_y = self._load_or_build_client_features(client_id)
        else:
            loader = self._build_client_loader(client_id)
            client_num_samples = int(len(loader.dataset))
        local_head = get_fc(num_feats=self.num_feats, num_classes=self.num_classes, distributed=False)
        local_head.load_state_dict(global_state)
        local_head.train()
        global_head = get_fc(num_feats=self.num_feats, num_classes=self.num_classes, distributed=False)
        global_head.load_state_dict(global_state)
        global_head.eval()
        assert_head_only_trainable(self.backbone, local_head)
        fl_optimizer = str(getattr(self.args, "fl_optimizer", "sgd")).lower()
        fl_scheduler = str(getattr(self.args, "fl_scheduler", "constant")).lower()
        if fl_optimizer == "adam":
            optimizer, effective_lr = build_linear_head_optimizer(
                local_head.parameters(),
                base_lr=float(self.args.local_lr),
                batch_size=int(self.args.local_batch_size),
                momentum=float(self.args.local_momentum),
                weight_decay=float(self.args.local_weight_decay),
            )
        elif fl_optimizer == "sgd":
            optimizer, effective_lr = build_sgd_linear_head_optimizer(
                local_head.parameters(),
                base_lr=float(self.args.local_lr),
                batch_size=int(self.args.local_batch_size),
                momentum=float(self.args.local_momentum),
                weight_decay=float(self.args.local_weight_decay),
            )
        else:
            raise ValueError(f"Unsupported FL optimizer: {self.args.fl_optimizer}")
        if fl_scheduler == "cosine":
            scheduler = build_cosine_scheduler(optimizer, total_epochs=max(int(self.args.local_epochs), 1))
        elif fl_scheduler == "constant":
            scheduler = build_constant_scheduler(optimizer)
        else:
            raise ValueError(f"Unsupported FL scheduler: {self.args.fl_scheduler}")
        scaler = GradScaler(enabled=bool(self.args.use_amp))
        prof = ProfileMeter()
        aux = {}
        steps = 0
        for _ in range(int(self.args.local_epochs)):
            max_train_batches = max(int(self.args.smoke_max_train_batches), 0)
            seen_batches = 0
            if self.use_feature_cache:
                if int(cached_y.shape[0]) > 0:
                    order = torch.randperm(int(cached_y.shape[0]))
                else:
                    order = torch.zeros((0,), dtype=torch.long)
                for start in range(0, int(order.shape[0]), int(self.args.local_batch_size)):
                    batch_idx = order[start : start + int(self.args.local_batch_size)]
                    z = cached_z.index_select(0, batch_idx).cuda(non_blocking=True)
                    y = cached_y.index_select(0, batch_idx).cuda(non_blocking=True)
                    with autocast(enabled=bool(self.args.use_amp)):
                        logits = local_head(z)
                        ce = nn.functional.cross_entropy(logits, y)
                        loss = ce + self._extra_loss(
                            logits=logits,
                            labels=y,
                            z=z.detach(),
                            local_head=local_head,
                            global_head=global_head,
                            method_state=method_state,
                            prof=prof,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if bool(self.args.use_amp):
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        self._apply_scaffold_correction(local_head, method_state, client_id, prof)
                        if float(self.args.grad_clip_norm) > 0:
                            torch.nn.utils.clip_grad_norm_(local_head.parameters(), max_norm=float(self.args.grad_clip_norm))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self._apply_scaffold_correction(local_head, method_state, client_id, prof)
                        if float(self.args.grad_clip_norm) > 0:
                            torch.nn.utils.clip_grad_norm_(local_head.parameters(), max_norm=float(self.args.grad_clip_norm))
                        optimizer.step()
                    self._collect_client_aux(z=z.detach(), y=y.detach(), aux=aux)
                    prof.add_local_flops(
                        linear_head_step_flops(
                            batch_size=int(y.shape[0]),
                            in_dim=int(self.num_feats),
                            out_dim=int(self.num_classes),
                        ),
                        bucket="base",
                    )
                    steps += 1
                    seen_batches += 1
                    if max_train_batches > 0 and seen_batches >= max_train_batches:
                        break
            else:
                for x, y in loader:
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
                    with autocast(enabled=bool(self.args.use_amp)):
                        with torch.no_grad():
                            x = self._train_preprocess(x)
                            z = self.backbone(x)
                        logits = local_head(z)
                        ce = nn.functional.cross_entropy(logits, y)
                        loss = ce + self._extra_loss(
                            logits=logits,
                            labels=y,
                            z=z.detach(),
                            local_head=local_head,
                            global_head=global_head,
                            method_state=method_state,
                            prof=prof,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if bool(self.args.use_amp):
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        self._apply_scaffold_correction(local_head, method_state, client_id, prof)
                        if float(self.args.grad_clip_norm) > 0:
                            torch.nn.utils.clip_grad_norm_(local_head.parameters(), max_norm=float(self.args.grad_clip_norm))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self._apply_scaffold_correction(local_head, method_state, client_id, prof)
                        if float(self.args.grad_clip_norm) > 0:
                            torch.nn.utils.clip_grad_norm_(local_head.parameters(), max_norm=float(self.args.grad_clip_norm))
                        optimizer.step()
                    self._collect_client_aux(z=z.detach(), y=y.detach(), aux=aux)
                    prof.add_local_flops(
                        linear_head_step_flops(
                            batch_size=int(y.shape[0]),
                            in_dim=int(self.num_feats),
                            out_dim=int(self.num_classes),
                        ),
                        bucket="base",
                    )
                    steps += 1
                    seen_batches += 1
                    if max_train_batches > 0 and seen_batches >= max_train_batches:
                        break
            scheduler.step()
        local_state = _clone_state_dict_to_cpu(local_head)
        upload = {
            "client_id": int(client_id),
            "state": local_state,
            "num_samples": int(client_num_samples),
            "flops": prof.local_flops,
            "flops_breakdown": prof.flops_breakdown_dict(),
            "steps": steps,
        }
        if self.args.method == "scaffold":
            c_g = method_state["c_global"]
            c_i = method_state["c_local"][client_id]
            new_c_i = {}
            delta_c = {}
            denom = max(int(steps), 1) * max(float(effective_lr), 1e-12)
            for name in local_state.keys():
                g = global_state[name].to(dtype=torch.float32)
                l = local_state[name].to(dtype=torch.float32)
                delta = (g - l) / float(denom)
                new_c = c_i[name].to(dtype=torch.float32) - c_g[name].to(dtype=torch.float32) + delta
                delta_c[name] = (new_c - c_i[name].to(dtype=torch.float32)).cpu()
                new_c_i[name] = new_c.cpu()
            upload["new_c_i"] = new_c_i
            upload["delta_c"] = delta_c
        if "proto_sum" in aux:
            upload["proto_sum"] = aux["proto_sum"].detach().cpu()
            upload["proto_cnt"] = aux["proto_cnt"].detach().cpu()
        if "class_cnt" in aux:
            upload["class_cnt"] = aux["class_cnt"].detach().cpu()
        return upload

    def _server_update(self, global_state: dict[str, torch.Tensor], uploads: list[dict], method_state: dict) -> tuple[dict, int]:
        weights_raw = [float(u["num_samples"]) for u in uploads]
        total = max(sum(weights_raw), 1.0)
        weights = [w / total for w in weights_raw]
        new_state = _weighted_average([u["state"] for u in uploads], weights)
        algo_flops = 0
        if self.args.method == "scaffold":
            c_global = method_state["c_global"]
            c_local = method_state["c_local"]
            for k in c_global.keys():
                avg_delta = None
                for u in uploads:
                    cur = u["delta_c"][k].to(dtype=torch.float32)
                    avg_delta = cur if avg_delta is None else (avg_delta + cur)
                avg_delta = avg_delta / float(len(uploads))
                c_global[k] = (c_global[k].to(dtype=torch.float32) + avg_delta).cpu()
                algo_flops += int(avg_delta.numel() * 2)
            for u in uploads:
                c_local[int(u["client_id"])] = {k: v.cpu() for k, v in u["new_c_i"].items()}
        if self.args.method == "fedpcl":
            old_protos = method_state.get("prototypes", torch.zeros((self.num_classes, self.num_feats), dtype=torch.float32))
            proto_sum = torch.zeros((self.num_classes, self.num_feats), dtype=torch.float32)
            proto_cnt = torch.zeros((self.num_classes,), dtype=torch.float32)
            for u in uploads:
                if "proto_sum" in u:
                    proto_sum += u["proto_sum"].to(dtype=torch.float32)
                    proto_cnt += u["proto_cnt"].to(dtype=torch.float32)
            valid = proto_cnt > 0
            protos = torch.zeros_like(proto_sum)
            protos[valid] = proto_sum[valid] / proto_cnt[valid][:, None]
            protos[valid] = nn.functional.normalize(protos[valid], dim=1)
            m = min(max(float(self.args.pcl_momentum), 0.0), 1.0)
            if m > 0:
                mixed = old_protos.to(dtype=torch.float32).clone()
                mixed[valid] = m * mixed[valid] + (1.0 - m) * protos[valid]
                mixed[valid] = nn.functional.normalize(mixed[valid], dim=1)
                method_state["prototypes"] = mixed
            else:
                method_state["prototypes"] = protos
            method_state["proto_counts"] = proto_cnt
            algo_flops += int((self.num_classes * self.num_feats + self.num_classes) * (2 * len(uploads) + 2))
        if self.args.method == "fedproto":
            proto_sum = torch.zeros((self.num_classes, self.num_feats), dtype=torch.float32)
            proto_cnt = torch.zeros((self.num_classes,), dtype=torch.float32)
            for u in uploads:
                if "proto_sum" in u:
                    proto_sum += u["proto_sum"].to(dtype=torch.float32)
                    proto_cnt += u["proto_cnt"].to(dtype=torch.float32)
            valid = proto_cnt > 0
            protos = torch.zeros_like(proto_sum)
            protos[valid] = proto_sum[valid] / proto_cnt[valid][:, None].clamp_min(1e-12)
            method_state["prototypes"] = protos
            method_state["proto_counts"] = proto_cnt
            algo_flops += int((self.num_classes * self.num_feats + self.num_classes) * (2 * len(uploads) + 1))
        if self.args.method == "ccvr":
            cls_cnt = torch.zeros((self.num_classes,), dtype=torch.float32)
            for u in uploads:
                if "class_cnt" in u:
                    cls_cnt += u["class_cnt"].to(dtype=torch.float32)
            if float(cls_cnt.sum().item()) > 0:
                prior = cls_cnt / cls_cnt.sum().clamp_min(1e-12)
                bias_name = None
                for name in new_state.keys():
                    if name.endswith("linear.bias") or name.endswith("bias"):
                        bias_name = name
                if bias_name is not None and int(new_state[bias_name].numel()) == self.num_classes:
                    new_state[bias_name] = new_state[bias_name].to(dtype=torch.float32) + prior.clamp_min(1e-12).log()
                    algo_flops += int(self.num_classes * 4)
            method_state["class_counts"] = cls_cnt
        return {k: v.cpu() for k, v in new_state.items()}, int(algo_flops)

    def _round_comm_cost(self, global_state: dict[str, torch.Tensor], uploads: list[dict], method_state: dict) -> tuple[int, int]:
        model_bytes = state_dict_bytes(global_state)
        down = model_bytes * len(uploads)
        up = 0
        for u in uploads:
            up += state_dict_bytes(u["state"])
        if self.args.method == "scaffold":
            c_bytes = state_dict_bytes(method_state["c_global"])
            down += c_bytes * len(uploads)
            for u in uploads:
                up += state_dict_bytes(u["delta_c"])
        if self.args.method in {"fedpcl", "fedproto"}:
            proto = method_state.get("prototypes", torch.zeros((self.num_classes, self.num_feats), dtype=torch.float32))
            cnt = method_state.get("proto_counts", torch.zeros((self.num_classes,), dtype=torch.float32))
            down += int(proto.numel() * proto.element_size() + cnt.numel() * cnt.element_size()) * len(uploads)
            for u in uploads:
                if "proto_sum" in u:
                    up += int(u["proto_sum"].numel() * u["proto_sum"].element_size())
                    up += int(u["proto_cnt"].numel() * u["proto_cnt"].element_size())
        if self.args.method == "ccvr":
            for u in uploads:
                if "class_cnt" in u:
                    up += int(u["class_cnt"].numel() * u["class_cnt"].element_size())
        return int(down), int(up)

    def _round_comm_cost_one_client(self, global_state: dict[str, torch.Tensor], upload: dict, method_state: dict) -> tuple[int, int]:
        down = state_dict_bytes(global_state)
        up = state_dict_bytes(upload["state"])
        if self.args.method == "scaffold":
            down += state_dict_bytes(method_state["c_global"])
            up += state_dict_bytes(upload["delta_c"])
        if self.args.method in {"fedpcl", "fedproto"}:
            proto = method_state.get("prototypes", torch.zeros((self.num_classes, self.num_feats), dtype=torch.float32))
            cnt = method_state.get("proto_counts", torch.zeros((self.num_classes,), dtype=torch.float32))
            down += int(proto.numel() * proto.element_size() + cnt.numel() * cnt.element_size())
            if "proto_sum" in upload:
                up += int(upload["proto_sum"].numel() * upload["proto_sum"].element_size())
                up += int(upload["proto_cnt"].numel() * upload["proto_cnt"].element_size())
        if self.args.method == "ccvr":
            if "class_cnt" in upload:
                up += int(upload["class_cnt"].numel() * upload["class_cnt"].element_size())
        return int(down), int(up)

    def _estimate_server_flops_round(self, global_state: dict[str, torch.Tensor], uploads: list[dict], method_state: dict) -> dict[str, int]:
        num_clients = max(len(uploads), 1)
        params = _state_numel(global_state)
        base_flops = params * (2 * num_clients + 1)
        algo_flops = 0
        if self.args.method == "scaffold":
            algo_flops += params * (num_clients + 2)
        if self.args.method in {"fedpcl", "fedproto"}:
            proto_elems = int(self.num_classes * self.num_feats)
            cnt_elems = int(self.num_classes)
            algo_flops += (proto_elems + cnt_elems) * (2 * num_clients + 1)
        if self.args.method == "ccvr":
            algo_flops += int(self.num_classes * (2 * num_clients + 8))
        return {
            "base": int(base_flops),
            "extra": int(algo_flops),
            "setup": 0,
            "total": int(base_flops + algo_flops),
        }

    def _select_clients_for_round(self, rng: np.random.Generator) -> list[int]:
        all_ids = list(range(int(self.args.num_clients)))
        if len(all_ids) == 0:
            return []
        join_ratio = float(getattr(self.args, "join_ratio", 1.0))
        join_ratio = min(max(join_ratio, 0.0), 1.0)
        fixed_num = max(int(int(self.args.num_clients) * join_ratio), 1)
        if bool(getattr(self.args, "random_join_ratio", False)):
            num_join = int(rng.integers(fixed_num, int(self.args.num_clients) + 1))
        else:
            num_join = fixed_num
        picked = rng.choice(np.asarray(all_ids, dtype=np.int64), size=num_join, replace=False)
        return [int(v) for v in np.atleast_1d(picked).tolist()]

    def _ccvr_collect_stats(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        sum_feat = torch.zeros((self.num_classes, self.num_feats), device="cuda", dtype=torch.float64)
        sumsq_feat = torch.zeros((self.num_classes, self.num_feats), device="cuda", dtype=torch.float64)
        counts = torch.zeros((self.num_classes,), device="cuda", dtype=torch.float64)
        stats_flops = 0
        for client_id in range(self.args.num_clients):
            loader = self._build_client_loader(client_id)
            max_train_batches = max(int(self.args.smoke_max_train_batches), 0)
            seen_batches = 0
            for x, y in loader:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                with torch.no_grad():
                    x = self._train_preprocess(x)
                    z = self.backbone(x).to(dtype=torch.float64)
                sum_feat.index_add_(0, y, z)
                sumsq_feat.index_add_(0, y, z * z)
                stats_flops += int(y.shape[0]) * int(self.num_feats) * 4
                counts += torch.bincount(y, minlength=self.num_classes).to(device=counts.device, dtype=torch.float64)
                seen_batches += 1
                if max_train_batches > 0 and seen_batches >= max_train_batches:
                    break
        valid = counts > 0
        means = torch.zeros_like(sum_feat)
        means[valid] = sum_feat[valid] / counts[valid].unsqueeze(1).clamp_min(1e-12)
        vars_diag = torch.ones_like(sum_feat) * 1e-6
        vars_diag[valid] = (
            sumsq_feat[valid] / counts[valid].unsqueeze(1).clamp_min(1e-12) - means[valid] * means[valid]
        ).clamp_min(1e-6)
        return means.to(dtype=torch.float32), vars_diag.to(dtype=torch.float32), int(stats_flops)

    def _run_ccvr_calibration(self, global_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        calib_epochs = max(int(self.args.ccvr_calib_epochs), 0)
        samples_per_class = max(int(self.args.ccvr_calib_samples_per_class), 1)
        if calib_epochs <= 0:
            return global_state, {"base": 0, "extra": 0, "setup": 0, "total": 0}
        means, vars_diag, stats_flops = self._ccvr_collect_stats()
        head = get_fc(num_feats=self.num_feats, num_classes=self.num_classes, distributed=False)
        head.load_state_dict(global_state)
        head.train()
        optimizer, _ = build_sgd_linear_head_optimizer(
            head.parameters(),
            base_lr=float(self.args.ccvr_calib_lr),
            batch_size=int(self.args.ccvr_calib_samples_per_class),
        )
        scheduler = build_constant_scheduler(optimizer)
        total_flops = 0
        for _ in range(calib_epochs):
            for cls in range(self.num_classes):
                mu = means[cls]
                var = vars_diag[cls]
                if float(var.mean().item()) <= 0:
                    continue
                std = var.sqrt()
                z = torch.randn((samples_per_class, self.num_feats), device="cuda", dtype=torch.float32)
                z = z * std[None, :] + mu[None, :]
                y = torch.full((samples_per_class,), int(cls), device="cuda", dtype=torch.long)
                logits = head(z)
                loss = nn.functional.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(self.args.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=float(self.args.grad_clip_norm))
                optimizer.step()
                total_flops += linear_head_step_flops(samples_per_class, int(self.num_feats), int(self.num_classes))
            scheduler.step()
        calib_total = int(total_flops + stats_flops)
        return _clone_state_dict_to_cpu(head), {
            "base": 0,
            "extra": 0,
            "setup": int(calib_total),
            "total": int(calib_total),
        }

    def _afl_build_head(self) -> nn.Module:
        head = _AFLLinearHead(self.num_feats, self.num_classes).cuda()
        head.eval()
        return head

    def _afl_local_update(self, client_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
        corr_rep = torch.zeros((self.num_feats, self.num_feats), device="cuda", dtype=torch.float64)
        corr_label = torch.zeros((self.num_feats, self.num_classes), device="cuda", dtype=torch.float64)
        prof = ProfileMeter()
        if self.use_feature_cache:
            z_all, y_all = self._load_or_build_client_features(client_id)
            max_train_batches = max(int(self.args.smoke_max_train_batches), 0)
            if max_train_batches > 0:
                limit = int(max_train_batches) * int(self.args.local_batch_size)
                z_all = z_all[:limit]
                y_all = y_all[:limit]
            if int(y_all.shape[0]) > 0:
                z = z_all.cuda(non_blocking=True).to(dtype=torch.float64)
                y = y_all.cuda(non_blocking=True)
                onehot = torch.zeros((y.shape[0], self.num_classes), device="cuda", dtype=torch.float64)
                onehot.scatter_(1, y[:, None], 1.0)
                corr_rep += z.T @ z
                corr_label += z.T @ onehot
                prof.add_local_flops(
                    linear_head_step_flops(int(y.shape[0]), int(self.num_feats), int(self.num_classes)),
                    bucket="base",
                )
                prof.add_local_flops(
                    int(y.shape[0]) * int(self.num_feats) * int(self.num_classes) * 2,
                    bucket="extra",
                )
        else:
            loader = self._build_client_loader(client_id)
            max_train_batches = max(int(self.args.smoke_max_train_batches), 0)
            seen_batches = 0
            with torch.no_grad():
                for x, y in loader:
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
                    x = self._train_preprocess(x)
                    z = self.backbone(x).to(dtype=torch.float64)
                    onehot = torch.zeros((y.shape[0], self.num_classes), device="cuda", dtype=torch.float64)
                    onehot.scatter_(1, y[:, None], 1.0)
                    corr_rep += z.T @ z
                    corr_label += z.T @ onehot
                    prof.add_local_flops(
                        linear_head_step_flops(int(y.shape[0]), int(self.num_feats), int(self.num_classes)),
                        bucket="base",
                    )
                    prof.add_local_flops(
                        int(y.shape[0]) * int(self.num_feats) * int(self.num_classes) * 2,
                        bucket="extra",
                    )
                    seen_batches += 1
                    if max_train_batches > 0 and seen_batches >= max_train_batches:
                        break
        reg = float(self.args.afl_ri_reg)
        eye = torch.eye(self.num_feats, device="cuda", dtype=torch.float64)
        r_mat = torch.linalg.inv(corr_rep + reg * eye)
        w_mat = r_mat @ corr_label
        c_mat = torch.linalg.inv(r_mat)
        prof.add_local_flops(int(self.num_feats**3 + self.num_feats * self.num_feats * self.num_classes), bucket="extra")
        return w_mat.detach().cpu(), r_mat.detach().cpu(), c_mat.detach().cpu(), prof.flops_breakdown_dict()

    def _afl_aggregate(
        self,
        weights: list[torch.Tensor],
        r_mats: list[torch.Tensor],
        c_mats: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(weights) < 2:
            return (
                weights[0].cuda(non_blocking=True).to(dtype=torch.float64),
                r_mats[0].cuda(non_blocking=True).to(dtype=torch.float64),
                c_mats[0].cuda(non_blocking=True).to(dtype=torch.float64),
            )
        r0 = r_mats[0].cuda(non_blocking=True).to(dtype=torch.float64)
        c0 = c_mats[0].cuda(non_blocking=True).to(dtype=torch.float64)
        w0 = weights[0].cuda(non_blocking=True).to(dtype=torch.float64)
        r1 = r_mats[1].cuda(non_blocking=True).to(dtype=torch.float64)
        c1 = c_mats[1].cuda(non_blocking=True).to(dtype=torch.float64)
        w1 = weights[1].cuda(non_blocking=True).to(dtype=torch.float64)
        eye = torch.eye(r0.shape[0], device="cuda", dtype=torch.float64)
        wt = (eye - r0 @ c1 + r0 @ c1 @ torch.linalg.inv(c0 + c1) @ c1) @ w0 + (
            eye - r1 @ c0 + r1 @ c0 @ torch.linalg.inv(c0 + c1) @ c0
        ) @ w1
        ct = c0 + c1
        rt = torch.linalg.inv(ct)
        for i in range(1, len(weights) - 1):
            r_next = r_mats[i + 1].cuda(non_blocking=True).to(dtype=torch.float64)
            c_next = c_mats[i + 1].cuda(non_blocking=True).to(dtype=torch.float64)
            w_next = weights[i + 1].cuda(non_blocking=True).to(dtype=torch.float64)
            inv_term = torch.linalg.inv(ct + c_next)
            wt = (eye - rt @ c_next + rt @ c_next @ inv_term @ c_next) @ wt + (
                eye - r_next @ ct + r_next @ ct @ inv_term @ ct
            ) @ w_next
            ct = ct + c_next
            rt = torch.linalg.inv(ct)
        return wt, rt, ct

    def _afl_clean_regularization(self, weight: torch.Tensor, c_mat: torch.Tensor) -> torch.Tensor:
        reg = float(self.args.afl_ri_reg)
        eye = torch.eye(self.num_feats, device=weight.device, dtype=torch.float64)
        r_origin = torch.linalg.inv(c_mat - int(self.args.num_clients) * reg * eye)
        return weight + (int(self.args.num_clients) * reg * r_origin) @ weight

    def _run_seed(self, seed: int, seed_rank: int) -> dict:
        set_global_seed(seed)
        out_dir = os.path.join(
            self.args.output_root,
            self.args.method,
            self.args.dataset,
            self.args.model,
            f"seed_{seed}",
        )
        ensure_dir(out_dir)
        head = get_fc(num_feats=self.num_feats, num_classes=self.num_classes, distributed=False)
        head.train()
        global_state = _clone_state_dict_to_cpu(head)
        method_state = self._init_method_state(global_state)
        stopper = EarlyStopper(
            patience_rounds=self.args.patience_rounds,
            min_delta=self.args.min_delta,
            warmup_rounds=self.args.warmup_rounds,
        )
        enable_early_stop = str(self.args.method).lower() != "fedavg"
        rounds = []
        curve_x = []
        curve_y = []
        meter = ProfileMeter()
        stop_round = self.args.max_rounds
        converged_round = self.args.max_rounds
        converged_acc = 0.0
        rng_round = np.random.default_rng(int(seed) + 13)
        for r in range(1, int(self.args.max_rounds) + 1):
            selected_clients = self._select_clients_for_round(rng_round)
            uploads = []
            for client_id in selected_clients:
                up = self._train_one_client(client_id=client_id, global_state=global_state, method_state=method_state)
                uploads.append(up)
                meter.merge_flops_breakdown(up.get("flops_breakdown", {}))
            if len(uploads) == 0:
                continue
            drop_rate = min(max(float(getattr(self.args, "client_drop_rate", 0.0)), 0.0), 0.99)
            if drop_rate > 0:
                keep_num = max(int((1.0 - drop_rate) * len(uploads)), 1)
                keep_idx = rng_round.choice(np.arange(len(uploads)), size=keep_num, replace=False)
                uploads = [uploads[int(i)] for i in np.atleast_1d(keep_idx).tolist()]
            down_b, up_b = self._round_comm_cost(global_state=global_state, uploads=uploads, method_state=method_state)
            one_down, one_up = self._round_comm_cost_one_client(
                global_state=global_state,
                upload=uploads[0],
                method_state=method_state,
            )
            server_flops_round = self._estimate_server_flops_round(
                global_state=global_state,
                uploads=uploads,
                method_state=method_state,
            )
            client_base_round_total = int(sum(int(u.get("flops_breakdown", {}).get("base", 0)) for u in uploads))
            client_extra_round_total = int(sum(int(u.get("flops_breakdown", {}).get("extra", 0)) for u in uploads))
            client_setup_round_total = int(sum(int(u.get("flops_breakdown", {}).get("setup", 0)) for u in uploads))
            client_flops_round_total = int(client_base_round_total + client_extra_round_total + client_setup_round_total)
            one_breakdown = uploads[0].get("flops_breakdown", {})
            client_flops_round_one = int(uploads[0]["flops"])
            global_state, server_algo_update_flops = self._server_update(
                global_state=global_state,
                uploads=uploads,
                method_state=method_state,
            )
            server_flops_round["extra"] = int(server_flops_round["extra"] + server_algo_update_flops)
            server_flops_round["total"] = int(
                server_flops_round["base"] + server_flops_round["extra"] + server_flops_round["setup"]
            )
            head.load_state_dict(global_state)
            top1, top5 = self._evaluate(head)
            meter.add_down(down_b)
            meter.add_up(up_b)
            meter_breakdown = meter.flops_breakdown_dict()
            row = {
                "round": r,
                "val_top1": float(top1),
                "val_top5": float(top5),
                "server_flops_round_est": int(server_flops_round["total"]),
                "server_flops_base_round_est": int(server_flops_round["base"]),
                "server_flops_extra_round_est": int(server_flops_round["extra"]),
                "server_flops_setup_round_est": int(server_flops_round["setup"]),
                "client_flops_round_total": int(client_flops_round_total),
                "client_flops_round_one": int(client_flops_round_one),
                "client_flops_base_round_total": int(client_base_round_total),
                "client_flops_extra_round_total": int(client_extra_round_total),
                "client_flops_setup_round_total": int(client_setup_round_total),
                "client_flops_base_round_one": int(one_breakdown.get("base", 0)),
                "client_flops_extra_round_one": int(one_breakdown.get("extra", 0)),
                "client_flops_setup_round_one": int(one_breakdown.get("setup", 0)),
                "client_bytes_up_round_one": int(one_up),
                "client_bytes_down_round_one": int(one_down),
                "client_bytes_total_round_one": int(one_up + one_down),
                "local_flops_cum": int(meter.local_flops),
                "local_flops_base_cum": int(meter_breakdown.get("base", 0)),
                "local_flops_extra_cum": int(meter_breakdown.get("extra", 0)),
                "local_flops_setup_cum": int(meter_breakdown.get("setup", 0)),
                "bytes_up_cum": int(meter.bytes_up),
                "bytes_down_cum": int(meter.bytes_down),
                "bytes_total_cum": int(meter.bytes_total),
            }
            rounds.append(row)
            write_jsonl(os.path.join(out_dir, "round_metrics.jsonl"), rounds)
            if seed_rank == 0:
                curve_x.append(r)
                curve_y.append(float(top1))
            if enable_early_stop:
                stop_info = stopper.update(round_id=r, acc=top1, chance_acc=(1.0 / max(self.num_classes, 1)))
                if stop_info["is_converged"]:
                    stop_round = int(r)
                    break
            else:
                stopper.best_acc = max(float(stopper.best_acc), float(top1))
                if float(top1) >= float(stopper.best_acc):
                    stopper.best_round = int(r)
            stop_round = int(r)
        if self.args.method == "ccvr":
            global_state, ccvr_flops = self._run_ccvr_calibration(global_state)
            head.load_state_dict(global_state)
            top1, top5 = self._evaluate(head)
            meter.add_local_flops(int(ccvr_flops["setup"]), bucket="setup")
            meter_breakdown = meter.flops_breakdown_dict()
            rounds.append(
                {
                    "round": int(stop_round),
                    "stage": "ccvr_calibration",
                    "val_top1": float(top1),
                    "val_top5": float(top5),
                    "server_flops_round_est": int(ccvr_flops["total"]),
                    "server_flops_base_round_est": int(ccvr_flops["base"]),
                    "server_flops_extra_round_est": int(ccvr_flops["extra"]),
                    "server_flops_setup_round_est": int(ccvr_flops["setup"]),
                    "client_flops_round_total": 0,
                    "client_flops_round_one": 0,
                    "client_flops_base_round_total": 0,
                    "client_flops_extra_round_total": 0,
                    "client_flops_setup_round_total": 0,
                    "client_flops_base_round_one": 0,
                    "client_flops_extra_round_one": 0,
                    "client_flops_setup_round_one": 0,
                    "client_bytes_up_round_one": 0,
                    "client_bytes_down_round_one": 0,
                    "client_bytes_total_round_one": 0,
                    "local_flops_cum": int(meter.local_flops),
                    "local_flops_base_cum": int(meter_breakdown.get("base", 0)),
                    "local_flops_extra_cum": int(meter_breakdown.get("extra", 0)),
                    "local_flops_setup_cum": int(meter_breakdown.get("setup", 0)),
                    "bytes_up_cum": int(meter.bytes_up),
                    "bytes_down_cum": int(meter.bytes_down),
                    "bytes_total_cum": int(meter.bytes_total),
                }
            )
            write_jsonl(os.path.join(out_dir, "round_metrics.jsonl"), rounds)
            if top1 > float(stopper.best_acc):
                stopper.best_acc = float(top1)
                stopper.best_round = int(stop_round)
        if int(stopper.best_round) > 0:
            converged_round = int(stopper.best_round)
            converged_acc = float(stopper.best_acc)
        else:
            converged_round = int(stop_round)
            converged_acc = 0.0
        if seed_rank == 0:
            write_json(os.path.join(out_dir, "curve_acc_vs_round.json"), {"rounds": curve_x, "val_top1": curve_y})
            write_curve_csv(os.path.join(out_dir, "curve_acc_vs_round.csv"), curve_x, curve_y)
        result = {
            "seed": int(seed),
            "method": self.args.method,
            "dataset": self.args.dataset,
            "model": self.args.model,
            "best_acc": float(stopper.best_acc),
            "best_round": int(stopper.best_round),
            "converged_round": int(converged_round),
            "converged_acc": float(converged_acc),
            "stop_round": int(stop_round),
            "local_flops": int(meter.local_flops),
            "local_flops_base": int(meter.flops_breakdown_dict().get("base", 0)),
            "local_flops_extra": int(meter.flops_breakdown_dict().get("extra", 0)),
            "local_flops_setup": int(meter.flops_breakdown_dict().get("setup", 0)),
            "bytes_up": int(meter.bytes_up),
            "bytes_down": int(meter.bytes_down),
            "bytes_total": int(meter.bytes_total),
        }
        write_json(os.path.join(out_dir, "result_seed.json"), result)
        write_json(
            os.path.join(out_dir, "run_config.json"),
            {
                "args": self.args.__dict__,
                "partition_payload": self.partition_payload,
            },
        )
        return result

    def run(self) -> dict:
        if self.args.method == "afl":
            return self._run_afl()
        seed_results = []
        for i, seed in enumerate(self.args.seeds):
            seed_results.append(self._run_seed(seed=int(seed), seed_rank=i))
        accs = [float(v["best_acc"]) for v in seed_results]
        convs = [float(v["converged_round"]) for v in seed_results]
        flops = [float(v["local_flops"]) for v in seed_results]
        flops_base = [float(v.get("local_flops_base", 0.0)) for v in seed_results]
        flops_extra = [float(v.get("local_flops_extra", 0.0)) for v in seed_results]
        flops_setup = [float(v.get("local_flops_setup", 0.0)) for v in seed_results]
        comm = [float(v["bytes_total"]) for v in seed_results]
        acc_mean, acc_std = mean_std(accs)
        conv_mean, conv_std = mean_std(convs)
        flops_mean, flops_std = mean_std(flops)
        flops_base_mean, flops_base_std = mean_std(flops_base)
        flops_extra_mean, flops_extra_std = mean_std(flops_extra)
        flops_setup_mean, flops_setup_std = mean_std(flops_setup)
        comm_mean, comm_std = mean_std(comm)
        summary = {
            "method": self.args.method,
            "dataset": self.args.dataset,
            "model": self.args.model,
            "acc_mean": float(acc_mean),
            "acc_std": float(acc_std),
            "converged_round_mean": float(conv_mean),
            "converged_round_std": float(conv_std),
            "local_flops_mean": float(flops_mean),
            "local_flops_std": float(flops_std),
            "local_flops_base_mean": float(flops_base_mean),
            "local_flops_base_std": float(flops_base_std),
            "local_flops_extra_mean": float(flops_extra_mean),
            "local_flops_extra_std": float(flops_extra_std),
            "local_flops_setup_mean": float(flops_setup_mean),
            "local_flops_setup_std": float(flops_setup_std),
            "comm_bytes_mean": float(comm_mean),
            "comm_bytes_std": float(comm_std),
            "seed_results": seed_results,
        }
        write_json(
            os.path.join(self.args.output_root, self.args.method, self.args.dataset, self.args.model, "result_summary.json"),
            summary,
        )
        return summary

    def _run_afl(self) -> dict:
        seed_results = []
        for i, seed in enumerate(self.args.seeds):
            set_global_seed(seed)
            head = self._afl_build_head()
            local_weights = []
            local_r = []
            local_c = []
            prof = ProfileMeter()
            for client_id in range(self.args.num_clients):
                w_mat, r_mat, c_mat, flops_breakdown = self._afl_local_update(client_id)
                local_weights.append(w_mat)
                local_r.append(r_mat)
                local_c.append(c_mat)
                prof.merge_flops_breakdown(flops_breakdown)
            global_weight, _global_r, global_c = self._afl_aggregate(local_weights, local_r, local_c)
            if bool(self.args.afl_clean_reg):
                global_weight = self._afl_clean_regularization(global_weight, global_c)
            head.linear.weight.data.copy_(global_weight.T.to(dtype=head.linear.weight.dtype))
            top1, top5 = self._evaluate(head)
            upload_bytes_total = 0
            for w_mat, r_mat, c_mat in zip(local_weights, local_r, local_c):
                upload_bytes_total += int(w_mat.numel() * w_mat.element_size())
                upload_bytes_total += int(r_mat.numel() * r_mat.element_size())
                upload_bytes_total += int(c_mat.numel() * c_mat.element_size())
            prof.add_up(upload_bytes_total)
            out_dir = os.path.join(
                self.args.output_root,
                self.args.method,
                self.args.dataset,
                self.args.model,
                f"seed_{seed}",
            )
            ensure_dir(out_dir)
            afl_breakdown = prof.flops_breakdown_dict()
            upload_one = int(prof.bytes_up / max(self.args.num_clients, 1))
            write_json(os.path.join(out_dir, "curve_acc_vs_round.json"), {"rounds": [1], "val_top1": [float(top1)]})
            write_curve_csv(os.path.join(out_dir, "curve_acc_vs_round.csv"), [1], [float(top1)])
            write_jsonl(
                os.path.join(out_dir, "round_metrics.jsonl"),
                [
                    {
                        "round": 1,
                        "val_top1": float(top1),
                        "val_top5": float(top5),
                        "server_flops_round_est": 0,
                        "server_flops_base_round_est": 0,
                        "server_flops_extra_round_est": 0,
                        "server_flops_setup_round_est": 0,
                        "client_flops_round_total": int(prof.local_flops),
                        "client_flops_round_one": int(prof.local_flops / max(self.args.num_clients, 1)),
                        "client_flops_base_round_total": int(afl_breakdown.get("base", 0)),
                        "client_flops_extra_round_total": int(afl_breakdown.get("extra", 0)),
                        "client_flops_setup_round_total": int(afl_breakdown.get("setup", 0)),
                        "client_flops_base_round_one": int(afl_breakdown.get("base", 0) / max(self.args.num_clients, 1)),
                        "client_flops_extra_round_one": int(afl_breakdown.get("extra", 0) / max(self.args.num_clients, 1)),
                        "client_flops_setup_round_one": int(afl_breakdown.get("setup", 0) / max(self.args.num_clients, 1)),
                        "client_bytes_up_round_one": int(upload_one),
                        "client_bytes_down_round_one": 0,
                        "client_bytes_total_round_one": int(upload_one),
                        "local_flops_cum": int(prof.local_flops),
                        "local_flops_base_cum": int(afl_breakdown.get("base", 0)),
                        "local_flops_extra_cum": int(afl_breakdown.get("extra", 0)),
                        "local_flops_setup_cum": int(afl_breakdown.get("setup", 0)),
                        "bytes_up_cum": int(prof.bytes_up),
                        "bytes_down_cum": int(prof.bytes_down),
                        "bytes_total_cum": int(prof.bytes_total),
                    }
                ],
            )
            seed_result = {
                "seed": int(seed),
                "method": "afl",
                "dataset": self.args.dataset,
                "model": self.args.model,
                "best_acc": float(top1),
                "best_round": 1,
                "converged_round": 1,
                "converged_acc": float(top1),
                "local_flops": int(prof.local_flops),
                "local_flops_base": int(prof.flops_breakdown_dict().get("base", 0)),
                "local_flops_extra": int(prof.flops_breakdown_dict().get("extra", 0)),
                "local_flops_setup": int(prof.flops_breakdown_dict().get("setup", 0)),
                "bytes_up": int(prof.bytes_up),
                "bytes_down": int(prof.bytes_down),
                "bytes_total": int(prof.bytes_total),
            }
            write_json(os.path.join(out_dir, "result_seed.json"), seed_result)
            write_json(
                os.path.join(out_dir, "run_config.json"),
                {
                    "args": self.args.__dict__,
                    "partition_payload": self.partition_payload,
                },
            )
            seed_results.append(seed_result)
        accs = [float(v["best_acc"]) for v in seed_results]
        flops = [float(v["local_flops"]) for v in seed_results]
        flops_base = [float(v.get("local_flops_base", 0.0)) for v in seed_results]
        flops_extra = [float(v.get("local_flops_extra", 0.0)) for v in seed_results]
        flops_setup = [float(v.get("local_flops_setup", 0.0)) for v in seed_results]
        comm = [float(v["bytes_total"]) for v in seed_results]
        acc_mean, acc_std = mean_std(accs)
        flops_mean, flops_std = mean_std(flops)
        flops_base_mean, flops_base_std = mean_std(flops_base)
        flops_extra_mean, flops_extra_std = mean_std(flops_extra)
        flops_setup_mean, flops_setup_std = mean_std(flops_setup)
        comm_mean, comm_std = mean_std(comm)
        summary = {
            "method": "afl",
            "dataset": self.args.dataset,
            "model": self.args.model,
            "acc_mean": float(acc_mean),
            "acc_std": float(acc_std),
            "converged_round_mean": 1.0,
            "converged_round_std": 0.0,
            "local_flops_mean": float(flops_mean),
            "local_flops_std": float(flops_std),
            "local_flops_base_mean": float(flops_base_mean),
            "local_flops_base_std": float(flops_base_std),
            "local_flops_extra_mean": float(flops_extra_mean),
            "local_flops_extra_std": float(flops_extra_std),
            "local_flops_setup_mean": float(flops_setup_mean),
            "local_flops_setup_std": float(flops_setup_std),
            "comm_bytes_mean": float(comm_mean),
            "comm_bytes_std": float(comm_std),
            "seed_results": seed_results,
        }
        write_json(
            os.path.join(self.args.output_root, self.args.method, self.args.dataset, self.args.model, "result_summary.json"),
            summary,
        )
        return summary
