from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass
from typing import Callable

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class FederatedFeatureCacheSpec:
    dataset: str
    model: str
    real_res: int
    crop_res: int
    train_crop_mode: str
    num_clients: int
    partition: str
    dirichlet_alpha: float
    dirichlet_balance: bool
    dirichlet_min_size: int
    shard_per_client: int
    classes_per_client: int
    seed: int


def _sanitize_cache_token(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "", str(x))


def shared_feature_cache_dir(output_root: str) -> str:
    cur = os.path.abspath(str(output_root))
    while True:
        if os.path.basename(cur) == "output":
            return os.path.join(cur, "feature_cache")
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.join(os.path.abspath(str(output_root)), "feature_cache")
        cur = parent


def federated_feature_cache_client_file_name(client_id: int) -> str:
    return f"client_{int(client_id):04d}.pt"


def _client_indices_fingerprint(indices: list[int]) -> str:
    joined = ",".join(str(int(v)) for v in indices)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def federated_feature_cache_hparams_dir_name(spec: FederatedFeatureCacheSpec) -> str:
    partition = _sanitize_cache_token(spec.partition)
    crop_mode = _sanitize_cache_token(spec.train_crop_mode)
    alpha = f"{float(spec.dirichlet_alpha):.4f}".replace(".", "p")
    balance = "bal1" if bool(spec.dirichlet_balance) else "bal0"
    min_size = f"ms{int(spec.dirichlet_min_size)}"
    shards = f"sh{int(spec.shard_per_client)}"
    classes = f"cpc{int(spec.classes_per_client)}"
    return (
        f"rr{int(spec.real_res)}_cr{int(spec.crop_res)}"
        f"_cm{crop_mode}_nc{int(spec.num_clients)}"
        f"_p{partition}_a{alpha}_{balance}_{min_size}_{shards}_{classes}_s{int(spec.seed)}"
    )


def federated_feature_cache_scope_dir(cache_root: str, spec: FederatedFeatureCacheSpec) -> str:
    dataset = _sanitize_cache_token(spec.dataset)
    model = _sanitize_cache_token(spec.model)
    return os.path.join(cache_root, dataset, model, federated_feature_cache_hparams_dir_name(spec))


def _federated_feature_cache_file_name(spec: FederatedFeatureCacheSpec, client_id: int) -> str:
    dataset = _sanitize_cache_token(spec.dataset)
    model = _sanitize_cache_token(spec.model)
    partition = _sanitize_cache_token(spec.partition)
    crop_mode = _sanitize_cache_token(spec.train_crop_mode)
    alpha = f"{float(spec.dirichlet_alpha):.4f}".replace(".", "p")
    return (
        f"{dataset}_{model}_c{int(client_id)}"
        f"_rr{int(spec.real_res)}_cr{int(spec.crop_res)}"
        f"_cm{crop_mode}_nc{int(spec.num_clients)}"
        f"_p{partition}_a{alpha}_s{int(spec.seed)}.pt"
    )


def _legacy_federated_feature_cache_file_name(spec: FederatedFeatureCacheSpec, client_id: int) -> str:
    return f"{spec.dataset}_{spec.model}_c{int(client_id)}_r{int(spec.crop_res)}.pt"


def _federated_feature_cache_name_candidates(spec: FederatedFeatureCacheSpec, client_id: int) -> list[str]:
    return [
        federated_feature_cache_client_file_name(client_id),
        _federated_feature_cache_file_name(spec, client_id),
        _legacy_federated_feature_cache_file_name(spec, client_id),
    ]


def find_federated_cache_file_in_dir(cache_dir: str, *, spec: FederatedFeatureCacheSpec, client_id: int) -> str | None:
    dir_candidates = [cache_dir]
    scoped_dir = federated_feature_cache_scope_dir(cache_dir, spec)
    if os.path.abspath(scoped_dir) != os.path.abspath(cache_dir):
        dir_candidates.append(scoped_dir)
    for dir_path in dir_candidates:
        if not os.path.isdir(dir_path):
            continue
        for filename in _federated_feature_cache_name_candidates(spec, client_id):
            path = os.path.join(dir_path, filename)
            if os.path.exists(path):
                return path
    return None


def _enclosing_output_root(path: str) -> str:
    cur = os.path.abspath(str(path))
    while True:
        if os.path.basename(cur) == "output":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(str(path))
        cur = parent


def _partition_payload_matches(
    payload: dict,
    *,
    spec: FederatedFeatureCacheSpec,
    expected_client_sizes: list[int],
    expected_client_indices: list[list[int]] | None = None,
) -> bool:
    if str(payload.get("dataset")) != str(spec.dataset):
        return False
    if int(payload.get("num_clients", -1)) != int(spec.num_clients):
        return False
    if str(payload.get("partition")) != str(spec.partition):
        return False
    if abs(float(payload.get("dirichlet_alpha", -1.0)) - float(spec.dirichlet_alpha)) > 1e-8:
        return False
    if bool(payload.get("dirichlet_balance", False)) != bool(spec.dirichlet_balance):
        return False
    if int(payload.get("dirichlet_min_size", -1)) != int(spec.dirichlet_min_size):
        return False
    if int(payload.get("shard_per_client", -1)) != int(spec.shard_per_client):
        return False
    if int(payload.get("classes_per_client", -1)) != int(spec.classes_per_client):
        return False
    if int(payload.get("seed", -1)) != int(spec.seed):
        return False
    got_sizes = [int(v) for v in payload.get("client_sizes", [])]
    if got_sizes != [int(v) for v in expected_client_sizes]:
        return False
    if expected_client_indices is not None:
        got_indices_raw = payload.get("client_indices", [])
        try:
            got_indices = [[int(x) for x in client] for client in got_indices_raw]
            exp_indices = [[int(x) for x in client] for client in expected_client_indices]
        except Exception:
            return False
        if got_indices != exp_indices:
            return False
    return True


def _has_matching_partition_cache(
    run_root: str,
    *,
    spec: FederatedFeatureCacheSpec,
    expected_client_sizes: list[int],
    expected_client_indices: list[list[int]] | None = None,
) -> bool:
    partition_dir = os.path.join(run_root, "partition_cache")
    if not os.path.isdir(partition_dir):
        return False
    for fn in os.listdir(partition_dir):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(partition_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if isinstance(payload, dict) and _partition_payload_matches(
            payload,
            spec=spec,
            expected_client_sizes=expected_client_sizes,
            expected_client_indices=expected_client_indices,
        ):
            return True
    return False


def discover_readable_federated_feature_cache_dirs(
    *,
    output_root: str,
    current_cache_dir: str,
    spec: FederatedFeatureCacheSpec,
    expected_client_sizes: list[int],
    expected_client_indices: list[list[int]] | None = None,
) -> list[str]:
    out_root = _enclosing_output_root(output_root)
    readable = [os.path.abspath(current_cache_dir)]
    candidates = []
    if os.path.isdir(out_root):
        for root, dirs, _files in os.walk(out_root):
            if "feature_cache" not in dirs:
                continue
            cache_dir = os.path.join(root, "feature_cache")
            if os.path.abspath(cache_dir) == os.path.abspath(current_cache_dir):
                continue
            if not _has_matching_partition_cache(
                root,
                spec=spec,
                expected_client_sizes=expected_client_sizes,
                expected_client_indices=expected_client_indices,
            ):
                continue
            candidates.append(cache_dir)
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for cache_dir in candidates:
        abs_path = os.path.abspath(cache_dir)
        if abs_path not in readable:
            readable.append(abs_path)
    return readable


def build_feature_loader(dataset, *, batch_size: int, num_workers: int, is_heavy_dataset: bool) -> DataLoader:
    actual_workers = min(int(num_workers), 4) if is_heavy_dataset else int(num_workers)
    loader_kwargs = dict(
        shuffle=False,
        num_workers=actual_workers,
        batch_size=int(batch_size),
        drop_last=False,
        pin_memory=actual_workers > 0,
        persistent_workers=actual_workers > 0 and (not is_heavy_dataset),
    )
    if actual_workers > 0:
        loader_kwargs["prefetch_factor"] = 1 if is_heavy_dataset else 2
    return DataLoader(dataset, **loader_kwargs)


def forward_features(
    *,
    loader: DataLoader,
    backbone,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor],
    feature_dim: int,
    use_amp: bool = True,
    progress_desc: str | None = None,
    progress_log_interval: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    feats = []
    labels = []
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    seen_samples = 0
    with torch.no_grad():
        iterator = loader
        if progress_desc is not None:
            if total_batches is not None:
                print(f"[feature_cache] {progress_desc}: start total_batches={int(total_batches)}", flush=True)
            else:
                print(f"[feature_cache] {progress_desc}: start", flush=True)
            iterator = tqdm(loader, desc=progress_desc, leave=False)
        for batch_idx, (x, y) in enumerate(iterator, start=1):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            x = preprocess_fn(x)
            with autocast(enabled=bool(use_amp)):
                z = backbone(x)
            feats.append(z.detach().to(dtype=torch.float32).cpu())
            labels.append(y.detach().cpu())
            seen_samples += int(y.shape[0])
            if progress_desc is not None and (
                batch_idx == 1
                or (progress_log_interval > 0 and batch_idx % int(progress_log_interval) == 0)
                or (total_batches is not None and batch_idx == int(total_batches))
            ):
                if total_batches is not None:
                    print(
                        f"[feature_cache] {progress_desc}: batch={batch_idx}/{int(total_batches)} "
                        f"samples={seen_samples}",
                        flush=True,
                    )
                else:
                    print(
                        f"[feature_cache] {progress_desc}: batch={batch_idx} samples={seen_samples}",
                        flush=True,
                    )
    z_all = torch.cat(feats, dim=0) if feats else torch.zeros((0, int(feature_dim)), dtype=torch.float32)
    y_all = torch.cat(labels, dim=0) if labels else torch.zeros((0,), dtype=torch.long)
    return z_all, y_all


def load_cached_federated_client_features(
    *,
    readable_cache_dirs: list[str],
    spec: FederatedFeatureCacheSpec,
    client_id: int,
    expected_num_samples: int,
    expected_num_features: int,
    expected_client_indices: list[int] | None = None,
) -> tuple[str, torch.Tensor, torch.Tensor] | None:
    expected_fingerprint = (
        _client_indices_fingerprint(expected_client_indices) if expected_client_indices is not None else None
    )
    for cache_dir in readable_cache_dirs:
        cache_path = find_federated_cache_file_in_dir(cache_dir, spec=spec, client_id=client_id)
        if cache_path is None:
            continue
        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        feats = payload.get("features")
        labels = payload.get("labels")
        meta = payload.get("meta", {})
        if not torch.is_tensor(feats) or not torch.is_tensor(labels):
            continue
        if int(labels.shape[0]) != int(expected_num_samples):
            continue
        if feats.ndim != 2 or int(feats.shape[1]) != int(expected_num_features):
            continue
        if expected_fingerprint is not None:
            got_fingerprint = str(meta.get("client_indices_fingerprint", ""))
            if got_fingerprint != expected_fingerprint:
                continue
        return cache_path, feats.to(dtype=torch.float32, copy=False), labels.to(dtype=torch.long, copy=False)
    return None


def save_federated_client_features(
    *,
    cache_dir: str,
    spec: FederatedFeatureCacheSpec,
    client_id: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    client_indices: list[int] | None = None,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, federated_feature_cache_client_file_name(client_id))
    torch.save(
        {
            "features": features,
            "labels": labels,
            "meta": {
                "dataset": str(spec.dataset),
                "model": str(spec.model),
                "client_id": int(client_id),
                "num_clients": int(spec.num_clients),
                "partition": str(spec.partition),
                "dirichlet_alpha": float(spec.dirichlet_alpha),
                "seed": int(spec.seed),
                "real_res": int(spec.real_res),
                "crop_res": int(spec.crop_res),
                "train_crop_mode": str(spec.train_crop_mode),
                "num_samples": int(labels.shape[0]),
                "client_indices_fingerprint": (
                    _client_indices_fingerprint(client_indices) if client_indices is not None else ""
                ),
            },
        },
        cache_path,
    )
    return cache_path
