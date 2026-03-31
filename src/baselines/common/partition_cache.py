from __future__ import annotations

import json
import os
import shutil
from json import JSONDecodeError

import fcntl

import numpy as np
import torch
from torch.utils.data import Subset


def _indices_to_list(indices) -> list[int]:
    if isinstance(indices, torch.Tensor):
        return [int(v) for v in indices.reshape(-1).tolist()]
    return [int(v) for v in list(indices)]


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
        idx = _indices_to_list(ds.ds.indices)
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
    raise RuntimeError("Unable to extract labels for partitioning.")


def split_iid(n: int, k: int, rng: np.random.Generator) -> list[list[int]]:
    idx = np.arange(n)
    rng.shuffle(idx)
    splits = np.array_split(idx, k)
    return [s.tolist() for s in splits]


def split_shards(labels: list[int], k: int, shard_per_client: int, rng: np.random.Generator) -> list[list[int]]:
    labels_np = np.asarray(labels, dtype=np.int64)
    total_shards = int(k) * max(int(shard_per_client), 1)
    sorted_indices = np.argsort(labels_np, kind="stable")
    shards = np.array_split(sorted_indices, total_shards)
    order = np.arange(total_shards)
    rng.shuffle(order)
    per_client = [[] for _ in range(k)]
    cursor = 0
    for client_id in range(k):
        for _ in range(max(int(shard_per_client), 1)):
            shard_id = int(order[cursor])
            per_client[client_id].extend(int(v) for v in shards[shard_id].tolist())
            cursor += 1
    for client_id in range(k):
        rng.shuffle(per_client[client_id])
    return per_client


def split_random_classes(
    labels: list[int],
    k: int,
    classes_per_client: int,
    rng: np.random.Generator,
    *,
    min_size: int,
    max_retries: int = 1024,
) -> list[list[int]]:
    labels_np = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels_np.max()) + 1
    classes_per_client = max(int(classes_per_client), 1)
    min_size = max(int(min_size), 0)
    if classes_per_client > num_classes:
        raise ValueError(
            f"classes_per_client={classes_per_client} exceeds num_classes={num_classes}."
        )
    all_classes = np.arange(num_classes, dtype=np.int64)
    for _try in range(max(int(max_retries), 1)):
        assigned = [[] for _ in range(k)]
        class_to_clients = {int(c): [] for c in all_classes.tolist()}
        for client_id in range(k):
            chosen = rng.choice(all_classes, size=classes_per_client, replace=False)
            chosen_list = [int(v) for v in np.atleast_1d(chosen).tolist()]
            assigned[client_id] = chosen_list
            for cls in chosen_list:
                class_to_clients[cls].append(client_id)
        missing = [cls for cls, clients in class_to_clients.items() if len(clients) == 0]
        if missing:
            continue
        per_client = [[] for _ in range(k)]
        for cls in all_classes.tolist():
            idx_c = np.where(labels_np == int(cls))[0].tolist()
            rng.shuffle(idx_c)
            owners = class_to_clients[int(cls)]
            chunks = np.array_split(np.asarray(idx_c, dtype=np.int64), len(owners))
            for client_id, chunk in zip(owners, chunks):
                per_client[client_id].extend(int(v) for v in chunk.tolist())
        for client_id in range(k):
            rng.shuffle(per_client[client_id])
        if min((len(v) for v in per_client), default=0) >= min_size:
            return per_client
    raise RuntimeError(
        f"Failed to build random_classes partition with classes_per_client={classes_per_client}, "
        f"min_size={min_size} after {int(max_retries)} retries."
    )


def split_dirichlet(
    labels: list[int],
    k: int,
    alpha: float,
    rng: np.random.Generator,
    *,
    balanced: bool,
    min_size: int,
    max_retries: int = 1024,
) -> list[list[int]]:
    labels_np = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels_np.max()) + 1
    total_samples = int(labels_np.shape[0])
    min_size = max(int(min_size), 0)
    if bool(balanced):
        class_pools = []
        for c in range(num_classes):
            idx_c = np.where(labels_np == c)[0].tolist()
            rng.shuffle(idx_c)
            class_pools.append(idx_c)
        target_sizes = [int(total_samples // k) for _ in range(k)]
        for i in range(int(total_samples % k)):
            target_sizes[i] += 1
        per_client = [[] for _ in range(k)]
        for client_id in range(k):
            need = int(target_sizes[client_id])
            class_pref = rng.dirichlet(alpha * np.ones(num_classes))
            while need > 0:
                available = np.asarray([len(pool) > 0 for pool in class_pools], dtype=bool)
                if not bool(available.any()):
                    raise RuntimeError("Balanced Dirichlet allocation ran out of samples unexpectedly.")
                probs = class_pref.copy()
                probs[~available] = 0.0
                if float(probs.sum()) <= 0:
                    probs = available.astype(np.float64)
                probs = probs / probs.sum()
                draw = rng.multinomial(need, probs)
                moved = 0
                for cls in np.where(draw > 0)[0]:
                    take = min(int(draw[cls]), len(class_pools[int(cls)]))
                    if take <= 0:
                        continue
                    moved += take
                    per_client[client_id].extend(class_pools[int(cls)][-take:])
                    del class_pools[int(cls)][-take:]
                if moved <= 0:
                    for cls in np.where(available)[0]:
                        per_client[client_id].append(class_pools[int(cls)].pop())
                        moved = 1
                        break
                need -= int(moved)
            rng.shuffle(per_client[client_id])
        if min((len(v) for v in per_client), default=0) < min_size:
            raise RuntimeError(
                f"Balanced Dirichlet allocation produced min client size below required min_size={min_size}."
            )
        return per_client
    last_per_client = None
    for _try in range(max(int(max_retries), 1)):
        per_client = [[] for _ in range(k)]
        for c in range(num_classes):
            idx_c = np.where(labels_np == c)[0]
            rng.shuffle(idx_c)
            if len(idx_c) == 0:
                continue
            props = rng.dirichlet(alpha * np.ones(k))
            counts = (props * len(idx_c)).astype(np.int64)
            diff = len(idx_c) - int(counts.sum())
            if diff > 0:
                for i in rng.choice(k, size=diff, replace=True):
                    counts[i] += 1
            elif diff < 0:
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
        last_per_client = per_client
        if min((len(v) for v in per_client), default=0) >= min_size:
            return per_client
    if last_per_client is None:
        raise RuntimeError("Dirichlet partition generation failed before producing any split.")
    if min_size <= 0:
        return last_per_client
    repaired = [list(v) for v in last_per_client]
    deficits = [i for i, v in enumerate(repaired) if len(v) < min_size]
    donors = sorted(range(k), key=lambda i: len(repaired[i]), reverse=True)
    for client_id in deficits:
        need = int(min_size - len(repaired[client_id]))
        for donor_id in donors:
            if donor_id == client_id:
                continue
            available = int(len(repaired[donor_id]) - min_size)
            if available <= 0:
                continue
            take = min(need, available)
            picked = rng.choice(len(repaired[donor_id]), size=take, replace=False)
            picked = sorted((int(v) for v in np.atleast_1d(picked)), reverse=True)
            moved = []
            for pos in picked:
                moved.append(repaired[donor_id].pop(pos))
            repaired[client_id].extend(moved)
            need -= take
            if need <= 0:
                break
        if need > 0:
            raise RuntimeError(
                f"Failed to repair Dirichlet partition to min_size={min_size}, "
                f"balanced={bool(balanced)} after {int(max_retries)} retries."
            )
    for i in range(k):
        rng.shuffle(repaired[i])
    return repaired


def split_dirichlet_afl(
    labels: list[int],
    k: int,
    alpha: float,
    rng: np.random.Generator,
    *,
    min_size: int,
    max_retries: int = 100000,
) -> list[list[int]]:
    labels_np = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels_np.max()) + 1
    total_samples = int(labels_np.shape[0])
    min_size = max(int(min_size), 0)
    if k <= 0:
        raise ValueError(f"num_clients must be positive, got {k}.")
    if total_samples <= 0:
        return [[] for _ in range(k)]

    retry_budget = max(int(max_retries), 1)
    for _try in range(retry_budget):
        idx_batch = [[] for _ in range(k)]
        min_client_size = 0
        for cls in range(num_classes):
            idx_c = np.where(labels_np == cls)[0]
            rng.shuffle(idx_c)
            if len(idx_c) == 0:
                continue
            proportions = rng.dirichlet(np.repeat(float(alpha), k))
            proportions = np.asarray(
                [p * (len(idx_j) < (total_samples / k)) for p, idx_j in zip(proportions, idx_batch)],
                dtype=np.float64,
            )
            if float(proportions.sum()) <= 0:
                proportions = np.ones((k,), dtype=np.float64)
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            chunks = np.split(idx_c, split_points)
            idx_batch = [idx_j + chunk.tolist() for idx_j, chunk in zip(idx_batch, chunks)]
            min_client_size = min((len(idx_j) for idx_j in idx_batch), default=0)
        if min_client_size >= min_size:
            for client_id in range(k):
                rng.shuffle(idx_batch[client_id])
            return idx_batch
    raise RuntimeError(
        f"Failed to build AFL-style Dirichlet partition with alpha={alpha}, min_size={min_size} "
        f"after {retry_budget} retries."
    )


def build_or_load_partitions(
    cache_path: str,
    dataset_name: str,
    num_clients: int,
    partition: str,
    dirichlet_alpha: float,
    dirichlet_balance: bool,
    dirichlet_min_size: int,
    shard_per_client: int,
    classes_per_client: int,
    labels: list[int],
    seed: int,
) -> dict:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    lock_path = f"{cache_path}.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return payload
            except (OSError, JSONDecodeError):
                backup_path = f"{cache_path}.corrupt"
                try:
                    shutil.move(cache_path, backup_path)
                except OSError:
                    pass

        rng = np.random.default_rng(seed)
        if partition == "iid":
            splits = split_iid(n=len(labels), k=num_clients, rng=rng)
        elif partition == "dirichlet":
            if bool(dirichlet_balance):
                splits = split_dirichlet(
                    labels=labels,
                    k=num_clients,
                    alpha=dirichlet_alpha,
                    rng=rng,
                    balanced=True,
                    min_size=int(dirichlet_min_size),
                )
            else:
                splits = split_dirichlet_afl(
                    labels=labels,
                    k=num_clients,
                    alpha=dirichlet_alpha,
                    rng=rng,
                    min_size=int(dirichlet_min_size),
                )
        elif partition in {"dirichlet_balanced", "dirichlet_legacy"}:
            splits = split_dirichlet(
                labels=labels,
                k=num_clients,
                alpha=dirichlet_alpha,
                rng=rng,
                balanced=bool(dirichlet_balance),
                min_size=int(dirichlet_min_size),
            )
        elif partition in {"dirichlet_afl", "dirichlet_pfl", "afl_dirichlet"}:
            splits = split_dirichlet_afl(
                labels=labels,
                k=num_clients,
                alpha=dirichlet_alpha,
                rng=rng,
                min_size=int(dirichlet_min_size),
            )
        elif partition == "shards":
            splits = split_shards(
                labels=labels,
                k=num_clients,
                shard_per_client=int(shard_per_client),
                rng=rng,
            )
        elif partition in {"random_classes", "label_quantity"}:
            splits = split_random_classes(
                labels=labels,
                k=num_clients,
                classes_per_client=int(classes_per_client),
                rng=rng,
                min_size=int(dirichlet_min_size),
            )
        else:
            raise NotImplementedError(partition)
        payload = {
            "dataset": dataset_name,
            "num_clients": int(num_clients),
            "partition": str(partition),
            "dirichlet_alpha": float(dirichlet_alpha),
            "dirichlet_balance": bool(dirichlet_balance),
            "dirichlet_min_size": int(dirichlet_min_size),
            "shard_per_client": int(shard_per_client),
            "classes_per_client": int(classes_per_client),
            "seed": int(seed),
            "client_indices": splits,
            "client_sizes": [int(len(v)) for v in splits],
        }
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cache_path)
        return payload


def validate_partition_payload(payload: dict, expected_num_samples: int):
    client_indices_raw = payload.get("client_indices", [])
    client_indices = [[int(x) for x in client] for client in client_indices_raw]
    flat = [idx for client in client_indices for idx in client]
    if len(flat) != int(expected_num_samples):
        raise RuntimeError(
            "Partition sample count mismatch: "
            f"expected {int(expected_num_samples)}, got {len(flat)}."
        )
    if len(set(flat)) != len(flat):
        raise RuntimeError("Partition leakage detected: duplicate sample indices across clients.")
    expected = set(range(int(expected_num_samples)))
    actual = set(flat)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            "Partition coverage mismatch: "
            f"missing={missing[:10]} extra={extra[:10]} "
            f"(showing up to 10 indices each)."
        )
    sizes = [int(len(v)) for v in client_indices]
    payload_sizes = [int(v) for v in payload.get("client_sizes", [])]
    if payload_sizes and payload_sizes != sizes:
        raise RuntimeError(
            "Partition metadata mismatch: client_sizes does not match client_indices lengths."
        )
