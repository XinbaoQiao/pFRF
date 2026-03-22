from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import ot
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class ClientStats:
    sum_per_class: torch.Tensor
    count_per_class: torch.Tensor
    sum_all: torch.Tensor
    count_all: int


@dataclass(frozen=True)
class ClientLossStats:
    nu_local: list[torch.Tensor]
    nu_weights: list[torch.Tensor]
    count_per_class: torch.Tensor


class FederatedClient:
    def __init__(self, client_id: int, dataset, num_classes: int):
        self.client_id = client_id
        self.dataset = dataset
        self.num_classes = num_classes

    @torch.no_grad()
    def compute_local_stats(
        self,
        backbone: nn.Module,
        normalize,
        num_feats: int,
        batch_size: int,
        num_workers: int,
    ) -> ClientStats:
        loader = DataLoader(
            self.dataset,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        )

        sum_per_class = torch.zeros(
            (self.num_classes, num_feats), device="cuda", dtype=torch.float64
        )
        count_per_class = torch.zeros(
            (self.num_classes,), device="cuda", dtype=torch.long
        )
        sum_all = torch.zeros((num_feats,), device="cuda", dtype=torch.float64)
        count_all = 0

        for x, y in tqdm(loader, desc=f"Client {self.client_id} Stats", leave=False):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            x = normalize(x)
            with autocast(enabled=False):
                z = backbone(x)
            z = z.to(dtype=torch.float64)

            sum_all += z.sum(dim=0, dtype=torch.float64)
            count_all += int(z.shape[0])

            sum_per_class.index_add_(0, y, z)
            count_per_class += torch.bincount(y, minlength=self.num_classes)

        return ClientStats(
            sum_per_class=sum_per_class.detach().cpu(),
            count_per_class=count_per_class.detach().cpu(),
            sum_all=sum_all.detach().cpu(),
            count_all=count_all,
        )

    @torch.no_grad()
    def _project_simplex(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[x < 0] = 0
        s = float(x.sum())
        if np.isclose(s, 0.0):
            return np.zeros_like(x)
        return x / s

    @torch.no_grad()
    def _fixed_support_barycenter(
        self,
        B: np.ndarray,
        M: np.ndarray,
        weights: Optional[np.ndarray],
        eta: float,
        num_itermax: int,
        stop_thr: float,
        norm: str,
    ) -> np.ndarray:
        a = ot.unif(int(M.shape[0]))
        a_prev = a.copy()
        weights = ot.unif(int(B.shape[0])) if weights is None else weights

        if str(norm) == "max":
            _M = M / np.max(M)
        elif str(norm) == "median":
            _M = M / np.median(M)
        elif str(norm) == "none":
            _M = M
        else:
            raise ValueError(f"Unknown norm: {norm}")

        for _ in range(int(num_itermax)):
            potentials = []
            for i in range(int(B.shape[0])):
                _, ret = ot.emd(a, B[i], _M, log=True)
                potentials.append(ret["u"])
            f_star = sum(potentials) / len(potentials)
            a = a * np.exp(-float(eta) * f_star)
            a = self._project_simplex(a)
            da = np.abs(a - a_prev).sum()
            if float(da) < float(stop_thr):
                return a
            a_prev = a.copy()
        return a

    @staticmethod
    @torch.no_grad()
    def interpolate_to_local_measure(
        xi: np.ndarray,
        a: np.ndarray,
        b_c: np.ndarray,
        w_c: np.ndarray,
        interpolation_t: float,
        ot_solver: str = "emd",
        sinkhorn_reg: float = 1e-2,
        warm_state: tuple[np.ndarray, np.ndarray] | None = None,
        dp_enable: bool = False,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        n_c_global: float = 1.0,
        dp_rng_seed: int | None = None,
        return_plan_score: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, tuple[np.ndarray, np.ndarray] | None]:
        xi = np.asarray(xi, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        b_c = np.asarray(b_c, dtype=np.float64)
        w_c = np.asarray(w_c, dtype=np.float64)

        # Guard degenerate inputs; keep current estimate unchanged.
        if int(b_c.shape[0]) == 0 or int(w_c.shape[0]) == 0:
            plan_fallback = np.array(a, copy=True)
            return np.array(xi, copy=True), (plan_fallback if bool(return_plan_score) else None), None

        a = np.clip(a, 0.0, None)
        a_sum = float(np.sum(a))
        if not np.isfinite(a_sum) or a_sum <= 0.0:
            a = np.ones_like(a, dtype=np.float64) / float(max(int(a.shape[0]), 1))
        else:
            a = a / a_sum
        w_c = np.clip(w_c, 0.0, None)
        w_sum = float(np.sum(w_c))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            w_c = np.ones_like(w_c, dtype=np.float64) / float(max(int(w_c.shape[0]), 1))
        else:
            w_c = w_c / w_sum

        M = ot.dist(xi, b_c, metric="sqeuclidean")
        M = np.nan_to_num(M, nan=1e6, posinf=1e6, neginf=0.0)
        ot_solver = str(ot_solver).lower()
        used_sinkhorn = False
        if ot_solver == "sinkhorn":
            try:
                if warm_state is not None:
                    P, log_info = ot.sinkhorn(a, w_c, M, reg=float(sinkhorn_reg), warmstart=warm_state, log=True)
                else:
                    P, log_info = ot.sinkhorn(a, w_c, M, reg=float(sinkhorn_reg), log=True)
            except TypeError:
                P, log_info = ot.sinkhorn(a, w_c, M, reg=float(sinkhorn_reg), log=True)
            used_sinkhorn = True
            next_warm = None
            if isinstance(log_info, dict):
                u = log_info.get("u")
                v = log_info.get("v")
                if u is not None and v is not None:
                    next_warm = (u, v)
        else:
            P = ot.emd(a, w_c, M)
            next_warm = None
        P = np.asarray(P, dtype=np.float64)
        if (not np.all(np.isfinite(P))) or float(np.sum(P)) <= 0.0:
            # Sinkhorn can become numerically unstable on some classes; robust fallback.
            P = np.asarray(ot.emd(a, w_c, M), dtype=np.float64)
            next_warm = None
            used_sinkhorn = False
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        P = np.clip(P, 0.0, None)
        p_sum = float(np.sum(P))
        if p_sum <= 0.0:
            P = np.outer(a, w_c)

        row_mass = np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
        mapped = (P @ b_c) / row_mass
        mapped = np.where(np.isfinite(mapped), mapped, xi)
        xi_k = (1.0 - float(interpolation_t)) * xi + float(interpolation_t) * mapped
        xi_k = np.where(np.isfinite(xi_k), xi_k, xi)
        if bool(dp_enable):
            omega = np.clip(a, 1e-12, None)
            sens = (2.0 * float(interpolation_t)) / (max(float(n_c_global), 1.0) * omega)
            sigma = sens * np.sqrt(2.0 * np.log(1.25 / float(dp_delta))) / float(dp_epsilon)
            if dp_rng_seed is None:
                rng_dp = np.random.default_rng()
            else:
                rng_dp = np.random.default_rng(int(dp_rng_seed))
            noise = rng_dp.normal(loc=0.0, scale=sigma[:, None], size=xi_k.shape)
            xi_k = xi_k + noise
            xi_k = np.where(np.isfinite(xi_k), xi_k, xi)
        if not bool(return_plan_score):
            return xi_k, None, next_warm
        # Use OT row marginals as support-weight signal (directly tied to transport plan).
        # For exact OT these match the source weights; for approximate solvers they provide
        # a numerically stable proxy without introducing extra heuristic reweighting.
        plan_score = np.sum(P, axis=1)
        plan_score = np.nan_to_num(plan_score, nan=0.0, posinf=0.0, neginf=0.0)
        plan_sum = float(np.sum(plan_score))
        if np.isfinite(plan_sum) and plan_sum > 0:
            plan_score = plan_score / plan_sum
        else:
            plan_score = np.array(a, copy=True)
        if used_sinkhorn and not np.all(np.isfinite(plan_score)):
            plan_score = np.array(a, copy=True)
        return xi_k, plan_score, next_warm

    @torch.no_grad()
    def compute_local_loss_stats(
        self,
        backbone: nn.Module,
        normalize,
        num_feats: int,
        batch_size: int,
        num_workers: int,
        ipc: int,
        weighted: bool,
        max_iter: int,
        theta: float,
        tol: float,
        eta: float,
        num_itermax: int,
        stop_thr: float,
        norm: str,
        max_samples_per_class: int,
        seed: int,
    ) -> ClientLossStats:
        class_indices = [[] for _ in range(self.num_classes)]
        for local_idx in tqdm(
            range(len(self.dataset)),
            desc=f"Client {self.client_id} Label Scan",
            leave=False,
        ):
            _, y = self.dataset[local_idx]
            y_int = int(y.item()) if torch.is_tensor(y) else int(y)
            if 0 <= y_int < self.num_classes:
                class_indices[y_int].append(int(local_idx))

        nu_local_list = []
        nu_weight_list = []
        count_per_class = torch.zeros((self.num_classes,), dtype=torch.long)

        for c in tqdm(range(self.num_classes), desc=f"Client {self.client_id} Loss", leave=False):
            idx = class_indices[c]
            count_per_class[c] = int(len(idx))
            if int(max_samples_per_class) > 0 and len(idx) > int(max_samples_per_class):
                rng = np.random.default_rng(int(seed) + int(self.client_id) * 10007 + int(c))
                picked = rng.choice(
                    np.asarray(idx, dtype=np.int64),
                    size=int(max_samples_per_class),
                    replace=False,
                )
                idx = sorted(int(i) for i in picked.tolist())

            if len(idx) == 0:
                z_c = torch.empty((0, num_feats), dtype=torch.float32)
            else:
                subset = torch.utils.data.Subset(self.dataset, idx)
                loader = DataLoader(
                    subset,
                    shuffle=False,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    pin_memory=False,
                    drop_last=False,
                    persistent_workers=False,
                )
                z_parts = []
                for x, _ in tqdm(
                    loader,
                    desc=f"Client {self.client_id} C{c:03d}",
                    leave=False,
                ):
                    x = x.cuda(non_blocking=True)
                    x = normalize(x)
                    with autocast(enabled=True):
                        z = backbone(x)
                    z_parts.append(z.float().detach().cpu())
                z_c = torch.cat(z_parts, dim=0).contiguous() if len(z_parts) > 0 else torch.empty((0, num_feats), dtype=torch.float32)

            z_c = z_c.to(dtype=torch.float64).contiguous()
            n_c = int(z_c.shape[0])
            if n_c > 0:
                z_c = z_c / torch.linalg.norm(z_c, dim=1, keepdim=True).clamp_min(1e-12)
                b_c = z_c
                w_c = torch.full((n_c,), 1.0 / float(n_c), dtype=torch.float64)
            else:
                b_c = torch.empty((0, num_feats), dtype=torch.float64)
                w_c = torch.empty((0,), dtype=torch.float64)
            nu_local_list.append(b_c.detach().cpu())
            nu_weight_list.append(w_c.detach().cpu())

        return ClientLossStats(
            nu_local=nu_local_list,
            nu_weights=nu_weight_list,
            count_per_class=count_per_class.detach().cpu(),
        )
