from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from augmentation import get_augmentor
from federated.client import FederatedClient
from synsets import get_distilled_dataset


@dataclass(frozen=True)
class AggregatedStats:
    mu_target: torch.Tensor
    mu_all: torch.Tensor
    count_per_class: torch.Tensor
    count_all: int


@dataclass(frozen=True)
class LossTargets:
    b_star: torch.Tensor
    support_weights: torch.Tensor
    g_star: torch.Tensor
    class_weights: torch.Tensor
    mu_all: torch.Tensor
    count_per_class: torch.Tensor
    count_all: int
    ipc: int


@dataclass(frozen=True)
class DistillResult:
    images: torch.Tensor
    labels: torch.Tensor
    losses: list[float]


class FederatedServer:
    def __init__(self, experiment_name: str, output_root: str = "output"):
        self.experiment_name = experiment_name
        self.output_dir = os.path.join(output_root, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def save_json(self, name: str, payload: dict):
        path = os.path.join(self.output_dir, name)
        with open(path, "w") as f:
            f.write(json.dumps(payload, indent=4))

    @staticmethod
    def resolve_dp_delta(dp_delta: float | str, count_all: int) -> float:
        raw = str(dp_delta).strip().lower()
        if raw == "auto":
            return 1.0 / float(max(int(count_all), 1))
        try:
            return float(dp_delta)
        except (TypeError, ValueError) as exc:
            raise ValueError("dp_delta must be a positive float or 'auto'") from exc

    def aggregate_stats(self, client_stats_list) -> AggregatedStats:
        if len(client_stats_list) == 0:
            raise ValueError("client_stats_list is empty")

        sum_per_class = None
        count_per_class = None
        sum_all = None
        count_all = 0

        for st in client_stats_list:
            s_pc = st.sum_per_class.to(dtype=torch.float64)
            c_pc = st.count_per_class.to(dtype=torch.long)
            s_all = st.sum_all.to(dtype=torch.float64)

            if sum_per_class is None:
                sum_per_class = torch.zeros_like(s_pc)
                count_per_class = torch.zeros_like(c_pc)
                sum_all = torch.zeros_like(s_all)

            sum_per_class += s_pc
            count_per_class += c_pc
            sum_all += s_all
            count_all += int(st.count_all)

        mu_target = (sum_per_class / count_per_class.clamp_min(1)[:, None]).to(dtype=torch.float64)
        denom = (count_all - count_per_class).clamp_min(1)[:, None]
        mu_all = (sum_all / max(count_all, 1)).to(dtype=torch.float64)

        return AggregatedStats(
            mu_target=mu_target,
            mu_all=mu_all,
            count_per_class=count_per_class,
            count_all=count_all,
        )

    def aggregate_loss_targets(
        self,
        client_loss_list,
        aggregated_stats: AggregatedStats,
        ipc: int,
        interpolation_rounds: int,
        interpolation_t: float,
        init_mode: str = "mean",
        gaussian_std: float = 1.0,
        random_seed: int = 0,
        update_support_weights: bool = True,
        stop_tol_xi: float | None = None,
        stop_tol_a: float | None = None,
        stop_patience: int = 1,
        stop_eps: float = 1e-12,
        dp_enable: bool = False,
        dp_epsilon: float = 1.0,
        dp_delta: float | str = "auto",
        ot_solver: str = "emd",
        sinkhorn_reg: float = 1e-2,
        ot_warmstart: bool = True,
    ) -> LossTargets:
        num_classes = int(aggregated_stats.mu_target.shape[0])
        num_feats = int(aggregated_stats.mu_target.shape[1])
        ipc = int(ipc)
        interpolation_rounds = int(max(interpolation_rounds, 1))
        interpolation_t = float(interpolation_t)
        init_mode = str(init_mode).lower()
        gaussian_std = float(gaussian_std)
        random_seed = int(random_seed)
        update_support_weights = bool(update_support_weights)
        stop_tol_xi = None if stop_tol_xi is None else float(stop_tol_xi)
        stop_tol_a = None if stop_tol_a is None else float(stop_tol_a)
        stop_patience = int(max(stop_patience, 1))
        stop_eps = float(stop_eps)
        dp_enable = bool(dp_enable)
        dp_epsilon = float(dp_epsilon)
        ot_solver = str(ot_solver).lower()
        sinkhorn_reg = float(sinkhorn_reg)
        ot_warmstart = bool(ot_warmstart)
        dp_delta = self.resolve_dp_delta(dp_delta, aggregated_stats.count_all)
        if dp_enable:
            if not (dp_epsilon > 0.0):
                raise ValueError("dp_epsilon must be > 0 when dp_enable is True")
            if not (0.0 < dp_delta < 1.0):
                raise ValueError("dp_delta must be in (0, 1) when dp_enable is True")
        if ot_solver not in {"emd", "sinkhorn"}:
            raise ValueError(f"Unknown ot_solver: {ot_solver}")
        if ot_solver == "sinkhorn" and not (sinkhorn_reg > 0.0):
            raise ValueError("sinkhorn_reg must be > 0 when ot_solver == sinkhorn")

        count_per_class = aggregated_stats.count_per_class.detach().cpu().to(dtype=torch.float64)
        count_all = float(max(int(aggregated_stats.count_all), 1))
        class_weights = (count_per_class / count_all).to(dtype=torch.float64)
        mu_all = aggregated_stats.mu_all.detach().cpu().to(dtype=torch.float64)
        mu_target = aggregated_stats.mu_target.detach().cpu().to(dtype=torch.float64)

        b_star = torch.zeros((num_classes, ipc, num_feats), dtype=torch.float64)
        support_weights = torch.zeros((num_classes, ipc), dtype=torch.float64)
        if int(ipc) == 1:
            b_star[:, 0, :] = mu_target
            support_weights[:, 0] = 1.0
            if dp_enable:
                sens = (2.0 * float(interpolation_t)) / count_per_class.clamp_min(1).to(dtype=torch.float64).numpy()
                sigma = sens * np.sqrt(2.0 * np.log(1.25 / float(dp_delta))) / float(dp_epsilon)
                rng_dp = np.random.default_rng(int(random_seed))
                noise = rng_dp.normal(loc=0.0, scale=sigma[:, None], size=(num_classes, num_feats))
                b_star[:, 0, :] = b_star[:, 0, :] + torch.from_numpy(noise).to(dtype=torch.float64)
            g_star = (mu_all[None, None, :] / float(num_classes)) - (class_weights[:, None, None] * b_star)
            return LossTargets(
                b_star=b_star,
                support_weights=support_weights,
                g_star=g_star,
                class_weights=class_weights,
                mu_all=mu_all,
                count_per_class=aggregated_stats.count_per_class.detach().cpu(),
                count_all=int(aggregated_stats.count_all),
                ipc=ipc,
            )

        if len(client_loss_list) == 0:
            raise ValueError("client_loss_list is empty")

        def _unpack_client_loss(item):
            if isinstance(item, (str, os.PathLike)):
                payload = torch.load(item, map_location="cpu")
                return payload["nu_local"], payload["nu_weights"], payload["count_per_class"]
            return item.nu_local, item.nu_weights, item.count_per_class

        client_payloads = [_unpack_client_loss(item) for item in client_loss_list]
        class_entries_all = []
        total_n_all = np.zeros((num_classes,), dtype=np.float64)
        xi_states: list[np.ndarray | None] = [None] * num_classes
        a_states: list[np.ndarray | None] = [None] * num_classes
        warm_states_all = []
        stable_counts = np.zeros((num_classes,), dtype=np.int64)
        done_mask = np.zeros((num_classes,), dtype=bool)

        for c in range(num_classes):
            class_entries = []
            total_n = 0.0
            pooled_count = 0
            pooled_sum = np.zeros((num_feats,), dtype=np.float64)
            pooled_sumsq = np.zeros((num_feats,), dtype=np.float64)
            for nu_local, nu_weights, count_per_class_local in client_payloads:
                n_c = int(count_per_class_local[c].item())
                if n_c <= 0:
                    continue
                b_c = nu_local[c].to(dtype=torch.float64).cpu().numpy()
                if dp_enable and int(b_c.shape[0]) > 0:
                    b_norm = np.linalg.norm(b_c, axis=1, keepdims=True)
                    b_norm = np.clip(b_norm, 1e-12, None)
                    b_c = b_c / b_norm
                w_c = nu_weights[c].to(dtype=torch.float64).cpu().numpy()
                w_c = w_c / max(float(np.sum(w_c)), 1e-12)
                class_entries.append((float(n_c), b_c, w_c))
                total_n += float(n_c)
                if init_mode == "gaussian" and int(b_c.shape[0]) > 0:
                    pooled_count += int(b_c.shape[0])
                    pooled_sum += b_c.sum(axis=0)
                    pooled_sumsq += np.square(b_c).sum(axis=0)

            class_entries_all.append(class_entries)
            total_n_all[c] = total_n
            warm_states_all.append([None] * len(class_entries))

            if len(class_entries) == 0:
                b_star[c] = mu_target[c][None, :].repeat(ipc, 1)
                support_weights[c] = torch.full((ipc,), 1.0 / float(ipc), dtype=torch.float64)
                done_mask[c] = True
                continue

            rng = np.random.default_rng(random_seed + c)
            a = np.ones((ipc,), dtype=np.float64) / float(ipc)
            if init_mode == "mean":
                xi = np.repeat(mu_target[c][None, :].cpu().numpy(), ipc, axis=0)
            elif init_mode == "gaussian":
                base = mu_target[c].cpu().numpy()
                if pooled_count > 0:
                    pooled_mean = pooled_sum / float(pooled_count)
                    pooled_var = np.maximum((pooled_sumsq / float(pooled_count)) - np.square(pooled_mean), 0.0)
                    base_std = float(np.mean(np.sqrt(pooled_var)))
                else:
                    base_std = 1.0
                sigma = max(base_std * gaussian_std, 1e-12)
                xi = base[None, :] + rng.normal(loc=0.0, scale=sigma, size=(ipc, num_feats))
            elif init_mode == "proxy":
                pooled_points = []
                pooled_probs = []
                for n_c, b_c, w_c in class_entries:
                    lam = float(n_c) / max(total_n, 1e-12)
                    pooled_points.append(b_c)
                    pooled_probs.append(lam * w_c)
                pooled_points = np.concatenate(pooled_points, axis=0)
                pooled_probs = np.concatenate(pooled_probs, axis=0)
                prob_sum = float(np.sum(pooled_probs))
                if prob_sum <= 0.0:
                    pooled_probs = np.ones((int(pooled_points.shape[0]),), dtype=np.float64) / float(
                        pooled_points.shape[0]
                    )
                else:
                    pooled_probs = pooled_probs / prob_sum
                replace = bool(int(pooled_points.shape[0]) < ipc)
                sel = rng.choice(int(pooled_points.shape[0]), size=ipc, replace=replace, p=pooled_probs)
                xi = np.array(pooled_points[sel], dtype=np.float64, copy=True)
            else:
                raise ValueError(f"Unknown init_mode: {init_mode}")
            if dp_enable:
                xi_norm = np.linalg.norm(xi, axis=1, keepdims=True)
                xi_norm = np.clip(xi_norm, 1e-12, None)
                xi = xi / xi_norm

            xi_states[c] = xi
            a_states[c] = a

        done_count = int(done_mask.sum())
        for round_idx in tqdm(range(interpolation_rounds), desc="Server Loss CommRound", leave=False):
            if done_count >= num_classes:
                break
            for c in range(num_classes):
                if done_mask[c]:
                    continue
                class_entries = class_entries_all[c]
                total_n = float(total_n_all[c])
                xi = xi_states[c]
                a = a_states[c]
                if xi is None or a is None:
                    continue
                xi_prev = xi
                a_prev = a
                xi_running = None
                a_running = None
                lambda_running = 0.0
                warm_states = warm_states_all[c]
                for client_idx, (n_c, b_c, w_c) in enumerate(class_entries):
                    lam = float(n_c) / max(total_n, 1e-12)
                    warm_state = warm_states[client_idx] if (ot_warmstart and ot_solver == "sinkhorn") else None
                    xi_k, plan_score, next_warm_state = FederatedClient.interpolate_to_local_measure(
                        xi=xi,
                        a=a,
                        b_c=b_c,
                        w_c=w_c,
                        interpolation_t=interpolation_t,
                        ot_solver=ot_solver,
                        sinkhorn_reg=sinkhorn_reg,
                        warm_state=warm_state,
                        dp_enable=dp_enable,
                        dp_epsilon=dp_epsilon,
                        dp_delta=dp_delta,
                        n_c_global=max(float(count_per_class[c].item()), 1.0),
                        dp_rng_seed=(
                            int(random_seed) + int(c) * 1000003 + int(round_idx) * 1009 + int(client_idx) * 17
                        ),
                        return_plan_score=update_support_weights,
                    )
                    if ot_warmstart and ot_solver == "sinkhorn":
                        warm_states[client_idx] = next_warm_state
                    lambda_next = lambda_running + float(lam)
                    if xi_running is None:
                        xi_running = np.array(xi_k, copy=True)
                    else:
                        alpha = float(lam) / max(lambda_next, 1e-12)
                        xi_running = xi_running + alpha * (xi_k - xi_running)
                    if update_support_weights:
                        if a_running is None:
                            a_running = np.array(plan_score, copy=True)
                        else:
                            alpha = float(lam) / max(lambda_next, 1e-12)
                            a_running = a_running + alpha * (plan_score - a_running)
                    lambda_running = lambda_next
                xi = xi_running if xi_running is not None else np.array(xi, copy=True)
                if update_support_weights:
                    if a_running is None:
                        a_running = np.array(a, copy=True)
                    a_sum = float(np.sum(a_running))
                    if a_sum > 0:
                        a = a_running / a_sum

                xi_states[c] = xi
                a_states[c] = a

                if stop_tol_xi is not None:
                    xi_delta = float(np.linalg.norm(xi - xi_prev) / (np.linalg.norm(xi_prev) + stop_eps))
                    if update_support_weights and stop_tol_a is not None:
                        a_delta = float(np.linalg.norm(a - a_prev, ord=1))
                        is_stable = (xi_delta < stop_tol_xi) and (a_delta < stop_tol_a)
                    else:
                        is_stable = xi_delta < stop_tol_xi
                    if is_stable:
                        stable_counts[c] += 1
                        if stable_counts[c] >= stop_patience:
                            done_mask[c] = True
                            done_count += 1
                    else:
                        stable_counts[c] = 0

        for c in range(num_classes):
            xi = xi_states[c]
            a = a_states[c]
            if xi is None or a is None:
                continue
            b_star[c] = torch.from_numpy(xi).to(dtype=torch.float64)
            support_weights[c] = torch.from_numpy(a).to(dtype=torch.float64)

        support_weights = support_weights / support_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        g_star = (mu_all[None, None, :] / float(num_classes)) - (class_weights[:, None, None] * b_star)

        return LossTargets(
            b_star=b_star,
            support_weights=support_weights,
            g_star=g_star,
            class_weights=class_weights,
            mu_all=mu_all,
            count_per_class=aggregated_stats.count_per_class.detach().cpu(),
            count_all=int(aggregated_stats.count_all),
            ipc=ipc,
        )

    def distill(
        self,
        cfg,
        backbone: nn.Module,
        loss_targets: LossTargets,
        normalize,
        num_classes: int,
        num_feats: int,
        log_it: int = 50,
    ) -> DistillResult:
        class DatasetProxy:
            def __init__(self, num_classes: int, normalize):
                self.num_classes = num_classes
                self.normalize = normalize

        train_dataset = DatasetProxy(num_classes=num_classes, normalize=normalize)
        distilled_dataset = get_distilled_dataset(train_dataset=train_dataset, cfg=cfg)
        syn_augmentor = get_augmentor(aug_mode=cfg.aug_mode, crop_res=cfg.crop_res)

        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()

        if int(cfg.ipc) != int(loss_targets.ipc):
            raise RuntimeError("loss requires ipc to match loss target support size")
        class_weights = loss_targets.class_weights.to(device="cuda", dtype=torch.float32)
        syn_class_weights = torch.full(
            (num_classes,),
            1.0 / float(max(num_classes, 1)),
            device="cuda",
            dtype=torch.float32,
        )
        support_weights = loss_targets.support_weights.to(device="cuda", dtype=torch.float32)
        g_star = loss_targets.g_star.to(device="cuda", dtype=torch.float32)
        losses = []
        scaler = GradScaler()
        aug_repeats = int(cfg.augs_per_batch)
        if aug_repeats <= 0:
            raise ValueError("augs_per_batch must be positive")
        simultaneous_classes = int(getattr(cfg, "distill_simultaneous_classes", -1))
        if simultaneous_classes == 0 or simultaneous_classes < -1:
            raise ValueError("distill_simultaneous_classes must be -1 or a positive integer")

        forward_bs = int(getattr(cfg, "forward_batch_size", 0) or 0)
        use_save_on_cpu = bool(getattr(cfg, "force_save_on_cpu_acts", False))
        # Chunk synthetic forward when forward_batch_size is set, regardless of IPC.
        # This prevents OOM for large-class datasets (e.g., ImageNet-1k with IPC=1).
        chunk_aug_for_high_ipc = (forward_bs > 0)
        loss_expected = torch.arange(num_classes, device="cuda").repeat_interleave(int(cfg.ipc))
        loss_order_idx = None
        fixed_labels = None
        fixed_counts = None
        fixed_cnt_all = None
        use_grouped_distill = (
            int(cfg.ipc) > 1
            and simultaneous_classes > 0
            and simultaneous_classes < int(num_classes)
        )
        class_groups = []
        image_groups = []
        steps_per_group = 0
        cached_z_slots = None
        if use_grouped_distill:
            class_groups = [
                torch.arange(start, min(start + simultaneous_classes, num_classes), device="cuda", dtype=torch.long)
                for start in range(0, num_classes, simultaneous_classes)
            ]
            class_image_slots = torch.arange(num_classes * int(cfg.ipc), device="cuda", dtype=torch.long).view(
                num_classes, int(cfg.ipc)
            )
            image_groups = [class_image_slots.index_select(0, cls_idx).reshape(-1) for cls_idx in class_groups]
            total_steps = int(cfg.iterations) + 1
            steps_per_group = max((total_steps + len(class_groups) - 1) // len(class_groups), 1)
            cached_z_slots = loss_targets.b_star.to(device="cuda", dtype=torch.float32).clone()

        for step in tqdm(range(cfg.iterations + 1), desc="Server Distill", leave=True):
            distilled_dataset.upkeep(step=step)

            x_syn, y_syn = distilled_dataset.get_data()
            y_syn_base = y_syn
            if int(loss_expected.shape[0]) != int(y_syn_base.shape[0]):
                raise RuntimeError("loss requires deterministic class-major ordering for synset")
            if loss_order_idx is None and not bool(torch.equal(loss_expected, y_syn_base)):
                loss_order_idx = torch.argsort(y_syn_base, stable=True)
                y_sorted = y_syn_base[loss_order_idx]
                if not bool(torch.equal(loss_expected, y_sorted)):
                    raise RuntimeError("loss requires deterministic class-major ordering for synset")
            if loss_order_idx is not None:
                x_syn = x_syn[loss_order_idx]
                y_syn_base = loss_expected

            if fixed_labels is None:
                fixed_labels = y_syn_base.detach().clone()
                fixed_counts = torch.bincount(fixed_labels, minlength=num_classes)
                fixed_cnt_all = int(fixed_labels.numel())

            if use_grouped_distill:
                group_idx = min(step // steps_per_group, len(class_groups) - 1)
                active_classes = class_groups[group_idx]
                active_image_idx = image_groups[group_idx]
                x_syn_active = x_syn.index_select(0, active_image_idx)
                y_syn_active = y_syn_base.index_select(0, active_image_idx)
                expected_active = active_classes.repeat_interleave(int(cfg.ipc))
                if not bool(torch.equal(y_syn_active, expected_active)):
                    raise RuntimeError("grouped loss requires deterministic class-major ordering for synset")

                z_slots_sum = torch.zeros(
                    (int(active_classes.numel()), int(cfg.ipc), num_feats),
                    device="cuda",
                    dtype=torch.float32,
                )

                for _ in range(aug_repeats):
                    if chunk_aug_for_high_ipc and int(x_syn_active.shape[0]) > forward_bs:
                        z_parts = []
                        for start in range(0, int(x_syn_active.shape[0]), forward_bs):
                            end = min(start + forward_bs, int(x_syn_active.shape[0]))
                            with autocast(enabled=True):
                                x_aug_chunk = syn_augmentor(x_syn_active[start:end])
                                x_aug_chunk = normalize(x_aug_chunk)
                                if use_save_on_cpu:
                                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                                        z_chunk = backbone(x_aug_chunk)
                                else:
                                    z_chunk = backbone(x_aug_chunk)
                            z_parts.append(z_chunk)
                        z_aug = torch.cat(z_parts, dim=0)
                    else:
                        with autocast(enabled=True):
                            x_aug = syn_augmentor(x_syn_active)
                            x_aug = normalize(x_aug)
                            if use_save_on_cpu:
                                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                                    z_aug = backbone(x_aug)
                            else:
                                z_aug = backbone(x_aug)

                    z_aug = z_aug.to(dtype=torch.float32)
                    z_slots_sum += z_aug.view(int(active_classes.numel()), int(cfg.ipc), num_feats)

                z_slots_active = z_slots_sum / float(cfg.augs_per_batch)
                z_slots_full = cached_z_slots.clone()
                z_slots_full.index_copy_(0, active_classes, z_slots_active)
                mu_target_syn = torch.einsum("ci,cid->cd", support_weights, z_slots_full)
                mu_all_syn = torch.einsum("c,cd->d", syn_class_weights, mu_target_syn)
                syn_class_weights_active = syn_class_weights.index_select(0, active_classes)
                support_weights_active = support_weights.index_select(0, active_classes)
                g_star_active = g_star.index_select(0, active_classes)
                g_syn_active = (mu_all_syn[None, None, :] / float(num_classes)) - (
                    syn_class_weights_active[:, None, None] * z_slots_active
                )
                cos = torch.nn.functional.cosine_similarity(g_syn_active, g_star_active, dim=-1)
                loss = 1 - (cos * support_weights_active).sum(dim=1).mean(dim=0)

                distilled_dataset.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(distilled_dataset.optimizer)
                scaler.update()
                cached_z_slots.index_copy_(0, active_classes, z_slots_active.detach())

                losses.append(float(loss.detach().item()))

                if log_it > 0 and step % log_it == 0:
                    self.save_json(
                        "distill_progress.json",
                        {
                            "step": step,
                            "loss": losses[-1],
                            "experiment": self.experiment_name,
                            "active_class_group": int(group_idx),
                            "active_class_start": int(active_classes[0].item()),
                            "active_class_end": int(active_classes[-1].item()),
                            "active_num_classes": int(active_classes.numel()),
                        },
                    )
                continue

            sum_per_class = torch.zeros((num_classes, num_feats), device="cuda", dtype=torch.float32)
            cnt_per_class = torch.zeros((num_classes,), device="cuda", dtype=torch.long)
            sum_all = torch.zeros((num_feats,), device="cuda", dtype=torch.float32)
            z_slots_sum = torch.zeros((num_classes, int(cfg.ipc), num_feats), device="cuda", dtype=torch.float32)
            cnt_all = 0

            for _ in range(aug_repeats):
                # Chunk augmentation itself to avoid a single huge x_aug allocation.
                if chunk_aug_for_high_ipc and int(x_syn.shape[0]) > forward_bs:
                    z_parts = []
                    for start in range(0, int(x_syn.shape[0]), forward_bs):
                        end = min(start + forward_bs, int(x_syn.shape[0]))
                        with autocast(enabled=True):
                            x_aug_chunk = syn_augmentor(x_syn[start:end])
                            x_aug_chunk = normalize(x_aug_chunk)
                            if use_save_on_cpu:
                                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                                    z_chunk = backbone(x_aug_chunk)
                            else:
                                z_chunk = backbone(x_aug_chunk)
                        z_parts.append(z_chunk)
                    z_aug = torch.cat(z_parts, dim=0)
                else:
                    with autocast(enabled=True):
                        x_aug = syn_augmentor(x_syn)
                        x_aug = normalize(x_aug)
                        if use_save_on_cpu:
                            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                                z_aug = backbone(x_aug)
                        else:
                            z_aug = backbone(x_aug)

                z_aug = z_aug.to(dtype=torch.float32)
                sum_per_class.index_add_(0, y_syn_base, z_aug)
                if fixed_labels is not None and bool(torch.equal(fixed_labels, y_syn_base)):
                    cnt_per_class += fixed_counts
                    cnt_all += fixed_cnt_all
                else:
                    cnt_per_class += torch.bincount(y_syn_base, minlength=num_classes)
                    cnt_all += int(y_syn_base.shape[0])
                sum_all += z_aug.sum(dim=0)
                z_slots_sum += z_aug.view(num_classes, int(cfg.ipc), num_feats)

            z_slots = z_slots_sum / float(cfg.augs_per_batch)
            # Match Eq.(6): class synthetic centroid should be weighted by support weights omega.
            mu_target_syn = torch.einsum("ci,cid->cd", support_weights, z_slots)
            mu_all_syn = torch.einsum("c,cd->d", syn_class_weights, mu_target_syn)
            g_syn = (mu_all_syn[None, None, :] / float(num_classes)) - (
                syn_class_weights[:, None, None] * z_slots
            )
            cos = torch.nn.functional.cosine_similarity(g_syn, g_star, dim=-1)
            loss = 1 - (cos * support_weights).sum(dim=1).mean(dim=0)

            distilled_dataset.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(distilled_dataset.optimizer)
            scaler.update()

            losses.append(float(loss.detach().item()))

            if log_it > 0 and step % log_it == 0:
                self.save_json(
                    "distill_progress.json",
                    {
                        "step": step,
                        "loss": losses[-1],
                        "experiment": self.experiment_name,
                    },
                )

        with torch.no_grad():
            syn_images, syn_labels = distilled_dataset.get_data()
            syn_images = syn_images.detach().cpu()
            syn_labels = syn_labels.detach().cpu()

        vis_dir = os.path.join(self.output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        images_01 = syn_images.clamp(0, 1)
        vutils.save_image(
            images_01,
            os.path.join(vis_dir, "all.png"),
            nrow=min(10, int(images_01.shape[0])),
            padding=2,
        )
        for i in range(int(images_01.shape[0])):
            y = int(syn_labels[i].item())
            vutils.save_image(images_01[i], os.path.join(vis_dir, f"img_{i:05d}_y{y}.png"))
        for c in range(num_classes):
            idx = (syn_labels == c).nonzero(as_tuple=False).flatten()
            if int(idx.numel()) == 0:
                continue
            vutils.save_image(
                images_01[idx],
                os.path.join(vis_dir, f"class_{c:03d}.png"),
                nrow=int(idx.numel()),
                padding=2,
            )

        out = DistillResult(
            images=syn_images,
            labels=syn_labels,
            losses=losses,
        )
        torch.save(out.__dict__, os.path.join(self.output_dir, "data.pth"))
        return out
