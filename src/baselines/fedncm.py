from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from baselines.common import ProfileMeter, mean_std, set_global_seed, write_curve_csv, write_json, write_jsonl
from baselines.federated.base_runner import BaseFederatedRunner, _clone_state_dict_to_cpu
from models import get_fc


class FedNCMRunner(BaseFederatedRunner):
    def _client_class_sums(self, client_id: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        cls_sum = torch.zeros((self.num_classes, self.num_feats), device="cuda", dtype=torch.float32)
        cls_cnt = torch.zeros((self.num_classes,), device="cuda", dtype=torch.float32)
        prof = ProfileMeter()
        if bool(getattr(self.args, "cache_features", False)):
            z_all, y_all = self._load_or_build_client_features(client_id)
            max_train_batches = max(int(getattr(self.args, "smoke_max_train_batches", 0)), 0)
            if max_train_batches > 0:
                limit = int(max_train_batches) * int(self.args.local_batch_size)
                z_all = z_all[:limit]
                y_all = y_all[:limit]
            if int(y_all.shape[0]) > 0:
                z = z_all.cuda(non_blocking=True).to(dtype=torch.float32)
                y = y_all.cuda(non_blocking=True)
                cls_sum.index_add_(0, y, z)
                cls_cnt += torch.bincount(y, minlength=self.num_classes).to(device=cls_cnt.device, dtype=torch.float32)
                prof.add_local_flops(int(y.shape[0]) * int(self.num_feats) * 3, bucket="setup")
        else:
            ds = Subset(self.train_dataset, self.client_indices[client_id])
            loader_kwargs = dict(
                shuffle=False,
                num_workers=self.args.local_num_workers,
                batch_size=self.args.local_batch_size,
                drop_last=False,
                pin_memory=self.args.local_num_workers > 0,
                persistent_workers=self.args.local_num_workers > 0,
            )
            if self.args.local_num_workers > 0:
                loader_kwargs["prefetch_factor"] = 2
            loader = DataLoader(ds, **loader_kwargs)
            max_train_batches = max(int(getattr(self.args, "smoke_max_train_batches", 0)), 0)
            seen_batches = 0
            for x, y in loader:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                with torch.no_grad():
                    x = self._train_preprocess(x)
                    z = self.backbone(x).to(dtype=torch.float32)
                cls_sum.index_add_(0, y, z)
                cls_cnt += torch.bincount(y, minlength=self.num_classes).to(device=cls_cnt.device, dtype=torch.float32)
                prof.add_local_flops(int(y.shape[0]) * int(self.num_feats) * 3, bucket="setup")
                seen_batches += 1
                if max_train_batches > 0 and seen_batches >= max_train_batches:
                    break
        return cls_sum.cpu(), cls_cnt.cpu(), prof.flops_breakdown_dict()

    def _build_ncm_state(self) -> tuple[dict[str, torch.Tensor], ProfileMeter]:
        head = get_fc(num_feats=self.num_feats, num_classes=self.num_classes, distributed=False)
        state = _clone_state_dict_to_cpu(head)
        weight_key = None
        bias_key = None
        for name, tensor in state.items():
            if tensor.dim() == 2 and tuple(tensor.shape) == (self.num_classes, self.num_feats):
                if name.endswith("linear.weight") or name.endswith("weight"):
                    weight_key = name
            if tensor.dim() == 1 and int(tensor.shape[0]) == self.num_classes:
                if name.endswith("linear.bias") or name.endswith("bias"):
                    bias_key = name
        if weight_key is None:
            raise RuntimeError("Cannot locate classifier weight for FedNCM.")

        total_sum = torch.zeros((self.num_classes, self.num_feats), dtype=torch.float32)
        total_cnt = torch.zeros((self.num_classes,), dtype=torch.float32)
        meter = ProfileMeter()
        upload_bytes_total = 0
        for client_id in range(self.args.num_clients):
            cls_sum, cls_cnt, flops_breakdown = self._client_class_sums(client_id)
            total_sum += cls_sum.to(dtype=torch.float32)
            total_cnt += cls_cnt.to(dtype=torch.float32)
            meter.merge_flops_breakdown(flops_breakdown)
            upload_bytes_total += int(cls_sum.numel() * cls_sum.element_size() + cls_cnt.numel() * cls_cnt.element_size())

        means = torch.zeros_like(total_sum)
        valid = total_cnt > 0
        means[valid] = total_sum[valid] / total_cnt[valid].unsqueeze(1).clamp_min(1e-12)
        means[valid] = F.normalize(means[valid], dim=1)

        state[weight_key] = means.to(dtype=state[weight_key].dtype)
        if bias_key is not None:
            state[bias_key] = torch.zeros_like(state[bias_key])

        meter.add_local_flops(int(self.num_classes * self.num_feats * 2), bucket="setup")
        meter.add_up(upload_bytes_total)
        return state, meter

    def _run_seed(self, seed: int, seed_rank: int) -> dict:
        set_global_seed(seed)
        out_dir = os.path.join(
            self.args.output_root,
            self.args.method,
            self.args.dataset,
            self.args.model,
            f"seed_{seed}",
        )
        os.makedirs(out_dir, exist_ok=True)

        global_state, meter = self._build_ncm_state()
        head = get_fc(num_feats=self.num_feats, num_classes=self.num_classes, distributed=False)
        head.load_state_dict(global_state)
        head.eval()
        top1, top5 = self._evaluate(head)

        write_json(os.path.join(out_dir, "curve_acc_vs_round.json"), {"rounds": [1], "val_top1": [float(top1)]})
        write_curve_csv(os.path.join(out_dir, "curve_acc_vs_round.csv"), [1], [float(top1)])

        meter_breakdown = meter.flops_breakdown_dict()
        upload_one = int(meter.bytes_up / max(self.args.num_clients, 1))
        round_row = {
            "round": 1,
            "val_top1": float(top1),
            "val_top5": float(top5),
            "server_flops_round_est": 0,
            "server_flops_base_round_est": 0,
            "server_flops_extra_round_est": 0,
            "server_flops_setup_round_est": 0,
            "client_flops_round_total": int(meter.local_flops),
            "client_flops_round_one": int(meter.local_flops / max(self.args.num_clients, 1)),
            "client_flops_base_round_total": int(meter_breakdown.get("base", 0)),
            "client_flops_extra_round_total": int(meter_breakdown.get("extra", 0)),
            "client_flops_setup_round_total": int(meter_breakdown.get("setup", 0)),
            "client_flops_base_round_one": int(meter_breakdown.get("base", 0) / max(self.args.num_clients, 1)),
            "client_flops_extra_round_one": int(meter_breakdown.get("extra", 0) / max(self.args.num_clients, 1)),
            "client_flops_setup_round_one": int(meter_breakdown.get("setup", 0) / max(self.args.num_clients, 1)),
            "client_bytes_up_round_one": int(upload_one),
            "client_bytes_down_round_one": 0,
            "client_bytes_total_round_one": int(upload_one),
            "local_flops_cum": int(meter.local_flops),
            "local_flops_base_cum": int(meter_breakdown.get("base", 0)),
            "local_flops_extra_cum": int(meter_breakdown.get("extra", 0)),
            "local_flops_setup_cum": int(meter_breakdown.get("setup", 0)),
            "bytes_up_cum": int(meter.bytes_up),
            "bytes_down_cum": int(meter.bytes_down),
            "bytes_total_cum": int(meter.bytes_total),
        }
        write_jsonl(os.path.join(out_dir, "round_metrics.jsonl"), [round_row])

        result = {
            "seed": int(seed),
            "method": self.args.method,
            "dataset": self.args.dataset,
            "model": self.args.model,
            "best_acc": float(top1),
            "best_round": 1,
            "converged_round": 1,
            "converged_acc": float(top1),
            "stop_round": 1,
            "local_flops": int(meter.local_flops),
            "local_flops_base": int(meter_breakdown.get("base", 0)),
            "local_flops_extra": int(meter_breakdown.get("extra", 0)),
            "local_flops_setup": int(meter_breakdown.get("setup", 0)),
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
        seed_results = []
        for i, seed in enumerate(self.args.seeds):
            seed_results.append(self._run_seed(seed=int(seed), seed_rank=i))
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
            "method": self.args.method,
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


def build_runner(args):
    return FedNCMRunner(args)
