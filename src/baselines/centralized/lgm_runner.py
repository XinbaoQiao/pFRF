from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

from augmentation import AugBasic, get_augmentor
from baselines.common import (
    build_cosine_scheduler,
    build_linear_head_optimizer,
    ensure_dir,
    mean_std,
    set_global_seed,
    write_json,
    write_jsonl,
)
from baselines.common.profiling import linear_head_step_flops
from config import DistillCfg
from data.dataloaders import get_dataset
from models import get_fc, get_model
from synsets import get_distilled_dataset


def _estimate_lgm_flops(
    dataset: str,
    data_root: str,
    real_res: int,
    crop_res: int,
    train_crop_mode: str,
    num_feats: int,
    ipc: int,
    augs_per_batch: int,
    steps: int,
    backbone_forward_flops_per_sample_est: int,
) -> int:
    train_dataset, _ = get_dataset(
        name=dataset,
        res=real_res,
        crop_res=crop_res,
        train_crop_mode=train_crop_mode,
        data_root=data_root,
    )
    num_classes = int(train_dataset.num_classes)
    batch = int(max(ipc, 1) * max(augs_per_batch, 1) * num_classes)
    per_step_head = 2 * batch * int(num_feats) * int(num_classes)
    per_step_backbone = 2 * batch * int(backbone_forward_flops_per_sample_est)
    per_step = int(per_step_head + per_step_backbone)
    return int(per_step * max(steps, 1))


def _evaluate(backbone, head, loader, normalize, num_classes: int) -> tuple[float, float]:
    backbone.eval()
    head.eval()
    c1 = 0
    c5 = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            x = normalize(x)
            z = backbone(x)
            logits = head(z)
            pred = logits.argmax(dim=1)
            c1 += int((pred == y).sum().item())
            if num_classes >= 5:
                top5 = logits.topk(k=5, dim=1).indices
                c5 += int((top5 == y[:, None]).any(dim=1).sum().item())
            total += int(y.shape[0])
    return float(c1 / max(total, 1)), (float(c5 / max(total, 1)) if num_classes >= 5 else 0.0)


def _parse_seed_list(raw) -> list[int]:
    if isinstance(raw, str):
        vals = [v.strip() for v in raw.split(",") if v.strip()]
        return [int(v) for v in vals]
    return [int(v) for v in raw]


def _build_distill_cfg(args) -> DistillCfg:
    cfg = DistillCfg()
    requested_ipc = int(getattr(args, "ipc", 1))
    if requested_ipc != 1:
        raise ValueError(f"LGM baseline only supports ipc=1 in this project, but got ipc={requested_ipc}")
    cfg.dataset = args.dataset
    cfg.model = args.model
    cfg.data_root = args.data_root
    cfg.real_res = int(args.real_res)
    cfg.crop_res = int(args.crop_res)
    cfg.syn_res = int(getattr(args, "syn_res", args.real_res))
    cfg.train_crop_mode = args.train_crop_mode
    cfg.ipc = 1
    cfg.augs_per_batch = max(int(getattr(args, "augs_per_batch", 3)), 1)
    cfg.iterations = max(int(getattr(args, "lgm_iterations", 5000)), 1)
    cfg.objective = "lgm"
    cfg.skip_if_exists = False
    cfg.distill_mode = str(getattr(args, "distill_mode", "pyramid"))
    cfg.aug_mode = str(getattr(args, "aug_mode", "standard"))
    cfg.decorrelate_color = bool(getattr(args, "decorrelate_color", True))
    cfg.init_mode = str(getattr(args, "init_mode", "noise"))
    cfg.lr = float(getattr(args, "distill_lr", getattr(cfg, "lr", 2e-3)))
    cfg.num_workers = int(getattr(args, "local_num_workers", getattr(cfg, "num_workers", 16)))
    cfg.image_log_it = int(getattr(args, "image_log_it", getattr(cfg, "image_log_it", 500)))
    cfg.checkpoint_it = int(getattr(args, "checkpoint_it", getattr(cfg, "checkpoint_it", 1000)))
    cfg.pyramid_extent_it = int(getattr(args, "pyramid_extent_it", getattr(cfg, "pyramid_extent_it", 200)))
    cfg.pyramid_start_res = int(getattr(args, "pyramid_start_res", getattr(cfg, "pyramid_start_res", 1)))
    cfg.distill_opt = str(getattr(args, "distill_opt", "adam"))
    return cfg


def _next_batch(data_iter, data_loader):
    batch = next(data_iter, None)
    if batch is None:
        data_iter = iter(data_loader)
        batch = next(data_iter)
    return batch, data_iter


def _distill_lgm(cfg: DistillCfg, train_dataset, backbone, num_feats: int, use_amp: bool) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    real_batch_size = max(int(cfg.ipc) * int(cfg.augs_per_batch) * int(train_dataset.num_classes), 1)
    train_loader_kwargs = dict(
        shuffle=True,
        num_workers=int(cfg.num_workers),
        batch_size=real_batch_size,
        pin_memory=bool(int(cfg.num_workers) > 0),
        drop_last=True,
        persistent_workers=bool(int(cfg.num_workers) > 0 and int(train_dataset.num_classes) <= 50),
    )
    if int(cfg.num_workers) > 0 and int(train_dataset.num_classes) <= 50:
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    train_iter = iter(train_loader)
    distilled_dataset = get_distilled_dataset(train_dataset=train_dataset, cfg=cfg)
    syn_augmentor = get_augmentor(aug_mode=cfg.aug_mode, crop_res=cfg.crop_res)
    real_augmentor = get_augmentor(aug_mode=cfg.aug_mode, crop_res=cfg.crop_res)
    losses: list[float] = []
    amp_scale = 1024.0

    for step in range(int(cfg.iterations) + 1):
        distilled_dataset.upkeep(step=step)
        fc = get_fc(
            num_feats=num_feats,
            num_classes=int(train_dataset.num_classes),
            distributed=False,
        )

        batch_real, train_iter = _next_batch(train_iter, train_loader)
        x_real, y_real = batch_real
        x_real = x_real.cuda(non_blocking=True).detach()
        y_real = y_real.cuda(non_blocking=True).detach()

        with autocast(enabled=use_amp):
            x_real = real_augmentor(x_real)
            x_real = train_dataset.normalize(x_real)
            z_real = backbone(x_real)
            out_real = fc(z_real)
            loss_real = nn.functional.cross_entropy(out_real, y_real)

        grad_real_w, grad_real_b = torch.autograd.grad(
            loss_real,
            [fc.linear.weight, fc.linear.bias],
            retain_graph=False,
            create_graph=False,
        )
        grad_real = torch.cat(
            [grad_real_w.detach().flatten(), grad_real_b.detach().flatten()],
            dim=0,
        )

        x_syn, y_syn = distilled_dataset.get_data()
        with autocast(enabled=use_amp):
            x_syn = syn_augmentor(torch.cat([x_syn for _ in range(int(cfg.augs_per_batch))], dim=0))
            y_syn = torch.cat([y_syn for _ in range(int(cfg.augs_per_batch))], dim=0)
            x_syn = train_dataset.normalize(x_syn)
            z_syn = backbone(x_syn)
            out_syn = fc(z_syn)
            loss_syn = nn.functional.cross_entropy(out_syn, y_syn)

        grad_syn_w, grad_syn_b = torch.autograd.grad(
            loss_syn,
            [fc.linear.weight, fc.linear.bias],
            retain_graph=True,
            create_graph=True,
        )
        grad_syn = torch.cat([grad_syn_w.flatten(), grad_syn_b.flatten()], dim=0)

        match_loss = 1 - torch.nn.functional.cosine_similarity(grad_real, grad_syn, dim=0)
        scaled_match_loss = match_loss * amp_scale

        distilled_dataset.optimizer.zero_grad(set_to_none=True)
        scaled_match_loss.backward()
        for group in distilled_dataset.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.div_(amp_scale)
        distilled_dataset.optimizer.step()

        losses.append(float(match_loss.detach().item()))

    with torch.no_grad():
        syn_images, syn_labels = distilled_dataset.get_data()
        syn_images = syn_images.detach().cpu()
        syn_labels = syn_labels.detach().cpu()
    return syn_images, syn_labels, losses


def run_lgm(args) -> dict:
    seed_list = _parse_seed_list(args.seeds)
    if len(seed_list) == 0:
        raise ValueError("At least one seed is required for lgm baseline")

    cfg = _build_distill_cfg(args)
    out_dir = os.path.join(args.output_root, "lgm", args.dataset, args.model)
    ensure_dir(out_dir)

    train_dataset, test_dataset = get_dataset(
        name=args.dataset,
        res=args.real_res,
        crop_res=args.crop_res,
        train_crop_mode=args.train_crop_mode,
        data_root=args.data_root,
    )
    num_classes = int(train_dataset.num_classes)
    test_kwargs = dict(
        shuffle=False,
        num_workers=args.eval_num_workers,
        batch_size=args.eval_batch_size,
        drop_last=False,
        pin_memory=args.eval_num_workers > 0,
        persistent_workers=args.eval_num_workers > 0,
    )
    if args.eval_num_workers > 0:
        test_kwargs["prefetch_factor"] = 2
    test_loader = DataLoader(test_dataset, **test_kwargs)

    set_global_seed(int(seed_list[0]))
    backbone, num_feats = get_model(name=args.model, distributed=False)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    syn_images, syn_labels, distill_losses = _distill_lgm(
        cfg=cfg,
        train_dataset=train_dataset,
        backbone=backbone,
        num_feats=int(num_feats),
        use_amp=bool(getattr(args, "use_amp", True)),
    )
    torch.save({"images": syn_images, "labels": syn_labels}, os.path.join(out_dir, "data.pth"))

    backbone_forward_flops_per_sample_est = int(
        getattr(args, "backbone_forward_flops_per_sample_est", 43897295616)
    )
    distill_steps = int(len(distill_losses))
    distill_flops = _estimate_lgm_flops(
        dataset=cfg.dataset,
        data_root=cfg.data_root,
        real_res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode=cfg.train_crop_mode,
        num_feats=int(num_feats),
        ipc=cfg.ipc,
        augs_per_batch=cfg.augs_per_batch,
        steps=distill_steps,
        backbone_forward_flops_per_sample_est=backbone_forward_flops_per_sample_est,
    )

    syn_ds = TensorDataset(syn_images.clone(), syn_labels.clone())
    syn_batch_size = max(1, min(int(args.local_batch_size), len(syn_ds)))
    syn_loader = DataLoader(
        syn_ds,
        shuffle=True,
        num_workers=0,
        batch_size=syn_batch_size,
        drop_last=False,
        pin_memory=False,
    )
    train_augmentor = AugBasic(crop_res=int(args.crop_res)).cuda()

    seed_results = []
    for seed in seed_list:
        set_global_seed(int(seed))
        head = get_fc(num_feats=num_feats, num_classes=num_classes, distributed=False)
        head.train()
        optimizer, _ = build_linear_head_optimizer(
            head.parameters(),
            base_lr=float(args.local_lr),
            batch_size=int(syn_batch_size),
        )
        scheduler = build_cosine_scheduler(optimizer, total_epochs=int(args.max_rounds))

        best_top1 = 0.0
        best_top5 = 0.0
        bad = 0
        chance_acc = 1.0 / max(num_classes, 1)
        head_flops = 0
        rounds = []
        for epoch in range(1, int(args.max_rounds) + 1):
            flops_round = 0
            for x, y in syn_loader:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                with autocast(enabled=bool(getattr(args, "use_amp", True))):
                    with torch.no_grad():
                        x = train_augmentor(x)
                        x = train_dataset.normalize(x)
                        z = backbone(x).float()
                    logits = head(z)
                    loss = nn.functional.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(getattr(args, "grad_clip_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        head.parameters(), max_norm=float(getattr(args, "grad_clip_norm", 0.0))
                    )
                optimizer.step()
                step_flops = linear_head_step_flops(int(y.shape[0]), int(num_feats), int(num_classes))
                flops_round += int(step_flops)
                head_flops += int(step_flops)
            scheduler.step()
            top1, top5 = _evaluate(backbone, head, test_loader, test_dataset.normalize, num_classes)
            rounds.append(
                {
                    "round": int(epoch),
                    "val_top1": float(top1),
                    "val_top5": float(top5),
                    "head_train_flops_round": int(flops_round),
                    "head_train_flops_cum": int(head_flops),
                    "distill_flops_round_est": int(distill_flops if epoch == 1 else 0),
                }
            )
            if top1 > (best_top1 + float(args.min_delta)):
                best_top1 = float(top1)
                best_top5 = float(top5)
                bad = 0
            else:
                if epoch >= int(args.warmup_rounds) and float(top1) >= float(chance_acc):
                    bad += 1
            if epoch >= int(args.warmup_rounds) and float(top1) >= float(chance_acc) and bad >= int(args.patience_rounds):
                break

        seed_dir = os.path.join(out_dir, f"seed_{int(seed)}")
        ensure_dir(seed_dir)
        write_jsonl(os.path.join(seed_dir, "round_metrics.jsonl"), rounds)
        seed_results.append(
            {
                "seed": int(seed),
                "top1": float(best_top1),
                "top5": float(best_top5),
                "head_train_flops_total": int(head_flops),
                "train_flops_total_with_distill_est": int(head_flops + distill_flops),
            }
        )

    top1_mean, top1_std = mean_std([float(v["top1"]) for v in seed_results])
    top5_mean, top5_std = mean_std([float(v["top5"]) for v in seed_results])
    head_flops_mean, head_flops_std = mean_std([float(v["head_train_flops_total"]) for v in seed_results])
    total_flops_mean, total_flops_std = mean_std(
        [float(v["train_flops_total_with_distill_est"]) for v in seed_results]
    )
    summary = {
        "method": "lgm",
        "dataset": cfg.dataset,
        "model": cfg.model,
        "objective": cfg.objective,
        "distill_seed": int(seed_list[0]),
        "distill_iterations": int(cfg.iterations),
        "distill_steps_actual": int(distill_steps),
        "ipc": int(cfg.ipc),
        "augs_per_batch": int(cfg.augs_per_batch),
        "distill_flops_total_estimated": int(distill_flops),
        "distill_loss_final": float(distill_losses[-1]) if len(distill_losses) > 0 else None,
        "top1_mean": float(top1_mean),
        "top1_std": float(top1_std),
        "top5_mean": float(top5_mean),
        "top5_std": float(top5_std),
        "head_train_flops_mean": float(head_flops_mean),
        "head_train_flops_std": float(head_flops_std),
        "train_flops_mean": float(total_flops_mean),
        "train_flops_std": float(total_flops_std),
        "seed_results": seed_results,
    }
    write_json(os.path.join(out_dir, "result_summary.json"), summary)
    return summary
