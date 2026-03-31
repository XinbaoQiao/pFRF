from __future__ import annotations

from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from baselines.common import (
    build_cosine_scheduler,
    build_linear_head_optimizer,
    ensure_dir,
    mean_std,
    sanitize_path_for_log,
    set_global_seed,
    write_json,
    write_jsonl,
)
from baselines.common.profiling import linear_head_step_flops
from data.feature_cache import build_feature_loader, forward_features, shared_feature_cache_dir
from data.dataloaders import get_dataset
from models import get_fc, get_model


def _normalize_train_mode(raw: str) -> str:
    low = str(raw).lower().strip()
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
    mode = alias.get(low, low)
    if mode not in {"linear_probe", "finetune"}:
        raise ValueError(f"Unsupported full_dataset train_mode: {raw}")
    return mode


def _train_mode_output_dir_name(train_mode: str) -> str:
    if train_mode == "linear_probe":
        return "linearprobing"
    if train_mode == "finetune":
        return "finetuning"
    raise ValueError(f"Unsupported full_dataset train_mode: {train_mode}")


def _full_dataset_out_dir(args, train_mode: str) -> str:
    layout = str(getattr(args, "full_dataset_output_layout", "legacy")).lower().strip()
    if layout == "by_mode":
        return os.path.join(
            args.output_root,
            "full_dataset",
            _train_mode_output_dir_name(train_mode),
            args.dataset,
            args.model,
        )
    return os.path.join(args.output_root, "full_dataset", args.dataset, args.model)


def _log_progress(args, message: str) -> None:
    ts = datetime.now().strftime("%F %T")
    train_mode = _normalize_train_mode(getattr(args, "train_mode", "linear_probe"))
    print(
        f"[{ts}] [full_dataset] train_mode={train_mode} dataset={args.dataset} model={args.model} {message}",
        flush=True,
    )


def _shared_feature_cache_dir(output_root: str) -> str:
    return shared_feature_cache_dir(output_root)


def _shared_full_dataset_cache_path(args) -> str:
    cache_root = _shared_feature_cache_dir(args.output_root)
    scope_dir = os.path.join(
        cache_root,
        str(args.dataset),
        str(args.model),
        (
            f"rr{int(args.real_res)}_cr{int(args.crop_res)}"
            f"_cm{str(args.train_crop_mode)}_full"
        ),
    )
    ensure_dir(scope_dir)
    return os.path.join(scope_dir, "feature_cache.pt")


def _evaluate(backbone, head, loader, normalize, num_classes: int, use_amp: bool = True) -> tuple[float, float]:
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
            with autocast(enabled=bool(use_amp)):
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


def run_full_dataset(args) -> dict:
    train_mode = _normalize_train_mode(getattr(args, "train_mode", "linear_probe"))
    _log_progress(args, "starting run")
    train_dataset, test_dataset = get_dataset(
        name=args.dataset,
        res=args.real_res,
        crop_res=args.crop_res,
        train_crop_mode=args.train_crop_mode,
        data_root=args.data_root,
    )
    num_classes = int(train_dataset.num_classes)
    _log_progress(
        args,
        (
            f"loaded datasets train_size={len(train_dataset)} test_size={len(test_dataset)} "
            f"num_classes={num_classes}"
        ),
    )
    train_kwargs = dict(
        shuffle=True,
        num_workers=args.local_num_workers,
        batch_size=args.local_batch_size,
        drop_last=False,
        pin_memory=args.local_num_workers > 0,
        persistent_workers=args.local_num_workers > 0,
    )
    if args.local_num_workers > 0:
        train_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, **train_kwargs)
    train_loader_no_shuffle = build_feature_loader(
        train_dataset,
        batch_size=int(getattr(args, "feature_batch_size", args.local_batch_size)),
        num_workers=int(args.local_num_workers),
        is_heavy_dataset=str(args.dataset).lower().startswith("imagenet"),
    )
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
    backbone, num_feats = get_model(name=args.model, distributed=False)
    _log_progress(args, f"loaded backbone num_feats={int(num_feats)}")
    if train_mode == "linear_probe":
        for p in backbone.parameters():
            p.requires_grad_(False)
    use_feature_cache = bool(getattr(args, "cache_features", False)) and train_mode == "linear_probe"
    if bool(getattr(args, "cache_features", False)) and train_mode != "linear_probe":
        _log_progress(args, "cache_features ignored for finetune mode")
    train_epochs = max(int(getattr(args, "max_rounds", 200)), 1)
    _log_progress(
        args,
        (
            f"config train_mode={train_mode} cache_features={use_feature_cache} "
            f"train_batch_size={int(args.local_batch_size)} eval_batch_size={int(args.eval_batch_size)} "
            f"train_epochs={train_epochs}"
        ),
    )

    def _train_preprocess(x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[-1]) != int(args.crop_res) or int(x.shape[-2]) != int(args.crop_res):
            x = F.interpolate(
                x,
                size=(int(args.crop_res), int(args.crop_res)),
                mode="bilinear",
                align_corners=False,
            )
        x = train_dataset.normalize(x)
        return x

    seed_list = _parse_seed_list(args.seeds)
    out_dir = _full_dataset_out_dir(args, train_mode)
    ensure_dir(out_dir)
    feature_cache_path = _shared_full_dataset_cache_path(args)
    cached_feats = None
    cached_labels = None
    if use_feature_cache:
        if os.path.exists(feature_cache_path):
            _log_progress(args, f"loading cached features from {sanitize_path_for_log(feature_cache_path)}")
            payload = torch.load(feature_cache_path, map_location="cpu")
            cached_feats = payload["features"]
            cached_labels = payload["labels"]
            _log_progress(
                args,
                (
                    f"loaded cached features num_samples={int(cached_labels.shape[0])} "
                    f"feature_dim={int(cached_feats.shape[1]) if cached_feats.ndim > 1 else int(num_feats)}"
                ),
            )
        else:
            _log_progress(args, f"building feature cache at {sanitize_path_for_log(feature_cache_path)}")
            cached_feats, cached_labels = forward_features(
                loader=train_loader_no_shuffle,
                backbone=backbone,
                preprocess_fn=_train_preprocess,
                feature_dim=int(num_feats),
                use_amp=bool(getattr(args, "use_amp", True)),
            )
            torch.save({"features": cached_feats, "labels": cached_labels}, feature_cache_path)
            _log_progress(
                args,
                (
                    f"saved feature cache num_samples={int(cached_labels.shape[0])} "
                    f"feature_dim={int(cached_feats.shape[1]) if cached_feats.ndim > 1 else int(num_feats)}"
                ),
            )
    seed_results = []
    # Keep centralized full-dataset training horizon aligned with the
    # main/baseline global training rounds rather than client local epochs.
    for seed in seed_list:
        _log_progress(args, f"seed={int(seed)} start")
        set_global_seed(int(seed))
        if train_mode == "finetune":
            if seed_results:
                del backbone
                torch.cuda.empty_cache()
                backbone, _ = get_model(name=args.model, distributed=False)
            for p in backbone.parameters():
                p.requires_grad_(True)
            backbone.train()
        else:
            backbone.eval()
        head = get_fc(num_feats=num_feats, num_classes=num_classes, distributed=False)
        head.train()
        if train_mode == "linear_probe":
            optimizer, _ = build_linear_head_optimizer(
                head.parameters(),
                base_lr=float(args.local_lr),
                batch_size=int(args.local_batch_size),
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": list(backbone.parameters()), "lr": float(getattr(args, "backbone_lr", 1e-4))},
                    {"params": list(head.parameters()), "lr": float(args.local_lr)},
                ],
                weight_decay=float(getattr(args, "weight_decay", 0.0)),
            )
        scheduler = build_cosine_scheduler(optimizer, total_epochs=train_epochs)
        scaler = GradScaler(enabled=bool(getattr(args, "use_amp", True)))
        best_top1 = 0.0
        best_top5 = 0.0
        bad = 0
        chance_acc = max(1.0 / max(num_classes, 1), 0.10)
        flops = 0
        rounds = []
        for epoch in range(1, train_epochs + 1):
            flops_round = 0
            max_train_batches = max(int(getattr(args, "smoke_max_train_batches", 0)), 0)
            seen_batches = 0
            if use_feature_cache:
                if int(cached_labels.shape[0]) > 0:
                    order = torch.randperm(int(cached_labels.shape[0]))
                else:
                    order = torch.zeros((0,), dtype=torch.long)
                for start in range(0, int(order.shape[0]), int(args.local_batch_size)):
                    idx = order[start : start + int(args.local_batch_size)]
                    z = cached_feats.index_select(0, idx).cuda(non_blocking=True)
                    y = cached_labels.index_select(0, idx).cuda(non_blocking=True)
                    with autocast(enabled=bool(getattr(args, "use_amp", True))):
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
                    flops += int(step_flops)
                    seen_batches += 1
                    if max_train_batches > 0 and seen_batches >= max_train_batches:
                        break
            elif train_mode == "linear_probe":
                for x, y in train_loader:
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
                    with autocast(enabled=bool(getattr(args, "use_amp", True))):
                        with torch.no_grad():
                            x = _train_preprocess(x)
                            z = backbone(x)
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
                    flops += int(step_flops)
                    seen_batches += 1
                    if max_train_batches > 0 and seen_batches >= max_train_batches:
                        break
            else:
                backbone.train()
                for x, y in train_loader:
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(enabled=bool(getattr(args, "use_amp", True))):
                        x = _train_preprocess(x)
                        z = backbone(x)
                        logits = head(z)
                        loss = nn.functional.cross_entropy(logits, y)
                    scaler.scale(loss).backward()
                    if float(getattr(args, "grad_clip_norm", 0.0)) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(backbone.parameters()) + list(head.parameters()),
                            max_norm=float(getattr(args, "grad_clip_norm", 0.0)),
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    seen_batches += 1
                    if max_train_batches > 0 and seen_batches >= max_train_batches:
                        break
            scheduler.step()
            top1, top5 = _evaluate(
                backbone,
                head,
                test_loader,
                test_dataset.normalize,
                num_classes,
                use_amp=bool(getattr(args, "use_amp", True)),
            )
            if epoch == 1 or epoch == train_epochs or epoch % 10 == 0:
                _log_progress(
                    args,
                    (
                        f"seed={int(seed)} epoch={epoch}/{train_epochs} "
                        f"batches={seen_batches} val_top1={float(top1):.4f} val_top5={float(top5):.4f} "
                        f"best_top1={float(best_top1):.4f} bad_epochs={bad}"
                    ),
                )
            rounds.append(
                {
                    "round": int(epoch),
                    "val_top1": float(top1),
                    "val_top5": float(top5),
                    "head_train_flops_round": int(flops_round),
                    "head_train_flops_cum": int(flops),
                    "train_mode": train_mode,
                    "distill_flops_round_est": 0,
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
                _log_progress(
                    args,
                    (
                        f"seed={int(seed)} early_stop epoch={epoch} "
                        f"best_top1={float(best_top1):.4f} best_top5={float(best_top5):.4f}"
                    ),
                )
                break
        seed_dir = os.path.join(out_dir, f"seed_{int(seed)}")
        ensure_dir(seed_dir)
        write_jsonl(os.path.join(seed_dir, "round_metrics.jsonl"), rounds)
        _log_progress(
            args,
            (
                f"seed={int(seed)} finished best_top1={float(best_top1):.4f} "
                f"best_top5={float(best_top5):.4f} total_flops={int(flops)}"
            ),
        )
        seed_results.append(
            {
                "seed": int(seed),
                "train_mode": train_mode,
                "top1": float(best_top1),
                "top5": float(best_top5),
                "train_flops_total": int(flops),
            }
        )
    top1_mean, top1_std = mean_std([float(v["top1"]) for v in seed_results])
    top5_mean, top5_std = mean_std([float(v["top5"]) for v in seed_results])
    flops_mean, flops_std = mean_std([float(v["train_flops_total"]) for v in seed_results])
    summary = {
        "method": "full_dataset",
        "train_mode": train_mode,
        "dataset": args.dataset,
        "model": args.model,
        "top1_mean": float(top1_mean),
        "top1_std": float(top1_std),
        "top5_mean": float(top5_mean),
        "top5_std": float(top5_std),
        "train_flops_mean": float(flops_mean),
        "train_flops_std": float(flops_std),
        "seed_results": seed_results,
    }
    write_json(os.path.join(out_dir, "result_summary.json"), summary)
    _log_progress(
        args,
        f"saved summary to {sanitize_path_for_log(os.path.join(out_dir, 'result_summary.json'))}",
    )
    return summary
