import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from augmentation import AugBasic
from config import PrecomputeCfg
from data.dataloaders import get_dataset
from models import get_fc, get_model


@torch.no_grad()
def compute_centers(
    loader: DataLoader,
    model: nn.Module,
    normalize,
    num_classes: int,
    num_feats: int,
) -> dict:
    sum_per_class = torch.zeros(
        (num_classes, num_feats), device="cuda", dtype=torch.float64
    )
    cnt_per_class = torch.zeros((num_classes,), device="cuda", dtype=torch.long)

    sum_all = torch.zeros((num_feats,), device="cuda", dtype=torch.float64)
    cnt_all = 0

    for x, y in tqdm(loader, desc="Computing Centers", leave=True):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        x = normalize(x)

        with autocast(enabled=True):
            z = model(x)
        z = z.float()

        sum_all += z.sum(dim=0, dtype=torch.float64)
        cnt_all += int(z.shape[0])

        sum_per_class.index_add_(0, y, z.to(dtype=torch.float64))
        cnt_per_class += torch.bincount(y, minlength=num_classes)

    mu_target = (sum_per_class / cnt_per_class.clamp_min(1)[:, None]).to(
        dtype=torch.float32
    )

    denom = (cnt_all - cnt_per_class).clamp_min(1)[:, None]
    mu_non_target = ((sum_all[None, :] - sum_per_class) / denom).to(dtype=torch.float32)

    flow_star = mu_non_target - mu_target

    return {
        "mu_target": mu_target,
        "mu_non_target": mu_non_target,
        "flow_star": flow_star,
        "counts_per_class": cnt_per_class,
        "count_all": cnt_all,
    }


def train_golden_classifier(
    train_loader: DataLoader,
    test_loader: DataLoader,
    backbone: nn.Module,
    normalize,
    num_feats: int,
    num_classes: int,
    crop_res: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    eval_it: int,
) -> dict:
    augmentor = AugBasic(crop_res=crop_res).cuda()
    fc = get_fc(num_feats=num_feats, num_classes=num_classes, distributed=False)
    fc.train()

    optimizer = torch.optim.Adam(fc.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    scaler = GradScaler()

    best_top1 = -1.0
    best_state = None

    for epoch in tqdm(range(epochs), desc="Training Golden Classifier", leave=True):
        for x, y in tqdm(train_loader, desc="Train Epoch", leave=False):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            with autocast(enabled=True):
                with torch.no_grad():
                    x = augmentor(x)
                    x = normalize(x)
                    z = backbone(x)

                out = fc(z)
                loss = nn.functional.cross_entropy(out, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        if eval_it > 0 and (epoch % eval_it == 0 or epoch == epochs - 1):
            fc.eval()
            top1_metric = MulticlassAccuracy(
                average="micro", num_classes=num_classes, top_k=1
            ).cuda()
            for x, y in tqdm(test_loader, desc="Eval Epoch", leave=False):
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                x = normalize(x)
                z = backbone(x)
                out = fc(z)
                top1_metric.update(out, y)
            top1 = float(top1_metric.compute().item())
            if top1 > best_top1:
                best_top1 = top1
                best_state = {k: v.detach().cpu() for k, v in fc.state_dict().items()}
            fc.train()

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in fc.state_dict().items()}

    return {"state_dict": best_state, "best_top1": best_top1}


def main(cfg: PrecomputeCfg):
    save_dir = os.path.join(
        "logged_files", cfg.job_tag, cfg.dataset, cfg.model, "precompute"
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "cfg.json"), "w") as f:
        f.write(json.dumps(cfg.as_dict(), indent=4))
    torch.save(cfg, os.path.join(save_dir, "cfg.pth"))

    train_dataset, test_dataset = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode=cfg.train_crop_mode,
        data_root=cfg.data_root,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        drop_last=False,
    )

    backbone, num_feats = get_model(name=cfg.model, distributed=False)

    centers = compute_centers(
        loader=train_loader,
        model=backbone,
        normalize=train_dataset.normalize,
        num_classes=train_dataset.num_classes,
        num_feats=num_feats,
    )

    stats_dict = {
        **{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in centers.items()},
        "meta": {
            "dataset": cfg.dataset,
            "model": cfg.model,
            "num_classes": train_dataset.num_classes,
            "num_feats": num_feats,
        },
    }
    torch.save(stats_dict, os.path.join(save_dir, "stats.pt"))

    golden = train_golden_classifier(
        train_loader=train_loader,
        test_loader=test_loader,
        backbone=backbone,
        normalize=train_dataset.normalize,
        num_feats=num_feats,
        num_classes=train_dataset.num_classes,
        crop_res=cfg.crop_res,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.train_epochs,
        eval_it=cfg.eval_it,
    )
    torch.save(
        {"meta": stats_dict["meta"], **golden},
        os.path.join(save_dir, "golden_fc.pth"),
    )

    print("Saved stats to", os.path.join(save_dir, "stats.pt"))
    print("Saved golden classifier to", os.path.join(save_dir, "golden_fc.pth"))


if __name__ == "__main__":
    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = PrecomputeCfg(explicit_bool=True).parse_args()
    main(args)
