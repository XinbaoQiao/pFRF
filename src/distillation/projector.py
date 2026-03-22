import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from augmentation import AugBasic
from config import ProjectorCfg
from data.dataloaders import get_dataset
from models import get_fc, get_model


def _next_batch(data_iter, data_loader):
    batch = next(data_iter, None)
    if batch is None:
        data_iter = iter(data_loader)
        batch = next(data_iter)
    return batch, data_iter


@torch.no_grad()
def evaluate(
    test_loader: DataLoader,
    deployment_backbone: nn.Module,
    projector: nn.Module | None,
    classifier: nn.Module,
    normalize,
):
    num_classes = test_loader.dataset.num_classes
    top1_metric = MulticlassAccuracy(
        average="micro", num_classes=num_classes, top_k=1
    ).cuda()
    if num_classes >= 5:
        top5_metric = MulticlassAccuracy(
            average="micro", num_classes=num_classes, top_k=5
        ).cuda()

    for x, y in tqdm(test_loader, desc="Evaluating", leave=False):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        x = normalize(x)
        z = deployment_backbone(x)
        if projector is not None:
            z = projector(z)
        logits = classifier(z)
        top1_metric.update(logits, y)
        if num_classes >= 5:
            top5_metric.update(logits, y)

    top1 = float(top1_metric.compute().item())
    if num_classes >= 5:
        top5 = float(top5_metric.compute().item())
    else:
        top5 = 0.0
    return top1, top5


def train_projector_on_synthetic(
    syn_loader: DataLoader,
    normalize,
    augmentor: nn.Module,
    original_backbone: nn.Module,
    deployment_backbone: nn.Module,
    projector: nn.Module,
    steps: int,
    lr: float,
):
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    scaler = GradScaler()
    syn_iter = iter(syn_loader)
    projector.train()

    for _ in tqdm(range(steps), desc="Stage1 Projector", leave=True):
        batch, syn_iter = _next_batch(syn_iter, syn_loader)
        x, _ = batch
        x = x.cuda(non_blocking=True)

        with torch.no_grad():
            x = augmentor(x)
            x = normalize(x)
            z_o = original_backbone(x).float()
            z_d = deployment_backbone(x).float()

        with autocast(enabled=True):
            pred = projector(z_d)
            loss = nn.functional.mse_loss(pred, z_o)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    projector.eval()


def train_classifier_on_synthetic(
    syn_loader: DataLoader,
    normalize,
    augmentor: nn.Module,
    deployment_backbone: nn.Module,
    projector: nn.Module | None,
    classifier: nn.Module,
    steps: int,
    lr: float,
):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    scaler = GradScaler()
    syn_iter = iter(syn_loader)
    classifier.train()

    for _ in tqdm(range(steps), desc="Stage2 Classifier", leave=True):
        batch, syn_iter = _next_batch(syn_iter, syn_loader)
        x, y = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        with torch.no_grad():
            x = augmentor(x)
            x = normalize(x)
            z = deployment_backbone(x).float()
            if projector is not None:
                z = projector(z).float()

        with autocast(enabled=True):
            logits = classifier(z)
            loss = nn.functional.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    classifier.eval()


def main(cfg: ProjectorCfg):
    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)

    _, test_dataset = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode=cfg.train_crop_mode,
        data_root=cfg.data_root,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=cfg.num_workers,
        batch_size=cfg.real_batch_size,
        drop_last=False,
    )

    syn_data_path = cfg.syn_data_path
    run_dir = None
    if syn_data_path is None:
        model_dir = os.path.join(
            "logged_files", cfg.job_tag, cfg.dataset, cfg.distill_model
        )
        syn_set_files = sorted(
            list(glob.glob(os.path.join(model_dir, "**", "data.pth"), recursive=True))
        )
        if len(syn_set_files) == 0:
            raise FileNotFoundError(f"No data.pth found under {model_dir}")
        syn_data_path = syn_set_files[0]
        run_dir = "/".join(syn_data_path.split("/")[:-1])
    else:
        run_dir = "/".join(syn_data_path.split("/")[:-1])

    save_dir = os.path.join(run_dir, "projector")
    save_file = os.path.join(save_dir, "{}.pth".format(cfg.eval_model))
    projector_file = os.path.join(save_dir, "projector_{}.pth".format(cfg.eval_model))
    classifier_file = os.path.join(save_dir, "classifier_{}.pth".format(cfg.eval_model))
    if os.path.exists(save_file) and cfg.skip_if_exists:
        print("This projector eval already done.")
        print("Exiting...")
        return

    syn_set = torch.load(syn_data_path, weights_only=False)
    if "images" not in syn_set or "labels" not in syn_set:
        raise KeyError("synthetic data must contain both images and labels")
    syn_images = syn_set["images"].cuda()
    syn_labels = syn_set["labels"].long().cuda()
    if int(syn_images.shape[0]) != int(syn_labels.shape[0]):
        raise ValueError("synthetic images and labels size mismatch")

    original_backbone, num_feats_o = get_model(
        cfg.distill_model, distributed=torch.cuda.device_count() > 1
    )
    deployment_backbone, num_feats_d = get_model(
        cfg.eval_model, distributed=torch.cuda.device_count() > 1
    )

    for p in original_backbone.parameters():
        p.requires_grad_(False)
    for p in deployment_backbone.parameters():
        p.requires_grad_(False)
    original_backbone.eval()
    deployment_backbone.eval()

    num_classes = int(test_dataset.num_classes)
    ds = TensorDataset(syn_images.detach().clone(), syn_labels.detach().clone())
    syn_loader = DataLoader(
        ds, batch_size=min(cfg.syn_batch_size, len(ds)), shuffle=True, drop_last=False
    )

    normalize = test_dataset.normalize
    augmentor = AugBasic(crop_res=cfg.crop_res).cuda()
    projector = None
    if cfg.use_projector:
        projector = nn.Linear(num_feats_d, num_feats_o).cuda()
        train_projector_on_synthetic(
            syn_loader=syn_loader,
            normalize=normalize,
            augmentor=augmentor,
            original_backbone=original_backbone,
            deployment_backbone=deployment_backbone,
            projector=projector,
            steps=cfg.projector_steps,
            lr=cfg.projector_lr,
        )
        for p in projector.parameters():
            p.requires_grad_(False)
        feat_dim = num_feats_o
    else:
        feat_dim = num_feats_d

    classifier = get_fc(num_feats=feat_dim, num_classes=num_classes, distributed=False)
    train_classifier_on_synthetic(
        syn_loader=syn_loader,
        normalize=normalize,
        augmentor=augmentor,
        deployment_backbone=deployment_backbone,
        projector=projector,
        classifier=classifier,
        steps=cfg.classifier_steps,
        lr=cfg.classifier_lr,
    )
    top1, top5 = evaluate(
        test_loader=test_loader,
        deployment_backbone=deployment_backbone,
        projector=projector,
        classifier=classifier,
        normalize=normalize,
    )

    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "top1": top1,
            "top5": top5,
            "syn_data_path": syn_data_path,
            "use_projector": bool(cfg.use_projector),
            "classifier_steps": int(cfg.classifier_steps),
            "projector_steps": int(cfg.projector_steps) if cfg.use_projector else 0,
        },
        save_file,
    )
    torch.save({"state_dict": classifier.state_dict()}, classifier_file)
    if cfg.use_projector and projector is not None:
        torch.save({"state_dict": projector.state_dict()}, projector_file)

    print(f"Results saved to {save_file}")
    print("Top 1: {:.2f}".format(top1 * 100))
    print("Top 5: {:.2f}".format(top5 * 100))


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = ProjectorCfg(explicit_bool=True).parse_args()
    main(args)
