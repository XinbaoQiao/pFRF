from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from augmentation import AugBasic
from models import get_fc
from .interface import build_semantic_translator


@dataclass(frozen=True)
class PersonalizedEvalResult:
    client_id: int
    train_top1: float
    test_top1: float
    test_top5: float
    personal_semantic_head_state_dict: dict[str, torch.Tensor]
    semantic_translator_state_dict: dict[str, torch.Tensor]


def _state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def train_local_feature_head(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_feats: int,
    num_classes: int,
    lr: float,
    epochs: int,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], float]:
    ds = TensorDataset(
        features.detach().cpu().to(dtype=torch.float32),
        labels.detach().cpu().to(dtype=torch.long),
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, min(int(batch_size), len(ds))),
        shuffle=True,
        drop_last=False,
    )

    fc = get_fc(num_feats=num_feats, num_classes=num_classes, distributed=False)
    fc.train()
    optimizer = torch.optim.Adam(fc.parameters(), lr=float(lr), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(int(epochs), 1), eta_min=0.0)
    scaler = GradScaler()

    for _ in tqdm(range(int(epochs)), desc="Local Head Train", leave=False):
        for z, y in loader:
            z = z.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with autocast(enabled=True):
                logits = fc(z)
                loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    fc.eval()
    top1_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=1).cuda()
    with torch.no_grad():
        for z, y in loader:
            z = z.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            logits = fc(z)
            top1_metric.update(logits, y)
    train_top1 = float(top1_metric.compute().item())
    return _state_dict_to_cpu(fc.state_dict()), train_top1


def _train_semantic_translator(
    *,
    syn_images: torch.Tensor,
    syn_labels: torch.Tensor,
    test_loader: DataLoader,
    backbone: nn.Module,
    normalize,
    num_feats: int,
    num_classes: int,
    crop_res: int,
    translator_type: str,
    personal_semantic_head_state_dict: dict[str, torch.Tensor],
    lr: float,
    epochs: int,
    batch_size: int,
    identity_lambda: float,
) -> tuple[dict[str, torch.Tensor], float, float]:
    syn_ds = TensorDataset(syn_images.detach().cpu(), syn_labels.detach().cpu())
    syn_loader = DataLoader(
        syn_ds,
        batch_size=max(1, min(int(batch_size), len(syn_ds))),
        shuffle=True,
        drop_last=False,
    )

    augmentor = AugBasic(crop_res=crop_res).cuda()
    semantic_translator = build_semantic_translator(translator_type=translator_type, dim=num_feats).cuda()
    personal_semantic_head = get_fc(num_feats=num_feats, num_classes=num_classes, distributed=False)
    personal_semantic_head.load_state_dict(personal_semantic_head_state_dict)
    personal_semantic_head.eval()
    for p in personal_semantic_head.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(semantic_translator.parameters(), lr=float(lr), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(int(epochs), 1), eta_min=0.0)
    scaler = GradScaler()

    for _ in tqdm(range(int(epochs)), desc="Semantic Translator Train", leave=False):
        for x, y in syn_loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with autocast(enabled=True):
                with torch.no_grad():
                    x = augmentor(x)
                    x = normalize(x)
                    z = backbone(x)
                z_adapt = semantic_translator(z)
                logits = personal_semantic_head(z_adapt)
                loss = nn.functional.cross_entropy(logits, y)
                if float(identity_lambda) > 0.0:
                    loss = loss + float(identity_lambda) * nn.functional.mse_loss(z_adapt, z)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    semantic_translator.eval()
    personal_semantic_head.eval()
    top1_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=1).cuda()
    top5_metric = None
    if num_classes >= 5:
        top5_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=5).cuda()

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Personalized Eval", leave=False):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            x = normalize(x)
            z = backbone(x)
            logits = personal_semantic_head(semantic_translator(z))
            top1_metric.update(logits, y)
            if top5_metric is not None:
                top5_metric.update(logits, y)

    top1 = float(top1_metric.compute().item())
    top5 = float(top5_metric.compute().item()) if top5_metric is not None else 0.0
    return _state_dict_to_cpu(semantic_translator.state_dict()), top1, top5


def train_personalized_clients(
    *,
    output_dir: str,
    client_features: list[tuple[int, torch.Tensor, torch.Tensor]],
    client_test_loaders: list[DataLoader],
    syn_images: torch.Tensor,
    syn_labels: torch.Tensor,
    backbone: nn.Module,
    normalize,
    num_feats: int,
    num_classes: int,
    crop_res: int,
    classifier_lr: float,
    classifier_epochs: int,
    interface_type: str,
    interface_lr: float,
    interface_epochs: int,
    batch_size: int,
    identity_lambda: float,
) -> list[PersonalizedEvalResult]:
    results = []
    clients_dir = os.path.join(output_dir, "clients")
    os.makedirs(clients_dir, exist_ok=True)

    for client_id, feats, labels in client_features:
        personal_semantic_head_state_dict, train_top1 = train_local_feature_head(
            features=feats,
            labels=labels,
            num_feats=num_feats,
            num_classes=num_classes,
            lr=classifier_lr,
            epochs=classifier_epochs,
            batch_size=batch_size,
        )
        semantic_translator_state_dict, test_top1, test_top5 = _train_semantic_translator(
            syn_images=syn_images,
            syn_labels=syn_labels,
            test_loader=client_test_loaders[client_id],
            backbone=backbone,
            normalize=normalize,
            num_feats=num_feats,
            num_classes=num_classes,
            crop_res=crop_res,
            translator_type=interface_type,
            personal_semantic_head_state_dict=personal_semantic_head_state_dict,
            lr=interface_lr,
            epochs=interface_epochs,
            batch_size=batch_size,
            identity_lambda=identity_lambda,
        )
        result = PersonalizedEvalResult(
            client_id=client_id,
            train_top1=train_top1,
            test_top1=test_top1,
            test_top5=test_top5,
            personal_semantic_head_state_dict=personal_semantic_head_state_dict,
            semantic_translator_state_dict=semantic_translator_state_dict,
        )
        client_dir = os.path.join(clients_dir, f"client_{client_id:04d}")
        os.makedirs(client_dir, exist_ok=True)
        torch.save(
            {"fc_state_dict": personal_semantic_head_state_dict},
            os.path.join(client_dir, "personal_semantic_head.pth"),
        )
        torch.save(
            {"semantic_translator_state_dict": semantic_translator_state_dict},
            os.path.join(client_dir, "semantic_translator.pth"),
        )
        with open(os.path.join(client_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "client_id": client_id,
                    "personal_semantic_head_train_top1": train_top1,
                    "personalized_test_top1": test_top1,
                    "personalized_test_top5": test_top5,
                },
                f,
                indent=4,
            )
        results.append(result)
    return results
