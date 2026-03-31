import glob
import json
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
from config import PrototypeHeadCfg
from data.dataloaders import get_dataset
from models import get_model


def _resolve_run_dir_from_syn_path(syn_data_path: str) -> str:
    syn_parent_dir = os.path.dirname(os.path.abspath(syn_data_path))
    if os.path.basename(syn_parent_dir) == "artifacts":
        return os.path.dirname(syn_parent_dir)
    return syn_parent_dir


def _resolve_input_paths(cfg: PrototypeHeadCfg) -> tuple[str, str, str]:
    if cfg.syn_data_path is not None:
        syn_data_path = os.path.abspath(cfg.syn_data_path)
        if not os.path.exists(syn_data_path):
            raise FileNotFoundError(f"syn_data_path does not exist: {syn_data_path}")
        run_dir = _resolve_run_dir_from_syn_path(syn_data_path)
    else:
        model_dir = os.path.join("logged_files", cfg.job_tag, cfg.dataset, cfg.distill_model)
        syn_set_files = sorted(
            list(glob.glob(os.path.join(model_dir, "**", "data.pth"), recursive=True))
        )
        if len(syn_set_files) == 0:
            raise FileNotFoundError(f"No data.pth found under {model_dir}")
        syn_data_path = os.path.abspath(syn_set_files[0])
        run_dir = _resolve_run_dir_from_syn_path(syn_data_path)

    if cfg.barycenter_path is not None:
        barycenter_path = os.path.abspath(cfg.barycenter_path)
    else:
        barycenter_path = os.path.join(run_dir, "artifacts", "barycenter_targets.pth")
        if not os.path.exists(barycenter_path):
            barycenter_path = os.path.join(run_dir, "barycenter_targets.pth")

    if not os.path.exists(barycenter_path):
        raise FileNotFoundError(f"barycenter_targets.pth does not exist: {barycenter_path}")

    return syn_data_path, barycenter_path, run_dir


def _next_batch(data_iter, data_loader):
    batch = next(data_iter, None)
    if batch is None:
        data_iter = iter(data_loader)
        batch = next(data_iter)
    return batch, data_iter


def _build_teacher_prototypes(
    barycenter_payload: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b_star = barycenter_payload["b_star"].float()
    support_weights = barycenter_payload["support_weights"].float()
    if b_star.ndim != 3 or support_weights.ndim != 2:
        raise ValueError("Expected b_star to be [C, I, D] and support_weights to be [C, I]")
    if b_star.shape[:2] != support_weights.shape:
        raise ValueError("b_star and support_weights shape mismatch on [C, I]")
    support_weights = support_weights.clamp_min(1e-12)
    support_weights = support_weights / support_weights.sum(dim=1, keepdim=True)
    class_prototypes = torch.einsum("ci,cid->cd", support_weights, b_star)
    return b_star, support_weights, class_prototypes


def _prototype_logits(
    features: torch.Tensor,
    class_prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    features = nn.functional.normalize(features, dim=-1)
    class_prototypes = nn.functional.normalize(class_prototypes, dim=-1)
    return features @ class_prototypes.t() / max(float(temperature), 1e-6)


def _multi_prototype_logits(
    features: torch.Tensor,
    prototype_bank: torch.Tensor,
    support_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    features = nn.functional.normalize(features, dim=-1)
    prototype_bank = nn.functional.normalize(prototype_bank, dim=-1)
    sim = torch.einsum("bd,cid->bci", features, prototype_bank)
    log_weights = torch.log(support_weights.clamp_min(1e-12))
    scaled = sim / max(float(temperature), 1e-6) + log_weights.unsqueeze(0)
    return torch.logsumexp(scaled, dim=-1)


def _apply_logit_prior(logits: torch.Tensor, log_prior: torch.Tensor | None, prior_tau: float) -> torch.Tensor:
    if log_prior is None or float(prior_tau) == 0.0:
        return logits
    return logits + float(prior_tau) * log_prior.unsqueeze(0)


def _build_projector(student_dim: int, teacher_dim: int, cfg: PrototypeHeadCfg) -> nn.Module:
    if cfg.projector_type == "mlp":
        hidden_dim = max(int(cfg.projector_hidden_dim), 1)
        return nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim),
        )
    return nn.Linear(student_dim, teacher_dim)


def _resolve_log_prior(
    barycenter_payload: dict,
    syn_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    counts = barycenter_payload.get("count_per_class", None)
    if counts is None:
        counts_tensor = torch.bincount(syn_labels.detach().cpu().long(), minlength=int(num_classes)).float()
    else:
        counts_tensor = torch.as_tensor(counts, dtype=torch.float32).flatten()
        if counts_tensor.numel() < int(num_classes):
            pad = torch.zeros(int(num_classes) - counts_tensor.numel(), dtype=torch.float32)
            counts_tensor = torch.cat([counts_tensor, pad], dim=0)
        elif counts_tensor.numel() > int(num_classes):
            counts_tensor = counts_tensor[: int(num_classes)]
    probs = counts_tensor.clamp_min(1e-12)
    probs = probs / probs.sum().clamp_min(1e-12)
    return torch.log(probs)


@torch.no_grad()
def _auto_select_prior_tau(
    syn_loader: DataLoader,
    normalize,
    student_backbone: nn.Module,
    projector: nn.Module,
    prototype_bank: torch.Tensor,
    support_weights: torch.Tensor,
    class_prototypes: torch.Tensor,
    use_multi_prototypes: bool,
    temperature: float,
    log_prior: torch.Tensor | None,
    tau_min: float,
    tau_max: float,
    tau_steps: int,
) -> float:
    if log_prior is None:
        return 0.0
    tau_steps = max(int(tau_steps), 1)
    if tau_steps == 1:
        candidates = [float(tau_min)]
    else:
        candidates = np.linspace(float(tau_min), float(tau_max), num=tau_steps).tolist()
    student_backbone.eval()
    projector.eval()
    best_tau = float(candidates[0])
    best_loss = float("inf")
    log_prior_cuda = log_prior.cuda(non_blocking=True)
    for tau in candidates:
        total_loss = 0.0
        total_n = 0
        for x, y in syn_loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            x = normalize(x)
            z = projector(student_backbone(x).float()).float()
            if use_multi_prototypes and prototype_bank.shape[1] > 1:
                logits = _multi_prototype_logits(z, prototype_bank, support_weights, temperature)
            else:
                logits = _prototype_logits(z, class_prototypes, temperature)
            logits = _apply_logit_prior(logits, log_prior_cuda, tau)
            loss = nn.functional.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_n += int(y.shape[0])
        mean_loss = total_loss / max(total_n, 1)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_tau = float(tau)
    return best_tau


@torch.no_grad()
def _init_projector_with_ridge(
    syn_loader: DataLoader,
    normalize,
    augmentor: nn.Module,
    student_backbone: nn.Module,
    class_prototypes: torch.Tensor,
    projector: nn.Module,
    ridge_lambda: float,
    max_samples: int,
):
    if not isinstance(projector, nn.Linear):
        return
    student_backbone.eval()
    xs = []
    ys = []
    collected = 0
    max_samples = int(max_samples)
    for x, y in syn_loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        x = augmentor(x)
        x = normalize(x)
        z = student_backbone(x).float()
        t = class_prototypes[y].float()
        xs.append(z)
        ys.append(t)
        collected += int(x.shape[0])
        if collected >= max_samples:
            break
    if len(xs) == 0:
        return
    x_mat = torch.cat(xs, dim=0)
    y_mat = torch.cat(ys, dim=0)
    if x_mat.shape[0] > max_samples:
        x_mat = x_mat[:max_samples]
        y_mat = y_mat[:max_samples]
    gram = x_mat.t() @ x_mat
    reg = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * float(ridge_lambda)
    rhs = x_mat.t() @ y_mat
    weight_t = torch.linalg.solve(gram + reg, rhs)
    projector.weight.data.copy_(weight_t.t())
    if projector.bias is not None:
        projector.bias.data.zero_()


@torch.no_grad()
def _auto_select_temperature(
    syn_loader: DataLoader,
    normalize,
    student_backbone: nn.Module,
    projector: nn.Module,
    prototype_bank: torch.Tensor,
    support_weights: torch.Tensor,
    class_prototypes: torch.Tensor,
    use_multi_prototypes: bool,
    t_min: float,
    t_max: float,
    t_steps: int,
) -> float:
    t_steps = max(int(t_steps), 1)
    if t_steps == 1:
        candidates = [float(t_min)]
    else:
        candidates = np.geomspace(float(t_min), float(t_max), num=t_steps).tolist()
    student_backbone.eval()
    projector.eval()
    best_t = float(candidates[0])
    best_loss = float("inf")
    for t in candidates:
        total_loss = 0.0
        total_n = 0
        for x, y in syn_loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            x = normalize(x)
            z = projector(student_backbone(x).float()).float()
            if use_multi_prototypes and prototype_bank.shape[1] > 1:
                logits = _multi_prototype_logits(z, prototype_bank, support_weights, t)
            else:
                logits = _prototype_logits(z, class_prototypes, t)
            loss = nn.functional.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_n += int(y.shape[0])
        mean_loss = total_loss / max(total_n, 1)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_t = float(t)
    return best_t


def train_projector(
    syn_loader: DataLoader,
    normalize,
    augmentor: nn.Module,
    student_backbone: nn.Module,
    projector: nn.Module,
    prototype_bank: torch.Tensor,
    support_weights: torch.Tensor,
    class_prototypes: torch.Tensor,
    use_multi_prototypes: bool,
    steps: int,
    lr: float,
    weight_decay: float,
    align_weight: float,
    ce_weight: float,
    temperature: float,
):
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(steps), 1), eta_min=0.0)
    scaler = GradScaler()
    syn_iter = iter(syn_loader)
    projector.train()

    for _ in tqdm(range(int(steps)), desc="Prototype Head", leave=True):
        batch, syn_iter = _next_batch(syn_iter, syn_loader)
        x, y = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        with torch.no_grad():
            x = augmentor(x)
            x = normalize(x)
            z_student = student_backbone(x).float()

        with autocast(enabled=True):
            z_projected = projector(z_student)
            target = class_prototypes[y]
            loss_align = nn.functional.mse_loss(z_projected, target)
            if use_multi_prototypes and prototype_bank.shape[1] > 1:
                logits = _multi_prototype_logits(
                    z_projected.float(), prototype_bank, support_weights, temperature
                )
            else:
                logits = _prototype_logits(z_projected.float(), class_prototypes, temperature)
            loss_ce = nn.functional.cross_entropy(logits, y)
            loss = float(align_weight) * loss_align + float(ce_weight) * loss_ce

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    projector.eval()


@torch.no_grad()
def evaluate(
    test_loader: DataLoader,
    normalize,
    student_backbone: nn.Module,
    projector: nn.Module,
    prototype_bank: torch.Tensor,
    support_weights: torch.Tensor,
    class_prototypes: torch.Tensor,
    use_multi_prototypes: bool,
    temperature: float,
    log_prior: torch.Tensor | None,
    prior_tau: float,
) -> tuple[float, float]:
    num_classes = test_loader.dataset.num_classes
    top1_metric = MulticlassAccuracy(
        average="micro", num_classes=num_classes, top_k=1
    ).cuda()
    if num_classes >= 5:
        top5_metric = MulticlassAccuracy(
            average="micro", num_classes=num_classes, top_k=5
        ).cuda()

    student_backbone.eval()
    projector.eval()
    log_prior_cuda = None if log_prior is None else log_prior.cuda(non_blocking=True)
    for x, y in tqdm(test_loader, desc="Evaluating Prototype Head", leave=False):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        x = normalize(x)
        z = student_backbone(x).float()
        z = projector(z).float()
        if use_multi_prototypes and prototype_bank.shape[1] > 1:
            logits = _multi_prototype_logits(z, prototype_bank, support_weights, temperature)
        else:
            logits = _prototype_logits(z, class_prototypes, temperature)
        logits = _apply_logit_prior(logits, log_prior_cuda, prior_tau)
        top1_metric.update(logits, y)
        if num_classes >= 5:
            top5_metric.update(logits, y)

    top1 = float(top1_metric.compute().item())
    if num_classes >= 5:
        top5 = float(top5_metric.compute().item())
    else:
        top5 = 0.0
    return top1, top5


def main(cfg: PrototypeHeadCfg):
    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)

    syn_data_path, barycenter_path, run_dir = _resolve_input_paths(cfg)
    save_dir = os.path.join(run_dir, "eval", "prototype_head")
    save_file = os.path.join(save_dir, f"{cfg.eval_model}.pth")
    projector_file = os.path.join(save_dir, f"projector_{cfg.eval_model}.pth")
    if os.path.exists(save_file) and cfg.skip_if_exists:
        print("This prototype head eval already done.")
        print("Exiting...")
        return

    train_dataset, test_dataset = get_dataset(
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

    syn_set = torch.load(syn_data_path, weights_only=False)
    syn_images = syn_set["images"].cuda()
    syn_labels = syn_set["labels"].long().cuda()
    syn_loader = DataLoader(
        TensorDataset(syn_images.detach().clone(), syn_labels.detach().clone()),
        batch_size=min(int(cfg.syn_batch_size), len(syn_images)),
        shuffle=True,
        drop_last=False,
    )

    barycenter_payload = torch.load(barycenter_path, map_location="cpu", weights_only=False)
    prototype_bank, support_weights, class_prototypes = _build_teacher_prototypes(barycenter_payload)
    prototype_bank = prototype_bank.cuda()
    support_weights = support_weights.cuda()
    class_prototypes = class_prototypes.cuda()
    teacher_dim = int(class_prototypes.shape[-1])
    log_prior = _resolve_log_prior(
        barycenter_payload=barycenter_payload,
        syn_labels=syn_labels,
        num_classes=train_dataset.num_classes,
    )
    if not cfg.use_logit_prior:
        log_prior = None

    student_backbone, student_dim = get_model(cfg.eval_model, distributed=False)
    projector = _build_projector(student_dim=student_dim, teacher_dim=teacher_dim, cfg=cfg).cuda()

    augmentor = AugBasic(crop_res=cfg.crop_res).cuda()
    normalize = train_dataset.normalize

    if cfg.projector_init == "ridge":
        _init_projector_with_ridge(
            syn_loader=syn_loader,
            normalize=normalize,
            augmentor=augmentor,
            student_backbone=student_backbone,
            class_prototypes=class_prototypes,
            projector=projector,
            ridge_lambda=cfg.projector_init_ridge,
            max_samples=cfg.projector_init_max_samples,
        )

    print("loaded synthetic data from", syn_data_path)
    print("loaded barycenter targets from", barycenter_path)
    print("distill model is", cfg.distill_model)
    print("eval model is", cfg.eval_model)

    train_projector(
        syn_loader=syn_loader,
        normalize=normalize,
        augmentor=augmentor,
        student_backbone=student_backbone,
        projector=projector,
        prototype_bank=prototype_bank,
        support_weights=support_weights,
        class_prototypes=class_prototypes,
        use_multi_prototypes=cfg.use_multi_prototypes,
        steps=cfg.projector_steps,
        lr=cfg.projector_lr,
        weight_decay=cfg.projector_weight_decay,
        align_weight=cfg.align_weight,
        ce_weight=cfg.ce_weight,
        temperature=cfg.temperature,
    )
    selected_temperature = float(cfg.temperature)
    if cfg.auto_temperature:
        selected_temperature = _auto_select_temperature(
            syn_loader=syn_loader,
            normalize=normalize,
            student_backbone=student_backbone,
            projector=projector,
            prototype_bank=prototype_bank,
            support_weights=support_weights,
            class_prototypes=class_prototypes,
            use_multi_prototypes=cfg.use_multi_prototypes,
            t_min=cfg.temperature_min,
            t_max=cfg.temperature_max,
            t_steps=cfg.temperature_steps,
        )
    selected_prior_tau = float(cfg.prior_tau) if cfg.use_logit_prior else 0.0
    if cfg.use_logit_prior and cfg.auto_prior_tau:
        selected_prior_tau = _auto_select_prior_tau(
            syn_loader=syn_loader,
            normalize=normalize,
            student_backbone=student_backbone,
            projector=projector,
            prototype_bank=prototype_bank,
            support_weights=support_weights,
            class_prototypes=class_prototypes,
            use_multi_prototypes=cfg.use_multi_prototypes,
            temperature=selected_temperature,
            log_prior=log_prior,
            tau_min=cfg.prior_tau_min,
            tau_max=cfg.prior_tau_max,
            tau_steps=cfg.prior_tau_steps,
        )
    print("selected temperature is", selected_temperature)
    print("selected prior tau is", selected_prior_tau)
    top1, top5 = evaluate(
        test_loader=test_loader,
        normalize=normalize,
        student_backbone=student_backbone,
        projector=projector,
        prototype_bank=prototype_bank,
        support_weights=support_weights,
        class_prototypes=class_prototypes,
        use_multi_prototypes=cfg.use_multi_prototypes,
        temperature=selected_temperature,
        log_prior=log_prior,
        prior_tau=selected_prior_tau,
    )

    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "top1": top1,
            "top5": top5,
            "syn_data_path": syn_data_path,
            "barycenter_path": barycenter_path,
            "distill_model": cfg.distill_model,
            "eval_model": cfg.eval_model,
            "projector_steps": int(cfg.projector_steps),
            "projector_lr": float(cfg.projector_lr),
            "projector_weight_decay": float(cfg.projector_weight_decay),
            "align_weight": float(cfg.align_weight),
            "ce_weight": float(cfg.ce_weight),
            "temperature": float(selected_temperature),
            "temperature_train": float(cfg.temperature),
            "auto_temperature": bool(cfg.auto_temperature),
            "use_multi_prototypes": bool(cfg.use_multi_prototypes),
            "num_prototypes_per_class": int(prototype_bank.shape[1]),
            "projector_init": cfg.projector_init,
            "use_logit_prior": bool(cfg.use_logit_prior),
            "prior_tau": float(selected_prior_tau),
            "prior_tau_train": float(cfg.prior_tau),
            "auto_prior_tau": bool(cfg.auto_prior_tau),
            "teacher_dim": teacher_dim,
            "student_dim": int(student_dim),
        },
        save_file,
    )
    torch.save({"state_dict": projector.state_dict()}, projector_file)
    if cfg.save_readable_json:
        summary_file = os.path.join(save_dir, "result_summary.json")
        summary = {
            "top1": float(top1),
            "top5": float(top5),
            "top1_percent": float(top1 * 100.0),
            "top5_percent": float(top5 * 100.0),
            "dataset": cfg.dataset,
            "distill_model": cfg.distill_model,
            "eval_model": cfg.eval_model,
            "projector_type": cfg.projector_type,
            "projector_hidden_dim": int(cfg.projector_hidden_dim),
            "projector_init": cfg.projector_init,
            "use_multi_prototypes": bool(cfg.use_multi_prototypes),
            "num_prototypes_per_class": int(prototype_bank.shape[1]),
            "temperature": float(selected_temperature),
            "use_logit_prior": bool(cfg.use_logit_prior),
            "prior_tau": float(selected_prior_tau),
            "syn_data_path": syn_data_path,
            "barycenter_path": barycenter_path,
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Readable summary saved to {summary_file}")
    print(f"Results saved to {save_file}")
    print("Top 1: {:.2f}".format(top1 * 100))
    print("Top 5: {:.2f}".format(top5 * 100))


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = PrototypeHeadCfg(explicit_bool=True).parse_args()
    main(args)
