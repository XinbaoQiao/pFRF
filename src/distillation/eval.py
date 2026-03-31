import copy
import glob
import json
import os
import random
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from augmentation import AugBasic
from config import EvalCfg
from data.dataloaders import get_dataset
from models import get_fc, get_model


class Evaluator:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        checkpoint_path: str | None,
        augmentor: nn.Module,
        epochs: int,
        eval_it: int,
        patience: int,
        checkpoint_it: int,
        normalize: Callable[[Tensor], Tensor],
        num_feats: int,
        num_classes: int,
        num_eval: int,
        train_mode: str = "linear_probe",
        head_lr: float = 1e-3,
        backbone_lr: float = 1e-4,
        weight_decay: float = 0.0,
        random_seed: int = 3407,
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.augmentor = augmentor
        self.epochs = epochs
        self.eval_it = eval_it
        self.patience = patience
        self.checkpoint_it = checkpoint_it
        self.normalize = normalize
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.num_eval = num_eval
        self.train_mode = str(train_mode)
        self.head_lr = float(head_lr)
        self.backbone_lr = float(backbone_lr)
        self.weight_decay = float(weight_decay)
        self.random_seed = random_seed
        self.model_initial_state = copy.deepcopy(self.model.state_dict())

        self.top1_list = []
        self.top5_list = []

        self.reset()

        # this MUST be after self.reset()
        self.load_checkpoint()

    def reset(self):

        torch.manual_seed(self.random_seed + len(self.top1_list))
        random.seed(self.random_seed + len(self.top1_list))
        np.random.seed(self.random_seed + len(self.top1_list))

        self.model.load_state_dict(self.model_initial_state)
        self.fc = get_fc(
            num_feats=self.num_feats,
            num_classes=self.num_classes,
            distributed=torch.cuda.device_count() > 1,
        )
        self._configure_trainable_params()
        self.optimizer = self._build_optimizer()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.epochs, eta_min=0
        )

        self.current_epoch = 0
        self.patience_counter = 0
        self.top1_best = 0
        self.top5_best = 0

        self.scaler = GradScaler()
        self.best_fc = copy.deepcopy(self.fc)

    def _configure_trainable_params(self):
        if self.train_mode == "linear_probe":
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()
            for p in self.fc.parameters():
                p.requires_grad_(True)
            return
        if self.train_mode == "finetune":
            for p in self.model.parameters():
                p.requires_grad_(True)
            self.model.train()
            for p in self.fc.parameters():
                p.requires_grad_(True)
            return
        raise ValueError(f"Unsupported train_mode: {self.train_mode}")

    def _build_optimizer(self):
        world = max(torch.cuda.device_count(), 1)
        scaled_head_lr = self.head_lr * (self.train_loader.batch_size / world) / 256.0
        if self.train_mode == "linear_probe":
            return torch.optim.Adam(
                list(self.fc.parameters()),
                lr=scaled_head_lr,
                weight_decay=self.weight_decay,
            )
        scaled_backbone_lr = self.backbone_lr * (self.train_loader.batch_size / world) / 256.0
        return torch.optim.AdamW(
            [
                {"params": list(self.model.parameters()), "lr": scaled_backbone_lr},
                {"params": list(self.fc.parameters()), "lr": scaled_head_lr},
            ],
            weight_decay=self.weight_decay,
        )

    def train_and_eval(self):

        while len(self.top1_list) < self.num_eval:

            for e in tqdm(
                range(self.current_epoch, self.epochs),
                desc=f"Training {self.train_mode}",
                leave=True,
                initial=self.current_epoch,
                total=self.epochs,
            ):
                self.current_epoch = e

                if self.current_epoch % self.checkpoint_it == 0:
                    self.save_checkpoint()

                self.train_one_epoch()

                if (
                    self.current_epoch % self.eval_it == 0 and self.eval_it != -1
                ) or self.current_epoch == self.epochs - 1:
                    top1, top5 = self.evaluate()
                    print("Top1: {:.2f}".format(top1 * 100))
                    if top1 <= self.top1_best:
                        self.patience_counter += 1
                        print("Losing patience: {}".format(self.patience_counter))
                        if self.patience_counter == self.patience:
                            print("Out of patience! Stopping training.")
                            break

                    else:
                        self.best_fc = copy.deepcopy(self.fc)
                        self.patience_counter = 0
                        self.top1_best = top1
                        self.top5_best = top5

            self.top1_list.append(self.top1_best)
            self.top5_list.append(self.top5_best)
            self.reset()

        print("Finished!")
        print(self.top1_list)

    def train_one_epoch(self):
        if self.train_mode == "finetune":
            self.model.train()
        else:
            self.model.eval()
        self.fc.train()

        for x, y in tqdm(self.train_loader, desc="Epoch Progress", leave=False):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with autocast():
                x = self.augmentor(x)
                x = self.normalize(x)
                if self.train_mode == "linear_probe":
                    with torch.no_grad():
                        z = self.model(x)
                else:
                    z = self.model(x)

                out = self.fc(z)

                loss = nn.functional.cross_entropy(out, y)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            self.scaler.update()

        self.scheduler.step()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.fc.eval()
        num_classes = self.test_loader.dataset.num_classes
        top1_metric = MulticlassAccuracy(
            average="micro", num_classes=num_classes, top_k=1
        ).cuda()
        if self.test_loader.dataset.num_classes >= 5:
            top5_metric = MulticlassAccuracy(
                average="micro", num_classes=num_classes, top_k=5
            ).cuda()

        for x, y in tqdm(self.test_loader, desc="Evaluating Linear Head", leave=False):
            x = x.cuda()
            y = y.cuda()
            x = self.normalize(x)
            z = self.model(x)

            out = self.fc(z)

            top1_metric.update(out, y)

            if self.test_loader.dataset.num_classes >= 5:
                top5_metric.update(out, y)

        top1 = top1_metric.compute().item()
        if self.test_loader.dataset.num_classes >= 5:
            top5 = top5_metric.compute().item()
        else:
            top5 = 0.0

        return top1, top5

    def save_checkpoint(self):
        if self.checkpoint_path is None:
            return
        print("Saving checkpoint...")
        random_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
        save_dict = {
            "model": self.model.state_dict() if self.train_mode == "finetune" else None,
            "fc": self.fc.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "current_epoch": self.current_epoch,
            "top1_best": self.top1_best,
            "top5_best": self.top5_best,
            "patience_counter": self.patience_counter,
            "random_state": random_state,
            "top1_list": self.top1_list,
            "top5_list": self.top5_list,
        }

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(save_dict, self.checkpoint_path+".tmp")
        os.rename(
            self.checkpoint_path+".tmp", self.checkpoint_path
        )
        print("Saved!")

    def load_checkpoint(self):
        if self.checkpoint_path is None:
            return
        if os.path.exists(self.checkpoint_path):
            print("Checkpoint found! Resuming...")
            load_dict = torch.load(self.checkpoint_path, weights_only=False)
            if self.train_mode == "finetune" and load_dict.get("model") is not None:
                self.model.load_state_dict(load_dict["model"])
            self.fc.load_state_dict(load_dict["fc"])
            self.optimizer.load_state_dict(load_dict["optimizer"])
            self.scheduler.load_state_dict(load_dict["scheduler"])
            self.current_epoch = load_dict["current_epoch"]

            self.top1_best = load_dict["top1_best"]
            self.top5_best = load_dict["top5_best"]
            self.patience_counter = load_dict["patience_counter"]

            torch.set_rng_state(load_dict["random_state"]["torch"])
            torch.cuda.set_rng_state_all(load_dict["random_state"]["cuda"])
            random.setstate(load_dict["random_state"]["python"])
            np.random.set_state(load_dict["random_state"]["numpy"])

            self.top1_list = load_dict["top1_list"]
            self.top5_list = load_dict["top5_list"]

        else:
            print("No checkpoint found, starting from scratch...")


@torch.no_grad()
def _run_direct_prototype_inference(
    model: nn.Module,
    normalize: Callable[[Tensor], Tensor],
    prototype_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    temperature: float,
) -> tuple[float, float, int]:
    model.eval()
    proto_sum = None
    proto_count = torch.zeros(num_classes, device="cuda", dtype=torch.float32)
    total_proto_samples = 0

    for x, y in tqdm(prototype_loader, desc="Building Prototypes", leave=False):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()
        z = model(normalize(x)).float()
        if proto_sum is None:
            proto_sum = torch.zeros(num_classes, z.shape[1], device=z.device, dtype=torch.float32)
        for c in range(num_classes):
            mask = y == c
            if mask.any():
                proto_sum[c] += z[mask].sum(dim=0)
                proto_count[c] += float(mask.sum())
        total_proto_samples += int(y.shape[0])

    if proto_sum is None:
        raise ValueError("Prototype source is empty.")

    fallback = proto_sum.sum(dim=0) / proto_count.sum().clamp_min(1.0)
    prototypes = proto_sum / proto_count.clamp_min(1.0).unsqueeze(1)
    prototypes[proto_count == 0] = fallback
    prototypes = nn.functional.normalize(prototypes, dim=-1)

    top1_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=1).cuda()
    if num_classes >= 5:
        top5_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=5).cuda()

    for x, y in tqdm(test_loader, desc="Evaluating Prototypes", leave=False):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()
        z = nn.functional.normalize(model(normalize(x)).float(), dim=-1)
        logits = z @ prototypes.t() / max(float(temperature), 1e-6)
        top1_metric.update(logits, y)
        if num_classes >= 5:
            top5_metric.update(logits, y)

    top1 = top1_metric.compute().item()
    top5 = top5_metric.compute().item() if num_classes >= 5 else 0.0
    return top1, top5, total_proto_samples


@torch.no_grad()
def _run_fixed_prototype_inference(
    model: nn.Module,
    normalize: Callable[[Tensor], Tensor],
    test_loader: DataLoader,
    prototypes: torch.Tensor,
    temperature: float,
) -> tuple[float, float]:
    model.eval()
    num_classes = int(prototypes.shape[0])
    prototypes = nn.functional.normalize(prototypes.float().cuda(), dim=-1)
    top1_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=1).cuda()
    if num_classes >= 5:
        top5_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=5).cuda()

    for x, y in tqdm(test_loader, desc="Evaluating FL Prototypes", leave=False):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()
        z = nn.functional.normalize(model(normalize(x)).float(), dim=-1)
        if z.shape[1] != prototypes.shape[1]:
            raise ValueError(
                f"Feature dim mismatch: eval_model gives {int(z.shape[1])}, "
                f"prototype dim is {int(prototypes.shape[1])}. "
                "For direct prototype inference, model and FL prototype backbone must match."
            )
        logits = z @ prototypes.t() / max(float(temperature), 1e-6)
        top1_metric.update(logits, y)
        if num_classes >= 5:
            top5_metric.update(logits, y)

    top1 = top1_metric.compute().item()
    top5 = top5_metric.compute().item() if num_classes >= 5 else 0.0
    return top1, top5


def _prototype_temperature_candidates(cfg: EvalCfg) -> list[float]:
    steps = max(int(cfg.prototype_temperature_steps), 1)
    if steps == 1:
        return [float(cfg.prototype_temperature)]
    t_min = float(cfg.prototype_temperature_min)
    t_max = float(cfg.prototype_temperature_max)
    if t_max < t_min:
        t_min, t_max = t_max, t_min
    return [float(v) for v in np.linspace(t_min, t_max, num=steps).tolist()]


if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy("file_system")

    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)

    cfg = EvalCfg(explicit_bool=True).parse_args()

    is_prototype_mode = cfg.train_mode == "cosineprototype"
    fl_proto_mode = is_prototype_mode and cfg.prototype_source in {"fl_mean", "fl_barycenter"}
    needs_syn_data = not is_prototype_mode
    syn_data_path: str | None = None
    barycenter_path: str | None = None
    run_dir: str | None = None
    if cfg.run_dir_override is not None:
        run_dir = os.path.abspath(cfg.run_dir_override)
    if cfg.barycenter_path is not None:
        barycenter_path = os.path.abspath(cfg.barycenter_path)
        if not os.path.exists(barycenter_path):
            raise FileNotFoundError(f"barycenter_path does not exist: {barycenter_path}")
        if run_dir is None:
            bary_parent_dir = os.path.dirname(barycenter_path)
            if os.path.basename(bary_parent_dir) == "artifacts":
                run_dir = os.path.dirname(bary_parent_dir)
            else:
                run_dir = bary_parent_dir
    if cfg.syn_data_path is not None:
        syn_data_path = os.path.abspath(cfg.syn_data_path)
        if not os.path.exists(syn_data_path):
            raise FileNotFoundError(f"syn_data_path does not exist: {syn_data_path}")
        if run_dir is None:
            syn_parent_dir = os.path.dirname(syn_data_path)
            if os.path.basename(syn_parent_dir) == "artifacts":
                run_dir = os.path.dirname(syn_parent_dir)
            else:
                run_dir = syn_parent_dir
    elif needs_syn_data and run_dir is None:
        model_dir = os.path.join("logged_files", cfg.job_tag, cfg.dataset, cfg.model)
        print("Searching for saved data in {}".format(model_dir))
        syn_set_files = sorted(
            list(glob.glob(os.path.join(model_dir, "**", "data.pth"), recursive=True))
        )
        if len(syn_set_files) == 0:
            print(f"No data found at {model_dir}.")
            print("Exiting...")
            exit()
        if len(list(syn_set_files)) > 1:
            print("Warning: multiple syn sets found. Using the first one.")
        syn_data_path = syn_set_files[0]
        run_dir = os.path.dirname(syn_data_path)
    elif run_dir is None:
        if cfg.barycenter_path is None:
            model_dir = os.path.join("logged_files", cfg.job_tag, cfg.dataset, cfg.model)
            run_candidates = sorted(
                list(
                    glob.glob(
                        os.path.join(model_dir, "**", "artifacts", "barycenter_targets.pth"),
                        recursive=True,
                    )
                )
            )
            if len(run_candidates) > 0:
                barycenter_path = os.path.abspath(run_candidates[0])
                run_dir = os.path.dirname(os.path.dirname(barycenter_path))
            else:
                run_dir = os.path.join("logged_files", cfg.job_tag, cfg.dataset, cfg.model)
        else:
            run_dir = os.path.dirname(os.path.dirname(barycenter_path)) if os.path.basename(os.path.dirname(barycenter_path)) == "artifacts" else os.path.dirname(barycenter_path)
    if run_dir is None:
        raise RuntimeError("run_dir could not be resolved")

    if fl_proto_mode and barycenter_path is None:
        candidate = os.path.join(run_dir, "artifacts", "barycenter_targets.pth")
        if os.path.exists(candidate):
            barycenter_path = candidate
        else:
            fallback = os.path.join(run_dir, "barycenter_targets.pth")
            if os.path.exists(fallback):
                barycenter_path = fallback
    if fl_proto_mode and barycenter_path is None:
        raise FileNotFoundError("prototype_source=fl_mean/fl_barycenter requires barycenter_targets.pth")

    train_mode_dir = "cosineprototype" if is_prototype_mode else cfg.train_mode
    save_dir = os.path.join(run_dir, "eval", train_mode_dir)
    save_file = os.path.join(save_dir, "{}.pth".format(cfg.eval_model))
    checkpoint_path = os.path.join(save_dir, "checkpoint_{}.pth".format(cfg.eval_model))
    if os.path.exists(save_file) and cfg.skip_if_exists:
        print("This eval already done.")
        print("Exiting...")
        exit()

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
    )

    if syn_data_path is not None:
        syn_set = torch.load(syn_data_path, weights_only=False)
        syn_images = syn_set["images"].cuda()
        syn_labels = syn_set["labels"].long().cuda()
    else:
        syn_images = None
        syn_labels = None

    print("loaded file from ", run_dir)
    print("eval model is ", cfg.eval_model)
    print("train mode is ", cfg.train_mode)
    eval_model, num_feats = get_model(
        cfg.eval_model, distributed=torch.cuda.device_count() > 1
    )

    os.makedirs(save_dir, exist_ok=True)

    if is_prototype_mode:
        selected_temperature = float(cfg.prototype_temperature)
        if cfg.prototype_source in {"fl_mean", "fl_barycenter"}:
            payload = torch.load(barycenter_path, map_location="cpu", weights_only=False)
            b_star = payload["b_star"].float()
            support_weights = payload["support_weights"].float()
            if b_star.ndim != 3 or support_weights.ndim != 2:
                raise ValueError("Expected b_star [C, I, D] and support_weights [C, I]")
            if cfg.prototype_source == "fl_mean" and int(b_star.shape[1]) == 1:
                prototypes = b_star[:, 0, :]
            else:
                sw = support_weights.clamp_min(1e-12)
                sw = sw / sw.sum(dim=1, keepdim=True)
                prototypes = torch.einsum("ci,cid->cd", sw, b_star)
            if bool(cfg.auto_prototype_temperature):
                best_t = float(cfg.prototype_temperature)
                best_top1 = -1.0
                best_top5 = -1.0
                for t in _prototype_temperature_candidates(cfg):
                    cand_top1, cand_top5 = _run_fixed_prototype_inference(
                        model=eval_model,
                        normalize=train_dataset.normalize,
                        test_loader=test_loader,
                        prototypes=prototypes,
                        temperature=t,
                    )
                    if cand_top1 > best_top1 or (cand_top1 == best_top1 and cand_top5 > best_top5):
                        best_top1 = float(cand_top1)
                        best_top5 = float(cand_top5)
                        best_t = float(t)
                top1, top5 = best_top1, best_top5
                selected_temperature = best_t
            else:
                top1, top5 = _run_fixed_prototype_inference(
                    model=eval_model,
                    normalize=train_dataset.normalize,
                    test_loader=test_loader,
                    prototypes=prototypes,
                    temperature=selected_temperature,
                )
            save_dict = {
                "train_mode": cfg.train_mode,
                "prototype_mode": "cosineprototype",
                "prototype_source": cfg.prototype_source,
                "prototype_temperature": float(selected_temperature),
                "prototype_temperature_train": float(cfg.prototype_temperature),
                "auto_prototype_temperature": bool(cfg.auto_prototype_temperature),
                "eval_model": cfg.eval_model,
                "top1": float(top1),
                "top5": float(top5),
                "barycenter_path": barycenter_path,
                "syn_data_path": syn_data_path,
                "num_synthetic_samples": 0 if syn_images is None else int(syn_images.shape[0]),
                "num_prototype_source_samples": int(prototypes.shape[0]),
                "num_built_samples": int(prototypes.shape[0]),
            }
            print(f"Results saved to {save_file}")
            torch.save(obj=save_dict, f=save_file)
            summary_file = os.path.join(save_dir, "result_summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train_mode": cfg.train_mode,
                        "prototype_mode": "cosineprototype",
                        "prototype_source": cfg.prototype_source,
                        "prototype_temperature": float(selected_temperature),
                        "prototype_temperature_train": float(cfg.prototype_temperature),
                        "auto_prototype_temperature": bool(cfg.auto_prototype_temperature),
                        "eval_model": cfg.eval_model,
                        "top1": float(top1),
                        "top5": float(top5),
                        "top1_percent": float(top1 * 100.0),
                        "top5_percent": float(top5 * 100.0),
                        "barycenter_path": barycenter_path,
                        "syn_data_path": syn_data_path,
                        "num_synthetic_samples": 0 if syn_images is None else int(syn_images.shape[0]),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Readable summary saved to {summary_file}")
            print("Top 1: {:.2f}".format(top1 * 100))
        elif cfg.prototype_source == "full_train":
            prototype_loader = DataLoader(
                train_dataset,
                shuffle=False,
                num_workers=cfg.num_workers,
                batch_size=cfg.real_batch_size,
            )
            prototype_source_samples = len(train_dataset)
            if bool(cfg.auto_prototype_temperature):
                best_t = float(cfg.prototype_temperature)
                best_top1 = -1.0
                best_top5 = -1.0
                best_built = 0
                for t in _prototype_temperature_candidates(cfg):
                    cand_top1, cand_top5, cand_built = _run_direct_prototype_inference(
                        model=eval_model,
                        normalize=train_dataset.normalize,
                        prototype_loader=prototype_loader,
                        test_loader=test_loader,
                        num_classes=train_dataset.num_classes,
                        temperature=t,
                    )
                    if cand_top1 > best_top1 or (cand_top1 == best_top1 and cand_top5 > best_top5):
                        best_top1 = float(cand_top1)
                        best_top5 = float(cand_top5)
                        best_built = int(cand_built)
                        best_t = float(t)
                top1, top5, built_samples = best_top1, best_top5, best_built
                selected_temperature = best_t
            else:
                top1, top5, built_samples = _run_direct_prototype_inference(
                    model=eval_model,
                    normalize=train_dataset.normalize,
                    prototype_loader=prototype_loader,
                    test_loader=test_loader,
                    num_classes=train_dataset.num_classes,
                    temperature=selected_temperature,
                )
            save_dict = {
                "train_mode": cfg.train_mode,
                "prototype_mode": "cosineprototype",
                "prototype_source": cfg.prototype_source,
                "prototype_temperature": float(selected_temperature),
                "prototype_temperature_train": float(cfg.prototype_temperature),
                "auto_prototype_temperature": bool(cfg.auto_prototype_temperature),
                "eval_model": cfg.eval_model,
                "top1": float(top1),
                "top5": float(top5),
                "syn_data_path": syn_data_path,
                "num_synthetic_samples": 0 if syn_images is None else int(syn_images.shape[0]),
                "num_prototype_source_samples": int(prototype_source_samples),
                "num_built_samples": int(built_samples),
            }
            print(f"Results saved to {save_file}")
            torch.save(obj=save_dict, f=save_file)
            summary_file = os.path.join(save_dir, "result_summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train_mode": cfg.train_mode,
                        "prototype_mode": "cosineprototype",
                        "prototype_source": cfg.prototype_source,
                        "prototype_temperature": float(selected_temperature),
                        "prototype_temperature_train": float(cfg.prototype_temperature),
                        "auto_prototype_temperature": bool(cfg.auto_prototype_temperature),
                        "eval_model": cfg.eval_model,
                        "top1": float(top1),
                        "top5": float(top5),
                        "top1_percent": float(top1 * 100.0),
                        "top5_percent": float(top5 * 100.0),
                        "syn_data_path": syn_data_path,
                        "num_synthetic_samples": 0 if syn_images is None else int(syn_images.shape[0]),
                        "num_prototype_source_samples": int(prototype_source_samples),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Readable summary saved to {summary_file}")
            print("Top 1: {:.2f}".format(top1 * 100))
        else:
            raise ValueError("prototype_source must be one of: fl_mean, fl_barycenter, full_train")
    else:
        if syn_images is None or syn_labels is None:
            raise ValueError("linear_probe/finetune require syn_data_path.")
        ds = TensorDataset(syn_images.detach().clone(), syn_labels.detach().clone())
        loader = DataLoader(
            ds,
            batch_size=min(int(cfg.syn_batch_size), len(syn_images)),
            shuffle=True,
        )
        augmentor = AugBasic(crop_res=cfg.crop_res).cuda()
        augmentor = augmentor.cuda()

        evaluator = Evaluator(
            train_loader=loader,
            test_loader=test_loader,
            model=eval_model,
            checkpoint_path=checkpoint_path,
            augmentor=augmentor,
            epochs=cfg.eval_epochs,
            eval_it=cfg.eval_it,
            patience=cfg.patience,
            checkpoint_it=cfg.checkpoint_it,
            normalize=train_dataset.normalize,
            num_feats=num_feats,
            num_classes=train_dataset.num_classes,
            num_eval=cfg.num_eval,
            train_mode=cfg.train_mode,
            head_lr=cfg.head_lr,
            backbone_lr=cfg.backbone_lr,
            weight_decay=cfg.weight_decay,
        )

        evaluator.train_and_eval()

        top1_mean = float(np.mean(evaluator.top1_list))
        top1_std = float(np.std(evaluator.top1_list))
        top5_mean = float(np.mean(evaluator.top5_list))
        top5_std = float(np.std(evaluator.top5_list))

        save_dict = {
            "top1_mean": top1_mean,
            "top1_std": top1_std,
            "top5_mean": top5_mean,
            "top5_std": top5_std,
            "train_mode": cfg.train_mode,
            "eval_model": cfg.eval_model,
            "syn_data_path": syn_data_path,
            "num_synthetic_samples": int(syn_images.shape[0]),
        }

        print(f"Results saved to {save_file}")
        torch.save(obj=save_dict, f=save_file)
        summary_file = os.path.join(save_dir, "result_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_mode": cfg.train_mode,
                    "eval_model": cfg.eval_model,
                    "top1_mean": top1_mean,
                    "top1_std": top1_std,
                    "top5_mean": top5_mean,
                    "top5_std": top5_std,
                    "top1_mean_percent": top1_mean * 100.0,
                    "top1_std_percent": top1_std * 100.0,
                    "top5_mean_percent": top5_mean * 100.0,
                    "top5_std_percent": top5_std * 100.0,
                    "syn_data_path": syn_data_path,
                    "num_synthetic_samples": int(syn_images.shape[0]),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Readable summary saved to {summary_file}")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        print("Top 1 Mean ± Std: {:.2f} ± {:.2f}".format(top1_mean * 100, top1_std * 100))
