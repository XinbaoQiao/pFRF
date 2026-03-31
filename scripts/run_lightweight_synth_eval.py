#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = [
    "mobilenetv3_large",
    "mobileone_s4",
    "repvit_m1_5",
    "efficientformer_l1",
]
DEFAULT_MODES = ["linear_probe", "finetune"]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run linear probing and/or finetuning on synthetic distilled data for lightweight backbones."
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--teacher_model", required=True, type=str)
    parser.add_argument("--syn_data_path", required=True, type=str)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--eval_epochs", type=int, default=100)
    parser.add_argument("--num_eval", type=int, default=1)
    parser.add_argument("--real_batch_size", type=int, default=256)
    parser.add_argument("--syn_batch_size", type=int, default=100)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    eval_script = project_root / "src" / "distillation" / "eval.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    for model in args.models:
        for mode in args.modes:
            cmd = [
                args.python,
                str(eval_script),
                "--dataset",
                args.dataset,
                "--model",
                args.teacher_model,
                "--eval_model",
                model,
                "--syn_data_path",
                args.syn_data_path,
                "--train_mode",
                mode,
                "--eval_epochs",
                str(args.eval_epochs),
                "--num_eval",
                str(args.num_eval),
                "--real_batch_size",
                str(args.real_batch_size),
                "--syn_batch_size",
                str(args.syn_batch_size),
                "--head_lr",
                str(args.head_lr),
                "--backbone_lr",
                str(args.backbone_lr),
                "--weight_decay",
                str(args.weight_decay),
                "--num_workers",
                str(args.num_workers),
                "--data_root",
                args.data_root,
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True, env=env, cwd=str(project_root))


if __name__ == "__main__":
    main()
