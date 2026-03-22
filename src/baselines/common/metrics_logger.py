from __future__ import annotations

import csv
import json
import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: list[dict]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_curve_csv(path: str, xs: list[int], ys: list[float]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "val_top1"])
        for x, y in zip(xs, ys):
            writer.writerow([int(x), float(y)])


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    n = float(len(values))
    mean_v = float(sum(values) / n)
    var = float(sum((v - mean_v) ** 2 for v in values) / n)
    return mean_v, float(var**0.5)

