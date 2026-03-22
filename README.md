<div align="center">

# pFRF: Personalized Federated Representation Fusion

### Semantic-Interface FRF with Post-Distillation Personalization

Research-oriented codebase for building compact synthetic datasets from federated, non-IID image data using pretrained visual backbones, feature statistics, offline distillation, and a post-distillation personalization stage.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square)
![Scope](https://img.shields.io/badge/Setting-Federated%20Dataset%20Distillation-4c8bf5?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20Code-6c757d?style=flat-square)

**Wasserstein barycenter-inspired feature aggregation + pretrained visual representations + compact synthetic datasets + client-specific semantic interfaces**

`Non-IID clients` -> `feature-space proxies` -> `compact synthetic dataset` -> `Personal Semantic Head + Semantic Translator`

[`Code`](#quick-start) • [`Method`](#method-overview) • [`Results`](#results-summary) • [`Qualitative Results`](#qualitative-results) • [`Citation`](#citation)

</div>

---

## Teaser

> Wasserstein Barycenter as Federated Knowledge Proxy studies federated dataset distillation with pretrained visual backbones, feature-space aggregation, and offline synthetic image optimization.

<div align="center">

![Flowers102 gallery](output/feddd_20260318_063115/flowers102/flowers102_ipc1_dpFalse_noniid_dirichlet0.05_clients10/dinov2_vitb/vis/all.png)

</div>

<div align="center">

Synthetic gallery from the inspected `Flowers102 / dinov2_vitb` snapshot.

</div>

### Project Snapshot

| What it is | What is verified in this repository |
|---|---|
| Setting | `main_fed.py` exposes client count, partition mode, Dirichlet non-IID control, feature cache support, and DP-related flags |
| Backbone families | CLIP, DINO/DINOv2/DINOv3, EVA02, and MoCo-v3 variants are present in the model registry |
| Available artifacts | Multi-GPU launchers, saved metrics, run configs, linear probe checkpoints, and synthetic image grids are present in the repository snapshot |

### Benchmark Card

| Dataset | Best Backbone | Top-1 |
|---|---|---:|
| Flowers102 | `dinov2_vitb` | 99.63 |
| CIFAR-10 | `dinov2_vitb` | 94.26 |
| Imagenette | `clip_vitb` | 100.00 |
| Waterbirds | `dinov2_vitb` | 88.61 |

### Method Summary

```text
Pretrained backbone features
        +
federated client statistics / barycenter-inspired proxies
        +
offline synthetic image distillation
        ->
compact synthetic dataset for downstream evaluation
```

## Contents

- [Teaser](#teaser)
- [Overview](#overview)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Training and Running](#training-and-running)
- [Results Summary](#results-summary)
- [Visualization Examples](#visualization-examples)
- [Citation](#citation)

## Overview

`pFRF` extends the original FRF pipeline toward personalization. Instead of changing the server-side distillation loss, the current implementation keeps the global synthetic set construction intact and introduces a post-distillation client adaptation stage.

The current terminology is:

- `Personal Semantic Head`: a client-specific classifier trained on that client's cached backbone features
- `Semantic Translator`: a lightweight client-specific mapping trained on the distilled synthetic set so the synthetic representation can be interpreted by the client's personal semantic head

This keeps the method positioned as `FRF + local semantic decision adaptation`, rather than as a new client-aware distillation objective.

The repository contains:

- a main training/evaluation entrypoint in `main_fed.py`
- dataset loaders for multiple standard and robustness-oriented vision benchmarks
- a model registry covering CLIP, DINOv2, EVA02, and MoCo-v3 style backbones
- multi-GPU batch scripts for large experiment sweeps
- saved experiment outputs with synthetic visualizations and evaluation metrics

This README is grounded in the current repository state and the inspected experiment snapshot under `output/feddd_20260318_063115`.

## Method Overview

The implementation and accompanying methodology notes indicate the following high-level pipeline.

```text
Client datasets
  -> pretrained backbone feature extraction
  -> federated client-wise feature/statistic aggregation
  -> server-side distillation of synthetic images
  -> global linear evaluation on held-out test data
  -> post-distillation personalization via Personal Semantic Head + Semantic Translator
```

### What is verified in the repository

- `main_fed.py` defines a federated experiment configuration with defaults such as `num_clients = 10`, `partition = "dirichlet"`, `dirichlet_alpha = 0.05`, and `ipc = 1`
- feature caching is explicitly supported through `src/data/feature_cache.py` and the `--only_build_feature_cache` flag
- distillation is implemented over synthetic images with configurable modes such as `pixel` and `pyramid`
- evaluation writes `metrics.json`, `run_config.json`, linear probe checkpoints, and when enabled per-client personalization artifacts into each experiment directory
- dedicated batch scripts are provided for standard runs and DP-enabled runs

### Method intuition

Based on `methodology.tex` and the current code organization:

1. Replace direct gradient matching with feature-space proxy matching inspired by Wasserstein barycenters.
2. Aggregate distributed client information in a federated manner without centralizing raw training data.
3. Distill synthetic images offline from those aggregated targets using pretrained visual backbones.
4. After distillation, adapt the global synthetic set to each client through a `Personal Semantic Head` and a lightweight `Semantic Translator`.

### Personalization Stage

The current personalized extension is intentionally post-distillation only.

```text
Cached client features
  -> train one Personal Semantic Head per client
Distilled synthetic images
  -> train one Semantic Translator per client
  -> evaluate each client with its own Personal Semantic Head
```

This design is mechanism-level similar to projector-plus-inherited-classifier style methods, but it is rewritten for the federated setting:

- the inherited decision module is client-specific rather than global
- the translator is used for federated personalization rather than only model-gap compensation
- the server-side synthetic data objective remains unchanged in the current version

### Optional privacy mode

The repository includes a dedicated DP batch launcher:

- `run_frf_all_datasets_ipc1_dp_multigpu.sh`

It enables:

- `--dp_enable`
- `--dp_epsilon`
- `--dp_delta` (`auto` by default, resolved to `1 / N` where `N` is the total federated training-set size)

The default non-DP launcher is:

- `run_frf_all_datasets_ipc1_multigpu.sh`

## Repository Structure

```text
pfrf_project/
├── main_fed.py                         # Main federated distillation entrypoint
├── methodology.tex                    # Method write-up and theoretical notes
├── requirements.txt                   # Python dependencies
├── run_frf_all_datasets_ipc1_multigpu.sh
├── run_frf_all_datasets_ipc1_dp_multigpu.sh
├── run_frf_all_datasets_ipc5_multigpu.sh
├── run_frf_all_datasets_ipc10_multigpu.sh
├── run_feature_cache_warmup_multigpu.sh
├── run_baselines_imagenet_multigpu.sh
├── resource_manifest/                 # Download manifest and helper scripts
├── src/
│   ├── data/                          # Dataset loaders and feature cache logic
│   ├── federated/                     # Client/server logic
│   ├── distillation/                  # Distillation modules
│   ├── augmentation/                  # Augmentation operators
│   ├── baselines/                     # Baseline runners
│   ├── personalized/                  # Personal Semantic Head and Semantic Translator
│   └── models/                        # Backbone/model registry
├── datasets/                          # Shared dataset root (defaults to FRF project datasets)
├── pretrained_models/                 # Shared weights/cache root
├── output/                            # Experiment outputs
└── tools/                             # Utility scripts
```

## Environment Setup

### Requirements

Verified from `requirements.txt`:

- `torch`
- `torchvision`
- `timm`
- `transformers`
- `torchmetrics`
- `POT`
- `wilds`
- `spawrious`
- `clip`
- `kornia`
- `typed-argument-parser`
- plus standard scientific Python packages such as `numpy`, `scipy`, `pandas`, and `matplotlib`

### Recommended setup

```bash
cd /path/to/pfrf_project

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Cache and checkpoint paths

`main_fed.py` in `pfrf_project` defaults to reusing the original FRF asset roots:

```bash
PFRF_SHARED_ROOT=../frf_project
TORCH_HOME=$PFRF_SHARED_ROOT/pretrained_models
HF_HOME=$PFRF_SHARED_ROOT/pretrained_models
```

The output feature cache is also intended to be shared with the original FRF runs.

## Data Preparation

### One-click resource helper

The repository includes a download helper:

```bash
bash resource_manifest/download_resources.sh
```

The companion documentation is in:

- `resource_manifest/README.md`

### Dataset root convention

Verified from the repository:

- datasets are expected under `datasets/`
- pretrained model assets are expected under `pretrained_models/`

### Datasets visible in the current batch scripts

The inspected multi-GPU experiment scripts enumerate the following datasets:

- `flowers102`
- `food101`
- `imagenette`
- `spawrious`
- `stanforddogs`
- `waterbirds`
- `artbench`
- `cifar10`
- `cifar100`
- `cub2011`
- `imagenet-1k`
- `imagenet-100`

### Important preparation notes

Verified from `resource_manifest/README.md`:

- `ImageNet-1k` / `ImageNet-100` are not auto-downloaded and require manual preparation
- `Stanford Dogs` and `CUB-200-2011` may require manual download/unpack
- `waterbirds` depends on `wilds`
- `spawrious` depends on the external `spawrious` package

## Quick Start

### Minimal release workflow

```text
install dependencies
  -> prepare datasets / model cache
  -> launch one experiment
  -> inspect metrics and synthetic grids
```

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare data and local model cache

```bash
bash resource_manifest/download_resources.sh
```

Then make sure the dataset you want to use is available under `datasets/`. Some datasets, including ImageNet variants, still require manual preparation.

### 3. Run one example experiment

```bash
python main_fed.py \
  --experiment_name quickstart_clip_cifar10 \
  --dataset cifar10 \
  --model clip_vitb \
  --data_root datasets \
  --output_root output/quickstart
```

### 4. Inspect outputs

Look under:

```text
output/quickstart/cifar10/...
```

Typical generated files include:

- `metrics.json`
- `run_config.json`
- `distill_progress.json`
- `linear_probe.pth`
- `vis/all.png`

## Training and Running

### 1. Run a single experiment

```bash
python main_fed.py \
  --experiment_name clip_vitb \
  --dataset cifar10 \
  --model clip_vitb \
  --data_root datasets \
  --output_root output/frf_manual
```

### 2. Run the provided multi-GPU IPC=1 sweep

```bash
bash run_frf_all_datasets_ipc1_multigpu.sh
```

This launcher is verified to sweep multiple datasets and assign models to fixed GPU IDs.

### 3. Run the DP-enabled IPC=1 sweep

```bash
bash run_frf_all_datasets_ipc1_dp_multigpu.sh
```

The DP launcher passes:

```bash
--dp_enable True
--dp_epsilon 1.0
--dp_delta auto
```

unless overridden by environment variables in the script.

### 4. Warm up feature caches

The repository also includes:

```bash
bash run_feature_cache_warmup_multigpu.sh
```

This is consistent with the feature cache support exposed by `main_fed.py` and `src/data/feature_cache.py`.

### 5. Common runtime outputs

Each successful experiment directory in the inspected result snapshot may contain:

- `metrics.json`
- `run_config.json`
- `distill_progress.json`
- `linear_probe.pth`
- `vis/` visualization images

## Results Summary

Best snapshot results: `Flowers102 99.63`, `CIFAR-10 94.26`, `Imagenette 100.00`, `Waterbirds 88.61`.

The table below is built only from `metrics.json` files under:

- `output/feddd_20260318_063115`

For each dataset, all available backbone subdirectories were scanned and the best available result was selected by `top1` accuracy.

> `top5` is shown as `N/A` for `spawrious` and `waterbirds` because the current result files record `top5 = 0.0`; this README treats that as unavailable or not meaningful for the present snapshot rather than as a poor score.

### Best Available Result Per Dataset

| Dataset | Best Backbone in Snapshot | Top-1 (%) | Top-5 (%) | Notes |
|---|---:|---:|---:|---|
| ArtBench | `clip_vitb` | 62.56 | 96.33 | best among 3 available backbones |
| CIFAR-10 | `dinov2_vitb` | 94.26 | 99.39 | best among 3 available backbones |
| CIFAR-100 | `mocov3_resnet50` | 57.64 | 84.45 | only available finalized backbone in snapshot |
| CUB-2011 | `mocov3_resnet50` | 51.05 | 78.86 | only available finalized backbone in snapshot |
| Flowers102 | `dinov2_vitb` | 99.63 | 99.84 | best among 4 available backbones |
| Food101 | `clip_vitb` | 85.61 | 97.20 | best among 4 available backbones |
| ImageNet-100 | `mocov3_resnet50` | 78.02 | 94.58 | only available finalized backbone in snapshot |
| Imagenette | `clip_vitb` | 100.00 | 100.00 | best among 4 available backbones |
| Spawrious | `dinov2_vitb` | 76.77 | N/A | best among 4 available backbones |
| Stanford Dogs | `dinov2_vitb` | 82.70 | 97.38 | best among 3 available backbones |
| Waterbirds | `dinov2_vitb` | 88.61 | N/A | best among 3 available backbones |

### Reading the table

- The table is intentionally snapshot-scoped, not a claim of final benchmark leadership
- Some datasets have only one finalized backbone result in the current output tree
- `imagenet-1k` is omitted because the inspected snapshot does not include a finalized metric file
- `spawrious` and `waterbirds` use `N/A` for top-5 in this README because current metric files record `0.0`

### Incomplete datasets in the inspected snapshot

- `imagenet-1k` appears in the batch scripts and output tree, but no finalized `metrics.json` was found in the inspected snapshot, so it is treated as pending here.

<details>
<summary><strong>Snapshot provenance</strong></summary>

All values above were selected from experiment folders under `output/feddd_20260318_063115`. The selected best-result files are:

- `artbench/artbench_ipc1_dpFalse_noniid_dirichlet0.05_clients10/clip_vitb/metrics.json`
- `cifar10/cifar10_ipc1_dpFalse_noniid_dirichlet0.05_clients10/dinov2_vitb/metrics.json`
- `cifar100/cifar100_ipc1_dpFalse_noniid_dirichlet0.05_clients10/mocov3_resnet50/metrics.json`
- `cub2011/cub2011_ipc1_dpFalse_noniid_dirichlet0.05_clients10/mocov3_resnet50/metrics.json`
- `flowers102/flowers102_ipc1_dpFalse_noniid_dirichlet0.05_clients10/dinov2_vitb/metrics.json`
- `food101/food101_ipc1_dpFalse_noniid_dirichlet0.05_clients10/clip_vitb/metrics.json`
- `imagenet-100/imagenet-100_ipc1_dpFalse_noniid_dirichlet0.05_clients10/mocov3_resnet50/metrics.json`
- `imagenette/imagenette_ipc1_dpFalse_noniid_dirichlet0.05_clients10/clip_vitb/metrics.json`
- `spawrious/spawrious_ipc1_dpFalse_noniid_dirichlet0.05_clients10/dinov2_vitb/metrics.json`
- `stanforddogs/stanforddogs_ipc1_dpFalse_noniid_dirichlet0.05_clients10/dinov2_vitb/metrics.json`
- `waterbirds/waterbirds_ipc1_dpFalse_noniid_dirichlet0.05_clients10/dinov2_vitb/metrics.json`

</details>

## Qualitative Results

The following images are embedded directly from the inspected output snapshot.

### CIFAR-10 synthetic prototypes

Best current result in snapshot: `dinov2_vitb`

![CIFAR-10 synthetic prototypes](output/feddd_20260318_063115/cifar10/cifar10_ipc1_dpFalse_noniid_dirichlet0.05_clients10/dinov2_vitb/vis/all.png)

### ArtBench qualitative example

Best current result in snapshot: `clip_vitb`

![ArtBench qualitative example](output/feddd_20260318_063115/artbench/artbench_ipc1_dpFalse_noniid_dirichlet0.05_clients10/clip_vitb/vis/all.png)

### Flowers102 gallery

Best current result in snapshot: `dinov2_vitb`

![Flowers102 gallery](output/feddd_20260318_063115/flowers102/flowers102_ipc1_dpFalse_noniid_dirichlet0.05_clients10/dinov2_vitb/vis/all.png)

### Waterbirds qualitative example

Best current result in snapshot: `dinov2_vitb`

![Waterbirds qualitative example](output/feddd_20260318_063115/waterbirds/waterbirds_ipc1_dpFalse_noniid_dirichlet0.05_clients10/dinov2_vitb/vis/all.png)

## Citation

### Placeholder

The repository contains methodology notes in `methodology.tex`, but a final canonical citation block is not yet confirmed in the current snapshot.

When the paper metadata is finalized, replace this section with the official BibTeX entry, for example:

```bibtex
@article{frf_placeholder_2026,
  title   = {FRF: Federated Representation Fusion with Feature-Space Aggregation},
  author  = {Placeholder},
  journal = {Placeholder},
  year    = {2026}
}
```

---

If you use or adapt this repository before the citation metadata is finalized, please cite the eventual paper release and reference this codebase revision in your project notes.
