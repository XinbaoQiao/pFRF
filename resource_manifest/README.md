# Resource Manifest and One-Click Download

This directory organizes the resources referenced by two papers and their reference codebases, including pretrained model weights/repositories and datasets, and provides reproducible download scripts.

## Sources

- Papers (text extracted with pdfminer and stored in `_paper_text/`)
  - `RefPaper/Dataset distillation for pre-trained self-supervised vision models.pdf`
  - `RefPaper/Dataset Distillation via Relative Distribution Matching and Cognitive Heritage.pdf`
- Reference code
  - `RefCode/Dataset Distillation for Pre-Trained Self-Supervised Vision Models`

## Generated Files

- `resources.json`: structured resource manifest (extensible fields, suitable for programmatic use)
- `resources.csv`: tabular format of the same manifest
- `download_resources.sh`: one-click download and validation script (resume support, retry on failure, proxy support)
- `download_report.json`: actual download report generated after script execution (including size, sha256, etc.)

## Resource Placement Convention (Aligned with FRF Project)

- Dataset root: `<project_root>/datasets`
- Pretrained and cache root: `<project_root>/pretrained_models`
  - `TORCH_HOME` and `HF_HOME` default to this directory to avoid writing into system-wide caches

## Important Notes (Manually Prepared Resources)

The following resources appear in papers/code but are not auto-downloaded by the script:

- ImageNet-1k / ImageNet-100
  - ImageNet requires registration and acceptance of non-commercial terms. The script only checks whether `datasets/imagenet` exists and does not auto-download it.
- Stanford Dogs, CUB-200-2011
  - The script provides official entry links, but does not enforce automatic download/unzip by default (due to link availability and policy-change risks).

## Run the Download Script

Run from the `pfrf_project` root:

```bash
bash resource_manifest/download_resources.sh
```

With proxy (restricted-network environment):

```bash
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
bash resource_manifest/download_resources.sh
```

## Mapping to Papers/Code (Examples)

- DINOv2: the reference code uses `torch.hub.load("facebookresearch/dinov2", ...)`; this script prefers local clone + local hub load to avoid GitHub timeout issues.
- MoCo-v3: the reference code uses Hugging Face repo id `nyu-visionx/moco-v3-vit-b`; this script downloads it via `huggingface_hub.snapshot_download`.
- Flowers-102/Food-101: the reference code uses `torchvision.datasets.*(download=True)`; this script also uses torchvision to download into `datasets/`.

## Migration to a New Server

1. Copy the entire `pfrf_project`, or at least:
   - `datasets/`
   - `pretrained_models/`
2. Optionally set environment variables:
   - `TORCH_HOME=<project_root>/pretrained_models`
   - `HF_HOME=<project_root>/pretrained_models`
3. Re-run `download_resources.sh` on the new server for incremental completion (resume supported).
