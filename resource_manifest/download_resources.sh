#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOCK_DIR="$ROOT_DIR/resource_manifest/.download_lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -f "$LOCK_DIR/pid" ]]; then
    lock_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    if [[ -n "${lock_pid:-}" ]] && kill -0 "$lock_pid" >/dev/null 2>&1; then
      echo "Another download_resources.sh is running (pid=${lock_pid})."
      exit 2
    fi
  fi
  rm -rf "$LOCK_DIR" 2>/dev/null || true
  mkdir "$LOCK_DIR"
fi
echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR" 2>/dev/null || true' EXIT

DATASETS_DIR="${DATASETS_DIR:-$ROOT_DIR/datasets}"
PRETRAINED_DIR="${PRETRAINED_DIR:-$ROOT_DIR/pretrained_models}"
MANIFEST_JSON="${MANIFEST_JSON:-$ROOT_DIR/resource_manifest/resources.json}"
REPORT_JSON="${REPORT_JSON:-$ROOT_DIR/resource_manifest/download_report.json}"
IMAGENET_DIR="${IMAGENET_DIR:-$DATASETS_DIR/imagenet}"
INCLUDE_IMAGENET="${INCLUDE_IMAGENET:-0}"
INCLUDE_T3="${INCLUDE_T3:-0}"
INCLUDE_TEST="${INCLUDE_TEST:-0}"

TRAIN_T12_MD5="${TRAIN_T12_MD5:-1d675b47d978889d74fa0da5fadfb00e}"
TRAIN_T3_MD5="${TRAIN_T3_MD5:-ccaf1013018ac1037801578038d370da}"
VAL_MD5="${VAL_MD5:-29b22e2961454d5413ddabcf34fc5622}"
TEST_MD5="${TEST_MD5:-e1b8681fff3d63731c599df9b4b6fc02}"

mkdir -p "$DATASETS_DIR" "$PRETRAINED_DIR" "$ROOT_DIR/resource_manifest"

export TORCH_HOME="${TORCH_HOME:-$PRETRAINED_DIR}"
export HF_HOME="${HF_HOME:-$PRETRAINED_DIR}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT="https://huggingface.co"
export HF_HUB_DISABLE_XET=1

HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}"
NO_PROXY="${NO_PROXY:-${no_proxy:-}}"

CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-20}"
CURL_MAX_TIME="${CURL_MAX_TIME:-600}"
CURL_SPEED_TIME="${CURL_SPEED_TIME:-60}"
CURL_SPEED_LIMIT="${CURL_SPEED_LIMIT:-1024}"
CURL_IP_RESOLVE="${CURL_IP_RESOLVE:-4}"

can_reach_url() {
  local url="$1"
  curl -sSI \
    $([[ "$CURL_IP_RESOLVE" == "4" ]] && echo --ipv4) \
    $([[ "$CURL_IP_RESOLVE" == "6" ]] && echo --ipv6) \
    --connect-timeout 3 --max-time 5 \
    ${HTTP_PROXY:+--proxy "$HTTP_PROXY"} \
    "$url" >/dev/null 2>&1
}

download_url() {
  local url="$1"
  local out="$2"
  local tries="${3:-8}"
  mkdir -p "$(dirname "$out")"

  if command -v curl >/dev/null 2>&1; then
    for i in $(seq 1 "$tries"); do
      local_size="$(python -c "import os; print(os.path.getsize('$out') if os.path.exists('$out') else 0)")"
      code="$(curl -L --retry 0 \
        $([[ "$CURL_IP_RESOLVE" == "4" ]] && echo --ipv4) \
        $([[ "$CURL_IP_RESOLVE" == "6" ]] && echo --ipv6) \
        --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" \
        --speed-time "$CURL_SPEED_TIME" --speed-limit "$CURL_SPEED_LIMIT" \
        $([[ "$local_size" -gt 0 ]] && echo --continue-at -) \
        ${HTTP_PROXY:+--proxy "$HTTP_PROXY"} \
        -o "$out" -w "%{http_code}" -sS "$url" || true)"
      if [[ "$code" == "200" || "$code" == "206" ]]; then
        return 0
      fi
      if [[ "$code" == "416" ]]; then
        remote_len="$(curl -sSI ${HTTP_PROXY:+--proxy "$HTTP_PROXY"} "$url" | awk -F': ' 'tolower($1)=="content-length"{print $2}' | tail -n 1 | tr -d '\r' || true)"
        local_size="$(python -c "import os; print(os.path.getsize('$out') if os.path.exists('$out') else 0)")"
        if [[ -n "$remote_len" ]] && [[ "$remote_len" == "$local_size" ]]; then
          return 0
        fi
        rm -f "$out"
      fi
      sleep "$((i * 5))"
    done
  elif command -v wget >/dev/null 2>&1; then
    for i in $(seq 1 "$tries"); do
      wget -c -O "$out" "$url" && return 0
      sleep "$((i * 5))"
    done
  else
    echo "Missing curl/wget"
    return 2
  fi

  return 1
}

sha256_file() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    python - "$f" <<'PY'
import hashlib, sys
p=sys.argv[1]
h=hashlib.sha256()
with open(p,'rb') as fp:
    for b in iter(lambda: fp.read(1024*1024), b''):
        h.update(b)
print(h.hexdigest())
PY
  fi
}

md5_check() {
  local file="$1"
  local expected="$2"
  local got
  got="$(md5sum "$file" | awk '{print $1}')"
  if [[ "$got" != "$expected" ]]; then
    echo "MD5 mismatch: $file expected=$expected got=$got"
    return 1
  fi
  return 0
}

json_escape() {
  python -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

report_begin() {
  printf "{\n  \"started_at\": \"%s\",\n  \"root_dir\": \"%s\",\n  \"datasets_dir\": \"%s\",\n  \"pretrained_dir\": \"%s\",\n  \"items\": [\n" "$(date -Iseconds)" "$ROOT_DIR" "$DATASETS_DIR" "$PRETRAINED_DIR" > "$REPORT_JSON"
}

report_add_item() {
  local name="$1"
  local category="$2"
  local path="$3"
  local url="$4"
  local sha256="$5"
  local size="$6"
  local status="$7"
  local comma="$8"
  printf "    {\n      \"category\": %s,\n      \"name\": %s,\n      \"path\": %s,\n      \"source_url\": %s,\n      \"sha256\": %s,\n      \"size_bytes\": %s,\n      \"status\": %s\n    }%s\n" \
    "$(printf "%s" "$category" | json_escape)" \
    "$(printf "%s" "$name" | json_escape)" \
    "$(printf "%s" "$path" | json_escape)" \
    "$(printf "%s" "$url" | json_escape)" \
    "$(printf "%s" "$sha256" | json_escape)" \
    "$(printf "%s" "$size" | json_escape)" \
    "$(printf "%s" "$status" | json_escape)" \
    "$comma" >> "$REPORT_JSON"
}

report_end() {
  printf "  ],\n  \"finished_at\": \"%s\"\n}\n" "$(date -Iseconds)" >> "$REPORT_JSON"
}

download_flowers102() {
  CUDA_VISIBLE_DEVICES="" python - <<PY
import torchvision
import os
root=os.environ.get("DATASETS_DIR")
torchvision.datasets.Flowers102(root=f"{root}/flowers102", split="train", download=True)
torchvision.datasets.Flowers102(root=f"{root}/flowers102", split="test", download=True)
print("ok")
PY
}

download_food101() {
  CUDA_VISIBLE_DEVICES="" python - <<PY
import torchvision
import os
root=os.environ.get("DATASETS_DIR")
torchvision.datasets.Food101(root=f"{root}/food101", split="train", download=True)
torchvision.datasets.Food101(root=f"{root}/food101", split="test", download=True)
print("ok")
PY
}

download_artbench() {
  local tar_path="$DATASETS_DIR/artbench/artbench-10-imagefolder-split.tar"
  download_url "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar" "$tar_path" 8
  if [[ ! -d "$DATASETS_DIR/artbench/train" ]]; then
    mkdir -p "$DATASETS_DIR/artbench"
    if ! tar -xf "$tar_path" -C "$DATASETS_DIR/artbench"; then
      rm -f "$tar_path"
      return 1
    fi
    if [[ -d "$DATASETS_DIR/artbench/artbench-10-imagefolder-split" ]]; then
      mv "$DATASETS_DIR/artbench/artbench-10-imagefolder-split/train" "$DATASETS_DIR/artbench/train"
      mv "$DATASETS_DIR/artbench/artbench-10-imagefolder-split/test" "$DATASETS_DIR/artbench/test"
      rm -rf "$DATASETS_DIR/artbench/artbench-10-imagefolder-split"
    fi
  fi
}

download_imagenet_ilsvrc2012() {
  mkdir -p "$IMAGENET_DIR"

  local devkit_t12_url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
  local devkit_t3_url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t3.tar.gz"
  local train_t12_url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
  local train_t3_url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar"
  local val_url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
  local test_url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar"

  echo "[resource] imagenet_ilsvrc2012_archives"
  download_url "$devkit_t12_url" "$IMAGENET_DIR/ILSVRC2012_devkit_t12.tar.gz" 8
  download_url "$train_t12_url" "$IMAGENET_DIR/ILSVRC2012_img_train.tar" 8
  download_url "$val_url" "$IMAGENET_DIR/ILSVRC2012_img_val.tar" 8

  md5_check "$IMAGENET_DIR/ILSVRC2012_img_train.tar" "$TRAIN_T12_MD5"
  md5_check "$IMAGENET_DIR/ILSVRC2012_img_val.tar" "$VAL_MD5"

  if [[ "$INCLUDE_T3" -eq 1 ]]; then
    download_url "$devkit_t3_url" "$IMAGENET_DIR/ILSVRC2012_devkit_t3.tar.gz" 8
    download_url "$train_t3_url" "$IMAGENET_DIR/ILSVRC2012_img_train_t3.tar" 8
    md5_check "$IMAGENET_DIR/ILSVRC2012_img_train_t3.tar" "$TRAIN_T3_MD5"
  fi

  if [[ "$INCLUDE_TEST" -eq 1 ]]; then
    download_url "$test_url" "$IMAGENET_DIR/ILSVRC2012_img_test_v10102019.tar" 8
    md5_check "$IMAGENET_DIR/ILSVRC2012_img_test_v10102019.tar" "$TEST_MD5"
  fi

  echo "[resource] imagenet_ilsvrc2012_extract"
  CUDA_VISIBLE_DEVICES="" TORCH_HOME="$PRETRAINED_DIR" HF_HOME="$PRETRAINED_DIR" python - <<'PY'
from torchvision.datasets import ImageNet
ImageNet(root="datasets/imagenet", split="train")
ImageNet(root="datasets/imagenet", split="val")
print("ok")
PY
}

prepare_dinov2_repo_and_weights() {
  local repo_dir="$PRETRAINED_DIR/hub/facebookresearch_dinov2_main"
  mkdir -p "$PRETRAINED_DIR/hub"
  if [[ ! -d "$repo_dir" ]]; then
    if command -v git >/dev/null 2>&1; then
      git clone --depth 1 --branch main https://github.com/facebookresearch/dinov2.git "$repo_dir"
    fi
  fi
  local w="$PRETRAINED_DIR/hub/checkpoints/dinov2_vitb14_pretrain.pth"
  download_url "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" "$w" 8
}

prepare_clip_vitb16() {
  mkdir -p "$PRETRAINED_DIR/clip"
  local w="$PRETRAINED_DIR/clip/ViT-B-16.pt"
  if download_url "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e4c7b2c6a0c70d3b7e83a9c6db/ViT-B-16.pt" "$w" 2; then
    return 0
  fi
  rm -f "$w"
  if ! can_reach_url "https://huggingface.co/api/models/openai/clip-vit-base-patch16"; then
    return 1
  fi
  if CUDA_VISIBLE_DEVICES="" python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("openai/clip-vit-base-patch16", local_dir="pretrained_models/hf/openai/clip-vit-base-patch16", local_dir_use_symlinks=False)
print("ok")
PY
  then
    return 0
  fi
  return 1
}

prepare_moco_v3_resnet50() {
  local w="$PRETRAINED_DIR/hub/checkpoints/mocov3_r50_1000ep.pth.tar"
  download_url "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar" "$w" 8
}

prepare_eva02_vitb() {
  if ! can_reach_url "https://huggingface.co/api/models/timm/eva02_base_patch14_224.mim_in22k"; then
    return 1
  fi
  CUDA_VISIBLE_DEVICES="" python - <<PY
import os
from pathlib import Path
import timm
timm.create_model("eva02_base_patch14_224.mim_in22k", pretrained=True)
hf_home = os.environ.get("HF_HOME", "")
hub = Path(hf_home) / "hub"
hits = list(hub.rglob("*eva02*")) if hub.exists() else []
if len(hits) == 0:
    raise SystemExit(2)
print("ok")
PY
}

export DATASETS_DIR
export PRETRAINED_DIR

report_begin
any_failed=0

status="ok"
echo "[resource] dinov2"
prepare_dinov2_repo_and_weights || status="failed"
if [[ "$status" != "ok" ]]; then any_failed=1; fi
path="$PRETRAINED_DIR/hub/checkpoints/dinov2_vitb14_pretrain.pth"
if [[ "$status" == "ok" ]] && [[ -f "$path" ]]; then
  s="$(sha256_file "$path")"
  z="$(python -c "import os; print(os.path.getsize('$path'))")"
  report_add_item "DINOv2 ViT-B/14 weights" "model" "$path" "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" "$s" "$z" "$status" ","
else
  report_add_item "DINOv2 ViT-B/14 weights" "model" "$path" "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" "" "" "$status" ","
fi

status="ok"
echo "[resource] clip_vitb16"
prepare_clip_vitb16 || status="failed"
path="$PRETRAINED_DIR/clip/ViT-B-16.pt"
if [[ -f "$path" ]]; then
  size="$(python -c "import os; print(os.path.getsize('$path'))")"
  if [[ "$size" -lt 50000000 ]]; then
    rm -f "$path"
    status="failed"
  fi
fi
if [[ "$status" != "ok" ]]; then any_failed=1; fi
if [[ "$status" == "ok" ]] && [[ -f "$path" ]]; then
  s="$(sha256_file "$path")"
  z="$(python -c "import os; print(os.path.getsize('$path'))")"
  report_add_item "CLIP ViT-B/16 weights" "model" "$path" "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e4c7b2c6a0c70d3b7e83a9c6db/ViT-B-16.pt" "$s" "$z" "$status" ","
else
  report_add_item "CLIP ViT-B/16 weights" "model" "$path" "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e4c7b2c6a0c70d3b7e83a9c6db/ViT-B-16.pt" "" "" "$status" ","
fi

status="ok"
echo "[resource] moco_v3_resnet50"
prepare_moco_v3_resnet50 || status="failed"
if [[ "$status" != "ok" ]]; then any_failed=1; fi
path="$PRETRAINED_DIR/hub/checkpoints/mocov3_r50_1000ep.pth.tar"
if [[ -f "$path" ]]; then
  s="$(sha256_file "$path")"
  z="$(python -c "import os; print(os.path.getsize('$path'))")"
  report_add_item "MoCo-v3 ResNet-50 (1000ep checkpoint)" "model" "$path" "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar" "$s" "$z" "$status" ","
else
  report_add_item "MoCo-v3 ResNet-50 (1000ep checkpoint)" "model" "$path" "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar" "" "" "$status" ","
fi

status="ok"
echo "[resource] eva02_vitb_timm"
prepare_eva02_vitb || status="failed"
if [[ "$status" != "ok" ]]; then any_failed=1; fi
report_add_item "EVA-02 ViT-B (timm pretrained cache)" "model" "$PRETRAINED_DIR" "timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=True)" "" "" "$status" ","

status="ok"
echo "[resource] flowers102"
download_flowers102 || status="failed"
if [[ "$status" != "ok" ]]; then any_failed=1; fi
path="$DATASETS_DIR/flowers102"
if [[ -d "$path" ]]; then
  z="$(python -c "import pathlib; p=pathlib.Path('$path'); print(sum(f.stat().st_size for f in p.rglob('*') if f.is_file()))")"
  report_add_item "Flowers-102 dataset" "dataset" "$path" "torchvision.datasets.Flowers102(download=True)" "" "$z" "$status" ","
else
  report_add_item "Flowers-102 dataset" "dataset" "$path" "torchvision.datasets.Flowers102(download=True)" "" "" "$status" ","
fi

status="ok"
echo "[resource] food101"
download_food101 || status="failed"
if [[ "$status" != "ok" ]]; then any_failed=1; fi
path="$DATASETS_DIR/food101"
if [[ -d "$path" ]]; then
  z="$(python -c "import pathlib; p=pathlib.Path('$path'); print(sum(f.stat().st_size for f in p.rglob('*') if f.is_file()))")"
  report_add_item "Food-101 dataset" "dataset" "$path" "torchvision.datasets.Food101(download=True)" "" "$z" "$status" ","
else
  report_add_item "Food-101 dataset" "dataset" "$path" "torchvision.datasets.Food101(download=True)" "" "" "$status" ","
fi

status="ok"
echo "[resource] artbench10"
download_artbench || status="failed"
if [[ "$status" != "ok" ]]; then any_failed=1; fi
path="$DATASETS_DIR/artbench"
artbench_comma=""
if [[ "$INCLUDE_IMAGENET" -eq 1 ]]; then
  artbench_comma=","
fi
if [[ -d "$path" ]]; then
  z="$(python -c "import pathlib; p=pathlib.Path('$path'); print(sum(f.stat().st_size for f in p.rglob('*') if f.is_file()))")"
  report_add_item "ArtBench-10 dataset" "dataset" "$path" "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar" "" "$z" "$status" "$artbench_comma"
else
  report_add_item "ArtBench-10 dataset" "dataset" "$path" "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar" "" "" "$status" "$artbench_comma"
fi

if [[ "$INCLUDE_IMAGENET" -eq 1 ]]; then
  status="ok"
  download_imagenet_ilsvrc2012 || status="failed"
  if [[ "$status" != "ok" ]]; then any_failed=1; fi
  path="$IMAGENET_DIR"
  if [[ -d "$path" ]]; then
    z="$(python -c "import pathlib; p=pathlib.Path('$path'); print(sum(f.stat().st_size for f in p.rglob('*') if f.is_file()))")"
    report_add_item "ImageNet ILSVRC2012 dataset" "dataset" "$path" "https://image-net.org/data/ILSVRC/2012/" "" "$z" "$status" ""
  else
    report_add_item "ImageNet ILSVRC2012 dataset" "dataset" "$path" "https://image-net.org/data/ILSVRC/2012/" "" "" "$status" ""
  fi
fi

report_end

echo "WROTE $REPORT_JSON"
if [[ "$any_failed" -ne 0 ]]; then
  exit 1
fi
