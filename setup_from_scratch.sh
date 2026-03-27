#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash setup_from_scratch.sh [options]

Options:
  --with-models      Download required model weights from ModelScope.
  --skip-flash-attn  Skip flash-attn install and use xformers instead.
  --force-xformers   Force xformers even if flash-attn can be installed.
  -h, --help         Show this help message.

Environment variables:
  ENV_NAME               Conda env name (default: t23d)
  CUDA_HOME              CUDA toolkit path (default: /usr/local/cuda)
  TORCH_CUDA_ARCH_LIST   GPU arch list for building o_voxel (default: 8.0)
EOF
}

log() {
  echo "[setup] $*"
}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-t23d}"
WITH_MODELS=0
SKIP_FLASH_ATTN=0
FORCE_XFORMERS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-models)
      WITH_MODELS=1
      shift
      ;;
    --skip-flash-attn)
      SKIP_FLASH_ATTN=1
      shift
      ;;
    --force-xformers)
      FORCE_XFORMERS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$PROJECT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install conda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Creating conda env: $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.10 -y
else
  log "Conda env already exists: $ENV_NAME"
fi

conda activate "$ENV_NAME"

log "Installing base python tooling"
python -m pip install -U pip setuptools wheel

log "Installing PyTorch (CUDA 12.4)"
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0

log "Installing runtime dependencies"
python -m pip install \
  numpy scipy pillow safetensors accelerate transformers huggingface-hub \
  sentencepiece timm einops tqdm opencv-python-headless trimesh imageio \
  imageio-ffmpeg easydict zstandard plyfile filelock regex requests ftfy ninja \
  modelscope

if [[ "$SKIP_FLASH_ATTN" -eq 1 || "$FORCE_XFORMERS" -eq 1 ]]; then
  log "Installing xformers (flash-attn skipped/disabled)"
  python -m pip install xformers
  USE_XFORMERS=1
else
  log "Installing flash-attn (required by default backend)"
  if python -m pip install flash-attn==2.7.3; then
    USE_XFORMERS=0
  else
    log "flash-attn install failed; falling back to xformers"
    python -m pip install xformers
    USE_XFORMERS=1
  fi
fi

log "Installing TRELLIS GPU dependencies"
python -m pip install git+https://github.com/JeffreyXiang/CuMesh.git
python -m pip install git+https://github.com/JeffreyXiang/FlexGEMM.git
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git

log "Installing local diffusers (editable)"
python -m pip install -e ./repos/diffusers --no-build-isolation

EIGEN_DIR="repos/TRELLIS.2/o-voxel/third_party/eigen"
if [[ ! -d "$EIGEN_DIR" || -z "$(ls -A "$EIGEN_DIR")" ]]; then
  log "Cloning Eigen into $EIGEN_DIR"
  git clone --depth 1 https://gitlab.com/libeigen/eigen.git "$EIGEN_DIR"
else
  log "Eigen already present: $EIGEN_DIR"
fi

log "Building o_voxel extension"
pushd repos/TRELLIS.2/o-voxel >/dev/null
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
python setup.py build_ext --inplace
popd >/dev/null

if [[ "$WITH_MODELS" -eq 1 ]]; then
  log "Downloading model weights from ModelScope"
  python -m pip install -U modelscope

  mkdir -p models_cache/Qwen models_cache/trellis models_cache/hf_models/facebook models_cache/hf_models/briaai models_cache/trellis/TRELLIS24B/microsoft

  python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download

downloads = [
    ("Qwen/Qwen-Image-2512", "models_cache/Qwen/Qwen-Image-2512"),
    ("microsoft/TRELLIS.2-4B", "models_cache/trellis/TRELLIS24B"),
    ("microsoft/TRELLIS-image-large", "models_cache/trellis/TRELLIS24B/microsoft/TRELLIS-image-large"),
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", "models_cache/hf_models/facebook/dinov3-vitl16-pretrain-lvd1689m"),
    ("briaai/RMBG-2.0", "models_cache/hf_models/briaai/RMBG-2.0"),
]

for model_id, local_dir in downloads:
    print(f"[download] {model_id} -> {local_dir}")
    snapshot_download(model_id=model_id, local_dir=local_dir)
PY
else
  log "Skipping model download (use --with-models to enable)"
fi

log "Checking required local model files"
if python - <<'PY'
from pathlib import Path

required = [
    Path("models_cache/Qwen/Qwen-Image-2512/model_index.json"),
    Path("models_cache/trellis/TRELLIS24B/pipeline.json"),
    Path("models_cache/trellis/TRELLIS24B/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json"),
    Path("models_cache/trellis/TRELLIS24B/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
    Path("models_cache/hf_models/facebook/dinov3-vitl16-pretrain-lvd1689m/config.json"),
    Path("models_cache/hf_models/briaai/RMBG-2.0/config.json"),
]

missing = [str(p) for p in required if not p.exists()]
if missing:
    print("missing:")
    for p in missing:
        print(" -", p)
    raise SystemExit(1)
print("all required model files found")
PY
then
  :
else
  log "Some model files are missing."
  if [[ "$WITH_MODELS" -eq 0 ]]; then
    log "Tip: rerun with --with-models to download them automatically."
  fi
fi

log "Quick import check"
python - <<'PY'
import sys
import importlib.util
import torch, diffusers, o_voxel, cumesh, flex_gemm

sys.path.insert(0, "repos/TRELLIS.2")
from trellis2.pipelines import Trellis2ImageTo3DPipeline

print("cuda:", torch.cuda.is_available())
print("qwen_pipe:", hasattr(diffusers, "QwenImagePipeline"))
print("ovoxel_ext:", hasattr(o_voxel, "_C"))
print("cumesh_ok:", cumesh is not None)
print("flex_gemm_ok:", flex_gemm is not None)
print("flash_attn:", importlib.util.find_spec("flash_attn") is not None)
print("xformers:", importlib.util.find_spec("xformers") is not None)
print("trellis_import_ok:", Trellis2ImageTo3DPipeline is not None)
PY

if [[ "${USE_XFORMERS:-0}" -eq 1 ]]; then
  cat <<'EOF'
[setup] xformers backend is active.
[setup] Before running inference in a new shell:
[setup]   export ATTN_BACKEND=xformers
[setup]   export SPARSE_ATTN_BACKEND=xformers
EOF
fi

cat <<EOF
[setup] Done.
[setup] Activate env:
[setup]   conda activate $ENV_NAME
[setup] Run:
[setup]   python run_text2glb.py "a red racing car toy"
EOF
