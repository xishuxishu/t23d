# t23d text-to-glb (cli)

## what it does
- input prompt
- stage 1: qwen-image generates png
- stage 2: trellis2 generates glb

## prerequisites
- linux
- nvidia gpu + cuda driver
- conda installed
- this project already contains:
  - `repos/TRELLIS.2`
  - `repos/diffusers`
  - `models_cache/Qwen/Qwen-Image-2512`
  - `models_cache/trellis/TRELLIS24B`
  - `models_cache/hf_models/facebook/dinov3-vitl16-pretrain-lvd1689m`
  - `models_cache/hf_models/briaai/RMBG-2.0`
- strict local mode is enabled in code: no fallback to external/remote cache.

## 1) create environment `t23d`
```bash
cd /path/to/t23d
conda create -n t23d python=3.10 -y
conda activate t23d
python -m pip install -U pip setuptools wheel
```

## 2) install pytorch (cuda 12.4)
```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
```

## 3) install runtime python deps (qwen + trellis inference)
```bash
python -m pip install \
  numpy scipy pillow safetensors accelerate transformers huggingface-hub \
  sentencepiece timm einops tqdm opencv-python-headless trimesh imageio \
  imageio-ffmpeg easydict zstandard plyfile filelock regex requests ftfy ninja \
  modelscope
```

## 4) install trellis attention backend (required)
preferred (official default):
```bash
python -m pip install flash-attn==2.7.3
```

fallback (if flash-attn build/install fails):
```bash
python -m pip install xformers
```

if using xformers fallback, set before running:
```bash
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
```

## 5) install trellis gpu deps (mesh/glb path)
```bash
python -m pip install git+https://github.com/JeffreyXiang/CuMesh.git
python -m pip install git+https://github.com/JeffreyXiang/FlexGEMM.git
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git
```

## 6) install local diffusers (needed for QwenImagePipeline)
```bash
python -m pip install -e ./repos/diffusers --no-build-isolation
```

## 7) prepare/build o_voxel
if `repos/TRELLIS.2/o-voxel/third_party/eigen` is empty:
```bash
git clone --depth 1 https://gitlab.com/libeigen/eigen.git \
  repos/TRELLIS.2/o-voxel/third_party/eigen
```

build extension:
```bash
cd repos/TRELLIS.2/o-voxel
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
python setup.py build_ext --inplace
cd ../../..
```

## 8) download model weights (if missing)
if models are already present under `models_cache/`, you can skip this step.

optional (for private/gated models):
```bash
modelscope login --token <your_modelscope_token>
```

download all required weights from ModelScope:
```bash
mkdir -p models_cache/Qwen models_cache/trellis models_cache/hf_models/facebook models_cache/hf_models/briaai

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
```

note:
- We explicitly download `microsoft/TRELLIS-image-large` to
  `models_cache/trellis/TRELLIS24B/microsoft/TRELLIS-image-large`
  to ensure `ss_dec_conv3d_16l8_fp16` is always available.
- The code auto-adapts both local layouts:
  `ckpts/` and `microsoft/TRELLIS-image-large/ckpts/`.
- If a model uses a different ModelScope ID, only replace `model_id`
  in `downloads`; keep local target directories unchanged.

## 9) verify local model files exist
```bash
python - <<'PY'
from pathlib import Path
from engine.pipeline import TextToGLBPipeline

required = [
    Path("models_cache/Qwen/Qwen-Image-2512/model_index.json"),
    Path("models_cache/trellis/TRELLIS24B/pipeline.json"),
    Path("models_cache/hf_models/facebook/dinov3-vitl16-pretrain-lvd1689m/config.json"),
    Path("models_cache/hf_models/briaai/RMBG-2.0/config.json"),
]
missing = [str(p) for p in required if not p.exists()]
print("missing basic files:" if missing else "basic files found")
for p in missing:
    print(" -", p)

if not missing:
    p = TextToGLBPipeline()
    p._validate_trellis_assets()
    print("trellis local asset check: OK")
PY
```

## 10) quick import check
```bash
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
```

## 11) run
seed is fixed to official Qwen-Image seed `42` in code (no cli override).

single command:
```bash
python run_text2glb.py "a red racing car toy"
```

interactive mode:
```bash
python run_text2glb.py
```
