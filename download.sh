#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  models_cache/Qwen \
  models_cache/trellis \
  models_cache/hf_models/facebook \
  models_cache/hf_models/briaai

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