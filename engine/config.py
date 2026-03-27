from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
IMAGES_DIR = STORAGE_DIR / "images"
MODELS_DIR = STORAGE_DIR / "models"
CACHE_DIR = BASE_DIR / "models_cache"
HF_CACHE_DIR = CACHE_DIR / "hf"
TORCH_CACHE_DIR = CACHE_DIR / "torch"

REPOS_DIR = BASE_DIR / "repos"
TRELLIS_REPO_DIR = REPOS_DIR / "TRELLIS.2"
DIFFUSERS_REPO_DIR = REPOS_DIR / "diffusers"
TRELLIS_MODEL_PATH = str(CACHE_DIR / "trellis" / "TRELLIS24B")
QWEN_MODEL_ID = str(CACHE_DIR / "Qwen" / "Qwen-Image-2512")
OFFICIAL_SEED = 42


def ensure_dirs() -> None:
    for path in [
        IMAGES_DIR,
        MODELS_DIR,
        HF_CACHE_DIR,
        HF_CACHE_DIR / "modules",
        TORCH_CACHE_DIR,
        CACHE_DIR / "trellis",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def apply_runtime_env() -> None:
    ensure_dirs()
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
    os.environ["HF_MODULES_CACHE"] = str(HF_CACHE_DIR / "modules")
    os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
