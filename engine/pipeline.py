from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from PIL import Image

from .config import (
    BASE_DIR,
    DIFFUSERS_REPO_DIR,
    IMAGES_DIR,
    MODELS_DIR,
    OFFICIAL_SEED,
    QWEN_MODEL_ID,
    TRELLIS_MODEL_PATH,
    TRELLIS_REPO_DIR,
    apply_runtime_env,
)


class TextToGLBPipeline:
    """Text -> image -> glb using local Qwen-Image and TRELLIS.2."""

    def __init__(self) -> None:
        apply_runtime_env()
        self._base = Path(BASE_DIR).resolve()
        self._qwen_pipe = None
        self._trellis_pipe = None
        self._o_voxel = None

        self._add_repo_path(Path(DIFFUSERS_REPO_DIR) / "src")
        self._add_repo_path(Path(TRELLIS_REPO_DIR))
        self._add_repo_path(Path(TRELLIS_REPO_DIR) / "o-voxel")

    def run(self, prompt: str, job_id: str) -> tuple[Path, Path]:
        image_path = IMAGES_DIR / f"{job_id}.png"
        glb_path = MODELS_DIR / f"{job_id}.glb"
        self._txt2img(prompt, image_path)
        self._img2glb(image_path, glb_path)
        return image_path, glb_path

    def _txt2img(self, prompt: str, image_path: Path) -> None:
        import torch

        if self._qwen_pipe is None:
            self._qwen_pipe = self._load_qwen_pipe()

        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(OFFICIAL_SEED)

        out = self._qwen_pipe(
            prompt=f"{prompt}, single centered object, isolated subject, pure white background",
            negative_prompt=(
                "blurry, out of focus, bokeh, shallow depth of field, motion blur, "
                "low detail, cluttered background, busy background, dark background, "
                "gradient background, extra objects, multiple objects"
            ),
            width=1328,
            height=1328,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=generator,
        )
        out.images[0].save(image_path)

    def _img2glb(self, image_path: Path, glb_path: Path) -> None:
        if self._trellis_pipe is None:
            self._trellis_pipe = self._load_trellis_pipe()
        if self._o_voxel is None:
            import o_voxel

            self._o_voxel = o_voxel

        image = Image.open(image_path).convert("RGBA")
        mesh = self._trellis_pipe.run(image)[0]
        if hasattr(mesh, "simplify"):
            mesh.simplify(16777216)

        glb = self._o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=300000,
            texture_size=2048,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        glb.export(str(glb_path))

    def _load_qwen_pipe(self):
        import torch

        model_path = Path(QWEN_MODEL_ID).resolve()
        if not (model_path / "model_index.json").exists():
            raise RuntimeError(f"Missing file: {model_path / 'model_index.json'}")

        importlib.invalidate_caches()
        import diffusers

        pipe = diffusers.QwenImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True,
        )
        return pipe.to("cuda") if torch.cuda.is_available() else pipe

    def _load_trellis_pipe(self):
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        model_root = Path(TRELLIS_MODEL_PATH).resolve()
        print(str(model_root))

        pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(model_root))
        if hasattr(pipe, "cuda"):
            pipe.cuda()
        return pipe


    @staticmethod
    def _resolve_model_stem(model_root: Path, rel_path: str) -> Path | None:
        candidates = [model_root / rel_path]
        if "ckpts/" in rel_path:
            candidates.append(model_root / "ckpts" / rel_path.split("ckpts/", 1)[1])
        for stem in candidates:
            if stem.with_suffix(".json").exists() and stem.with_suffix(".safetensors").exists():
                return stem
        return None


    def _add_repo_path(self, path: Path) -> None:
        p = path.resolve()
        if not p.exists():
            return
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

