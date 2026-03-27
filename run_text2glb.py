from __future__ import annotations

import argparse
import os
import sys
import uuid

from engine.pipeline import TextToGLBPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text -> Image + GLB")
    parser.add_argument("prompt", nargs="*", help="text prompt")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = input("Enter prompt: ").strip()
    if not prompt:
        print("prompt cannot be empty", file=sys.stderr)
        return 1

    pipe = TextToGLBPipeline()
    image_path, glb_path = pipe.run(prompt=prompt, job_id=uuid.uuid4().hex)

    print(f"image: {image_path}")
    print(f"glb:   {glb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
