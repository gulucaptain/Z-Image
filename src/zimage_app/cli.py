"""Command-line generation using the same service as Gradio."""
import argparse
from pathlib import Path

from .service import GenerationRequest, GenerationService
from .settings import Settings


def main(default_save_steps=False):
    settings = Settings()
    parser = argparse.ArgumentParser(description="Z-Image 图像生成 / 逐步保存")
    parser.add_argument("--model-path", default=settings.model_path)
    parser.add_argument("--prompt", default="Mona Lisa")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output-dir", type=Path, default=settings.output_dir)
    parser.add_argument("--step-dir", type=Path, default=settings.step_dir)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-backend", default=settings.attention)
    parser.add_argument("--compile", action="store_true", default=settings.compile)
    parser.add_argument("--save-steps", action="store_true", default=default_save_steps)
    args = parser.parse_args()
    from dataclasses import replace
    settings = replace(settings, model_path=args.model_path, attention=args.attention_backend,
                       compile=args.compile, output_dir=args.output_dir.expanduser().resolve(),
                       step_dir=args.step_dir.expanduser().resolve())
    request = GenerationRequest(args.prompt, args.height, args.width, args.steps,
                                args.guidance_scale, args.seed, args.save_steps, args.negative_prompt)
    request.validate()
    result = GenerationService(settings).generate(request)
    print(result["status"])
    if result["steps"]:
        print("Step 图像：", Path(result["steps"][0]).parent)
    if result.get("step_video"):
        print("Step 视频：", result["step_video"])
