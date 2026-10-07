"""Generate one prompt repeatedly and export a ZIP and ordered case video."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zimage_app.repeat import repeat_generate
from zimage_app.service import GenerationRequest, GenerationService
from zimage_app.settings import Settings


def main():
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed-mode", choices=["increment", "random"], default="increment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=settings.model_path)
    parser.add_argument("--output-dir", type=Path, default=settings.step_dir,
                        help="重复任务根目录，默认 outputs2")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0)
    parser.add_argument("--save-steps", action="store_true")
    parser.add_argument("--fps", type=float, default=2, help="结果视频每秒播放的 case 数")
    parser.add_argument("--attention-backend", default=settings.attention)
    parser.add_argument("--compile", action="store_true", default=settings.compile)
    args = parser.parse_args()
    settings = replace(settings, model_path=args.model_path,
                       step_dir=args.output_dir.expanduser().resolve(),
                       attention=args.attention_backend, compile=args.compile)
    request = GenerationRequest(prompt=args.prompt, negative_prompt=args.negative_prompt,
                                height=args.height, width=args.width, steps=args.steps,
                                guidance_scale=args.guidance_scale, seed=args.seed,
                                save_steps=args.save_steps)
    result = None
    for result in repeat_generate(GenerationService(settings), request, args.count, args.seed_mode, args.fps):
        print(result["status"], flush=True)
    print("压缩包：", result["archive"])
    if result["video"]:
        print("视频：", result["video"])
    if not result["archive"] or len(result["gallery"]) != args.count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
