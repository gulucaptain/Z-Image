"""Generate reference images for all scenarios in a JSON case file."""

# Standalone archived scripts use the current checkout's source modules.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one reference image for every scenario in cases.json."
    )
    parser.add_argument("--cases", type=Path, default=Path("cases.json"))
    parser.add_argument("--output-dir", type=Path, default=_Path(__file__).resolve().parents[2] / "tmp" / "history" / "reference_images")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not regenerate images that already exist in output-dir.",
    )
    return parser.parse_args()


def load_scenarios(cases_path):
    with cases_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError(f"{cases_path} must contain a 'scenarios' list")

    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenarios[{index}] must be an object")
        if not scenario.get("id") or not scenario.get("reference_image"):
            raise ValueError(
                f"scenarios[{index}] must contain non-empty 'id' and 'reference_image' fields"
            )
    return scenarios


def main():
    args = parse_args()
    scenarios = load_scenarios(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = ensure_model_weights("/data/haoyuzhao/models/Z-Image-Turbo", verify=False)  # True to verify with md5
    dtype = torch.bfloat16
    compile = False  # default False for compatibility
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")

    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        device = "cuda"
        print("Chosen device: cuda")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            print("Chosen device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                device = "mps"
                print("Chosen device: mps")
            else:
                device = "cpu"
                print("Chosen device: cpu")
    # Load models
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    total_start = time.time()
    generated_count = 0
    for index, scenario in enumerate(scenarios):
        case_id = str(scenario["id"])
        output_path = args.output_dir / f"{case_id}.png"
        if args.skip_existing and output_path.exists():
            print(f"[{index + 1}/{len(scenarios)}] Skipping {case_id}: {output_path} exists")
            continue

        print(f"[{index + 1}/{len(scenarios)}] Generating {case_id}: {scenario['reference_image']}")
        start_time = time.time()
        images = generate(
            prompt=scenario["reference_image"],
            **components,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device).manual_seed(args.seed + index),
        )
        images[0].save(output_path)
        generated_count += 1
        print(f"Saved {output_path} ({time.time() - start_time:.2f} seconds)")

    print(
        f"Finished: generated {generated_count}/{len(scenarios)} images in "
        f"{time.time() - total_start:.2f} seconds"
    )

    ### !! For best speed performance, recommend to use `_flash_3` backend and set `compile=True`
    ### This would give you sub-second generation speed on Hopper GPU (H100/H200/H800) after warm-up


if __name__ == "__main__":
    main()
