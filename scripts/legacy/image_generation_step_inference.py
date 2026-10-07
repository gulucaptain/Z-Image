"""Save the decoded latent image after every sampling step to outputs2."""

# Standalone archived scripts use the current checkout's source modules.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Prefer this checkout's pipeline, including its step callback support.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/data/haoyuzhao/models/Z-Image-Turbo")
    parser.add_argument("--prompt", default="Mona Lisa")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs2")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    return args


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except (ImportError, RuntimeError):
        return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def save_step_image(vae, latents, output_path):
    # Match the pipeline's final decode without changing the sampling latents.
    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    decode_latents = latents.to(vae.dtype) / vae.config.scaling_factor + shift_factor
    image = vae.decode(decode_latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].cpu().permute(1, 2, 0).float().numpy()
    Image.fromarray((image * 255).round().astype("uint8")).save(output_path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = ensure_model_weights(args.model_path, verify=False)
    device = select_device()
    print(f"Chosen device: {device}")
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16, compile=False)
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    def save_step(step_index, timestep, latents):
        output_path = args.output_dir / f"step_{step_index + 1:03d}.png"
        save_step_image(components["vae"], latents, output_path)
        print(f"Saved step {step_index + 1} (t={timestep.item():.2f}): {output_path}")

    start_time = time.time()
    generate(
        prompt=args.prompt,
        **components,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device).manual_seed(args.seed),
        callback_on_step_end=save_step,
        output_type="latent",
    )
    print(f"Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
