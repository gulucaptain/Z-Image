"""Recreate images from Qwen-VL descriptions with Z-Image.

The script deliberately runs in two stages so the vision-language model and
Z-Image do not occupy GPU memory at the same time:

1. Qwen-VL turns every source image into an English text-to-image prompt.
2. Z-Image generates a new image from that prompt.

For every input image, ``generated/`` contains the recreated image and
``comparisons/`` contains a side-by-side source/generated comparison.  The
prompts are persisted in ``descriptions.json`` for inspection and resuming.
"""

from __future__ import annotations

# Standalone archived scripts use the current checkout's source modules.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))

import argparse
import gc
import json
import os
import time
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from utils import (
    AttentionBackend,
    ensure_model_weights,
    load_from_local_dir,
    set_attention_backend,
)
from zimage import generate


DEFAULT_QWEN_MODEL = "/data/haoyuzhao/models/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_ZIMAGE_MODEL = "/data/haoyuzhao/models/Z-Image-Turbo"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Asking for a plain English prompt makes the Qwen result directly usable by
# text-to-image models and avoids headings/reasoning leaking into the prompt.
DESCRIPTION_REQUEST = """Study the image carefully and write one detailed English prompt that can recreate it with a text-to-image model. Describe all visible people precisely: number of people, apparent age group, gender presentation, ethnicity/skin tone when visually evident, face shape and features, hair, body build, pose, expression, clothing, accessories, and spatial position. Describe the background, setting, objects, architecture, weather, time of day, lighting, colors, camera angle, framing, depth of field, composition, and visual style. Ignore all text, captions, subtitles, signs, logos, watermarks, and interface elements. Preserve spatial relationships and distinctive details. Do not identify real people, infer hidden facts, or mention that you are viewing an image. Output only the final prompt as one coherent paragraph, with no heading, bullets, or commentary."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe source images with Qwen-VL and recreate them with Z-Image."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="An image file or a directory of images.",
    )
    parser.add_argument("--output-dir", type=Path, default=_Path(__file__).resolve().parents[2] / "tmp" / "history" / "qwen_zimage_outputs")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--zimage-model", default=DEFAULT_ZIMAGE_MODEL)
    parser.add_argument("--description-request", default=DESCRIPTION_REQUEST)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attention-backend",
        default=os.environ.get("ZIMAGE_ATTENTION", "_native_flash"),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile Z-Image (faster after warm-up, slower startup).",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search an input directory recursively."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse saved descriptions and generated files when available.",
    )
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0 or args.steps <= 0:
        parser.error("--height, --width and --steps must be positive")
    return args


def discover_images(input_path: Path, recursive: bool) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No supported images found in {input_path}")
    return images


def output_key(path: Path, input_root: Path) -> str:
    """Build a stable, collision-resistant filename for directory inputs."""
    if input_root.is_file():
        return path.stem
    relative = path.relative_to(input_root.resolve()).with_suffix("")
    return "__".join(relative.parts)


def load_description_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Description cache must be a JSON object: {path}")
    return data


def save_json_atomic(data: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def describe_images(
    images: list[Path],
    keys: list[str],
    cache: dict[str, dict],
    args: argparse.Namespace,
    cache_path: Path,
) -> None:
    pending = [
        (image, key)
        for image, key in zip(images, keys)
        if not (args.skip_existing and cache.get(key, {}).get("description"))
    ]
    if not pending:
        print("All descriptions are cached; skipping Qwen-VL loading.")
        return

    print(f"Loading Qwen-VL from {args.qwen_model}")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.qwen_model, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.qwen_model)

    for index, (image_path, key) in enumerate(pending, start=1):
        print(f"[Qwen {index}/{len(pending)}] Describing {image_path}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": args.description_request},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        trimmed = generated_ids[:, inputs.input_ids.shape[1] :]
        description = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        if not description:
            raise RuntimeError(f"Qwen-VL returned an empty description for {image_path}")
        cache[key] = {"source": str(image_path), "description": description}
        save_json_atomic(cache, cache_path)
        print(f"Prompt: {description}")

    del processor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def make_comparison(source_path: Path, generated_path: Path, output_path: Path) -> None:
    with Image.open(source_path) as source_image, Image.open(generated_path) as generated_image:
        source = ImageOps.exif_transpose(source_image).convert("RGB")
        recreated = ImageOps.exif_transpose(generated_image).convert("RGB")
        panel_width = max(source.width, recreated.width)
        panel_height = max(source.height, recreated.height)
        label_height = 48

        def fit(image: Image.Image) -> Image.Image:
            return ImageOps.pad(
                image,
                (panel_width, panel_height),
                method=Image.Resampling.LANCZOS,
                color=(20, 20, 20),
            )

        canvas = Image.new("RGB", (panel_width * 2, panel_height + label_height), "white")
        canvas.paste(fit(source), (0, label_height))
        canvas.paste(fit(recreated), (panel_width, label_height))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default(size=24)
        except TypeError:  # Pillow < 10.1 does not expose the size argument.
            font = ImageFont.load_default()
        draw.text((16, 12), "Original", fill="black", font=font)
        draw.text((panel_width + 16, 12), "Generated", fill="black", font=font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)


def generate_images(
    images: list[Path], keys: list[str], cache: dict[str, dict], args: argparse.Namespace
) -> None:
    generated_dir = args.output_dir / "generated"
    comparison_dir = args.output_dir / "comparisons"
    generated_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for image_path, key in zip(images, keys):
        generated_path = generated_dir / f"{key}.png"
        comparison_path = comparison_dir / f"{key}_comparison.jpg"
        if args.skip_existing and generated_path.exists() and comparison_path.exists():
            print(f"Skipping existing result: {key}")
        else:
            pending.append((image_path, key, generated_path, comparison_path))
    if not pending:
        print("All generated images and comparisons exist; skipping Z-Image loading.")
        return

    device = choose_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print(f"Loading Z-Image on {device} from {args.zimage_model}")
    model_path = ensure_model_weights(args.zimage_model, verify=False)
    components = load_from_local_dir(
        model_path, device=device, dtype=dtype, compile=args.compile
    )
    AttentionBackend.print_available_backends()
    set_attention_backend(args.attention_backend)
    print(f"Chosen attention backend: {args.attention_backend}")

    for index, (source_path, key, generated_path, comparison_path) in enumerate(pending, start=1):
        extra_description = (
            " The background with the text '复旦大学附属眼耳鼻喉科医院'."
        )
        prompt = cache[key]["description"].strip() + extra_description
        # prompt = cache[key]["description"]

        print(f"[Z-Image {index}/{len(pending)}] Generating {key}")
        start = time.time()
        result = generate(
            prompt=prompt,
            **components,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=device).manual_seed(args.seed + images.index(source_path)),
        )
        result[0].save(generated_path)
        make_comparison(source_path, generated_path, comparison_path)
        print(f"Saved {generated_path} and {comparison_path} ({time.time() - start:.2f}s)")


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = discover_images(args.input, args.recursive)
    input_root = args.input.expanduser().resolve()
    keys = [output_key(path, input_root) for path in images]
    cache_path = args.output_dir / "descriptions.json"
    cache = load_description_cache(cache_path)

    started = time.time()
    print(f"Found {len(images)} image(s). Output: {args.output_dir}")
    describe_images(images, keys, cache, args, cache_path)
    generate_images(images, keys, cache, args)
    print(f"Finished {len(images)} image(s) in {time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
