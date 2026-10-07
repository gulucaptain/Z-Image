"""Generate OmniBench scene-preview images with Z-Image.

The default input is ``tmp/inputs/scene_preview_image_prompts_800.json``. Images are
stored under ``<output-dir>/<dataset>/<source_sample_id>.png`` so that the
mapping to the source benchmark remains stable and easy to inspect.
"""

from __future__ import annotations

# Standalone archived scripts use the current checkout's source modules.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CASES = SCRIPT_DIR / "tmp" / "inputs" / "scene_preview_image_prompts_800.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "tmp" / "history" / "minimax_bench_outputs" / "generated_scene_previews"
DEFAULT_MODEL_PATH = Path(
    os.environ.get("ZIMAGE_MODEL_PATH", "/data/haoyuzhao/models/Z-Image-Turbo")
)
DATASET_KEYS = ("MSR", "ADR", "VDR", "AVIR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate preview images for OmniBench's 800 scene prompts with Z-Image."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help=f"Prompt JSON path (default: {DEFAULT_CASES}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Image output root (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local Z-Image-Turbo weights. Can also be set with ZIMAGE_MODEL_PATH.",
    )

    selection = parser.add_argument_group("case selection")
    selection.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_KEYS,
        help="Only generate selected datasets, for example: --datasets MSR ADR.",
    )
    selection.add_argument(
        "--ids",
        nargs="+",
        help="Only generate exact source IDs, for example: --ids MSR-001 ADR-001.",
    )
    selection.add_argument(
        "--start",
        type=int,
        default=1,
        help="First global_index to include, 1-based and inclusive (default: 1).",
    )
    selection.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last global_index to include, 1-based and inclusive (default: no limit).",
    )
    selection.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate at most N cases after applying all other filters.",
    )

    image = parser.add_argument_group("image generation")
    image.add_argument("--height", type=int, default=576, help="Uniform image height.")
    image.add_argument("--width", type=int, default=1024, help="Uniform image width.")
    image.add_argument(
        "--aspect-mode",
        choices=("uniform", "recommended"),
        default="uniform",
        help=(
            "uniform uses --width/--height for every case; recommended uses a 32:9 "
            "canvas for MSR and --width/--height for the other datasets."
        ),
    )
    image.add_argument("--msr-height", type=int, default=432)
    image.add_argument("--msr-width", type=int, default=1536)
    image.add_argument("--steps", type=int, default=8)
    image.add_argument("--guidance-scale", type=float, default=0.0)
    image.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Each case uses base_seed + global_index - 1.",
    )
    image.add_argument(
        "--append-negative-constraints",
        action="store_true",
        help=(
            "Append each record's negative_prompt as natural-language constraints. "
            "Disabled by default because Z-Image Turbo has no separate negative-prompt input."
        ),
    )

    runtime = parser.add_argument_group("runtime and recovery")
    runtime.add_argument(
        "--device",
        choices=("auto", "cuda", "tpu", "mps", "cpu"),
        default="auto",
    )
    runtime.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    runtime.add_argument(
        "--attention-backend",
        default=os.environ.get("ZIMAGE_ATTENTION", "_native_flash"),
    )
    runtime.add_argument("--compile", action="store_true", help="Compile model components.")
    runtime.add_argument(
        "--verify-weights",
        action="store_true",
        help="Verify model weights when ensure_model_weights supports it.",
    )
    runtime.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing non-empty PNG files. By default they are skipped.",
    )
    runtime.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries after the first failed attempt (default: 2).",
    )
    runtime.add_argument("--retry-delay", type=float, default=2.0)
    runtime.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately after one case exhausts all retries.",
    )
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selection and print planned outputs without loading Z-Image.",
    )
    runtime.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print complete prompts instead of short previews.",
    )
    args = parser.parse_args()

    if args.start < 1:
        parser.error("--start must be >= 1")
    if args.end is not None and args.end < args.start:
        parser.error("--end must be >= --start")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.retry_delay < 0:
        parser.error("--retry-delay must be >= 0")
    for name in ("height", "width", "msr_height", "msr_width"):
        value = getattr(args, name)
        if value <= 0 or value % 16:
            parser.error(f"--{name.replace('_', '-')} must be a positive multiple of 16")
    return args


def load_prompt_records(cases_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not cases_path.exists():
        raise FileNotFoundError(f"Prompt JSON does not exist: {cases_path}")
    with cases_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{cases_path} must contain a 'samples' list")

    required = ("global_index", "dataset_key", "source_sample_id", "image_prompt")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for position, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{position}] must be an object")
        missing = [key for key in required if sample.get(key) in (None, "")]
        if missing:
            raise ValueError(f"samples[{position}] is missing: {', '.join(missing)}")

        dataset_key = str(sample["dataset_key"]).upper()
        case_id = str(sample["source_sample_id"])
        global_index = int(sample["global_index"])
        if dataset_key not in DATASET_KEYS:
            raise ValueError(f"samples[{position}] has unknown dataset_key: {dataset_key}")
        if not case_id.startswith(f"{dataset_key}-"):
            raise ValueError(
                f"samples[{position}] ID {case_id!r} does not match dataset {dataset_key}"
            )
        if case_id in seen_ids:
            raise ValueError(f"Duplicate source_sample_id: {case_id}")
        if global_index in seen_indices:
            raise ValueError(f"Duplicate global_index: {global_index}")
        seen_ids.add(case_id)
        seen_indices.add(global_index)

        item = dict(sample)
        item["dataset_key"] = dataset_key
        item["source_sample_id"] = case_id
        item["global_index"] = global_index
        normalized.append(item)

    normalized.sort(key=lambda item: item["global_index"])
    metadata = {key: value for key, value in data.items() if key != "samples"}
    return normalized, metadata


def select_records(records: Iterable[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_filter = set(args.datasets or DATASET_KEYS)
    id_filter = set(args.ids or [])
    selected = []
    for record in records:
        if record["dataset_key"] not in dataset_filter:
            continue
        if id_filter and record["source_sample_id"] not in id_filter:
            continue
        if record["global_index"] < args.start:
            continue
        if args.end is not None and record["global_index"] > args.end:
            continue
        selected.append(record)
    if args.limit is not None:
        selected = selected[: args.limit]
    if id_filter:
        found = {record["source_sample_id"] for record in selected}
        missing = sorted(id_filter - found)
        if missing:
            raise ValueError(
                "Requested IDs were not selected or do not exist: " + ", ".join(missing)
            )
    return selected


def dimensions_for(record: dict[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    if args.aspect_mode == "recommended" and record["dataset_key"] == "MSR":
        return args.msr_height, args.msr_width
    return args.height, args.width


def build_generation_prompt(record: dict[str, Any], append_negative: bool) -> str:
    prompt = str(record["image_prompt"]).strip()
    negative = str(record.get("negative_prompt") or "").strip()
    if negative and append_negative:
        prompt = f"{prompt}\n\n必须避免以下问题：{negative}"
    return prompt


def output_path_for(record: dict[str, Any], output_dir: Path) -> Path:
    return output_dir / record["dataset_key"] / f"{record['source_sample_id']}.png"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def resolve_device(torch: Any, requested: str) -> tuple[Any, str]:
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda", "cuda"
    if requested == "cuda":
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    if requested in ("auto", "tpu"):
        try:
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            return device, "tpu"
        except (ImportError, RuntimeError):
            if requested == "tpu":
                raise RuntimeError("--device tpu was requested, but torch_xla is unavailable")

    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps", "mps"
    if requested == "mps":
        raise RuntimeError("--device mps was requested, but MPS is unavailable")
    return "cpu", "cpu"


def resolve_dtype(torch: Any, requested: str, device_label: str) -> Any:
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    if requested == "float32":
        return torch.float32
    if device_label in ("cuda", "tpu"):
        return torch.bfloat16
    if device_label == "mps":
        return torch.float16
    return torch.float32


def initialize_zimage(args: argparse.Namespace) -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from utils import (
            AttentionBackend,
            ensure_model_weights,
            load_from_local_dir,
            set_attention_backend,
        )
        from zimage import generate
    except ImportError as error:
        raise RuntimeError(
            "Z-Image runtime is unavailable. Run this script from the Z-Image environment "
            "where torch, utils, and zimage can be imported."
        ) from error

    device, device_label = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device_label)
    print(f"Chosen device: {device_label}")
    print(f"Chosen dtype: {dtype}")

    model_path = ensure_model_weights(str(args.model_path), verify=args.verify_weights)
    components = load_from_local_dir(
        model_path,
        device=device,
        dtype=dtype,
        compile=args.compile,
    )
    AttentionBackend.print_available_backends()
    set_attention_backend(args.attention_backend)
    print(f"Chosen attention backend: {args.attention_backend}")
    return torch, generate, components, device


def seeded_generator(torch: Any, device: Any, seed: int) -> Any:
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except (RuntimeError, TypeError):
        return torch.Generator().manual_seed(seed)


def plan_line(
    record: dict[str, Any],
    output_path: Path,
    height: int,
    width: int,
    prompt: str,
    print_prompt: bool,
) -> str:
    prompt_display = prompt if print_prompt else " ".join(prompt.split())[:180]
    if not print_prompt and len(prompt) > 180:
        prompt_display += "…"
    return (
        f"{record['source_sample_id']} [{record['dataset_key']}] {width}x{height}\n"
        f"  output: {output_path}\n"
        f"  prompt: {prompt_display}"
    )


def run_generation(args: argparse.Namespace) -> int:
    records, source_metadata = load_prompt_records(args.cases)
    selected = select_records(records, args)
    if not selected:
        raise ValueError("No prompt records match the requested filters")

    print(f"Loaded {len(records)} prompt records from {args.cases}")
    print(f"Selected {len(selected)} records")
    print(f"Output root: {args.output_dir}")
    if args.dry_run:
        for record in selected:
            height, width = dimensions_for(record, args)
            prompt = build_generation_prompt(record, args.append_negative_constraints)
            print(plan_line(record, output_path_for(record, args.output_dir), height, width, prompt, args.print_prompts))
        print("Dry run complete; the model was not loaded and no images were written.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "generation_manifest.jsonl"
    summary_path = args.output_dir / "generation_summary.json"
    torch, generate, components, device = initialize_zimage(args)

    run_started_at = utc_now()
    total_start = time.time()
    generated_count = 0
    skipped_count = 0
    failed_count = 0
    failures: list[dict[str, Any]] = []
    interrupted = False

    try:
        for position, record in enumerate(selected, start=1):
            case_id = record["source_sample_id"]
            output_path = output_path_for(record, args.output_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = dimensions_for(record, args)
            seed = args.seed + record["global_index"] - 1
            prompt = build_generation_prompt(record, args.append_negative_constraints)

            if not args.overwrite and output_path.exists() and output_path.stat().st_size > 0:
                skipped_count += 1
                print(f"[{position}/{len(selected)}] Skip {case_id}: {output_path} exists")
                append_jsonl(
                    manifest_path,
                    {
                        "timestamp": utc_now(),
                        "status": "skipped_existing",
                        "source_sample_id": case_id,
                        "dataset_key": record["dataset_key"],
                        "global_index": record["global_index"],
                        "output_path": str(output_path),
                        "seed": seed,
                        "width": width,
                        "height": height,
                    },
                )
                continue

            print(f"[{position}/{len(selected)}] Generate {case_id} [{record['dataset_key']}] {width}x{height}, seed={seed}")
            if args.print_prompts:
                print(prompt)
            case_start = time.time()
            last_error: Exception | None = None
            for attempt in range(1, args.max_retries + 2):
                try:
                    images = generate(
                        prompt=prompt,
                        **components,
                        height=height,
                        width=width,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        generator=seeded_generator(torch, device, seed),
                    )
                    if not images:
                        raise RuntimeError("Z-Image returned no images")
                    temporary = output_path.with_suffix(".png.tmp")
                    images[0].save(temporary, format="PNG")
                    temporary.replace(output_path)
                    elapsed = time.time() - case_start
                    generated_count += 1
                    append_jsonl(
                        manifest_path,
                        {
                            "timestamp": utc_now(),
                            "status": "generated",
                            "source_sample_id": case_id,
                            "prompt_id": record.get("prompt_id"),
                            "lookup_key": record.get("lookup_key"),
                            "dataset_key": record["dataset_key"],
                            "category": record.get("category"),
                            "category_name": record.get("category_name"),
                            "counterfactual_id": record.get("counterfactual_id"),
                            "variant": record.get("variant"),
                            "global_index": record["global_index"],
                            "output_path": str(output_path),
                            "seed": seed,
                            "width": width,
                            "height": height,
                            "steps": args.steps,
                            "guidance_scale": args.guidance_scale,
                            "attempt": attempt,
                            "elapsed_seconds": round(elapsed, 3),
                        },
                    )
                    print(f"  Saved {output_path} ({elapsed:.2f}s)")
                    last_error = None
                    del images
                    break
                except Exception as error:  # Keep the 800-case job alive after isolated failures.
                    last_error = error
                    print(
                        f"  Attempt {attempt}/{args.max_retries + 1} failed: "
                        f"{type(error).__name__}: {error}",
                        file=sys.stderr,
                    )
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    if attempt <= args.max_retries:
                        time.sleep(args.retry_delay)

            if last_error is not None:
                failed_count += 1
                failure = {
                    "timestamp": utc_now(),
                    "status": "failed",
                    "source_sample_id": case_id,
                    "dataset_key": record["dataset_key"],
                    "global_index": record["global_index"],
                    "output_path": str(output_path),
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "error_type": type(last_error).__name__,
                    "error": str(last_error),
                    "traceback": "".join(
                        traceback.format_exception(
                            type(last_error), last_error, last_error.__traceback__
                        )
                    ),
                }
                failures.append(failure)
                append_jsonl(manifest_path, failure)
                if args.fail_fast:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user; writing summary before exit.", file=sys.stderr)

    elapsed_total = time.time() - total_start
    summary = {
        "run_started_at": run_started_at,
        "run_finished_at": utc_now(),
        "interrupted": interrupted,
        "cases_file": str(args.cases),
        "source_dataset_name": source_metadata.get("dataset_name"),
        "selected_count": len(selected),
        "generated_count": generated_count,
        "skipped_existing_count": skipped_count,
        "failed_count": failed_count,
        "elapsed_seconds": round(elapsed_total, 3),
        "output_dir": str(args.output_dir),
        "manifest_path": str(manifest_path),
        "configuration": {
            "datasets": args.datasets or list(DATASET_KEYS),
            "ids": args.ids,
            "start": args.start,
            "end": args.end,
            "limit": args.limit,
            "aspect_mode": args.aspect_mode,
            "uniform_width": args.width,
            "uniform_height": args.height,
            "msr_width": args.msr_width,
            "msr_height": args.msr_height,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "base_seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "attention_backend": args.attention_backend,
            "compile": args.compile,
            "overwrite": args.overwrite,
            "max_retries": args.max_retries,
            "negative_constraints_appended": args.append_negative_constraints,
        },
        "failures": failures,
    }
    write_json(summary_path, summary)
    print(
        f"Finished in {elapsed_total:.2f}s: generated={generated_count}, "
        f"skipped={skipped_count}, failed={failed_count}."
    )
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    if interrupted:
        return 130
    return 1 if failed_count else 0


def main() -> int:
    args = parse_args()
    try:
        return run_generation(args)
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
