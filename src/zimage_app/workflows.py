"""Batch, image recreation and video adapters for the shared UI."""
import importlib
import json
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from .inputs import load_records
from .service import GenerationRequest
from .settings import MEDIA_DIR


def batch_generate(service, request, file_path, text, datasets, ids, limit, append_negative, progress=None):
    records = load_records(file_path, text, datasets, ids, limit)
    gallery, files = [], []
    # Hold the model lock across the whole batch, including configuration.
    with service.lock:
        for index, record in enumerate(records):
            prompt = record["prompt"]
            if append_negative and record.get("negative_prompt"):
                prompt += "\n\n必须避免以下问题：" + record["negative_prompt"]
            current = replace(request, prompt=prompt, seed=request.seed + index)
            def on_progress(value, desc=""):
                if progress:
                    progress((index + value) / len(records), desc=f"{index + 1}/{len(records)} · {record['id']} · {desc}")
            result = service.generate(current, progress=on_progress, kind="batch", extra=record)
            gallery.append((result["image"], record["id"]))
            files.extend([result["image"], result["metadata"], *result["steps"]])
            if result.get("step_video"):
                files.append(result["step_video"])
            yield gallery, files, f"已完成 {index + 1}/{len(records)}，seed={current.seed}\n{result['image']}"


def recreate(service, source, qwen_path, extra_prompt, request, progress=None):
    if not source:
        raise ValueError("请上传参考图。")
    if not qwen_path.strip():
        raise ValueError("请填写 Qwen-VL 模型路径。")
    source = Path(source)
    module = importlib.import_module("scripts.legacy.image_reverse_regeneration")
    work_dir = service.settings.output_dir / "reverse_sources" / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)
    copied = work_dir / ("source" + source.suffix.lower())
    shutil.copy2(source, copied)
    cache = {}
    args = SimpleNamespace(skip_existing=False, qwen_model=qwen_path.strip(),
                           description_request=module.DESCRIPTION_REQUEST, max_new_tokens=512)
    with service.lock:
        # Qwen-VL and Z-Image must not occupy GPU memory simultaneously.
        service.unload()
        if progress:
            progress(0, desc="Qwen-VL 分析参考图")
        module.describe_images([copied], ["source"], cache, args, work_dir / "descriptions.json")
        prompt = cache["source"]["description"]
        if extra_prompt.strip():
            prompt += " " + extra_prompt.strip()
        result = service.generate(replace(request, prompt=prompt), progress=progress, kind="reverse",
                                  extra={"source": str(copied), "description": cache["source"]["description"]})
        comparison = Path(result["image"]).parent / "comparison.jpg"
        module.make_comparison(copied, Path(result["image"]), comparison)
    files = [result["image"], str(comparison), result["metadata"], *result["steps"]]
    if result.get("step_video"):
        files.append(result["step_video"])
    return result["image"], str(comparison), prompt, files, result["status"]


def video_process(mode, input_dir, fps, recursive, skip_existing, progress=None):
    source = Path(input_dir).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("视频输入目录不存在。")
    if mode == "提取视频帧" and float(fps) <= 0:
        raise ValueError("FPS 必须大于 0。")
    frames = mode == "提取视频帧"
    module = importlib.import_module("scripts.legacy.extract_video_frames" if frames else "scripts.legacy.extract_video_audio")
    videos = module.find_videos(source, bool(recursive))
    if not videos:
        raise ValueError("输入目录中没有支持的视频。")
    output = MEDIA_DIR / ("video_frames" if frames else "douyin_audio")
    if not frames and not shutil.which("ffmpeg"):
        raise ValueError("音频提取需要安装 FFmpeg。")
    logs = []
    for i, video in enumerate(videos):
        # Preserve subdirectories and extensions so same stems do not collide.
        relative = video.relative_to(source)
        target = output / relative.parent / relative.name.replace(".", "_")
        if frames:
            if skip_existing and target.is_dir() and any(target.iterdir()):
                logs.append(f"跳过：{video.name}")
            else:
                target.mkdir(parents=True, exist_ok=True)
                count = module.extract_frames(video, target, float(fps), False, "jpg", 95)
                logs.append(f"{video.name}：{count} 帧")
        else:
            target = target.with_name(target.name + ".mp3")
            if skip_existing and target.exists():
                logs.append(f"跳过：{video.name}")
            else:
                ok = module.extract_audio(video, target, "192k", overwrite=True)
                logs.append(f"{video.name}：{'完成' if ok else '失败'}")
        if progress:
            progress((i + 1) / len(videos), desc=video.name)
    return "\n".join(logs) + f"\n输出目录：{output}"
