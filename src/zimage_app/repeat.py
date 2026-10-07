"""Repeat one prompt, preserving run order, seeds and downloadable artifacts."""
import json
import math
import secrets
import shutil
import subprocess
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone

from .step_video import create_sequence_video


def repeat_generate(service, request, count=100, seed_mode="increment", fps=2, progress=None):
    request.validate()
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("重复次数必须是大于 0 的整数。")
    count = int(count)
    if seed_mode not in {"increment", "random"}:
        raise ValueError("Seed 模式必须为 increment 或 random。")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("视频 FPS 必须是大于 0 的有限数值。")
    if seed_mode == "increment" and request.seed + count > 2**63:
        raise ValueError("递增后的 Seed 超出允许范围，请降低起始 Seed 或重复次数。")
    if seed_mode == "increment":
        seeds = list(range(request.seed, request.seed + count))
    else:
        seeds, used = [], set()
        while len(seeds) < count:
            seed = secrets.randbelow(2**63)
            if seed not in used:
                seeds.append(seed)
                used.add(seed)

    # Keep model configuration fixed throughout this prompt's repetitions.
    with service.lock:
        task_id = "repeat_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:6]
        root = service.settings.step_dir / task_id
        root.mkdir(parents=True)
        manifest = {"task_id": task_id, "request": asdict(request), "count": count,
                    "seed_mode": seed_mode, "seeds": seeds, "video_fps": fps,
                    "model_path": service.settings.model_path, "cases": [], "status": "running"}
        manifest_path = root / "manifest.json"

        def persist():
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)

        persist()
        gallery = []
        yield {"gallery": [], "video": None, "archive": None, "directory": str(root),
               "status": f"开始重复生成，共 {count} 次。\n{root}"}
        for index, seed in enumerate(seeds, 1):
            case_dir = root / "cases" / f"case_{index:04d}"

            def on_progress(value, desc=""):
                if progress:
                    progress((index - 1 + value) / count, desc=f"Case {index}/{count} · {desc}")

            try:
                result = service.generate(
                    replace(request, seed=seed), progress=on_progress, kind="repeat",
                    extra={"task_id": task_id, "case_index": index}, case_dir=case_dir,
                )
            except Exception as exc:
                manifest.update(status="failed", failed_case=index, error=str(exc))
                persist()
                break
            manifest["cases"].append({"index": index, "seed": seed,
                                      "image": str((case_dir / "final.png").relative_to(root)),
                                      "metadata": str((case_dir / "metadata.json").relative_to(root))})
            gallery.append((result["image"], f"Case {index} · seed={seed}"))
            persist()
            yield {"gallery": list(gallery), "video": None, "archive": None, "directory": str(root),
                   "status": f"已完成 {index}/{count} · seed={seed}\n{root}"}
        else:
            manifest["status"] = "completed"

        video = None
        warning = ""
        if gallery:
            if progress:
                progress(1, desc="合成 case 视频和压缩包")
            try:
                video = create_sequence_video(root / "cases" / "case_%04d" / "final.png",
                                              root / "cases.mp4", len(gallery), fps)
                manifest["video"] = "cases.mp4"
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                warning = f"视频合成失败：{exc}"
                manifest["video_error"] = warning
        manifest["completed_count"] = len(gallery)
        persist()
        # Put the ZIP beside the directory to avoid including the archive itself.
        archive = shutil.make_archive(str(root), "zip", root_dir=root.parent, base_dir=root.name)
        status = f"完成 {len(gallery)}/{count} 个 case\n{root}"
        if manifest["status"] == "failed":
            status += f"\nCase {manifest['failed_case']} 失败：{manifest['error']}，已打包保留已有结果。"
        if warning:
            status += f"\n{warning}"
        yield {"gallery": gallery, "video": video, "archive": archive,
               "directory": str(root), "status": status}
