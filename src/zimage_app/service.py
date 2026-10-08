"""Shared lazy model lifecycle, generation and result persistence."""
import gc
import json
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .settings import Settings


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    height: int = 720
    width: int = 1280
    steps: int = 8
    guidance_scale: float = 0.0
    seed: int = 42
    save_steps: bool = False
    negative_prompt: str = ""
    analyze_velocity: bool = False

    def validate(self):
        if not self.prompt.strip():
            raise ValueError("提示词不能为空。")
        if self.height <= 0 or self.width <= 0 or self.height % 16 or self.width % 16:
            raise ValueError("宽度和高度必须是正的 16 的倍数。")
        if self.steps < 1:
            raise ValueError("步数必须大于 0。")
        if not 0 <= self.seed < 2**63:
            raise ValueError("Seed 必须在 0 到 2^63-1 之间。")
        if self.guidance_scale < 0:
            raise ValueError("Guidance Scale 不能为负数。")


def decode_image(vae, latents):
    from PIL import Image

    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    image = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor + shift, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).float().numpy()
    return Image.fromarray((image * 255).round().astype("uint8"))


class GenerationService:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.lock = threading.RLock()
        self.components = None
        self.device = None
        self.loaded_config = None

    def configure(self, model_path, attention, compile_model):
        if not str(model_path).strip():
            raise ValueError("模型路径不能为空。")
        with self.lock:
            from dataclasses import replace
            updated = replace(self.settings, model_path=str(model_path).strip(), attention=attention, compile=bool(compile_model))
            if updated != self.settings:
                self.unload()
                self.settings = updated
        return "配置已更新，下次生成时加载模型。"

    def unload(self):
        with self.lock:
            self.components = None
            self.loaded_config = None
            gc.collect()
            # Importing the UI should not import torch or load weights.
            import sys
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return "模型已卸载。"

    def load(self):
        if self.components is not None:
            return
        import torch
        from utils import ensure_model_weights, load_from_local_dir, set_attention_backend

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            try:
                import torch_xla.core.xla_model as xm
                self.device = xm.xla_device()
            except (ImportError, RuntimeError):
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32 if str(self.device) == "cpu" else torch.bfloat16
        set_attention_backend(self.settings.attention)
        path = ensure_model_weights(self.settings.model_path, verify=False)
        self.components = load_from_local_dir(path, device=self.device, dtype=dtype, compile=self.settings.compile)
        self.loaded_config = self.settings.model_path

    def generate_stream(self, request, progress=None, dual_noise=None):
        """Yield saved step previews while the synchronous sampler is running."""
        request.validate()
        events = queue.Queue()
        stopped = threading.Event()

        def on_step(update):
            if stopped.is_set():
                raise RuntimeError("生成已取消。")
            events.put(("update", update))

        def worker():
            try:
                result = self.generate(request, progress, step_callback=on_step,
                                       kind="dual_noise" if dual_noise is not None else "single", dual_noise=dual_noise)
                events.put(("update", result))
            except Exception as exc:
                events.put(("error", exc))
            finally:
                events.put(("done", None))

        thread = threading.Thread(target=worker, name="zimage-preview", daemon=True)
        thread.start()
        try:
            while True:
                event, value = events.get()
                if event == "done":
                    break
                if event == "error":
                    raise value
                yield value
        finally:
            stopped.set()
            thread.join()

    def generate(self, request, progress=None, kind="single", extra=None, step_callback=None, case_dir=None, dual_noise=None):
        request.validate()
        if dual_noise is not None:
            dual_noise.validate(request.steps)
        with self.lock:
            import torch
            from zimage import generate

            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:6]
            run_dir = Path(case_dir) if case_dir is not None else self.settings.output_dir / kind / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            step_dir = run_dir / "steps" if case_dir is not None else self.settings.step_dir / run_id
            paths = []
            diagnostics = None
            if request.analyze_velocity:
                from .diagnostics import VelocityDiagnostics
                diagnostics = VelocityDiagnostics(run_dir / "velocity")
            metadata = {**asdict(request), "run_id": run_id, "kind": kind,
                        "model_path": self.settings.model_path, "attention": self.settings.attention,
                        "compile": self.settings.compile, "extra": extra or {}, "status": "running"}
            if dual_noise is not None:
                metadata["dual_noise"] = asdict(dual_noise)
                metadata["dual_noise_trace"] = []
                if dual_noise.mode == "staged_copy":
                    first, second = dual_noise.stage_steps(request.steps)
                    metadata.update(stage1_steps=first, stage2_steps=second,
                                    experiment="pretrained_generation_then_analytic_copy",
                                    preview_layout="left X, right Y; active branch switches at stage boundary",
                                    velocity_convention="diagnostics: local dz/dsigma; trace: joint dz/dt",
                                    output_component="Y", joint_model_trained=False)
                else:
                    metadata["velocity_convention"] = "dz/dsigma; latent merge step includes jump/delta_sigma"
            metadata_path = run_dir / "metadata.json"
            def persist():
                temporary = metadata_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(metadata_path)
            persist()
            started = time.perf_counter()
            try:
                if progress:
                    progress(0, desc="加载模型 / 准备生成")
                self.load()
                if request.save_steps:
                    step_dir.mkdir(parents=True, exist_ok=True)

                joint_preview = None

                def on_dual_state(index, x, y, trace):
                    nonlocal joint_preview
                    if not request.save_steps and index + 1 not in {metadata["stage1_steps"], request.steps}:
                        return
                    from PIL import Image, ImageDraw
                    left = decode_image(self.components["vae"], x)
                    right = decode_image(self.components["vae"], y)
                    joint_preview = Image.new("RGB", (left.width + right.width, max(left.height, right.height) + 32), "white")
                    joint_preview.paste(left, (0, 0))
                    joint_preview.paste(right, (left.width, 0))
                    ImageDraw.Draw(joint_preview).text(
                        (8, joint_preview.height - 24),
                        f"Stage {trace['stage']} | left: X ({'active' if trace['stage'] == 1 else 'frozen'}) | right: Y ({'frozen' if trace['stage'] == 1 else 'active'})",
                        fill="black",
                    )
                    if trace["stage"] == 1 and index + 1 == metadata["stage1_steps"]:
                        left.save(run_dir / "stage1_X.png")
                        metadata["stage1_image"] = str(run_dir / "stage1_X.png")
                    if index + 1 == request.steps:
                        joint_preview.save(run_dir / "final_XY.png")
                        metadata["joint_final_image"] = str(run_dir / "final_XY.png")

                def on_step(index, timestep, latents):
                    if request.save_steps:
                        path = step_dir / f"step_{index + 1:03d}.png"
                        if joint_preview is not None:
                            joint_preview.save(path)
                        else:
                            decode_image(self.components["vae"], latents).save(path)
                        paths.append(str(path))
                    if progress:
                        stage = metadata.get("dual_noise_trace", [])
                        phase = f"阶段 {stage[-1]['stage']} · 更新 {stage[-1]['active']} · " if stage and "stage" in stage[-1] else ""
                        progress(min((index + 1) / request.steps, 1), desc=f"{phase}采样 step {index + 1}/{request.steps}")
                    if step_callback:
                        step_callback({"image": paths[-1] if paths else None,
                                       "steps": list(paths), "metadata": str(metadata_path),
                                       "diagnostics": diagnostics.snapshot() if diagnostics else None,
                                       "stage1_image": metadata.get("stage1_image"),
                                       "joint_final_image": metadata.get("joint_final_image"),
                                       "dual_noise_trace": list(metadata.get("dual_noise_trace", [])),
                                       "status": f"采样 step {index + 1}/{request.steps}"})

                with torch.inference_mode():
                    analysis_kwargs = {"callback_on_velocity": diagnostics.record} if diagnostics else {}
                    if dual_noise is not None:
                        analysis_kwargs.update(dual_noise=dual_noise, callback_on_dual_step=metadata["dual_noise_trace"].append)
                        if dual_noise.mode == "staged_copy":
                            analysis_kwargs["callback_on_dual_state"] = on_dual_state
                    images = generate(**self.components, prompt=request.prompt,
                                      height=request.height, width=request.width,
                                      num_inference_steps=request.steps, guidance_scale=request.guidance_scale,
                                      negative_prompt=request.negative_prompt or None,
                                      generator=torch.Generator(self.device).manual_seed(request.seed),
                                      callback_on_step_end=on_step, **analysis_kwargs)
                final = run_dir / "final.png"
                images[0].save(final)
                step_video = None
                video_warning = ""
                if paths:
                    from .step_video import create_step_video
                    if progress:
                        progress(1, desc="合成逐步视频")
                    try:
                        step_video = create_step_video(step_dir, len(paths))
                    except (RuntimeError, OSError, TimeoutError) as exc:
                        video_warning = f"逐步视频未生成：{exc}"
                    except subprocess.TimeoutExpired:
                        video_warning = "逐步视频未生成：合成超时。"
                    metadata.update(step_video=step_video, step_video_fps=2)
                    if video_warning:
                        metadata["step_video_error"] = video_warning
                metadata.update(status="completed", elapsed_seconds=round(time.perf_counter() - started, 3),
                                final_image=str(final), step_images=paths)
                if diagnostics:
                    metadata["velocity_analysis"] = diagnostics.snapshot()
                persist()
                return {"image": str(final), "steps": paths, "metadata": str(metadata_path),
                        "stage1_image": metadata.get("stage1_image"),
                        "joint_final_image": metadata.get("joint_final_image"),
                        "dual_noise_trace": list(metadata.get("dual_noise_trace", [])),
                        "step_video": step_video,
                        "diagnostics": diagnostics.snapshot() if diagnostics else None,
                        "status": f"完成 | {request.width}×{request.height} | seed={request.seed} | {metadata['elapsed_seconds']:.2f} 秒\n{final}"
                                  + (f"\n{video_warning}" if video_warning else "")}
            except Exception as exc:
                metadata.update(status="failed", error=str(exc), step_images=paths)
                if diagnostics:
                    metadata["velocity_analysis"] = diagnostics.snapshot()
                persist()
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    raise RuntimeError("显存不足，请降低分辨率或卸载模型后重试。") from exc
                raise

    def history(self, limit=50):
        items = sorted(self.settings.output_dir.glob("*/*/metadata.json"), key=lambda p: p.parent.name, reverse=True)
        gallery, rows = [], []
        for path in items[:limit]:
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                image = item.get("final_image")
                if image and Path(image).is_file():
                    gallery.append((image, f"{item['kind']} | seed={item['seed']} | {item['prompt'][:60]}"))
                rows.append([item["run_id"], item["kind"], item["status"], item["seed"], item["prompt"], str(path)])
            except (OSError, ValueError, KeyError):
                continue
        return gallery, rows
