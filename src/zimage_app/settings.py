import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "tmp" / "history"
INPUT_DIR = ROOT / "tmp" / "inputs"
MEDIA_DIR = ROOT / "tmp" / "media"
BUILTIN_CASES = INPUT_DIR / "scene_preview_image_prompts_800.json"


@dataclass(frozen=True)
class Settings:
    model_path: str = os.environ.get("ZIMAGE_MODEL_PATH", "/data/haoyuzhao/models/Z-Image-Turbo")
    qwen_path: str = os.environ.get("ZIMAGE_QWEN_PATH", "/data/haoyuzhao/models/Qwen3-VL-30B-A3B-Instruct")
    attention: str = os.environ.get("ZIMAGE_ATTENTION", "native")
    compile: bool = os.environ.get("ZIMAGE_COMPILE", "false").lower() in {"1", "true", "yes"}
    output_dir: Path = Path(os.environ.get("ZIMAGE_OUTPUT_DIR", str(ROOT / "outputs"))).expanduser().resolve()
    step_dir: Path = ROOT / "outputs2"
    tmp_dir: Path = Path(os.environ.get("ZIMAGE_TMP_DIR", str(ROOT / "tmp" / "app"))).expanduser().resolve()


def prepare_temp(settings):
    """Configure Gradio's cache before importing Gradio."""
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    gradio_dir = settings.tmp_dir / "gradio"
    gradio_dir.mkdir(exist_ok=True)
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = str(settings.tmp_dir)
    os.environ["GRADIO_TEMP_DIR"] = str(gradio_dir)


def browse_directories(settings):
    return {"生成结果": settings.output_dir, "逐步图像": settings.step_dir,
            "历史文生图": HISTORY_DIR / "outputs",
            "参考图": HISTORY_DIR / "reference_images", "Qwen 重绘旧结果": HISTORY_DIR / "qwen_zimage_outputs",
            "OmniBench 旧结果": HISTORY_DIR / "minimax_bench_outputs", "视频帧": MEDIA_DIR / "video_frames"}
