
# Standalone archived scripts use the current checkout's source modules.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

# Gradio resolves its temporary directory when it is imported. Configure all
# temporary-file environment variables first so this process never writes to
# the system /tmp directory.
LOCAL_TMP_DIR = Path(
    os.environ.get("ZIMAGE_TMP_DIR", str(Path.cwd() / "tmp"))
).resolve()
GRADIO_TMP_DIR = LOCAL_TMP_DIR / "gradio"
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
GRADIO_TMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(LOCAL_TMP_DIR)
os.environ["TMP"] = str(LOCAL_TMP_DIR)
os.environ["TEMP"] = str(LOCAL_TMP_DIR)
os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TMP_DIR)

import gradio as gr
import torch

warnings.filterwarnings("ignore")

from utils import (  # noqa: E402
    AttentionBackend,
    ensure_model_weights,
    load_from_local_dir,
    set_attention_backend,
)
from zimage import generate  # noqa: E402


MODEL_PATH = os.environ.get("ZIMAGE_MODEL_PATH", "/data/haoyuzhao/models/Z-Image-Turbo")
ATTN_BACKEND = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
COMPILE = os.environ.get("ZIMAGE_COMPILE", "false").lower() in {"1", "true", "yes"}
OUTPUT_DIR = Path(os.environ.get("ZIMAGE_OUTPUT_DIR", "outputs"))

DEFAULT_PROMPT = (
    "Photorealistic hospital corridor safety scene, 16:9 landscape composition. "
    "A compact white indoor service robot with a dark sensor panel and a low wheeled "
    "base is moving toward an L-shaped blind corner in a clean hospital corridor. "
    "The corridor has smooth white and pale-gray walls, light-colored anti-slip floor "
    "tiles, protective wall handrails, and soft recessed ceiling lights. A large "
    "circular convex safety mirror with a black rim is mounted high on the inner wall "
    "of the corner. Inside the convex mirror, a white medical trolley with stainless-"
    "steel rails and caster wheels is clearly visible rapidly approaching the same "
    "intersection from the hidden perpendicular corridor. The trolley carries "
    "transparent medicine boxes, medical-record folders, a sealed infusion bottle, "
    "and a gently swinging IV bag beneath a pale-blue waterproof cover. Subtle motion "
    "blur around the trolley's wheels conveys its high speed. The convex mirror "
    "provides a wide, optically coherent reflection of the hidden corridor, with "
    "natural edge compression, curved floor-tile lines, and slight fisheye distortion. "
    "The reflected trolley becomes prominent near the center of the mirror, clearly "
    "indicating an imminent collision risk. The actual trolley remains completely "
    "hidden behind the corner and appears only in the mirror. The service robot has "
    "not yet reacted or changed direction. Static medium-wide camera view, keeping the "
    "service robot, blind corner, and entire convex mirror visible simultaneously. "
    "Clean, quiet hospital atmosphere with subtle cinematic tension. Realistic scale, "
    "perspective, reflections, materials, and lighting. No people, no doctors, no "
    "patients, no collision, no warning graphics, no readable text, no logos, no "
    "subtitles, and no watermark."
)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


print("Loading Z-Image-Turbo. The Gradio page will start after loading finishes.")
DEVICE = select_device()
DTYPE = torch.bfloat16
print(f"Chosen device: {DEVICE}")

resolved_model_path = ensure_model_weights(MODEL_PATH, verify=False)
COMPONENTS = load_from_local_dir(
    resolved_model_path,
    device=DEVICE,
    dtype=DTYPE,
    compile=COMPILE,
)
AttentionBackend.print_available_backends()
set_attention_backend(ATTN_BACKEND)
print(f"Chosen attention backend: {ATTN_BACKEND}")
print(f"torch.compile enabled: {COMPILE}")


def generate_image(prompt, height, width, steps, guidance_scale, seed):
    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("Prompt不能为空。")

    height = int(height)
    width = int(width)
    steps = int(steps)
    seed = int(seed)

    start_time = time.perf_counter()
    try:
        with torch.inference_mode():
            images = generate(
                prompt=prompt,
                **COMPONENTS,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=float(guidance_scale),
                generator=torch.Generator(device=DEVICE).manual_seed(seed),
            )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise gr.Error("GPU显存不足，请降低图像分辨率后重试。") from exc

    elapsed = time.perf_counter() - start_time
    image = images[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"zimage_{timestamp}_seed{seed}.png"
    image.save(output_path)

    status = (
        f"生成完成：{elapsed:.2f}秒 | {width}x{height} | "
        f"seed={seed} | 保存至{output_path}"
    )
    return image, status


with gr.Blocks(title="Z-Image-Turbo") as demo:
    gr.Markdown("# Z-Image-Turbo图像生成")

    with gr.Row():
        with gr.Column(scale=5):
            prompt_input = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                lines=16,
            )

            with gr.Row():
                height_input = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=720,
                    step=8,
                    label="Height",
                )
                width_input = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1280,
                    step=8,
                    label="Width",
                )

            with gr.Row():
                steps_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=8,
                    step=1,
                    label="Inference Steps",
                )
                guidance_input = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=0.0,
                    step=0.1,
                    label="Guidance Scale",
                )
                seed_input = gr.Number(value=42, precision=0, label="Seed")

            generate_button = gr.Button("Generate", variant="primary")

        with gr.Column(scale=5):
            output_image = gr.Image(label="Generated Image", type="pil")
            output_status = gr.Textbox(label="Status", interactive=False)

    generate_button.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            height_input,
            width_input,
            steps_input,
            guidance_input,
            seed_input,
        ],
        outputs=[output_image, output_status],
    )


if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
