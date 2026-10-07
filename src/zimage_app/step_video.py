"""Encode saved sampling previews as a browser-playable MP4."""
import shutil
import subprocess
from pathlib import Path


def create_step_video(step_dir, frame_count, fps=2):
    return create_sequence_video(
        Path(step_dir) / "step_%03d.png", Path(step_dir) / "steps.mp4",
        frame_count, fps, "Step",
    )


def create_sequence_video(input_pattern, output, frame_count, fps=2, label="Case"):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法合成逐步视频。")
    video_filter = (
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,"
        f"drawtext=text='{label} " + r"%{eif\:n+1\:d}"
        f" / {frame_count}':"
        r"fontsize=max(12\,h/25):fontcolor=white:"
        "box=1:boxcolor=black@0.65:boxborderw=6:x=12:y=12"
    )
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-framerate", str(fps), "-start_number", "1",
         "-i", str(input_pattern), "-frames:v", str(frame_count),
         "-vf", video_filter, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(f"逐步视频合成失败：{result.stderr.strip()}")
    return str(output)
