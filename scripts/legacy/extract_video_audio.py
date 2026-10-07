"""Extract audio tracks from videos in a directory and save them as MP3.

Examples:
    python extract_video_audio.py /data/videos /data/audio
    python extract_video_audio.py /data/videos /data/audio --recursive --bitrate 320k

FFmpeg must be installed and available on PATH.
"""

from __future__ import annotations

# Standalone archived scripts use the current checkout's source modules.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))

import argparse
import shutil
import subprocess
from pathlib import Path


VIDEO_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".ts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video audio tracks to MP3 files with FFmpeg."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing videos")
    parser.add_argument("output_dir", type=Path, help="Directory for output MP3 files")
    parser.add_argument(
        "--bitrate",
        default="192k",
        help="MP3 bitrate, such as 128k, 192k or 320k (default: 192k)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process videos in subdirectories recursively",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip MP3 files that already exist",
    )
    return parser.parse_args()


def find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def extract_audio(
    video_path: Path,
    output_path: Path,
    bitrate: str,
    overwrite: bool,
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as error:
        # Remove an incomplete file left by a failed conversion.
        if output_path.exists():
            output_path.unlink()
        print(f"[Failed] {video_path}: FFmpeg exited with code {error.returncode}")
        return False


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found. Install it first, for example: "
            "conda install -c conda-forge ffmpeg"
        )
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    videos = find_videos(input_dir, args.recursive)
    if not videos:
        raise RuntimeError(f"No supported videos found in: {input_dir}")

    succeeded = 0
    skipped = 0
    failed = 0
    print(f"Found {len(videos)} video(s).")

    for index, video_path in enumerate(videos, start=1):
        relative = video_path.relative_to(input_dir)
        # Preserve the input directory structure. Add the video extension to
        # prevent foo.mp4 and foo.mov from writing to the same MP3 file.
        output_path = (
            output_dir
            / relative.parent
            / f"{video_path.stem}_{video_path.suffix[1:].lower()}.mp3"
        )

        if args.skip_existing and output_path.exists():
            print(f"[{index}/{len(videos)}] Skipped existing: {output_path}")
            skipped += 1
            continue

        print(f"[{index}/{len(videos)}] Extracting: {video_path}")
        if extract_audio(video_path, output_path, args.bitrate, overwrite=True):
            print(f"Saved: {output_path}")
            succeeded += 1
        else:
            failed += 1

    print(
        f"Finished: {succeeded} succeeded, {skipped} skipped, "
        f"{failed} failed. Output: {output_dir}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
