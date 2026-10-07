
# Standalone archived scripts use the current checkout's source modules.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
import argparse
from pathlib import Path

import cv2


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
}


def parse_args():
    parser = argparse.ArgumentParser(description="批量提取视频帧")
    parser.add_argument("input_dir", type=Path, help="视频文件夹")
    parser.add_argument("output_dir", type=Path, help="视频帧输出文件夹")
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="每秒提取的帧数，默认 1",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="提取视频中的所有帧，设置后忽略 --fps",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理输入文件夹中的子目录",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default="jpg",
        help="输出图片格式，默认 jpg",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPG 保存质量，范围 1～100，默认 95",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已经输出过帧的目录",
    )
    return parser.parse_args()


def find_videos(input_dir: Path, recursive: bool):
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    all_frames: bool,
    image_format: str,
    jpg_quality: int,
):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"[失败] 无法打开视频：{video_path}")
        return 0

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0:
        source_fps = 25.0
        print(f"[警告] 无法读取视频帧率，使用默认帧率：{source_fps}")

    if all_frames:
        frame_interval = 1
    else:
        if target_fps <= 0:
            capture.release()
            raise ValueError("--fps 必须大于 0")

        # target_fps 高于原始帧率时，最多只能提取所有原始帧。
        frame_interval = max(1, round(source_fps / target_fps))

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    saved_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            timestamp_seconds = frame_index / source_fps
            filename = (
                f"frame_{saved_index:06d}"
                f"_source_{frame_index:08d}"
                f"_time_{timestamp_seconds:010.3f}.{image_format}"
            )
            output_path = output_dir / filename

            if image_format == "jpg":
                saved = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpg_quality],
                )
            else:
                saved = cv2.imwrite(str(output_path), frame)

            if not saved:
                print(f"[警告] 图片保存失败：{output_path}")
            else:
                saved_index += 1

        frame_index += 1

    capture.release()

    print(
        f"[完成] {video_path.name}: "
        f"原始帧率={source_fps:.3f}, "
        f"总帧数={total_frames}, "
        f"保存帧数={saved_index}"
    )

    return saved_index


def main():
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入文件夹不存在：{input_dir}")

    videos = find_videos(input_dir, args.recursive)

    if not videos:
        raise RuntimeError(f"没有在输入文件夹中找到视频：{input_dir}")

    print(f"找到 {len(videos)} 个视频")

    total_saved = 0

    for index, video_path in enumerate(videos, start=1):
        relative_path = video_path.relative_to(input_dir)

        # 保留子目录结构，并加入视频扩展名避免同名文件冲突。
        video_output_dir = (
            output_root
            / relative_path.parent
            / f"{video_path.stem}_{video_path.suffix[1:].lower()}"
        )

        if args.skip_existing and video_output_dir.exists():
            existing_frames = list(
                video_output_dir.glob(f"*.{args.image_format}")
            )
            if existing_frames:
                print(
                    f"[{index}/{len(videos)}] 跳过已有结果："
                    f"{video_output_dir}"
                )
                continue

        print(f"[{index}/{len(videos)}] 正在处理：{video_path}")

        saved_count = extract_frames(
            video_path=video_path,
            output_dir=video_output_dir,
            target_fps=args.fps,
            all_frames=args.all_frames,
            image_format=args.image_format,
            jpg_quality=args.jpg_quality,
        )
        total_saved += saved_count

    print(f"全部处理完成，共保存 {total_saved} 帧到：{output_root}")


if __name__ == "__main__":
    main()