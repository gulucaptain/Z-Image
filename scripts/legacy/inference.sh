#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# CUDA_VISIBLE_DEVICES=7 python scripts/legacy/inference.py \
#     --cases /home/haoyuzhao/code/DiffSynth-Studio/assets/reference_text_speech_video_audio_50_scenarios.json \
#     --output-dir /home/haoyuzhao/code/Z-Image/tmp/history/reference_images \
#     --height 480 \
#     --width 832 \
#     --steps 8 \
#     --seed 42

# python scripts/legacy/extract_video_frames.py \
#     /home/haoyuzhao/code/Z-Image/tmp/inputs/douyin_video \
#     tmp/media/video_frames \
#     --fps 1

# python scripts/legacy/extract_video_audio.py /home/haoyuzhao/code/Z-Image/tmp/inputs/douyin_video /home/haoyuzhao/code/Z-Image/tmp/media/douyin_audio


ENV_NAME=$(basename "$VIRTUAL_ENV")
cp -L "$VIRTUAL_ENV/bin/python" "$VIRTUAL_ENV/bin/python-$ENV_NAME"

CUDA_VISIBLE_DEVICES=2 "$VIRTUAL_ENV/bin/python-$ENV_NAME" scripts/image_generation_inference.py
