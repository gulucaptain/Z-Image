#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
CUDA_VISIBLE_DEVICES=7 python scripts/legacy/image_reverse_regeneration.py \
    /home/haoyuzhao/code/Z-Image/tmp/media/video_frames/14_mp4/frame_000004_source_00000120_time_000004.000.jpg \
    --height 480 \
    --width 832 \
    --steps 8 \
    --guidance-scale 0 \
    --seed 42
