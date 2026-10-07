#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# CUDA_VISIBLE_DEVICES=0 python scripts/legacy/minimax_bench_image_generation.py \
#     --ids MSR-001 ADR-001 VDR-001 AVIR-001 \
#     --output-dir /home/haoyuzhao/code/Z-Image/tmp/history/minimax_bench_outputs/generated_scene_previews_smoke

# CUDA_VISIBLE_DEVICES=0 python scripts/legacy/minimax_bench_image_generation.py \
#     --output-dir /home/haoyuzhao/code/Z-Image/tmp/history/minimax_bench_outputs/generated_scene_previews

# CUDA_VISIBLE_DEVICES=0 python scripts/legacy/minimax_bench_image_generation.py \
#     --datasets ADR \
#     --output-dir /home/haoyuzhao/code/Z-Image/tmp/history/minimax_bench_outputs/generated_adr_first_frames_v2

CUDA_VISIBLE_DEVICES=0 python scripts/legacy/minimax_bench_image_generation.py \
    --datasets VDR \
    --output-dir /home/haoyuzhao/code/Z-Image/tmp/history/minimax_bench_outputs/generated_vdr_first_frames_v2
