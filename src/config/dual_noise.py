"""Reproducible settings for two-noise sampling experiments."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DualNoiseConfig:
    mode: str = "staged_copy"
    seed_b: int = 43
    weight_b: float = 0.5
    mix_step: int = 4
    normalize_initial: bool = True
    velocity_rule: str = "coupled"
    stage1_steps: int | None = None
    stage_split: float = 0.5
    terminal_epsilon: float = 0.0

    def stage_steps(self, steps):
        first = self.stage1_steps if self.stage1_steps is not None else steps // 2
        return first, steps - first

    def validate(self, steps):
        if self.mode == "complementary":
            raise ValueError("互补任务 (C,Q) 需要读取 C 的条件速度模型及训练权重；当前 Z-Image 文生图权重没有该接口。")
        if self.mode not in {"staged_copy", "initial", "latent", "velocity"}:
            raise ValueError("未知双噪声实验模式。")
        if not isinstance(self.seed_b, int) or not 0 <= self.seed_b < 2**63:
            raise ValueError("Seed B 必须是 [0, 2^63) 内的整数。")
        if not math.isfinite(self.weight_b) or not 0 <= self.weight_b <= 1:
            raise ValueError("噪声 B 的权重必须在 0 到 1 之间。")
        if self.mode == "latent" and (not isinstance(self.mix_step, int) or not 1 <= self.mix_step <= steps):
            raise ValueError("latent 混合步必须在 1 到采样步数之间（完成该步后混合）。")
        if self.velocity_rule not in {"coupled", "independent"}:
            raise ValueError("速度规则必须为 coupled 或 independent。")
        if self.mode == "staged_copy":
            first, second = self.stage_steps(steps)
            if not isinstance(first, int) or first < 1 or second < 1:
                raise ValueError("分阶段实验至少需要 2 步，且阶段一和阶段二都至少分配 1 步。")
            if not math.isfinite(self.stage_split) or not 0 < self.stage_split < 1:
                raise ValueError("时间分界 r 必须在 0 和 1 之间。")
            if not math.isfinite(self.terminal_epsilon) or not 0 <= self.terminal_epsilon < 1 - self.stage_split:
                raise ValueError("终点 epsilon 必须满足 0 ≤ epsilon < 1-r。")
