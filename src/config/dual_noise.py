"""Reproducible settings for two-noise sampling experiments."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DualNoiseConfig:
    mode: str = "initial"
    seed_b: int = 43
    weight_b: float = 0.5
    mix_step: int = 4
    normalize_initial: bool = True
    velocity_rule: str = "coupled"

    def validate(self, steps):
        if self.mode not in {"initial", "latent", "velocity"}:
            raise ValueError("双噪声模式必须为 initial、latent 或 velocity。")
        if not isinstance(self.seed_b, int) or not 0 <= self.seed_b < 2**63:
            raise ValueError("Seed B 必须是 [0, 2^63) 内的整数。")
        if not math.isfinite(self.weight_b) or not 0 <= self.weight_b <= 1:
            raise ValueError("噪声 B 的权重必须在 0 到 1 之间。")
        if self.mode == "latent" and (not isinstance(self.mix_step, int) or not 1 <= self.mix_step <= steps):
            raise ValueError("latent 混合步必须在 1 到采样步数之间（完成该步后混合）。")
        if self.velocity_rule not in {"coupled", "independent"}:
            raise ValueError("速度规则必须为 coupled 或 independent。")
