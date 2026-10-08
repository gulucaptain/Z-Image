"""Run the staged (X0,Y0) -> (Qhat,Y0) -> (Qhat,Qhat) inference baseline."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config.dual_noise import DualNoiseConfig
from zimage_app.service import GenerationRequest, GenerationService
from zimage_app.settings import Settings


def main():
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--mode", choices=["staged_copy", "initial", "latent", "velocity"], default="staged_copy",
                        help="默认分阶段复制基准；initial/latent/velocity 仅保留用于旧实验复现")
    parser.add_argument("--seed-a", type=int, default=42)
    parser.add_argument("--seed-b", type=int, default=43)
    parser.add_argument("--weight-b", type=float, default=0.5, help="仅旧线性混合模式生效")
    parser.add_argument("--mix-step", type=int, default=4, help="完成第 k 步后混合 latent，仅 latent 模式生效")
    parser.add_argument("--raw-initial-mix", action="store_true", help="初始混合不做方差补偿")
    parser.add_argument("--velocity-rule", choices=["coupled", "independent"], default="coupled")
    parser.add_argument("--stage1-steps", type=int, default=None, help="阶段一模型采样步数，默认总步数的一半；其余为解析复制步数")
    parser.add_argument("--stage-split", type=float, default=0.5, help="联合时间分界 r，默认 0.5")
    parser.add_argument("--terminal-epsilon", type=float, default=0, help="在 t=1-epsilon 停止，默认 0 精确复制")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=8, help="分阶段模式的总步数 N=N1+N2；默认各分配 4 步")
    parser.add_argument("--guidance-scale", type=float, default=0)
    parser.add_argument("--model-path", default=settings.model_path)
    parser.add_argument("--output-dir", type=Path, default=settings.output_dir)
    parser.add_argument("--step-dir", type=Path, default=settings.step_dir)
    parser.add_argument("--save-steps", action="store_true")
    parser.add_argument("--analyze-velocity", action="store_true")
    parser.add_argument("--attention-backend", default=settings.attention)
    parser.add_argument("--compile", action="store_true", default=settings.compile)
    args = parser.parse_args()
    config = DualNoiseConfig(mode=args.mode, seed_b=args.seed_b, weight_b=args.weight_b,
                             mix_step=args.mix_step, normalize_initial=not args.raw_initial_mix,
                             velocity_rule=args.velocity_rule, stage1_steps=args.stage1_steps,
                             stage_split=args.stage_split, terminal_epsilon=args.terminal_epsilon)
    settings = replace(settings, model_path=args.model_path, attention=args.attention_backend,
                       compile=args.compile, output_dir=args.output_dir.expanduser().resolve(),
                       step_dir=args.step_dir.expanduser().resolve())
    request = GenerationRequest(prompt=args.prompt, height=args.height, width=args.width,
                                steps=args.steps, guidance_scale=args.guidance_scale, seed=args.seed_a,
                                save_steps=args.save_steps, negative_prompt=args.negative_prompt,
                                analyze_velocity=args.analyze_velocity)
    result = GenerationService(settings).generate(request, kind="dual_noise", dual_noise=config)
    print(result["status"])
    print("参数与轨迹记录：", result["metadata"])
    if result.get("step_video"):
        print("过程视频：", result["step_video"])


if __name__ == "__main__":
    main()
