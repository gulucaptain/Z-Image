# 双噪声分阶段 Flow Matching 实验

本入口按 [双噪声分阶段方案](双噪声分阶段Flow_Matching方案.md) 修正，依次更新两个分量：

```text
(X₀,Y₀) → (Q̂,Y₀) → (Q̂,Q̂)
第一阶段：更新 X，冻结 Y
第二阶段：冻结 X，更新 Y
最终输出：Y
```

当前可运行的是原始方案的**生成后复制基准**。第一阶段复用已有 Z-Image / Turbo 单图速度场；第二阶段使用文档中的解析复制速度。它没有训练新的联合速度模型，也没有实现结构条件下的新细节生成。

原来的线性混合不再是 Gradio 的实验选项；旧 CLI 模式 `initial`、`latent`、`velocity` 仍可显式选择以复现实验。旧模式的混合权重、混合步和速度规则不影响新的 `staged_copy` 模式。

## 1. 两阶段的计算规则

联合时间分界为 r，默认 r=0.5。顶部采样步数为总步数 N，阶段一分配 N₁ 步，阶段二分配 N₂=N−N₁ 步，两者都至少为 1。

### 第一阶段：模型生成 X，Y 保持初始噪声

X 从 Seed X 产生的噪声出发，完成一套从 sigma=1 到 sigma=0 的普通推理。Y 从 Seed Y 独立采样，在第一阶段保持不变。

模型的局部时间 u=1−sigma，映射到联合时间 t=r·u。原采样器给出 v_sigma=dX/dsigma，对应联合速度为：

```text
v_X_joint = −v_sigma / r
v_Y_joint = 0
```

r=0.5 时缩放因子为 2，与半时间区间相符。代码根据实际 sigma 映射联合时间；模型的 sigma shift 仍然生效，模型步不一定在联合时间上等长。

已有权重得到的是生成样本 Q̂，不是已知真实 Q。真实 Q 仅在训练目标构造时使用。

### 第二阶段：解析速度将 Y 拉向固定 X

固定 X=Q̂，使用：

```text
v_Y_joint = (X − Y) / (1 − t)
v_X_joint = 0
Y_next = Y + (t_next − t) · v_Y_joint
```

第二阶段不调用 transformer，不根据 prompt 生成新细节，也不会自动修正第一阶段的错误。

默认 epsilon=0，最终 X=Y。代码只在 t<1 计算速度，在最后一步完成精确复制，不在奇异点 t=1 求值。因此固定 Seed X、改变 Seed Y，最终图像应一致，中间复制轨迹可以不同。

epsilon>0 时在 t_end=1−epsilon 提前停止，要求 0≤epsilon<1−r。解析终点为：

```text
Y_end = X + [epsilon / (1 − r)] · (Y₀ − X)
```

提前停止保留的是复制残差，不代表第二个噪声产生了模型学习的细节。

## 2. Gradio 操作

1. 在顶部选择模型并应用配置，设置分辨率、总步数 N、CFG 和 Seed（即 Seed X）。
2. 打开「双噪声实验」，填写第一阶段 prompt 和负向提示词。
3. 设置 Seed Y、阶段一步数 N₁、时间分界 r 和终点 epsilon。
4. 开启顶部逐步保存，可查看左右并排的 X/Y 联合状态：左 X、右 Y，图下标注阶段及活动/冻结分量。
5. 点击「运行分阶段复制基准」，查看阶段一 X、最终 Y、过程视频和冻结检查表。
6. 下载图像、视频与 metadata；如需有效速度热力图，开启本页的统计开关。

默认总步数 8、N₁=4，表示 **4 步模型生成 + 4 步复制**，不等于 8 步模型生成。对 Turbo 保留 8 步生成预算时，可设置总步数 12、N₁=8。基础 Z-Image 示例为总步数 34、N₁=30、CFG 4。

页面的「互补任务 (C,Q) 的训练方案」说明后续研究方向；没有对应条件权重时不会提供伪运行入口。

## 3. 脚本示例

在项目根目录运行；未激活虚拟环境时，将 python 替换为 `.venv/bin/python`。

```bash
# Turbo：8 步生成 + 4 步复制，终点 X=Y
python scripts/dual_noise_inference.py \
  --prompt "Mona Lisa" --mode staged_copy \
  --seed-a 42 --seed-b 43 --steps 12 --stage1-steps 8 --save-steps

# 提前停止，并导出有效速度统计
python scripts/dual_noise_inference.py \
  --prompt "Mona Lisa" --steps 12 --stage1-steps 8 \
  --stage-split 0.5 --terminal-epsilon 0.01 --save-steps --analyze-velocity

# 基础 Z-Image：30 步生成 + 4 步复制
python scripts/dual_noise_inference.py \
  --prompt "Mona Lisa" --model-path /data/haoyuzhao/models/Z-Image \
  --guidance-scale 4 --steps 34 --stage1-steps 30 --save-steps
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--mode` | `staged_copy` | 当前分阶段复制基准 |
| `--seed-a` / `--seed-b` | 42 / 43 | X / Y 的初始噪声 seed，范围 [0,2^63) |
| `--steps` | 8 | 总步数 N=N₁+N₂ |
| `--stage1-steps` | 未指定时 N//2 | 模型生成步数；其余为解析复制步数 |
| `--stage-split` | 0.5 | 联合时间分界 r，满足 0<r<1 |
| `--terminal-epsilon` | 0 | 在 t=1−epsilon 停止；0 表示精确复制 |
| `--prompt` / `--negative-prompt` | prompt 必填 / 负向为空 | 第一阶段模型的提示词 |
| `--height` / `--width` | 720 / 1280 | 沿用模型的尺寸要求 |
| `--guidance-scale` | 0 | 只影响第一阶段，第二阶段不使用 CFG |
| `--save-steps` | 关闭 | 保存并排 X/Y 状态并合成视频 |
| `--analyze-velocity` | 关闭 | 导出有效速度统计与热力图 |
| `--model-path` | Settings 配置 | 默认本地 Turbo，可通过 ZIMAGE_MODEL_PATH 覆盖 |
| `--output-dir` / `--step-dir` | Settings 配置 | 默认分别为 outputs / outputs2 |
| `--attention-backend` / `--compile` | Settings 配置 | 第一阶段的模型推理选项 |

## 4. 输出与阶段记录

```text
outputs/dual_noise/<任务编号>/
  stage1_X.png       # 第一阶段结果，第二阶段固定
  final.png          # 最终 Y；epsilon=0 时与阶段一图像相同
  final_XY.png       # 最终 X/Y 并排图
  metadata.json
  velocity/          # 开启统计时生成

outputs2/<任务编号>/ # 开启逐步保存时生成
  step_001.png       # 左 X / 右 Y，含阶段标注
  ...
  steps.mp4          # 按运行顺序播放，附 Step 编号
```

关闭逐步保存时仍保存阶段终点图像。视频需要系统 ffmpeg，编码失败时保留图片并显示原因。

metadata 记录两阶段步数、两个 seed 及 `joint_model_trained=false`。`dual_noise_trace` 包含：

| 字段 | 用途 |
| --- | --- |
| `stage` / `active` / `frozen` | 阶段、活动与冻结分量 |
| `joint_time` / `joint_time_next` | 联合时间更新区间 |
| `joint_velocity_rms` | 活动分量的联合速度 RMS |
| `x_rms` / `y_rms` | 两个分量的 latent RMS |
| `xy_distance_rms` | X/Y 距离，复制阶段应逐渐减小 |
| `inactive_update_rms` | 冻结分量更新 RMS，按构造为 0 |
| `model_evaluations` | 累计 transformer 调用，阶段二不增加 |
| `terminal_copy` | 第二阶段最后一步是否精确复制 |

联合 trace 使用 dz/dt；`velocity/` 统计沿用局部 dz/dsigma。第二阶段的局部 sigma=(1−t)/(1−r)，对应速度为 `−(1−r)·v_Y_joint`。两个时间参数下的速度 RMS 不能直接当成同一数值。

总步数中只有 N₁ 步调用模型。比较质量和算力应使用相同 **N₁ 次模型调用** 的普通推理作基线，VAE 解码、统计和视频编码耗时另计。

## 5. 训练目标与互补任务的实现范围

[src/zimage/staged_flow.py](src/zimage/staged_flow.py) 中的 `build_staged_fm_targets` 按文档构造联合状态和带阶段掩码的 FM 目标：

```text
阶段一：X_t=(1−t/r)X₀+(t/r)C，Y_t=Y₀
         U_X=(C−X₀)/r，U_Y=0
阶段二：X_t=C，Y_t=(1−s)Y₀+sQ，s=(t−r)/(1−r)
         U_X=0，U_Y=(Q−Y₀)/(1−r)
```

未传 condition 时 C=Q，对应原始 (Q,Q) 路径；传入配对 condition=C 时对应互补 (C,Q) 路径。默认 r=0.5 时目标速度因子为 2；t=r 归入第二阶段，状态连续、速度允许切换。

该函数只构造训练样本与目标。完整互补任务还需要定义 C=A(Q)、提供配对训练数据、增加读取 C 的条件速度模型并训练相应权重。真实 C 与生成 Ĉ 的条件偏差也需要评估。现有文生图权重没有该条件接口，本次没有将线性混合当作条件生成实现。

## 6. 建议检查流程

- 第一阶段所有 Y 保持不变，第二阶段所有 X 保持不变。
- epsilon=0 时最终 X=Y，固定 Seed X、改变 Seed Y，最终图像保持一致。
- epsilon>0 时结果符合解析残差公式；残差不能解释为已学习的新细节。
- 固定模型、CFG、prompt 和 N₁，与普通推理比较，检查复制阶段是否只是增加开销。
- 训练互补模型后，再比较普通 FM、复制方案和互补方案，评估固定 C 改变 Seed Y 的细节多样性，以及真实/生成条件下的结果差距。

```bash
python -m unittest discover -s tests -p 'test_staged_flow.py' -v
```

测试覆盖冻结、seed 不变性、解析终点、提前停止、训练目标的阶段掩码和系数，以及 Gradio 并排预览和下载。它们使用 CPU 测试模型，尚未证明真实权重上的图像质量收益。
