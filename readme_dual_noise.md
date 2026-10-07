# 双噪声混合实验说明

本实验让同一个 prompt 使用两个 seed 产生的初始噪声，比较三种生成结构：先混合噪声再推理、中途混合 latent，以及每步组合两条轨迹的速度。它修改的是推理过程，不涉及模型训练或权重更新。

脚本入口：[scripts/dual_noise_inference.py](scripts/dual_noise_inference.py)。网页入口：Gradio 的「双噪声实验」标签页。三种结构共用当前选择的 Z-Image 或 Z-Image-Turbo 权重、提示词、负向提示词、CFG 和采样时间表。

## 1. 参数和更新约定

记初始噪声为 `εA`、`εB`，B 的权重为 `α`，混合函数为：

```text
mix(A, B) = (1 − α) A + α B
```

`α=0` 取 A，`α=1` 取 B。默认 `α=0.5`。

当前采样器使用 Euler 更新，sigma 从高到低递减：

```text
Δσᵢ = σᵢ₊₁ − σᵢ       # 通常为负数
zᵢ₊₁ = zᵢ + Δσᵢ · v(zᵢ, σᵢ)
```

本文中的 `v` 是代码完成符号转换及 CFG 后的 `dz/dsigma`，不能直接替换为 transformer 原始输出。两分支分别执行现有的 CFG 逻辑，再进行速度混合。

Gradio 顶部 Seed 是 **Seed A**；双噪声页面的 Seed B 控制另一个噪声。默认分别为 42 和 43。两者相同时生成相同初始噪声，可以作为检查基线。

## 2. 三种结构

### 模式 1：初始噪声混合 `initial`

先构造一个新的初始 latent，再执行普通单轨迹推理：

```text
z₀ = (1 − α) εA + α εB
zᵢ₊₁ = zᵢ + Δσᵢ · v(zᵢ, σᵢ)
```

默认开启「初始噪声混合后补偿方差」。对两个独立的标准高斯噪声，原始线性混合的方差为 `(1−α)²+α²`，因此补偿后的初始值为：

```text
z₀ = [(1 − α) εA + α εB] / sqrt((1 − α)² + α²)
```

当 `α=0.5` 时，原始混合方差为 0.5，补偿相当于除以 `sqrt(0.5)`。补偿恢复初始噪声的尺度；它不保证单张图像更好。对于独立高斯噪声，补偿后的混合仍具有标准高斯分布，因此仅凭使用两个 seed 不能推断生成质量会提升。

Seed A/B 相同时，代码不做该补偿，避免放大相同噪声。CLI 添加 `--raw-initial-mix` 可关闭补偿。该开关只影响模式 1。

### 模式 2：中途 latent 混合 `latent`

A/B 先各自普通推理；**完成第 k 个更新后**混合，再继续单轨迹普通推理：

```text
A₀ = εA, B₀ = εB

前 k 步：
    Aᵢ₊₁ = Aᵢ + Δσᵢ · v(Aᵢ, σᵢ)
    Bᵢ₊₁ = Bᵢ + Δσᵢ · v(Bᵢ, σᵢ)

完成第 k 步后：
    zₖ = (1 − α) Aₖ + α Bₖ

剩余步骤：
    zᵢ₊₁ = zᵢ + Δσᵢ · v(zᵢ, σᵢ)
```

`--mix-step 4` 表示两条分支各自完成 4 步后混合。有效范围是 `1 ≤ k ≤ steps`；如果 `k=steps`，最终结果就是两条完成轨迹的 latent 混合，不再进行后续模型计算。

中间 latent **不做方差补偿**。已经推理过的 latent 不是独立标准高斯噪声，不能沿用模式 1 的方差公式。

逐步预览在混合前展示 A；第 k 步展示混合结果；随后展示混合后的单轨迹。B 不单独导出图片。混合会导致输出相对 A 的轨迹发生跳变，可能改变构图、细节或产生伪影，需要通过实际图像判断。

### 模式 3：双轨迹速度组合 `velocity`

输出轨迹从未经方差补偿的线性混合开始。每步在 A/B 两个位置分别计算速度，然后用组合速度更新输出：

```text
A₀ = εA, B₀ = εB
z₀ = (1 − α) A₀ + α B₀

每步：
    vA = v(Aᵢ, σᵢ)
    vB = v(Bᵢ, σᵢ)
    vMix = (1 − α) vA + α vB
    zᵢ₊₁ = zᵢ + Δσᵢ · vMix
```

模型在 A/B 位置计算速度；不会再额外调用一次模型计算 `v(zᵢ, σᵢ)`。A/B 的推进方式有两个选项：

| 规则 | A/B 更新方式 | 实验含义 |
| --- | --- | --- |
| `coupled`，默认 | A/B 都使用 `vMix` 更新 | 混合速度反馈到两分支，改变它们后续的采样轨迹 |
| `independent` | A 用 `vA`，B 用 `vB` 更新 | 保留两条普通轨迹，输出累计它们的加权速度 |

**独立规则的等价关系**：固定 α、相同时间表、Euler 更新及上述线性初始值下，始终有：

```text
zᵢ = (1 − α) Aᵢ + α Bᵢ
```

因此 `independent` 的最终输出等价于两个普通最终 latent 的线性混合（浮点误差除外），不构成额外的非线性能力。它也与模式 2 的 `k=steps` 在数学上等价；与先解码两张图、再混合 RGB 像素通常不同。

**耦合规则的性质**：A/B 每步添加同一个增量，所以分支差 `Aᵢ−Bᵢ` 保持不变；输出仍等于这两条耦合分支的加权 latent，但分支已不再是普通独立推理轨迹。在非线性速度场下，该规则可以与初始噪声混合、独立速度组合产生不同结果，是否改善图像质量需要另行评估。

## 3. Gradio 操作

1. 运行 `bash gradio_inference.sh`，进入工作台。
2. 在顶部选择模型，点击「应用配置」，设置分辨率、步数、Guidance Scale 和 Seed A。
3. 打开「双噪声实验」，填写 prompt、Seed B 和 B 的权重。
4. 选择结构；模式 2 设置混合步 k，模式 3 选择速度规则。仅模式 1 使用方差补偿开关。
5. 如需中间图像和过程视频，开启顶部「逐步预览并保存到 outputs2」。
6. 点击「运行双噪声实验」。生成时查看实时预览，完成后播放带 Step 标注的视频，并下载结果与参数记录。

页面里的不适用参数不影响所选模式。例如，模式 1 忽略混合步 k，模式 2 忽略速度规则。

仓库已有模型使用说明给出的参数范围：Turbo 常用 8 步、Guidance Scale 0；Z-Image 使用 28–50 步、Guidance Scale 3–5。比较实验时应固定同一模型及同一组采样参数。

## 4. 命令行用法

以下命令从项目根目录执行。将 `python` 替换为项目虚拟环境的解释器，例如 `.venv/bin/python`，即可使用同一套依赖。

```bash
# 1. 初始混合，默认补偿方差
python scripts/dual_noise_inference.py \
  --prompt "Mona Lisa" --mode initial \
  --seed-a 42 --seed-b 43 --weight-b 0.5 --steps 8 --save-steps

# 2. 各自完成第 4 步后混合 latent
python scripts/dual_noise_inference.py \
  --prompt "Mona Lisa" --mode latent \
  --seed-a 42 --seed-b 43 --weight-b 0.5 --mix-step 4 --steps 8 --save-steps

# 3. 混合速度反馈到两条分支
python scripts/dual_noise_inference.py \
  --prompt "Mona Lisa" --mode velocity --velocity-rule coupled \
  --seed-a 42 --seed-b 43 --weight-b 0.5 --steps 8 --save-steps

# 对照：A/B 各自普通推进，输出累计混合速度
python scripts/dual_noise_inference.py \
  --prompt "Mona Lisa" --mode velocity --velocity-rule independent \
  --seed-a 42 --seed-b 43 --weight-b 0.5 --steps 8 --save-steps
```

切换到基础 Z-Image 时，额外设置：

```text
--model-path /data/haoyuzhao/models/Z-Image --steps 30 --guidance-scale 4
```

### 参数表

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--prompt` | 必填 | 所有分支共用的提示词 |
| `--negative-prompt` | 空字符串 | 沿用现有 CFG 逻辑；Guidance Scale > 1 时生效 |
| `--mode` | `initial` | `initial` / `latent` / `velocity` |
| `--seed-a` / `--seed-b` | 42 / 43 | 各自控制初始噪声，范围 `[0, 2^63)` |
| `--weight-b` | 0.5 | B 的权重，范围 `[0, 1]` |
| `--mix-step` | 4 | 仅模式 2 生效，完成第 k 步后混合 |
| `--raw-initial-mix` | 不启用 | 仅模式 1 生效，关闭方差补偿 |
| `--velocity-rule` | `coupled` | 仅模式 3 生效，也可选 `independent` |
| `--height` / `--width` | 720 / 1280 | 宽高需要满足模型要求；当前服务要求为 16 的倍数 |
| `--steps` | 8 | Euler 采样步数 |
| `--guidance-scale` | 0 | 分支各自执行相同 CFG，再混合速度 |
| `--model-path` | 当前 Settings 配置 | 默认本地 Turbo 路径，可通过 `ZIMAGE_MODEL_PATH` 覆盖 |
| `--save-steps` | 不启用 | 保存中间结果，并自动合成过程视频 |
| `--analyze-velocity` | 不启用 | 额外导出有效速度与更新统计、热力图 |
| `--output-dir` / `--step-dir` | 当前 Settings 配置 | 默认分别为项目 `outputs` / `outputs2` |
| `--attention-backend` | 当前 Settings 配置 | 默认 `native` |
| `--compile` | 当前 Settings 配置 | 启用 torch.compile，默认关闭 |

## 5. 输出文件和轨迹记录

默认目录结构如下，每次运行使用独立任务编号：

```text
outputs/dual_noise/<任务编号>/
  final.png                  # 输出轨迹的最终图像
  metadata.json              # 参数、seed、模型路径、耗时及双噪声轨迹记录
  velocity/                  # 仅 CLI 开启 --analyze-velocity 时生成
    velocity.json
    velocity.csv
    velocity_*.png / *.npy

outputs2/<任务编号>/          # 开启逐步保存时生成
  step_001.png
  step_002.png
  ...
  steps.mp4                  # 每步 0.5 秒，左上角标注 Step 当前步 / 总步数
```

视频编码依赖系统 `ffmpeg` 和 H.264 编码器。编码失败时保留图像，并在运行状态与 metadata 中记录原因。

`metadata.json` 顶层 `seed` 是 Seed A，`dual_noise.seed_b` 是 Seed B。`dual_noise` 保存模式、权重、混合步、方差补偿开关及速度规则。`dual_noise_trace` 逐步保存以下字段：

| 字段 | 含义 |
| --- | --- |
| `step` / `mode` | 从 1 开始的更新编号及实验结构 |
| `merged_now` | 是否在当前步执行中途 latent 混合 |
| `merge_jump_rms` | 模式 2 中，混合结果相对完成当前步的 A latent 的跳变 RMS；其余步为 0 |
| `model_evaluations` | 截至当前步累计的 transformer 调用次数，不包含文本编码和 VAE 解码 |
| `delta_sigma` | 当前更新使用的 sigma 差 |
| `output_latent_rms` | 当前步输出 latent 的 RMS |
| `effective_velocity_rms` | 实际用于输出更新的有效速度 RMS |
| `branch_distance_rms` | A/B 距离 RMS；仅模式 2、3 存在 |

模式 2 混合后不再推进辅助分支，因此其 `branch_distance_rms` 保持合并时的值，不能解读为后续输出与另一条轨迹的距离。

模式 2 在混合步使用 `(混合目标 − 当前输出) / Δσ` 表示有效速度，其中包含人为跳变。该步的速度尖峰不代表模型本身预测了同样强的速度，也不能直接解读为模型预测误差。

在普通、无跳过步骤的 N 步采样中，模式 1 调用 transformer N 次，模式 2 调用 N+k 次，模式 3 调用 2N 次。CFG 在一次调用中批量处理正负条件，因此调用次数不等于全部计算成本，运行时长还受编译、逐步 VAE 解码和视频合成影响。

## 6. 如何比较实验效果

先固定模型、prompt、负向提示词、分辨率、采样步数、Guidance Scale 和 Seed A/B。用同一配置运行三种模式，以免把模型或采样参数的差异归因于混合结构。

可按下列顺序比较：

1. 用普通推理分别生成 Seed A 和 Seed B 的基线图像。
2. 模式 1 比较开启与关闭方差补偿，区分初始噪声尺度变化的影响。
3. 扫描 `α=0、0.25、0.5、0.75、1`，观察端点复现及构图、身份、文字、细节变化。
4. 模式 2 在固定 α 下扫描 k，例如 8 步采样使用 `k=1、2、4、6、8`，观察早混合与晚混合的差别。
5. 模式 3 比较 `coupled` 与 `independent`，结合过程视频、分支距离和耗时判断变化。
6. 扩展到多组 prompt 和 seed 对，再统计提示词符合度、伪影、构图与细节质量；单个样本不足以证明整体效果更好。

边界检查：α=0/1 应分别复现 A/B 的普通推理结果；Seed A=B 时三种结构应退化到同一条普通轨迹，允许浮点舍入误差。模式 3 的独立规则与模式 2 的 `k=N` 应给出等价最终 latent。

当前测试使用可计算预期结果的 CPU 非线性模型验证 Euler 更新、CFG、方差补偿、混合时机、权重端点、同 seed 退化及 Gradio 输出。测试不等于真实 Z-Image 权重的图像质量评测，也没有证明任一种混合结构优于普通推理。

```bash
python -m unittest discover -s tests -p 'test_dual_noise.py' -v
```

## 7. 实现位置

- [src/config/dual_noise.py](src/config/dual_noise.py)：实验参数和合法性检查。
- [src/zimage/dual_noise.py](src/zimage/dual_noise.py)：初始混合、latent 合并及双轨迹速度更新。
- [src/zimage/pipeline.py](src/zimage/pipeline.py)：模型速度预测、时间表与 Euler 更新接口。
- [src/zimage_app/service.py](src/zimage_app/service.py)：模型生命周期、流式预览、图像和参数保存。
- [src/zimage_app/ui.py](src/zimage_app/ui.py)：Gradio 实验入口。
- [tests/test_dual_noise.py](tests/test_dual_noise.py)：数学更新和界面输出验证。
