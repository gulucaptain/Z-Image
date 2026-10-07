<<<<<<< HEAD
<h1 align="center">⚡️- Image<br><sub><sup>An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer</sup></sub></h1>

<div align="center">

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://tongyi-mai.github.io/Z-Image-blog/)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-Z--Image-yellow)](https://huggingface.co/Tongyi-MAI/Z-Image)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-Z--Image--Turbo-yellow)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Online_Demo-Z--Image-blue)](https://huggingface.co/spaces/Tongyi-MAI/Z-Image)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Online_Demo-Z--Image--Turbo-blue)](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo)&#160;
[![ModelScope Model](https://img.shields.io/badge/🤖%20Checkpoint-Z--Image-624aff)](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image)&#160;
[![ModelScope Model](https://img.shields.io/badge/🤖%20Checkpoint-Z--Image--Turbo-624aff)](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)&#160;
[![ModelScope Space](https://img.shields.io/badge/🤖%20Online_Demo-Z--Image-17c7a7)](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=569345&modelType=Checkpoint&sdVersion=Z_IMAGE&modelUrl=modelscope%3A%2F%2FTongyi-MAI%2FZ-Image%3Frevision%3Dmaster)&#160;
[![ModelScope Space](https://img.shields.io/badge/🤖%20Online_Demo-Z--Image--Turbo-17c7a7)](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=469191&modelType=Checkpoint&sdVersion=Z_IMAGE_TURBO&modelUrl=modelscope%3A%2F%2FTongyi-MAI%2FZ-Image-Turbo%3Frevision%3Dmaster)&#160;
[![Art Gallery PDF](https://img.shields.io/badge/%F0%9F%96%BC%20Art_Gallery-PDF-ff69b4)](assets/Z-Image-Gallery.pdf)&#160;
[![Web Art Gallery](https://img.shields.io/badge/%F0%9F%8C%90%20Web_Art_Gallery-online-00bfff)](https://modelscope.cn/studios/Tongyi-MAI/Z-Image-Gallery/summary)&#160;
<a href="https://arxiv.org/abs/2511.22699" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv" height="21px"></a>


Welcome to the official repository for the Z-Image（造相）project!

</div>



## ✨ Z-Image

### Gradio 工作台：远程启动，本地访问

1. **在远程服务器的终端中**进入本项目目录并启动服务：

   ```bash
   cd /path/to/Z-Image
   bash gradio_inference.sh --host 127.0.0.1 --port 7860
   # 或使用已经安装依赖的 Python 环境
   python app.py --host 127.0.0.1 --port 7860
   ```

   将 `/path/to/Z-Image` 替换为服务器上的实际项目路径，两个启动命令任选一个。
   启动脚本优先使用当前虚拟环境，其次使用项目 `.venv`。
   新环境需先在服务器上安装依赖：`pip install -e '.[ui,video]'`；音频提取另需系统安装 FFmpeg。
   等待服务启动完成，并保持服务进程运行。

2. **在本地电脑的终端中**建立 SSH 端口转发（不要在远程服务器终端中执行）：

   ```bash
   ssh -N -L 7860:127.0.0.1:7860 用户名@服务器地址
   ```

   将用户名和服务器地址替换为实际值。如果 SSH 使用非默认端口，添加 `-p SSH端口`。
   保持此 SSH 连接开启；`-N` 表示仅转发端口，不进入远程命令行。

3. **在本地电脑的浏览器中**打开 `http://127.0.0.1:7860`。

服务器上的 `127.0.0.1` 指服务器自身，本地浏览器中的 `127.0.0.1` 指本地电脑；
SSH 转发将两者连接起来。此方式只需能通过 SSH 连接服务器，无需开放服务器的公网 7860 端口。
如果本地 7860 端口已被占用，改用
`ssh -N -L 17860:127.0.0.1:7860 用户名@服务器地址`，然后在本地访问 `http://127.0.0.1:17860`。

如果使用 VS Code Remote SSH，也可以在“端口（Ports）”面板中转发远程端口 `7860`，
然后打开该面板显示的本地地址；此时无需另外执行上述 SSH 转发命令。
如果服务本身就在本地电脑运行，启动后直接访问 `http://127.0.0.1:7860` 即可，无需 SSH 转发。

工作台提供文生图、逐步图像保存、批量生成、Qwen-VL 参考图反推重绘、视频帧/音频提取、历史与已有结果浏览。
“逐步预览并保存到 outputs2”默认开启：文生图每完成一个采样 step，就更新预览和步骤画廊，完成后显示最终图像。
逐步解码会增加生成耗时；关闭该选项后仅显示最终图像。批量生成和参考图重绘的步骤图可在下载列表或“结果与历史”的“逐步图像”目录中查看。

文生图页面的“分析 Flow Matching 速度场”默认开启，可独立于逐步图片保存使用。
生成过程中，“Flow Matching 速度场分析”面板逐步更新以下内容：

- **空间热力图**：每个 latent 位置的通道速度 RMS；色标上限固定为首个有效步骤的最大值。
  超出上限的值显示为最高颜色，饱和比例标在图下；非有限值显示为洋红色。它不是图像平面的二维运动场。
- **强度曲线**：实际送给调度器的速度 RMS 和更新 RMS；启用 CFG 时另显示条件、无条件预测 RMS。
- **方向曲线**：相邻有效步骤的速度展平后的余弦相似度；首步、零范数或非有限速度记为缺失值。
- **逐步数值与历史**：速度均值、标准差、绝对值分位数、时间、sigma、步长、更新比例和各步热力图。
  跳过模型计算的步骤标为 `skipped`，不会重复统计速度。

速度采用调度器的 `dz/dsigma` 约定（模型原始输出经过符号转换和 CFG 后），sigma 随生成递减：
`z_next = z + (sigma_next - sigma) * velocity`。
`update_rms = abs(sigma_next - sigma) * velocity_rms`；
`relative_update = update_rms / latent_rms`，分母使用更新前的 latent。
表中 `model_time` 为模型接收的 `1 - timestep / 1000`，与 sigma 分别记录。

分析文件保存在 `outputs/single/<任务编号>/velocity/`，并加入当前任务的下载列表：
`velocity.csv`、`velocity.json`、各步热力图 PNG 和原始 RMS 数值 NPY。
JSON 包含色标和速度约定，`metadata.json` 记录分析文件及统计结果。
这里只保存统计和空间 RMS，不保存完整的速度张量；这些指标反映当前采样轨迹的行为，不能直接衡量速度预测准确性。
统计、CPU 数据传输和文件写入会增加耗时，关闭分析选项可避免这些开销；批量生成和参考图重绘暂不提供此分析面板。

宽高必须是 16 的倍数。共用参数适用于三个图像生成页面；批量任务的 seed 按记录顺序递增。
模型在首次生成时加载，后续复用；模型设置、卸载和图像生成共享串行队列。
反推重绘会先卸载 Z-Image，完成 Qwen-VL 描述后再加载 Z-Image。
反推使用文字描述进行重绘，不使用参考图像 latent 作为去噪输入。

```text
app.py                         # 统一 Gradio 入口
gradio_inference.sh             # 启动脚本
src/zimage_app/
  settings.py                  # 路径和环境配置
  service.py                   # 模型管理、图像生成、输出参数与历史
  inputs.py                    # TXT / JSON 提示词解析和筛选
  workflows.py                 # 批量生成、反推重绘、视频工具适配
  ui.py                        # Gradio 页面与事件
  cli.py                       # 共用命令行生成入口
src/zimage/                    # 原有模型与采样流程
scripts/generate.py            # 新命令行入口
scripts/legacy/                # 归档的独立推理、数据工具和启动脚本
tests/test_app.py              # 无需加载模型的功能测试
outputs/<类型>/<任务编号>/      # final.png 和 metadata.json
outputs2/<任务编号>/            # step_001.png、step_002.png 等
```

每次生成使用独立目录，`metadata.json` 记录提示词、seed、模型配置、耗时和任务状态。
开启逐步保存后，生成完成时会按 step 顺序自动合成 `outputs2/<任务编号>/steps.mp4`，
每步播放 0.5 秒，每帧左上角标注 `Step 当前步 / 总步数`。Gradio 的“文生图 / 逐步保存”页面提供视频播放器和下载入口；
`scripts/image_generation_step_inference.py` 也会生成该视频并打印路径。
视频编码使用系统 `ffmpeg`（H.264）；合成失败时保留最终图和全部 step 图像，并在状态中显示原因。
“同一提示词重复生成”页面支持同一 prompt 运行 n 次（默认 100 次），使用顶部共用参数。
Seed 支持从顶部 Seed 顺序递增或逐次随机生成不同的值。
每次任务保存到 `outputs2/repeat_<任务编号>/`，每个 case 位于 `cases/case_0001/` 等独立目录，
包含 `final.png`、`metadata.json`，开启逐步保存时还包含 `steps/`。
`manifest.json` 记录完整 seed 清单和运行顺序；`cases.mp4` 按运行顺序播放最终图像并标注 Case 编号。
页面可播放视频并下载全部结果的 ZIP（保存在任务目录旁）；中途失败时也会打包已有结果。
命令行入口：

```bash
python scripts/repeat_generation.py --prompt "Mona Lisa" --count 100 --seed-mode increment --seed 42
python scripts/repeat_generation.py --prompt "Mona Lisa" --count 100 --seed-mode random --fps 2
# 需要保存每个 case 的中间步骤时添加 --save-steps
```

“双噪声实验”页面支持三个结构：噪声 A 使用顶部 Seed，噪声 B 使用独立 Seed B，
α 是 B 的权重。两条分支共用提示词、CFG 和 sigma 时间表，速度采用原采样器的 `dz/dsigma` 约定。
完整操作、公式、参数及实验比较方法见 [双噪声实验说明](readme_dual_noise.md)。

- `initial`：先令 `z0=(1-α)εA+αεB`，再普通推理。默认除以 `sqrt((1-α)^2+α^2)`
  补偿独立高斯噪声混合后的方差；可关闭。同 seed 时不补偿。
- `latent`：A/B 独立完成前 k 个 Euler 更新，随后混合 latent，余下步骤只推理混合后的轨迹。
  中间 latent 不做方差补偿。该步的轨迹记录包含 `merge_jump_rms`；有效速度统计包含混合跳变除以 `delta_sigma`。
- `velocity`：每步先在 A/B 位置分别计算经过 CFG 的速度，再令 `v=(1-α)vA+αvB`，
  用 `z←z+delta_sigma*v` 更新输出，输出初始值为未经方差补偿的线性混合。
  默认 `coupled` 让 A/B 也用组合速度更新；`independent` 让 A/B 各自推进。
  在当前 Euler 更新和固定 α 下，独立规则的输出恒等于 A/B latent 的线性混合（浮点误差除外）；
  耦合规则会改变分支轨迹，但并不保证改善生成质量。

结果保存在 `outputs/dual_noise/<任务编号>/`，中间图像和标注 step 的视频仍保存在 `outputs2/<任务编号>/`。
页面可预览和下载图像、视频及 `metadata.json`；后者记录两个 seed、混合参数、逐步跳变、分支距离及模型调用次数。
CLI 支持相同功能：

```bash
python scripts/dual_noise_inference.py --prompt "Mona Lisa" --mode initial --seed-a 42 --seed-b 43 --weight-b 0.5 --save-steps
python scripts/dual_noise_inference.py --prompt "Mona Lisa" --mode latent --mix-step 4 --save-steps
python scripts/dual_noise_inference.py --prompt "Mona Lisa" --mode velocity --velocity-rule coupled --save-steps
# velocity 可改为 independent；initial 可添加 --raw-initial-mix；可添加 --analyze-velocity 导出有效更新统计
```

历史文生图、参考图、Qwen 重绘和 OmniBench 结果归档在 `tmp/history/`；
视频输入与 OmniBench 提示词位于 `tmp/inputs/`，视频帧和音频位于 `tmp/media/`。
可在“结果与历史”中选择目录浏览、下载；新任务继续写入 `outputs/`，逐步图像写入 `outputs2/`。
模型权重 `ckpts/` 和 README 使用的展示素材 `assets/` 保留原位。

批量页面支持每行一个提示词的 TXT、JSON 字符串数组，以及已有的
`scenarios[].reference_image`、`samples[].image_prompt` 和 `prompts` 格式。
可使用内置 `tmp/inputs/scene_preview_image_prompts_800.json`，按数据集、ID、条数筛选并先预览任务。
旧 OmniBench 脚本中的重试、断点跳过与推荐画幅选项仍可通过归档脚本使用：

```bash
python scripts/legacy/minimax_bench_image_generation.py --datasets VDR --dry-run --limit 5
python scripts/generate.py --prompt "Mona Lisa" --steps 8 --save-steps
# 保留的两个兼容入口也调用同一推理服务
python scripts/image_generation_inference.py --prompt "Mona Lisa"
python scripts/image_generation_step_inference.py --steps 8
python -m unittest discover -s tests -v
```

可设置 `ZIMAGE_MODEL_PATH`、`ZIMAGE_QWEN_PATH`、`ZIMAGE_ATTENTION`、`ZIMAGE_COMPILE`、
`ZIMAGE_OUTPUT_DIR`、`ZIMAGE_TMP_DIR`。默认使用已有本地 Z-Image-Turbo / Qwen3-VL 模型路径，
Attention 默认 `native`；可在“模型设置”页面选择后端、启用编译或卸载模型。
远程服务器运行、本地电脑访问的连接方式见上面的 SSH 端口转发步骤。

Z-Image is a powerful and highly efficient image generation model family with **6B** parameters. Currently there are four variants:

- 🚀 **Z-Image-Turbo** – A distilled version of Z-Image that matches or exceeds leading competitors with only **8 NFEs** (Number of Function Evaluations). It offers **⚡️sub-second inference latency⚡️** on enterprise-grade H800 GPUs and fits comfortably within **16G VRAM consumer devices**. It excels in photorealistic image generation, bilingual text rendering (English & Chinese), and robust instruction adherence.

- 🎨 **Z-Image** – The foundation model behind Z-Image-Turbo. Z-Image focuses on **high-quality generation**, **rich aesthetics**, **strong diversity**, and **controllability**, well-suited for creative generation, **fine-tuning**, and downstream development. It supports a wide range of artistic styles, effective negative prompting, and high diversity across identities, poses, compositions, and layouts.

- 🧱 **Z-Image-Omni-Base** – The versatile foundation model capable of both **generation and editing tasks**. By releasing this checkpoint, we aim to unlock the full potential for community-driven fine-tuning and custom development, providing the most "raw" and diverse starting point for the open-source community.

- ✍️ **Z-Image-Edit** – A variant fine-tuned on Z-Image specifically for image editing tasks. It supports creative image-to-image generation with impressive instruction-following capabilities, allowing for precise edits based on natural language prompts.

### 📣 News

*   **[2026-01-27]** 🔥 **Z-Image is released!** We have released the model checkpoint on [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image) and [ModelScope](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image). Try our [online demo](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=569345&modelType=Checkpoint&sdVersion=Z_IMAGE&modelUrl=modelscope%3A%2F%2FTongyi-MAI%2FZ-Image%3Frevision%3Dmaster)!
*   **[2025-12-08]** 🏆 Z-Image-Turbo ranked 8th overall on the **Artificial Analysis Text-to-Image Leaderboard**, making it the 🥇 <strong style="color: #FFC300;">#1 open-source model</strong>! [Check out the full leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image).
*   **[2025-12-01]** 🎉 Our technical report for Z-Image is now available on [arXiv](https://arxiv.org/abs/2511.22699).
*   **[2025-11-26]** 🔥 **Z-Image-Turbo is released!** We have released the model checkpoint on [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and [ModelScope](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo). Try our [online demo](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo)!

### 📥 Model Zoo

| Model | Pre-Training | SFT | RL | Step | CFG | Task | Visual Quality | Diversity | Fine-Tunability | Hugging Face | ModelScope |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Z-Image-Omni-Base** | ✅ | ❌ | ❌ | 50 | ✅ | Gen. / Editing | Medium | High | Easy | *To be released* | *To be released* |
| **Z-Image** | ✅ | ✅ | ❌ | 50 | ✅ | Gen. | High | Medium | Easy | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint%20-Z--Image-yellow)](https://huggingface.co/Tongyi-MAI/Z-Image) <br> [![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Z--Image-blue)](https://huggingface.co/spaces/Tongyi-MAI/Z-Image) | [![ModelScope Model](https://img.shields.io/badge/🤖%20%20Checkpoint-Z--Image-624aff)](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image) <br> [![ModelScope Space](https://img.shields.io/badge/%F0%9F%A4%96%20Demo-Z--Image-17c7a7)](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=569345&modelType=Checkpoint&sdVersion=Z_IMAGE&modelUrl=modelscope%3A%2F%2FTongyi-MAI%2FZ-Image%3Frevision%3Dmaster) |
| **Z-Image-Turbo** | ✅ | ✅ | ✅ | 8 | ❌ | Gen. | Very High | Low | N/A | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint%20-Z--Image--Turbo-yellow)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) <br> [![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Z--Image--Turbo-blue)](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo) | [![ModelScope Model](https://img.shields.io/badge/🤖%20%20Checkpoint-Z--Image--Turbo-624aff)](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) <br> [![ModelScope Space](https://img.shields.io/badge/%F0%9F%A4%96%20Demo-Z--Image--Turbo-17c7a7)](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=469191&modelType=Checkpoint&sdVersion=Z_IMAGE_TURBO&modelUrl=modelscope%3A%2F%2FTongyi-MAI%2FZ-Image-Turbo%3Frevision%3Dmaster) |
| **Z-Image-Edit** | ✅ | ✅ | ❌ | 50 | ✅ | Editing | High | Medium | Easy | *To be released* | *To be released* |

The figure below illustrates at which training stage each model is produced.

![Training Pipeline of Z-Image](assets/training_pipeline.jpg)

### 🖼️ Showcase

📸 **Photorealistic Quality**: **Z-Image-Turbo** delivers strong photorealistic image generation while maintaining excellent aesthetic quality.

![Showcase of Z-Image on Photo-realistic image Generation](assets/showcase_realistic.png)

📖 **Accurate Bilingual Text Rendering**: **Z-Image-Turbo** excels at accurately rendering complex Chinese and English text.

![Showcase of Z-Image on Bilingual Text Rendering](assets/showcase_rendering.png)

💡  **Prompt Enhancing & Reasoning**: Prompt Enhancer empowers the model with reasoning capabilities, enabling it to transcend surface-level descriptions and tap into underlying world knowledge.

![reasoning.jpg](assets/reasoning.png)

🧠 **Creative Image Editing**: **Z-Image-Edit** shows a strong understanding of bilingual editing instructions, enabling imaginative and flexible image transformations.

![Showcase of Z-Image-Edit on Image Editing](assets/showcase_editing.png)

### 🏗️ Model Architecture
We adopt a **Scalable Single-Stream DiT** (S3-DiT) architecture. In this setup, text, visual semantic tokens, and image VAE tokens are concatenated at the sequence level to serve as a unified input stream, maximizing parameter efficiency compared to dual-stream approaches.

![Architecture of Z-Image and Z-Image-Edit](assets/architecture.webp)

### 📈 Performance

Z-Image-Turbo's performance has been validated on multiple independent benchmarks, where it consistently demonstrates state-of-the-art results, especially as the leading open-source model.

#### Artificial Analysis Text-to-Image Leaderboard
On the highly competitive [Artificial Analysis Leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image), Z-Image-Turbo ranked **8th overall** and secured the top position as the 🥇 <strong style="color: gold;">#1 Open-Source Model</strong>, outperforming all other open-source alternatives.


<p align="center">
  <a href="https://artificialanalysis.ai/image/leaderboard/text-to-image">
    <img src="assets/image_arena_all.jpg" alt="Z-Image Rank on Artificial Analysis Leaderboard"/><br />
    <span style="font-size:1.05em; cursor:pointer; text-decoration:underline;"> Artificial Analysis Leaderboard</span>
  </a>
</p>

<p align="center">
  <a href="https://artificialanalysis.ai/image/leaderboard/text-to-image">
    <img src="assets/image_arena_os.jpg" alt="Z-Image Rank on Artificial Analysis Leaderboard (Open-Source Model Only)"/><br />
    <span style="font-size:1.05em; cursor:pointer; text-decoration:underline;"> Artificial Analysis Leaderboard (Open-Source Model Only)</span>
  </a>
</p>

#### Alibaba AI Arena Text-to-Image Leaderboard
According to the Elo-based Human Preference Evaluation on [*Alibaba AI Arena*](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=T2I), Z-Image-Turbo also achieves state-of-the-art results among open-source models and shows highly competitive performance against leading proprietary models.

<p align="center">
  <a href="https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=T2I">
    <img src="assets/leaderboard.png" alt="Z-Image Elo Rating on AI Arena"/><br />
    <span style="font-size:1.05em; cursor:pointer; text-decoration:underline;"> Alibaba AI Arena Text-to-Image Leaderboard</span>
  </a>
</p>


### 🚀 Quick Start
#### (1) PyTorch Native Inference
Build a virtual environment you like and then install the dependencies:
```bash
pip install -e .
```
Then run the following code to generate an image:
```bash
python scripts/generate.py
```

#### (2) Diffusers Inference
Install the latest version of diffusers, use the following command:
<details>
  <summary>Click here for details for why you need to install diffusers from source</summary>

  We have submitted two pull requests ([#12703](https://github.com/huggingface/diffusers/pull/12703) and [#12715](https://github.com/huggingface/diffusers/pull/12715)) to the 🤗 diffusers repository to add support for Z-Image. Both PRs have been merged into the latest official diffusers release.
  Therefore, you need to install diffusers from source for the latest features and Z-Image support.

</details>

```bash
pip install git+https://github.com/huggingface/diffusers
```

<details>
<summary><b>Z-Image-Turbo</b> - Click to expand</summary>

Then, try the following code to generate an image:
```python
import torch
from diffusers import ZImagePipeline

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

# 2. Generate Image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("example.png")
```

</details>

<details>
<summary><b>Z-Image</b> - Click to expand</summary>

Recommended Parameters:
- **Resolution:** 512×512 to 2048×2048 (total pixel area, any aspect ratio)
- **Guidance scale:** 3.0 – 5.0
- **Inference steps:** 28 – 50
- **Negative prompts:** Strongly recommended for better control
- **CFG normalization:** `False` for general stylism, `True` for realism

Then, try the following code to generate an image:
```python
import torch
from diffusers import ZImagePipeline

# Load the pipeline
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# Generate image
prompt = "两名年轻亚裔女性紧密站在一起，背景为朴素的灰色纹理墙面，可能是室内地毯地面。左侧女性留着长卷发，身穿藏青色毛衣，左袖有奶油色褶皱装饰，内搭白色立领衬衫，下身白色裤子；佩戴小巧金色耳钉，双臂交叉于背后。右侧女性留直肩长发，身穿奶油色卫衣，胸前印有"Tun the tables"字样，下方为"New ideas"，搭配白色裤子；佩戴银色小环耳环，双臂交叉于胸前。两人均面带微笑直视镜头。照片，自然光照明，柔和阴影，以藏青、奶油白为主的中性色调，休闲时尚摄影，中等景深，面部和上半身对焦清晰，姿态放松，表情友好，室内环境，地毯地面，纯色背景。"
negative_prompt = "" # Optional, but would be powerful when you want to remove some unwanted content

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1280,
    width=720,
    cfg_normalization=False,
    num_inference_steps=50,
    guidance_scale=4,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("example.png")
```

</details>

## 🔬 Decoupled-DMD: The Acceleration Magic Behind Z-Image

[![arXiv](https://img.shields.io/badge/arXiv-2511.22677-b31b1b.svg)](https://arxiv.org/abs/2511.22677)

Decoupled-DMD is the core few-step distillation algorithm that empowers the 8-step Z-Image model.

Our core insight in Decoupled-DMD  is that the success of existing DMD (Distribution Matching Distillation) methods is the result of two independent, collaborating mechanisms:

-   **CFG Augmentation (CA)**: The primary **engine** 🚀 driving the distillation process, a factor largely overlooked in previous work.
-   **Distribution Matching (DM)**: Acts more as a **regularizer** ⚖️, ensuring the stability and quality of the generated output.

By recognizing and decoupling these two mechanisms, we were able to study and optimize them in isolation. This ultimately motivated us to develop an improved distillation process that significantly enhances the performance of few-step generation.

![Diagram of Decoupled-DMD](assets/decoupled-dmd.webp)

## 🤖 DMDR: Fusing DMD with Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2511.13649-b31b1b.svg)](https://arxiv.org/abs/2511.13649)

Building upon the strong foundation of Decoupled-DMD, our 8-step Z-Image model has already demonstrated exceptional capabilities. To achieve further improvements in terms of semantic alignment, aesthetic quality, and structural coherence—while producing images with richer high-frequency details—we present **DMDR**.

Our core insight behind DMDR is that Reinforcement Learning (RL) and Distribution Matching Distillation (DMD) can be synergistically integrated during the post-training of few-step models. We demonstrate that:

-   **RL Unlocks the Performance of DMD** 🚀
-   **DMD Effectively Regularizes RL** ⚖️

![Diagram of DMDR](assets/DMDR.webp)

## 🎉 Community Works

- [Cache-DiT](https://github.com/vipshop/cache-dit) provides inference acceleration for **Z-Image** and **Z-Image-ControlNet** via DBCache, Context Parallelism and Tensor Parallelism. It achieves nearly **4x** speedup on 4 GPUs with negligible precision loss. Please visit their [example](https://github.com/vipshop/cache-dit/blob/main/examples) for more details.
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) is a pure C++ diffusion model inference engine that supports fast and memory-efficient Z-Image inference across multiple platforms (CUDA, Vulkan, etc.). You can use stable-diffusion.cpp to generate images with Z-Image on machines with as little as **4GB** of VRAM. For more information, please refer to [How to Use Z‐Image on a GPU with Only 4GB VRAM](https://github.com/leejet/stable-diffusion.cpp/wiki/How-to-Use-Z%E2%80%90Image-on-a-GPU-with-Only-4GB-VRAM).
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) is a pure C++ diffusion model inference engine that supports fast and memory-efficient Z-Image inference across multiple platforms (CUDA, Vulkan, etc.). You can use stable-diffusion.cpp to generate images with Z-Image on machines with as little as **4GB** of VRAM. For more information, please refer to [How to Use Z‐Image on a GPU with Only 4GB VRAM](https://github.com/leejet/stable-diffusion.cpp/wiki/How-to-Use-Z%E2%80%90Image-on-a-GPU-with-Only-4GB-VRAM).
- [LeMiCa](https://github.com/UnicomAI/LeMiCa) provides a training-free, timestep-level acceleration method that conveniently speeds up Z-Image inference. For more details, see [LeMiCa4Z-Image](https://github.com/UnicomAI/LeMiCa/tree/main/LeMiCa4Z-Image).
- [ComfyUI ZImageLatent](https://github.com/HellerCommaA/ComfyUI-ZImageLatent) provdes an easy to use latent of the official Z-Image resolutions.
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) has provided more support for Z-Image, including LoRA training, full training, distillation training, and low-VRAM inference. Please refer to the [document](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md) of DiffSynth-Studio.
- [vllm-omni](https://github.com/vllm-project/vllm-omni), a framework that extends its support for omni-modality model fast inference and serving, now [supports](https://github.com/vllm-project/vllm-omni/blob/main/docs/models/supported_models.md) Z-Image.
- [SGLang-Diffusion](https://lmsys.org/blog/2025-11-07-sglang-diffusion/) brings SGLang's state-of-the-art performance to accelerate image and video generation for diffusion models, now [supporting](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/zimage_pipeline.py) Z-Image.
- [Candle](https://github.com/huggingface/candle) is a minimalist machine learning (ML) framework launched by Huggingface for Rust, which now [supports](https://github.com/huggingface/candle/pull/3261) Z-Image.
- [MeanCache](https://github.com/UnicomAI/MeanCache), a training-free inference acceleration method for Flow Matching models by China Unicom Data Science and Artificial Intelligence Research Institute. Delivers up to **3.7x** speedup for **Z-Image** generation with plug-and-play integration while preserving output quality.

## 🚀 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tongyi-MAI/Z-Image&type=date&legend=top-left)](https://www.star-history.com/#Tongyi-MAI/Z-Image&type=date&legend=top-left)


## 📜 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{team2025zimage,
  title={Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer},
  author={Z-Image Team},
  journal={arXiv preprint arXiv:2511.22699},
  year={2025}
}

@article{liu2025decoupled,
  title={Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield},
  author={Dongyang Liu and Peng Gao and David Liu and Ruoyi Du and Zhen Li and Qilong Wu and Xin Jin and Sihan Cao and Shifeng Zhang and Hongsheng Li and Steven Hoi},
  journal={arXiv preprint arXiv:2511.22677},
  year={2025}
}

@article{jiang2025distribution,
  title={Distribution Matching Distillation Meets Reinforcement Learning},
  author={Jiang, Dengyang and Liu, Dongyang and Wang, Zanyi and Wu, Qilong and Jin, Xin and Liu, David and Li, Zhen and Wang, Mengmeng and Gao, Peng and Yang, Harry},
  journal={arXiv preprint arXiv:2511.13649},
  year={2025}
}

```

## 🤝 We're Hiring!

We're actively looking for **Research Scientists**, **Engineers**, and **Interns** to work on foundational generative models and their applications. Interested candidates please send your resume to: **jingpeng.gp@alibaba-inc.com**
=======
# Z-Image
Self-repo for Z-Image deployment.
>>>>>>> github/main
