"""Gradio components and event bindings; no models loaded at import time."""
import gradio as gr

from config.dual_noise import DualNoiseConfig

from .inputs import load_records
from .diagnostics import FIELDS, plot_data
from .repeat import repeat_generate
from .service import GenerationRequest, GenerationService
from .settings import BUILTIN_CASES, INPUT_DIR, Settings, browse_directories
from .workflows import batch_generate, recreate, video_process


def request_from(prompt, height, width, steps, guidance, seed, save_steps, negative=""):
    return GenerationRequest(str(prompt), int(height), int(width), int(steps), float(guidance), int(seed), bool(save_steps), str(negative))


def build_app(settings=None, service=None):
    settings = settings or Settings()
    service = service or GenerationService(settings)
    directories = browse_directories(settings)

    def browse(name):
        directory = directories[name]
        paths = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}] if directory.exists() else []
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        selected = paths[:100]
        return [(str(p), str(p.relative_to(directory))) for p in selected], [str(p) for p in selected], f"{directory}\n共 {len(paths)} 张，显示最近 {len(selected)} 张。"

    def generate_one(prompt, negative, height, width, steps, guidance, seed, save_steps, analyze_velocity=False, progress=gr.Progress()):
        try:
            from dataclasses import replace
            request = request_from(prompt, height, width, steps, guidance, seed, save_steps, negative)
            request = replace(request, analyze_velocity=bool(analyze_velocity))
            request.validate()
            strength, direction = plot_data([])
            yield None, [], [], "加载模型 / 准备生成", None, strength, direction, [], [], None
            for result in service.generate_stream(request, progress):
                files = [p for p in [result["image"], result["metadata"], *result["steps"]] if p]
                if result.get("step_video"):
                    files.append(result["step_video"])
                analysis = result.get("diagnostics") or {"rows": [], "maps": [], "files": []}
                strength, direction = plot_data(analysis["rows"])
                files.extend(analysis["files"])
                yield (result["image"], [(p, f"Step {i + 1}") for i, p in enumerate(result["steps"])],
                       list(dict.fromkeys(files)), result["status"],
                       analysis["maps"][-1][0] if analysis["maps"] else None, strength, direction,
                       [[row.get(field) for field in FIELDS] for row in analysis["rows"]], analysis["maps"],
                       result.get("step_video"))
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def batch(file, use_builtin, text, datasets, ids, limit, append_negative, height, width, steps, guidance, seed, save_steps, progress=gr.Progress()):
        path = file or (str(BUILTIN_CASES) if use_builtin else None)
        try:
            request = request_from("batch", height, width, steps, guidance, seed, save_steps)
            request.validate()
            yield from batch_generate(service, request, path, text, datasets, ids, limit, append_negative, progress)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def preview_batch(file, use_builtin, text, datasets, ids, limit):
        try:
            path = file or (str(BUILTIN_CASES) if use_builtin else None)
            records = load_records(path, text, datasets, ids, limit)
            return [[r["id"], r.get("dataset", ""), r["prompt"]] for r in records]
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def repeat(prompt, negative, count, seed_mode, fps, height, width, steps, guidance, seed, save_steps, progress=gr.Progress()):
        try:
            request = request_from(prompt, height, width, steps, guidance, seed, save_steps, negative)
            for result in repeat_generate(service, request, count, seed_mode, float(fps), progress):
                yield result["gallery"], result["video"], result["archive"], result["status"]
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def dual(prompt, negative, mode, seed_b, weight_b, mix_step, normalize, velocity_rule,
             height, width, steps, guidance, seed, save_steps, progress=gr.Progress()):
        try:
            request = request_from(prompt, height, width, steps, guidance, seed, save_steps, negative)
            config = DualNoiseConfig(mode, int(seed_b), float(weight_b), int(mix_step), bool(normalize), velocity_rule)
            config.validate(request.steps)
            request.validate()
            yield None, [], None, [], "加载模型 / 准备双噪声生成"
            for result in service.generate_stream(request, progress, dual_noise=config):
                files = [result["image"], result["metadata"], *result["steps"], result.get("step_video")]
                yield (result["image"], [(p, f"Step {i + 1}") for i, p in enumerate(result["steps"])],
                       result.get("step_video"), [p for p in files if p], result["status"])
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def reverse(source, qwen_path, extra, height, width, steps, guidance, seed, save_steps, progress=gr.Progress()):
        try:
            request = request_from("reverse", height, width, steps, guidance, seed, save_steps)
            request.validate()
            return recreate(service, source, qwen_path, extra, request, progress)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def video(mode, directory, fps, recursive, skip, progress=gr.Progress()):
        try:
            return video_process(mode, directory, fps, recursive, skip, progress)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def configure(path, attention, compile_model):
        try:
            return service.configure(path, attention, compile_model)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="Z-Image 工作台") as demo:
        gr.Markdown("# Z-Image 图像工作台\n文生图、逐步保存、批量生成与参考图重绘。")
        with gr.Accordion("共用生成参数", open=True):
            model_path = gr.Dropdown(
                choices=[
                    ("Z-Image-Turbo — /data/haoyuzhao/models/Z-Image-Turbo", "/data/haoyuzhao/models/Z-Image-Turbo"),
                    ("Z-Image — /data/haoyuzhao/models/Z-Image", "/data/haoyuzhao/models/Z-Image"),
                ],
                value=settings.model_path,
                allow_custom_value=True,
                label="Z-Image 模型 / 路径",
                info="选择模型后点击「应用配置」，下次生成时加载对应权重；也可输入自定义路径。",
            )
            gr.Markdown("采样参数建议：Turbo 通常使用 8 步、Guidance Scale 0；"
                        "Z-Image 建议 28–50 步、Guidance Scale 3–5。")
            attention = gr.Dropdown(["native", "_native_flash", "_native_math", "flash", "_flash_3", "mps_flash"], value=settings.attention, allow_custom_value=True, label="Attention 后端")
            compile_model = gr.Checkbox(label="启用 torch.compile", value=settings.compile)
            with gr.Row():
                apply_btn = gr.Button("应用配置")
                unload_btn = gr.Button("卸载模型 / 释放显存")
            config_status = gr.Textbox(label="状态", value="模型尚未加载，首次生成时按需加载。", interactive=False)
            apply_btn.click(configure, [model_path, attention, compile_model], config_status, concurrency_id="models", concurrency_limit=1)
            unload_btn.click(service.unload, outputs=config_status, concurrency_id="models", concurrency_limit=1)
            with gr.Row():
                height = gr.Slider(256, 2048, value=720, step=16, label="高度")
                width = gr.Slider(256, 2048, value=1280, step=16, label="宽度")
                steps = gr.Slider(1, 50, value=8, step=1, label="采样步数")
            with gr.Row():
                guidance = gr.Slider(0, 10, value=0, step=0.1, label="Guidance Scale", info="Turbo 通常使用 0；大于 1 时启用负向提示词。")
                seed = gr.Number(value=42, precision=0, label="Seed")
                save_steps = gr.Checkbox(label="逐步预览并保存到 outputs2", value=True)
                gr.Markdown("文生图生成时逐步更新预览，完成后自动合成逐步视频（每步 0.5 秒）；"
                            "关闭后仅显示最终图像。逐步解码会增加生成耗时。")
        parameters = [height, width, steps, guidance, seed, save_steps]

        with gr.Tab("文生图 / 逐步保存"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="提示词", value="Mona Lisa", lines=8)
                    negative = gr.Textbox(label="负向提示词（Guidance Scale > 1 时生效）", lines=2)
                    analyze_velocity = gr.Checkbox(label="分析 Flow Matching 速度场", value=True,
                                                  info="记录当前采样轨迹上的速度、实际更新量和方向变化；增加统计和文件写入耗时。")
                    generate_btn = gr.Button("生成图像", variant="primary")
                with gr.Column():
                    image = gr.Image(label="实时预览 / 最终图像", type="filepath")
                    status = gr.Textbox(label="状态", interactive=False)
            with gr.Row():
                step_gallery = gr.Gallery(
                    label="每一步的图像", columns=3, height=320,
                    scale=2, min_width=360, object_fit="contain",
                )
                step_video = gr.Video(
                    label="逐步生成过程视频", format="mp4", interactive=False,
                    height=320, scale=1, min_width=280,
                )
            with gr.Accordion("Flow Matching 速度场分析", open=True):
                gr.Markdown("热力图显示 latent 每个位置的通道速度 RMS，不代表图像二维运动。"
                            "色标固定为首个有效步骤的最大值；超出色标的比例标在图下，原始数值保存在 NPY。"
                            "速度使用调度器的 dz/dσ 约定（σ 递减），包含 CFG 的实际作用。")
                with gr.Row():
                    velocity_heatmap = gr.Image(label="当前速度强度 / 固定色标", type="filepath")
                    strength_plot = gr.LinePlot(x="step", y="value", color="metric", height=300,
                                                x_title="采样步骤", y_title="RMS", label="速度与实际更新强度")
                direction_plot = gr.LinePlot(x="step", y="cosine", height=220, y_lim=[-1, 1],
                                             x_title="采样步骤", y_title="余弦相似度", label="相邻步骤的速度方向")
                gr.Markdown("方向相似度越接近 1，方向越一致；首步和零速度没有定义。"
                            "更新 RMS = |Δσ| × 速度 RMS，relative_update 是更新 RMS / 更新前 latent RMS。"
                            "跳过模型计算的步骤不会重复统计速度；这些指标不能直接衡量预测准确性。")
                with gr.Accordion("逐步数值与热力图历史", open=False):
                    velocity_table = gr.Dataframe(headers=FIELDS, datatype="auto", interactive=False, label="逐步统计")
                    velocity_gallery = gr.Gallery(label="速度热力图历史", columns=4)
            downloads = gr.File(label="下载图像、视频与参数", file_count="multiple")
            generate_btn.click(generate_one, [prompt, negative, *parameters, analyze_velocity],
                               [image, step_gallery, downloads, status, velocity_heatmap, strength_plot,
                                direction_plot, velocity_table, velocity_gallery, step_video],
                               concurrency_id="models", concurrency_limit=1, show_progress="minimal")

        with gr.Tab("双噪声实验"):
            gr.Markdown("噪声 A 使用顶部 Seed，噪声 B 使用下方 Seed B；两条分支使用相同提示词与采样时间表。"
                        "混合权重 α 表示 B 的占比：α=0 取 A，α=1 取 B。")
            with gr.Row():
                with gr.Column(scale=2):
                    dual_prompt = gr.Textbox(label="双噪声提示词", value="Mona Lisa", lines=4)
                    dual_negative = gr.Textbox(label="负向提示词", lines=2)
                    dual_mode = gr.Radio(
                        choices=[("1 · 初始噪声混合", "initial"), ("2 · 中途 latent 混合", "latent"),
                                 ("3 · 双轨迹速度组合", "velocity")], value="initial", label="实验结构",
                    )
                    with gr.Row():
                        seed_b = gr.Number(value=43, precision=0, label="Seed B")
                        weight_b = gr.Slider(0, 1, value=0.5, step=0.01, label="B 的混合权重 α")
                    mix_step = gr.Slider(1, 50, value=4, step=1, label="完成第 k 步后混合 latent（仅模式 2）")
                    normalize = gr.Checkbox(value=True, label="初始噪声混合后补偿方差（仅模式 1）")
                    velocity_rule = gr.Radio(
                        choices=[("混合速度同时更新 A/B（耦合）", "coupled"),
                                 ("A/B 各自推进，输出累计混合速度", "independent")],
                        value="coupled", label="速度更新规则（仅模式 3）",
                    )
                    with gr.Accordion("三种结构的具体规则", open=False):
                        gr.Markdown(
                            "**1**：先混合初始噪声，再普通推理。方差补偿将混合除以 √((1−α)²+α²)；同 seed 时不补偿。\n\n"
                            "**2**：A/B 各自完成前 k 步，混合 latent 后继续单条普通轨迹；混合时不做方差补偿。\n\n"
                            "**3**：每步在 A/B 位置各计算速度，输出用加权速度更新。耦合规则让 A/B 也使用该速度更新；"
                            "独立规则让 A/B 各用自身速度推进。输出初始值为未补偿的线性混合。"
                            "独立规则在 Euler 更新下等价于逐步混合 A/B latent，不能视为额外的非线性生成能力。"
                        )
                    dual_btn = gr.Button("运行双噪声实验", variant="primary")
                with gr.Column(scale=1, min_width=280):
                    dual_image = gr.Image(label="双噪声结果 / 实时预览", type="filepath", height=320)
                    dual_status = gr.Textbox(label="双噪声状态", lines=4, interactive=False)
            with gr.Row():
                dual_gallery = gr.Gallery(label="双噪声每步结果", columns=3, height=320, scale=2, min_width=360, object_fit="contain")
                dual_video = gr.Video(label="双噪声过程视频", height=320, scale=1, min_width=280, interactive=False)
            dual_files = gr.File(label="下载实验图像、视频与参数轨迹记录", file_count="multiple")
            dual_btn.click(
                dual, [dual_prompt, dual_negative, dual_mode, seed_b, weight_b, mix_step, normalize, velocity_rule, *parameters],
                [dual_image, dual_gallery, dual_video, dual_files, dual_status],
                concurrency_id="models", concurrency_limit=1, show_progress="minimal",
            )

        with gr.Tab("同一提示词重复生成"):
            gr.Markdown("使用顶部共用参数重复生成同一提示词。每个 case 的最终图像按运行顺序合成视频，"
                        "左上角显示 Case 编号；ZIP 包含所有 case、参数和 seed 清单。")
            with gr.Row():
                with gr.Column(scale=2):
                    repeat_prompt = gr.Textbox(label="重复生成的提示词", value="Mona Lisa", lines=5)
                    repeat_negative = gr.Textbox(label="负向提示词", lines=2)
                    with gr.Row():
                        repeat_count = gr.Number(value=100, minimum=1, precision=0, label="重复次数 n")
                        repeat_seed_mode = gr.Radio(
                            choices=[("顺序递增", "increment"), ("随机 Seed", "random")],
                            value="increment", label="Seed 模式",
                        )
                    repeat_fps = gr.Number(value=2, minimum=0.01, label="视频每秒播放 case 数")
                    gr.Markdown("顺序递增从顶部 Seed 开始；随机模式为每次生成选择不同的 seed。"
                                "开启顶部逐步保存后，每个 case 也会保存 step 图像和过程视频。")
                    repeat_btn = gr.Button("开始重复生成", variant="primary")
                with gr.Column(scale=1, min_width=280):
                    repeat_video = gr.Video(label="按运行顺序播放 case", height=320, interactive=False, format="mp4")
                    repeat_archive = gr.File(label="下载全部 case（ZIP）", interactive=False)
                    repeat_status = gr.Textbox(label="重复生成状态 / 保存目录", lines=5, interactive=False)
            repeat_gallery = gr.Gallery(label="按运行顺序查看 case", columns=4, height=320, object_fit="contain")
            repeat_btn.click(
                repeat, [repeat_prompt, repeat_negative, repeat_count, repeat_seed_mode, repeat_fps, *parameters],
                [repeat_gallery, repeat_video, repeat_archive, repeat_status],
                concurrency_id="models", concurrency_limit=1, show_progress="minimal",
            )

        with gr.Tab("批量生成"):
            with gr.Row():
                with gr.Column():
                    batch_file = gr.File(label="上传 TXT / JSON（优先使用上传文件）", file_types=[".txt", ".json"], type="filepath")
                    builtin = gr.Checkbox(label="使用内置 OmniBench 800 条提示词", value=False)
                    batch_text = gr.Textbox(label="或输入提示词，每行一条", lines=6)
                    datasets = gr.CheckboxGroup(["MSR", "ADR", "VDR", "AVIR"], label="数据集筛选（不选则全部）")
                    ids = gr.Textbox(label="指定 ID（空格或逗号分隔）")
                    limit = gr.Number(value=10, minimum=0, precision=0, label="最多生成条数（0 表示全部）")
                    append_negative = gr.Checkbox(label="将数据中的避免事项追加到提示词", value=True)
                    with gr.Row():
                        preview_btn = gr.Button("预览任务")
                        batch_btn = gr.Button("开始批量生成", variant="primary")
                with gr.Column():
                    batch_gallery = gr.Gallery(label="批量结果", columns=3)
                    batch_status = gr.Textbox(label="状态", interactive=False)
                    batch_files = gr.File(label="下载批量结果", file_count="multiple")
            preview = gr.Dataframe(headers=["ID", "数据集", "提示词"], interactive=False, wrap=True)
            batch_inputs = [batch_file, builtin, batch_text, datasets, ids, limit]
            preview_btn.click(preview_batch, batch_inputs, preview)
            batch_btn.click(batch, [*batch_inputs, append_negative, *parameters], [batch_gallery, batch_files, batch_status], concurrency_id="models", concurrency_limit=1)

        with gr.Tab("参考图反推 / 重绘"):
            gr.Markdown("先用 Qwen-VL 描述参考图，再用描述生成新图。")
            with gr.Row():
                with gr.Column():
                    source = gr.Image(label="参考图", type="filepath")
                    qwen_path = gr.Textbox(label="Qwen-VL 模型路径", value=settings.qwen_path)
                    extra = gr.Textbox(label="追加提示词（可选）", lines=3)
                    reverse_btn = gr.Button("反推并重绘", variant="primary")
                with gr.Column():
                    recreated = gr.Image(label="重绘结果", type="filepath")
                    comparison = gr.Image(label="原图与生成图对比", type="filepath")
            description = gr.Textbox(label="实际生成提示词", lines=6)
            reverse_status = gr.Textbox(label="状态", interactive=False)
            reverse_files = gr.File(label="下载结果", file_count="multiple")
            reverse_btn.click(reverse, [source, qwen_path, extra, *parameters], [recreated, comparison, description, reverse_files, reverse_status], concurrency_id="models", concurrency_limit=1)

        with gr.Tab("视频辅助工具"):
            mode = gr.Radio(["提取视频帧", "提取音频"], value="提取视频帧", label="操作")
            directory = gr.Textbox(label="本机视频目录", value=str(INPUT_DIR / "douyin_video"))
            fps = gr.Number(value=1, minimum=0.01, label="每秒提取帧数")
            recursive = gr.Checkbox(label="递归处理子目录")
            skip = gr.Checkbox(label="跳过已有结果", value=True)
            video_btn = gr.Button("开始处理")
            video_log = gr.Textbox(label="处理结果", lines=12, interactive=False)
            video_btn.click(video, [mode, directory, fps, recursive, skip], video_log, concurrency_id="video", concurrency_limit=1)

        with gr.Tab("结果与历史"):
            refresh = gr.Button("刷新历史")
            history_gallery = gr.Gallery(label="最近 50 次生成", columns=4)
            history_table = gr.Dataframe(headers=["任务", "类型", "状态", "Seed", "提示词", "参数文件"], interactive=False, wrap=True)
            refresh.click(service.history, outputs=[history_gallery, history_table])
            gr.Markdown("### 浏览已有文件")
            folder = gr.Dropdown(list(directories), value="生成结果", label="结果目录")
            browse_btn = gr.Button("浏览目录")
            file_gallery = gr.Gallery(label="目录中的图像", columns=4)
            existing_files = gr.File(label="下载已有图像", file_count="multiple")
            folder_status = gr.Textbox(label="目录信息", interactive=False)
            browse_btn.click(browse, folder, [file_gallery, existing_files, folder_status])

    return demo
