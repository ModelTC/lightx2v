import os
import gradio as gr
import asyncio
import argparse
import json
import torch
import gc
from easydict import EasyDict
from datetime import datetime
from loguru import logger
import sys
from pathlib import Path

module_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(module_path)

from lightx2v.infer import init_runner
from lightx2v.utils.envs import *

# advance_ptq
logger.add(
    "inference_logs.log",
    rotation="100 MB",
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)


global_runner = None
current_config = None


def generate_unique_filename(base_dir="./saved_videos"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{model_cls}_{timestamp}.mp4")


def run_inference(
    model_type,
    task,
    prompt,
    negative_prompt,
    image_path,
    save_video_path,
    torch_compile,
    infer_steps,
    num_frames,
    resolution,
    seed,
    sample_shift,
    enable_teacache,
    teacache_thresh,
    enable_cfg,
    cfg_scale,
    dit_quant_scheme,
    t5_quant_scheme,
    clip_quant_scheme,
    fps,
    use_tiny_vae,
    use_tiling_vae,
    lazy_load,
    precision_mode,
    use_expandable_alloc,
    cpu_offload,
    offload_granularity,
    t5_offload_granularity,
    attention_type,
    quant_op,
    rotary_chunk,
    clean_cuda_cache,
):
    global global_runner, current_config, model_path

    if os.path.exists(os.path.join(model_path, "config.json")):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            model_config = json.load(f)

    if task == "文生视频":
        task = "t2v"
    elif task == "图生视频":
        task = "i2v"

    if task == "t2v":
        if model_type == "Wan2.1 1.3B":
            # 1.3B
            coefficient = [
                [
                    -5.21862437e04,
                    9.23041404e03,
                    -5.28275948e02,
                    1.36987616e01,
                    -4.99875664e-02,
                ],
                [
                    2.39676752e03,
                    -1.31110545e03,
                    2.01331979e02,
                    -8.29855975e00,
                    1.37887774e-01,
                ],
            ]
        else:
            # 14B
            coefficient = [
                [
                    -3.03318725e05,
                    4.90537029e04,
                    -2.65530556e03,
                    5.87365115e01,
                    -3.15583525e-01,
                ],
                [
                    -5784.54975374,
                    5449.50911966,
                    -1811.16591783,
                    256.27178429,
                    -13.02252404,
                ],
            ]
    elif task == "i2v":
        if resolution in [
            "1280x720",
            "720x1280",
            "1024x1024",
            "1280x544",
            "544x1280",
            "1104x832",
            "832x1104",
            "960x960",
        ]:
            # 720p
            coefficient = [
                [
                    8.10705460e03,
                    2.13393892e03,
                    -3.72934672e02,
                    1.66203073e01,
                    -4.17769401e-02,
                ],
                [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
            ]
        else:
            # 480p
            coefficient = [
                [
                    2.57151496e05,
                    -3.54229917e04,
                    1.40286849e03,
                    -1.35890334e01,
                    1.32517977e-01,
                ],
                [
                    -3.02331670e02,
                    2.23948934e02,
                    -5.25463970e01,
                    5.87348440e00,
                    -2.01973289e-01,
                ],
            ]

    save_video_path = generate_unique_filename()

    is_dit_quant = dit_quant_scheme != "bf16"
    is_t5_quant = t5_quant_scheme != "bf16"
    if is_t5_quant:
        if t5_quant_scheme == "int8":
            t5_quant_ckpt = os.path.join(model_path, "models_t5_umt5-xxl-enc-int8.pth")
        else:
            t5_quant_ckpt = os.path.join(model_path, "models_t5_umt5-xxl-enc-fp8.pth")
    else:
        t5_quant_ckpt = None

    is_clip_quant = clip_quant_scheme != "bf16"
    if is_clip_quant:
        if clip_quant_scheme == "int8":
            clip_quant_ckpt = os.path.join(model_path, "clip-int8.pth")
        else:
            clip_quant_ckpt = os.path.join(model_path, "clip-fp8.pth")
    else:
        clip_quant_ckpt = None

    needs_reinit = lazy_load or global_runner is None or current_config is None or current_config.get("model_path") != model_path

    if torch_compile:
        os.environ["ENABLE_GRAPH_MODE"] = "true"
    else:
        os.environ["ENABLE_GRAPH_MODE"] = "false"
    if precision_mode == "bf16":
        os.environ["DTYPE"] = "BF16"
    else:
        os.environ.pop("DTYPE", None)
    if use_expandable_alloc:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    if is_dit_quant:
        if quant_op == "vllm":
            mm_type = f"W-{dit_quant_scheme}-channel-sym-A-{dit_quant_scheme}-channel-sym-dynamic-Vllm"
        elif quant_op == "sgl":
            mm_type = f"W-{dit_quant_scheme}-channel-sym-A-{dit_quant_scheme}-channel-sym-dynamic-Sgl"
        elif quant_op == "q8f":
            mm_type = f"W-{dit_quant_scheme}-channel-sym-A-{dit_quant_scheme}-channel-sym-dynamic-Q8F"
    else:
        mm_type = "Default"

    config = {
        "infer_steps": infer_steps,
        "target_video_length": num_frames,
        "target_width": int(resolution.split("x")[0]),
        "target_height": int(resolution.split("x")[1]),
        "attention_type": attention_type,
        "seed": seed,
        "enable_cfg": enable_cfg,
        "sample_guide_scale": cfg_scale,
        "sample_shift": sample_shift,
        "cpu_offload": cpu_offload,
        "offload_granularity": offload_granularity,
        "t5_offload_granularity": t5_offload_granularity,
        "dit_quantized_ckpt": model_path if is_dit_quant else None,
        "mm_config": {
            "mm_type": mm_type,
        },
        "fps": fps,
        "feature_caching": "Tea" if enable_teacache else "NoCaching",
        "coefficients": coefficient,
        "use_ret_steps": True,
        "teacache_thresh": teacache_thresh,
        "t5_quantized": is_t5_quant,
        "t5_quantized_ckpt": t5_quant_ckpt,
        "t5_quant_scheme": t5_quant_scheme,
        "clip_quantized": is_clip_quant,
        "clip_quantized_ckpt": clip_quant_ckpt,
        "clip_quant_scheme": clip_quant_scheme,
        "use_tiling_vae": use_tiling_vae,
        "tiny_vae": use_tiny_vae,
        "tiny_vae_path": (os.path.join(model_path, "taew2_1.pth") if use_tiny_vae else None),
        "lazy_load": lazy_load,
        "do_mm_calib": False,
        "parallel_attn_type": None,
        "parallel_vae": False,
        "max_area": False,
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "lora_path": None,
        "strength_model": 1.0,
        "use_prompt_enhancer": False,
        "text_len": 512,
        "rotary_chunk": rotary_chunk,
        "clean_cuda_cache": clean_cuda_cache,
    }

    args = argparse.Namespace(
        model_cls=model_cls,
        task=task,
        model_path=model_path,
        prompt_enhancer=None,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path=image_path,
        save_video_path=save_video_path,
    )

    config.update({k: v for k, v in vars(args).items()})
    config = EasyDict(config)
    config["mode"] = "infer"
    config.update(model_config)

    print(config)
    logger.info(f"使用模型: {model_path}")
    logger.info(f"推理配置:\n{json.dumps(config, indent=4, ensure_ascii=False)}")

    # 初始化或复用runner
    runner = global_runner
    if needs_reinit:
        if runner is not None:
            del runner
            torch.cuda.empty_cache()
            gc.collect()

        runner = init_runner(config)
        current_config = config

        if not lazy_load:
            global_runner = runner

    asyncio.run(runner.run_pipeline())

    if lazy_load:
        del runner
        torch.cuda.empty_cache()
        gc.collect()

    return save_video_path


def main():
    parser = argparse.ArgumentParser(description="Light Video Generation")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件夹路径")
    parser.add_argument(
        "--model_cls",
        type=str,
        choices=["wan2.1"],
        default="wan2.1",
        help="使用的模型类别",
    )
    parser.add_argument("--server_port", type=int, default=7862, help="服务器端口")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="服务器名称")
    args = parser.parse_args()

    global model_path, model_cls
    model_path = args.model_path
    model_cls = args.model_cls

    def update_model_type(task_type):
        if task_type == "图生视频":
            return gr.update(choices=["Wan2.1 14B"], value="Wan2.1 14B")
        elif task_type == "文生视频":
            return gr.update(choices=["Wan2.1 14B", "Wan2.1 1.3B"], value="Wan2.1 14B")

    def toggle_image_input(task):
        return gr.update(visible=(task == "图生视频"))

    with gr.Blocks(
        title="Lightx2v(轻量级视频推理生成引擎)",
        css="""
        .main-content { max-width: 1400px; margin: auto; }
        .output-video { max-height: 650px; }
        .warning { color: #ff6b6b; font-weight: bold; }
        .advanced-options { background: #f9f9ff; border-radius: 10px; padding: 15px; }
        .tab-button { font-size: 16px; padding: 10px 20px; }
    """,
    ) as demo:
        gr.Markdown(f"# 🎬 {model_cls} 视频生成器")
        gr.Markdown(f"### 使用模型: {model_path}")

        with gr.Tabs() as tabs:
            with gr.Tab("基础设置", id=1):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Group():
                            gr.Markdown("## 📥 输入参数")

                            with gr.Row():
                                task = gr.Dropdown(
                                    choices=["图生视频", "文生视频"],
                                    value="图生视频",
                                    label="任务类型",
                                )
                                model_type = gr.Dropdown(
                                    choices=["Wan2.1 14B"],
                                    value="Wan2.1 14B",
                                    label="模型类型",
                                )
                                task.change(
                                    fn=update_model_type,
                                    inputs=task,
                                    outputs=model_type,
                                )

                            with gr.Row():
                                image_path = gr.Image(
                                    label="输入图片",
                                    type="filepath",
                                    height=300,
                                    interactive=True,
                                    visible=True,  # Initially visible
                                )

                                task.change(
                                    fn=toggle_image_input,
                                    inputs=task,
                                    outputs=image_path,
                                )

                            with gr.Row():
                                with gr.Column():
                                    prompt = gr.Textbox(
                                        label="提示词",
                                        lines=3,
                                        placeholder="描述视频内容...",
                                        max_lines=5,
                                    )
                                with gr.Column():
                                    negative_prompt = gr.Textbox(
                                        label="负向提示词",
                                        lines=3,
                                        placeholder="不希望视频出现的内容...",
                                        max_lines=5,
                                        value="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                                    )
                                with gr.Column():
                                    resolution = gr.Dropdown(
                                        choices=[
                                            # 720p
                                            ("1280x720 (16:9, 720p)", "1280x720"),
                                            ("720x1280 (9:16, 720p)", "720x1280"),
                                            ("1024x1024 (1:1, 720p)", "1024x1024"),
                                            ("1280x544 (21:9, 720p)", "1280x544"),
                                            ("544x1280 (9:21, 720p)", "544x1280"),
                                            ("1104x832 (4:3, 720p)", "1104x832"),
                                            ("832x1104 (3:4, 720p)", "832x1104"),
                                            ("960x960 (1:1, 720p)", "960x960"),
                                            # 480p
                                            ("960x544 (16:9, 540p)", "960x544"),
                                            ("544x960 (9:16, 540p)", "544x960"),
                                            ("832x480 (16:9, 480p)", "832x480"),
                                            ("480x832 (9:16, 480p)", "480x832"),
                                            ("832x624 (4:3, 480p)", "832x624"),
                                            ("624x832 (3:4, 480p)", "624x832"),
                                            ("720x720 (1:1, 480p)", "720x720"),
                                            ("512x512 (1:1, 480p)", "512x512"),
                                        ],
                                        value="832x480",
                                        label="最大分辨率",
                                    )
                                with gr.Column():
                                    seed = gr.Slider(
                                        label="随机种子",
                                        minimum=-10000000,
                                        maximum=10000000,
                                        step=1,
                                        value=42,
                                        info="固定随机种子以获得可复现的结果",
                                    )
                                    infer_steps = gr.Slider(
                                        label="推理步数",
                                        minimum=1,
                                        maximum=100,
                                        step=1,
                                        value=20,
                                        info="视频生成的推理步数，增加步数可能提高质量但会降低速度",
                                    )
                                    sample_shift = gr.Slider(
                                        label="分布偏移程度",
                                        value=5,
                                        minimum=0,
                                        maximum=10,
                                        step=1,
                                        info="用于控制样本的分布偏移程度，数值越大表示偏移越明显",
                                    )

                                fps = gr.Slider(
                                    label="帧率(FPS)",
                                    minimum=8,
                                    maximum=30,
                                    step=1,
                                    value=16,
                                    info="视频每秒帧数，更高的FPS生成更流畅的视频",
                                )
                                num_frames = gr.Slider(
                                    label="总帧数",
                                    minimum=16,
                                    maximum=120,
                                    step=1,
                                    value=81,
                                    info="视频总帧数，更多的帧数生成更长的视频",
                                )

                            save_video_path = gr.Textbox(
                                label="输出视频路径",
                                value=generate_unique_filename(),
                                info="必须包含.mp4后缀，如果留空或使用默认值，将自动生成唯一文件名",
                            )

                            infer_btn = gr.Button("生成视频", variant="primary", size="lg")

                    with gr.Column(scale=6):
                        gr.Markdown("## 📤 生成的视频")
                        output_video = gr.Video(
                            label="结果",
                            height=624,
                            width=360,
                            autoplay=True,
                            elem_classes=["output-video"],
                        )

            with gr.Tab("⚙️ 高级选项", id=2):
                with gr.Group(elem_classes="advanced-options"):
                    gr.Markdown("### 无分类器引导(CFG)")
                    with gr.Row():
                        enable_cfg = gr.Checkbox(
                            label="启用无分类器引导",
                            value=False,
                            info="启用分类器引导，用于控制提示词强度",
                        )
                        cfg_scale = gr.Slider(
                            label="CFG缩放系数",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=5,
                            info="控制提示词的影响强度，值越高提示词影响越大",
                        )

                    gr.Markdown("### 显存/内存优化")
                    with gr.Row():
                        lazy_load = gr.Checkbox(
                            label="启用延迟加载",
                            value=False,
                            info="推理时延迟加载模型组件，适用内存受限环境",
                        )

                        torch_compile = gr.Checkbox(
                            label="启用Torch编译",
                            value=False,
                            info="使用torch.compile加速推理过程",
                        )

                        use_expandable_alloc = gr.Checkbox(
                            label="启用可扩展显存分配",
                            value=False,
                            info="有助于减少显存碎片",
                        )

                        rotary_chunk = gr.Checkbox(
                            label="分块处理旋转位置编码",
                            value=False,
                            info="启用后，使用分块处理旋转位置编码节省显存。",
                        )

                        clean_cuda_cache = gr.Checkbox(
                            label="清理 CUDA 显存缓存",
                            value=False,
                            info="启用后，及时释放显存但推理速度变慢。",
                        )

                    with gr.Row():
                        cpu_offload = gr.Checkbox(
                            label="CPU卸载",
                            value=False,
                            info="将模型的部分计算从 GPU 卸载到 CPU，以降低显存占用",
                        )
                        offload_granularity = gr.Dropdown(
                            label="Dit 卸载粒度",
                            choices=["block", "phase"],
                            value="block",
                            info="控制 Dit 模型卸载到 CPU 时的粒度",
                        )
                        t5_offload_granularity = gr.Dropdown(
                            label="T5 Encoder 卸载粒度",
                            choices=["model", "block"],
                            value="block",
                            info="控制 T5 Encoder 模型卸载到 CPU 时的粒度",
                        )

                    gr.Markdown("### 低精度量化")
                    with gr.Row():
                        attention_type = gr.Dropdown(
                            label="attention 算子",
                            choices=["flash_attn2", "flash_attn3", "sage_attn2"],
                            value="flash_attn2",
                            info="使用合适的 attention 算子可加速推理",
                        )

                        quant_op = gr.Dropdown(
                            label="量化算子",
                            choices=["vllm", "sgl", "q8f"],
                            value="vllm",
                            info="使用合适的量化算子可加速推理",
                        )

                        dit_quant_scheme = gr.Dropdown(
                            label="Dit",
                            choices=["fp8", "int8", "bf16"],
                            value="bf16",
                            info="Dit模型的量化精度",
                        )
                        t5_quant_scheme = gr.Dropdown(
                            label="T5 Encoder",
                            choices=["fp8", "int8", "bf16"],
                            value="bf16",
                            info="T5 Encoder模型的量化精度",
                        )
                        clip_quant_scheme = gr.Dropdown(
                            label="Clip Encoder",
                            choices=["fp8", "int8", "fp16"],
                            value="fp16",
                            info="Clip Encoder的量化精度",
                        )
                        precision_mode = gr.Dropdown(
                            label="敏感层精度",
                            choices=["fp32", "bf16"],
                            value="bf16",
                            info="选择用于敏感层计算的数值精度。",
                        )

                    gr.Markdown("### 变分自编码器(VAE)")
                    with gr.Row():
                        use_tiny_vae = gr.Checkbox(
                            label="使用轻量级VAE",
                            value=False,
                            info="使用轻量级VAE模型加速解码过程",
                        )
                        use_tiling_vae = gr.Checkbox(
                            label="启用 VAE 平铺推理",
                            value=False,
                            info="使用 VAE 平铺推理以降低显存占用",
                        )

                    gr.Markdown("### 特征缓存")
                    with gr.Row():
                        enable_teacache = gr.Checkbox(
                            label="启用Tea Cache",
                            value=False,
                            info="在推理过程中缓存特征以减少推理步数",
                        )
                        teacache_thresh = gr.Slider(
                            label="Tea Cache阈值",
                            value=0.26,
                            minimum=0,
                            maximum=1,
                            info="加速越高，质量可能越差 —— 设置为 0.1 可获得约 2.0 倍加速，设置为 0.2 可获得约 3.0 倍加速",
                        )

        infer_btn.click(
            fn=run_inference,
            inputs=[
                model_type,
                task,
                prompt,
                negative_prompt,
                image_path,
                save_video_path,
                torch_compile,
                infer_steps,
                num_frames,
                resolution,
                seed,
                sample_shift,
                enable_teacache,
                teacache_thresh,
                enable_cfg,
                cfg_scale,
                dit_quant_scheme,
                t5_quant_scheme,
                clip_quant_scheme,
                fps,
                use_tiny_vae,
                use_tiling_vae,
                lazy_load,
                precision_mode,
                use_expandable_alloc,
                cpu_offload,
                offload_granularity,
                t5_offload_granularity,
                attention_type,
                quant_op,
                rotary_chunk,
                clean_cuda_cache,
            ],
            outputs=output_video,
        )

    demo.launch(share=True, server_port=args.server_port, server_name=args.server_name)


if __name__ == "__main__":
    main()
