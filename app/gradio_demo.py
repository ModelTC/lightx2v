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

    if task == "Text-to-Video":
        task = "t2v"
    elif task == "Image-to-Video":
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
    logger.info(f"Using model: {model_path}")
    logger.info(f"Inference configuration:\n{json.dumps(config, indent=4, ensure_ascii=False)}")

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
    parser.add_argument("--model_path", type=str, required=True, help="Model folder path")
    parser.add_argument(
        "--model_cls",
        type=str,
        choices=["wan2.1"],
        default="wan2.1",
        help="Model class to use",
    )
    parser.add_argument("--server_port", type=int, default=7862, help="Server port")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name")
    args = parser.parse_args()

    global model_path, model_cls
    model_path = args.model_path
    model_cls = args.model_cls

    def update_model_type(task_type):
        if task_type == "Image-to-Video":
            return gr.update(choices=["Wan2.1 14B"], value="Wan2.1 14B")
        elif task_type == "Text-to-Video":
            return gr.update(choices=["Wan2.1 14B", "Wan2.1 1.3B"], value="Wan2.1 14B")

    def toggle_image_input(task):
        return gr.update(visible=(task == "Image-to-Video"))

    with gr.Blocks(
        title="Lightx2v (Lightweight Video Inference Generation Engine)",
        css="""
        .main-content { max-width: 1400px; margin: auto; }
        .output-video { max-height: 650px; }
        .warning { color: #ff6b6b; font-weight: bold; }
        .advanced-options { background: #f9f9ff; border-radius: 10px; padding: 15px; }
        .tab-button { font-size: 16px; padding: 10px 20px; }
    """,
    ) as demo:
        gr.Markdown(f"# 🎬 {model_cls} Video Generator")
        gr.Markdown(f"### Using model: {model_path}")

        with gr.Tabs() as tabs:
            with gr.Tab("Basic Settings", id=1):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Group():
                            gr.Markdown("## 📥 Input Parameters")

                            with gr.Row():
                                task = gr.Dropdown(
                                    choices=["Image-to-Video", "Text-to-Video"],
                                    value="Image-to-Video",
                                    label="Task Type",
                                )

                                model_type = gr.Dropdown(
                                    choices=["Wan2.1 14B"],
                                    value="Wan2.1 14B",
                                    label="Model Type",
                                )
                                task.change(
                                    fn=update_model_type,
                                    inputs=task,
                                    outputs=model_type,
                                )

                            with gr.Row():
                                image_path = gr.Image(
                                    label="Input Image",
                                    type="filepath",
                                    height=300,
                                    interactive=True,
                                    visible=True,
                                )

                                task.change(
                                    fn=toggle_image_input,
                                    inputs=task,
                                    outputs=image_path,
                                )

                            with gr.Row():
                                with gr.Column():
                                    prompt = gr.Textbox(
                                        label="Prompt",
                                        lines=3,
                                        placeholder="Describe the video content...",
                                        max_lines=5,
                                    )
                                with gr.Column():
                                    negative_prompt = gr.Textbox(
                                        label="Negative Prompt",
                                        lines=3,
                                        placeholder="Content you don't want in the video...",
                                        max_lines=5,
                                        value="camera shake, garish colors, overexposure, static, blurry details, subtitles, style, work, painting, image, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, finger fusion, static frame, cluttered background, three legs, crowded background, walking backwards",
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
                                        value="480x832",
                                        label="Max Resolution",
                                    )
                                with gr.Column():
                                    seed = gr.Slider(
                                        label="Random Seed",
                                        minimum=-10000000,
                                        maximum=10000000,
                                        step=1,
                                        value=42,
                                        info="Fix the random seed for reproducible results",
                                    )
                                    infer_steps = gr.Slider(
                                        label="Inference Steps",
                                        minimum=1,
                                        maximum=100,
                                        step=1,
                                        value=20,
                                        info="Inference steps for video generation. More steps may improve quality but reduce speed",
                                    )
                                    sample_shift = gr.Slider(
                                        label="Distribution Shift",
                                        value=5,
                                        minimum=0,
                                        maximum=10,
                                        step=1,
                                        info="Controls the distribution shift of samples. Larger values mean more obvious shifts",
                                    )

                                fps = gr.Slider(
                                    label="Frame Rate (FPS)",
                                    minimum=8,
                                    maximum=30,
                                    step=1,
                                    value=16,
                                    info="Frames per second. Higher FPS produces smoother video",
                                )
                                num_frames = gr.Slider(
                                    label="Total Frames",
                                    minimum=16,
                                    maximum=120,
                                    step=1,
                                    value=81,
                                    info="Total number of frames. More frames produce longer video",
                                )

                            save_video_path = gr.Textbox(
                                label="Output Video Path",
                                value=generate_unique_filename(),
                                info="Must include .mp4 suffix. If left empty or using default, a unique filename will be automatically generated",
                            )

                            infer_btn = gr.Button("Generate Video", variant="primary", size="lg")

                    with gr.Column(scale=6):
                        gr.Markdown("## 📤 Generated Video")
                        output_video = gr.Video(
                            label="Result",
                            height=624,
                            width=360,
                            autoplay=True,
                            elem_classes=["output-video"],
                        )

            with gr.Tab("⚙️ Advanced Options", id=2):
                with gr.Group(elem_classes="advanced-options"):
                    gr.Markdown("### Classifier-Free Guidance (CFG)")
                    with gr.Row():
                        enable_cfg = gr.Checkbox(
                            label="Enable Classifier-Free Guidance",
                            value=False,
                            info="Enable classifier guidance to control prompt strength",
                        )
                        cfg_scale = gr.Slider(
                            label="CFG Scale",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=5,
                            info="Controls the influence strength of the prompt. Higher values mean stronger influence",
                        )

                    gr.Markdown("### Memory Optimization")
                    with gr.Row():
                        lazy_load = gr.Checkbox(
                            label="Enable Lazy Loading",
                            value=False,
                            info="Lazily load model components during inference, suitable for memory-constrained environments",
                        )

                        torch_compile = gr.Checkbox(
                            label="Enable Torch Compile",
                            value=False,
                            info="Use torch.compile to accelerate the inference process",
                        )

                        use_expandable_alloc = gr.Checkbox(
                            label="Enable Expandable Memory Allocation",
                            value=False,
                            info="Helps reduce memory fragmentation",
                        )

                        rotary_chunk = gr.Checkbox(
                            label="Chunked Rotary Position Encoding",
                            value=False,
                            info="When enabled, uses chunked processing for rotary position encoding to save memory.",
                        )

                        clean_cuda_cache = gr.Checkbox(
                            label="Clean CUDA Memory Cache",
                            value=False,
                            info="When enabled, frees up memory in a timely manner but slows down inference.",
                        )

                    with gr.Row():
                        cpu_offload = gr.Checkbox(
                            label="CPU Offload",
                            value=False,
                            info="Offload part of the model computation from GPU to CPU to reduce video memory usage",
                        )
                        offload_granularity = gr.Dropdown(
                            label="Dit Offload Granularity",
                            choices=["block", "phase"],
                            value="block",
                            info="Controls the granularity of Dit model offloading to CPU",
                        )
                        t5_offload_granularity = gr.Dropdown(
                            label="T5 Encoder Offload Granularity",
                            choices=["model", "block"],
                            value="block",
                            info="Controls the granularity of T5 Encoder model offloading to CPU",
                        )

                    gr.Markdown("### Low-Precision Quantization")
                    with gr.Row():
                        attention_type = gr.Dropdown(
                            label="Attention Operator",
                            choices=["flash_attn2", "flash_attn3", "sage_attn2"],
                            value="flash_attn2",
                            info="Using a suitable attention operator can accelerate inference",
                        )

                        quant_op = gr.Dropdown(
                            label="Quantization Operator",
                            choices=["vllm", "sgl", "q8f"],
                            value="vllm",
                            info="Using a suitable quantization operator can accelerate inference",
                        )

                        dit_quant_scheme = gr.Dropdown(
                            label="Dit",
                            choices=["fp8", "int8", "bf16"],
                            value="bf16",
                            info="Quantization precision for Dit model",
                        )
                        t5_quant_scheme = gr.Dropdown(
                            label="T5 Encoder",
                            choices=["fp8", "int8", "bf16"],
                            value="bf16",
                            info="Quantization precision for T5 Encoder model",
                        )
                        clip_quant_scheme = gr.Dropdown(
                            label="Clip Encoder",
                            choices=["fp8", "int8", "fp16"],
                            value="fp16",
                            info="Quantization precision for Clip Encoder",
                        )
                        precision_mode = gr.Dropdown(
                            label="Sensitive Layer Precision",
                            choices=["fp32", "bf16"],
                            value="bf16",
                            info="Select the numerical precision for sensitive layer calculations.",
                        )

                    gr.Markdown("### Variational Autoencoder (VAE)")
                    with gr.Row():
                        use_tiny_vae = gr.Checkbox(
                            label="Use Lightweight VAE",
                            value=False,
                            info="Use a lightweight VAE model to accelerate the decoding process",
                        )
                        use_tiling_vae = gr.Checkbox(
                            label="Enable VAE Tiling Inference",
                            value=False,
                            info="Use VAE tiling inference to reduce video memory usage",
                        )

                    gr.Markdown("### Feature Caching")
                    with gr.Row():
                        enable_teacache = gr.Checkbox(
                            label="Enable Tea Cache",
                            value=False,
                            info="Cache features during inference to reduce the number of inference steps",
                        )
                        teacache_thresh = gr.Slider(
                            label="Tea Cache Threshold",
                            value=0.26,
                            minimum=0,
                            maximum=1,
                            info="Higher acceleration may lead to lower quality - setting to 0.1 gives about 2.0x acceleration, setting to 0.2 gives about 3.0x acceleration",
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
