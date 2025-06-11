import os
import gradio as gr
import asyncio
import argparse
import json
import torch
import gc
from easydict import EasyDict
from loguru import logger
from lightx2v.infer import init_runner
from lightx2v.utils.envs import *


logger.add(
    "inference_logs.log",
    rotation="100 MB",
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

SUPPORTED_MODEL = "wan2.1"
TASK = "i2v"


def run_inference(
    model_path,
    prompt,
    negative_prompt,
    image_path,
    save_video_path,
    torch_compile,
    infer_steps,
    num_frames,
    width,
    height,
    seed,
    enable_teacache,
    enable_cfg,
    cfg_scale,
    quant_option,
    fps,
    use_tiny_vae,
    tiny_vae_path,
):
    """Wrapper for wan2.1 I2V inference logic with advanced options"""

    if torch_compile:
        os.environ["ENABLE_GRAPH_MODE"] = "true"

    config = {
        "infer_steps": infer_steps,
        "target_video_length": num_frames,
        "target_height": height,
        "target_width": width,
        "attention_type": "sage_attn2",
        "seed": seed,
        "enable_cfg": enable_cfg,
        "sample_guide_scale": cfg_scale,
        "sample_shift": 5,
        "cpu_offload": True,
        "offload_granularity": "phase",
        "t5_offload_granularity": "block",
        "dit_quantized_ckpt": model_path,
        "mm_config": {
            "mm_type": ("W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm" if quant_option == "fp8" else "W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm"),
        },
        "fps": fps,
        "feature_caching": "Tea" if enable_teacache else "NoCaching",
        "coefficients": [
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
        ],
        "use_ret_steps": True,
        "teacache_thresh": 0.26,
        "t5_quantized": True,
        "t5_quantized_ckpt": os.path.join(model_path, "models_t5_umt5-xxl-enc-int8.pth"),
        "t5_quant_scheme": "int8",
        "clip_quantized": True,
        "clip_quantized_ckpt": os.path.join(model_path, "clip-int8.pth"),
        "clip_quant_scheme": "int8",
        "use_tiling_vae": True,
        "tiny_vae": use_tiny_vae,
        "tiny_vae_path": tiny_vae_path if use_tiny_vae else None,
        "lazy_load": True,
        "do_mm_calib": False,
        "parallel_attn_type": None,
        "parallel_vae": False,
        "max_area": False,
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "teacache_thresh": 0.26,
        "use_bfloat16": True,
        "lora_path": None,
        "strength_model": 1.0,
        "use_prompt_enhancer": False,
        "text_len": 512,
    }

    args = argparse.Namespace(
        model_cls=SUPPORTED_MODEL,
        task=TASK,
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

    if os.path.exists(os.path.join(model_path, "config.json")):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            model_config = json.load(f)
        config.update(model_config)

    logger.info(f"Updated inference config:\n{json.dumps(config, indent=4, ensure_ascii=False)}")

    runner = init_runner(config)
    asyncio.run(runner.run_pipeline())

    del runner
    gc.collect()
    torch.cuda.empty_cache()

    return save_video_path


with gr.Blocks(
    title="Wan2.1 I2V Video Generation",
    css="""
    .advanced-options { background: #f9f9ff; border-radius: 10px; padding: 15px; }
    .output-video { max-height: 650px; }
    .warning { color: #ff6b6b; font-weight: bold; }
""",
) as demo:
    gr.Markdown("# 🎬 Wan2.1 Image-to-Video (I2V) Generator")

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("## 📥 Input Parameters")

                with gr.Row():
                    image_path = gr.Image(
                        label="Input Image",
                        type="filepath",
                        height=300,
                        interactive=True,
                    )

                model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="/your/path/to/wan2.1_quant_model",
                    info="Local model folder path (in8/fp8 quantization supported)",
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
                            placeholder="Unwanted content...",
                            max_lines=5,
                        )
                    with gr.Column():
                        tiny_vae_path = gr.Textbox(
                            label="Tiny vae path",
                            lines=3,
                            placeholder="/your/path/to/tiny_vae.pth",
                            max_lines=5,
                        )

                save_video_path = gr.Textbox(
                    label="Output Video Path",
                    value="./save_results/wan2.1_i2v_output.mp4",
                    info="Must include .mp4 suffix",
                )

                with gr.Accordion("⚙️ Advanced Options", open=False):
                    with gr.Group(elem_classes="advanced-options"):
                        gr.Markdown("### Performance Settings")
                        with gr.Row():
                            torch_compile = gr.Checkbox(
                                label="Enable Torch Compile",
                                value=False,
                                info="Use torch.compile for faster inference",
                            )

                            quant_option = gr.Radio(
                                label="Quantization Method",
                                choices=["fp8", "int8"],
                                value="fp8",
                                info="Select quantization method for model",
                            )

                            infer_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=20,
                                info="Infer steps for video generation",
                            )

                            enable_teacache = gr.Checkbox(
                                label="Enable Teacache",
                                value=False,
                                info="Teacache for caching features during inference",
                            )

                            enable_cfg = gr.Checkbox(
                                label="Enable CFG",
                                value=False,
                                info="Classifier-Free Guidance for prompt strength control",
                            )

                            use_tiny_vae = gr.Checkbox(
                                label="Use Tiny VAE",
                                value=False,
                                info="Tiny VAE for faster inference",
                            )

                            cfg_scale = gr.Slider(
                                label="CFG scale",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=5,
                                info="CFG scale for controlling the strength of the prompt",
                            )

                            seed = gr.Slider(
                                label="Seed",
                                minimum=-10000000,
                                maximum=10000000,
                                step=1,
                                value=42,
                                info="Random seed for reproducibility",
                            )

                        gr.Markdown("### Video Parameters")
                        with gr.Row():
                            fps = gr.Slider(
                                label="FPS (Frames Per Second)",
                                minimum=8,
                                maximum=30,
                                step=1,
                                value=16,
                                info="Higher FPS = smoother video",
                            )

                            num_frames = gr.Slider(
                                label="Number of Frames",
                                minimum=16,
                                maximum=120,
                                step=1,
                                value=81,
                                info="More frames = longer video",
                            )

                        with gr.Row():
                            width = gr.Number(
                                label="Width",
                                value=832,
                                precision=0,
                                minimum=320,
                                maximum=1920,
                                info="Output video width",
                            )

                            height = gr.Number(
                                label="Height",
                                value=480,
                                precision=0,
                                minimum=240,
                                maximum=1080,
                                info="Output video height",
                            )

                        gr.Markdown(
                            """
                        <div class="warning">
                            ⚠️ Note: Changing resolution may affect video quality and performance
                        </div>
                        """
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

    infer_btn.click(
        fn=run_inference,
        inputs=[
            model_path,
            prompt,
            negative_prompt,
            image_path,
            save_video_path,
            torch_compile,
            infer_steps,
            num_frames,
            width,
            height,
            seed,
            enable_teacache,
            enable_cfg,
            cfg_scale,
            quant_option,
            fps,
            use_tiny_vae,
            tiny_vae_path,
        ],
        outputs=output_video,
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, server_name="0.0.0.0")
