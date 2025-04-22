import argparse
import torch
import torch.distributed as dist
import os
import time
import gc
import json
from pathlib import Path
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

from lightx2v.utils.envs import *
from lightx2v.utils.utils import save_videos_grid, seed_all, cache_video
from lightx2v.utils.profiler import ProfilingContext, ProfilingContext4Debug
from lightx2v.utils.set_config import set_config

from lightx2v.models.input_encoders.hf.llama.model import TextEncoderHFLlamaModel
from lightx2v.models.input_encoders.hf.clip.model import TextEncoderHFClipModel
from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.llava.model import TextEncoderHFLlavaModel
from lightx2v.models.input_encoders.hf.xlm_roberta.model import CLIPModel

from lightx2v.models.schedulers.hunyuan.scheduler import HunyuanScheduler
from lightx2v.models.schedulers.hunyuan.feature_caching.scheduler import HunyuanSchedulerTaylorCaching, HunyuanSchedulerTeaCaching
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.schedulers.wan.feature_caching.scheduler import WanSchedulerTeaCaching

from lightx2v.models.networks.hunyuan.model import HunyuanModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper

from lightx2v.models.video_encoders.hf.autoencoder_kl_causal_3d.model import VideoEncoderKLCausal3DModel
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE

from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.graph_runner import GraphRunner

from lightx2v.common.ops import *


def load_models(config):
    if config["parallel_attn_type"]:
        cur_rank = dist.get_rank()  # 获取当前进程的 rank
        torch.cuda.set_device(cur_rank)  # 设置当前进程的 CUDA 设备
    image_encoder = None
    if config.cpu_offload:
        init_device = torch.device("cpu")
    else:
        init_device = torch.device("cuda")

    if config.model_cls == "hunyuan":
        if config.task == "t2v":
            text_encoder_1 = TextEncoderHFLlamaModel(os.path.join(config.model_path, "text_encoder"), init_device)
        else:
            text_encoder_1 = TextEncoderHFLlavaModel(os.path.join(config.model_path, "text_encoder_i2v"), init_device)
        text_encoder_2 = TextEncoderHFClipModel(os.path.join(config.model_path, "text_encoder_2"), init_device)
        text_encoders = [text_encoder_1, text_encoder_2]
        model = HunyuanModel(config.model_path, config, init_device, config)
        vae_model = VideoEncoderKLCausal3DModel(config.model_path, dtype=torch.float16, device=init_device, config=config)

    elif config.model_cls == "wan2.1":
        with ProfilingContext("Load Text Encoder"):
            text_encoder = T5EncoderModel(
                text_len=config["text_len"],
                dtype=torch.bfloat16,
                device=init_device,
                checkpoint_path=os.path.join(config.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=os.path.join(config.model_path, "google/umt5-xxl"),
                shard_fn=None,
            )
            text_encoders = [text_encoder]
        with ProfilingContext("Load Wan Model"):
            model = WanModel(config.model_path, config, init_device)

        if config.lora_path:
            lora_wrapper = WanLoraWrapper(model)
            with ProfilingContext("Load LoRA Model"):
                lora_name = lora_wrapper.load_lora(config.lora_path)
                lora_wrapper.apply_lora(lora_name, config.strength_model)
                print(f"Loaded LoRA: {lora_name}")

        with ProfilingContext("Load WAN VAE Model"):
            vae_model = WanVAE(vae_pth=os.path.join(config.model_path, "Wan2.1_VAE.pth"), device=init_device, parallel=config.parallel_vae)
        if config.task == "i2v":
            with ProfilingContext("Load Image Encoder"):
                image_encoder = CLIPModel(
                    dtype=torch.float16,
                    device=init_device,
                    checkpoint_path=os.path.join(config.model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                    tokenizer_path=os.path.join(config.model_path, "xlm-roberta-large"),
                )
    else:
        raise NotImplementedError(f"Unsupported model class: {config.model_cls}")

    return model, text_encoders, vae_model, image_encoder


def set_target_shape(config, image_encoder_output):
    if config.model_cls == "hunyuan":
        if config.task == "t2v":
            vae_scale_factor = 2 ** (4 - 1)
            config.target_shape = (
                1,
                16,
                (config.target_video_length - 1) // 4 + 1,
                int(config.target_height) // vae_scale_factor,
                int(config.target_width) // vae_scale_factor,
            )
        elif config.task == "i2v":
            vae_scale_factor = 2 ** (4 - 1)
            config.target_shape = (
                1,
                16,
                (config.target_video_length - 1) // 4 + 1,
                int(image_encoder_output["target_height"]) // vae_scale_factor,
                int(image_encoder_output["target_width"]) // vae_scale_factor,
            )
    elif config.model_cls == "wan2.1":
        if config.task == "i2v":
            config.target_shape = (16, 21, config.lat_h, config.lat_w)
        elif config.task == "t2v":
            config.target_shape = (
                16,
                (config.target_video_length - 1) // 4 + 1,
                int(config.target_height) // config.vae_stride[1],
                int(config.target_width) // config.vae_stride[2],
            )


def generate_crop_size_list(base_size=256, patch_size=32, max_ratio=4.0):
    num_patches = round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def get_closest_ratio(height: float, width: float, ratios: list, buckets: list):
    aspect_ratio = float(height) / float(width)
    diff_ratios = ratios - aspect_ratio

    if aspect_ratio >= 1:
        indices = [(index, x) for index, x in enumerate(diff_ratios) if x <= 0]
    else:
        indices = [(index, x) for index, x in enumerate(diff_ratios) if x > 0]

    closest_ratio_id = min(indices, key=lambda pair: abs(pair[1]))[0]
    closest_size = buckets[closest_ratio_id]
    closest_ratio = ratios[closest_ratio_id]

    return closest_size, closest_ratio


def run_image_encoder(config,  image_path, image_encoder, vae_model):
    if config.model_cls == "hunyuan":
        img = Image.open(image_path).convert("RGB")
        origin_size = img.size

        i2v_resolution = "720p"
        if i2v_resolution == "720p":
            bucket_hw_base_size = 960
        elif i2v_resolution == "540p":
            bucket_hw_base_size = 720
        elif i2v_resolution == "360p":
            bucket_hw_base_size = 480
        else:
            raise ValueError(f"i2v_resolution: {i2v_resolution} must be in [360p, 540p, 720p]")

        crop_size_list = generate_crop_size_list(bucket_hw_base_size, 32)
        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

        resize_param = min(closest_size)
        center_crop_param = closest_size

        ref_image_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(resize_param), torchvision.transforms.CenterCrop(center_crop_param), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5], [0.5])]
        )

        semantic_image_pixel_values = [ref_image_transform(img)]
        semantic_image_pixel_values = torch.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2).to(torch.float16).to(torch.device("cuda"))

        img_latents = vae_model.encode(semantic_image_pixel_values, config).mode()

        scaling_factor = 0.476986
        img_latents.mul_(scaling_factor)

        target_height, target_width = closest_size

        return {"img": img, "img_latents": img_latents, "target_height": target_height, "target_width": target_width}

    elif config.model_cls == "wan2.1":
        img = Image.open(image_path).convert("RGB")
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        clip_encoder_out = image_encoder.visual([img[:, None, :, :]], config).squeeze(0).to(torch.bfloat16)
        h, w = img.shape[1:]
        aspect_ratio = h / w
        max_area = config.target_height * config.target_width
        lat_h = round(np.sqrt(max_area * aspect_ratio) // config.vae_stride[1] // config.patch_size[1] * config.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // config.vae_stride[2] // config.patch_size[2] * config.patch_size[2])
        h = lat_h * config.vae_stride[1]
        w = lat_w * config.vae_stride[2]

        config.lat_h = lat_h
        config.lat_w = lat_w

        msk = torch.ones(1, 81, lat_h, lat_w, device=torch.device("cuda"))
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        vae_encode_out = vae_model.encode(
            [torch.concat([torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1), torch.zeros(3, 80, h, w)], dim=1).cuda()], config
        )[0]
        vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)
        return {"clip_encoder_out": clip_encoder_out, "vae_encode_out": vae_encode_out}

    else:
        raise NotImplementedError(f"Unsupported model class: {config.model_cls}")


def run_text_encoder(text, text_encoders, config, image_encoder_output):
    text_encoder_output = {}
    if config.model_cls == "hunyuan":
        for i, encoder in enumerate(text_encoders):
            if config.task == "i2v" and i == 0:
                text_state, attention_mask = encoder.infer(text, image_encoder_output["img"], config)
            else:
                text_state, attention_mask = encoder.infer(text, config)
            text_encoder_output[f"text_encoder_{i + 1}_text_states"] = text_state.to(dtype=torch.bfloat16)
            text_encoder_output[f"text_encoder_{i + 1}_attention_mask"] = attention_mask

    elif config.model_cls == "wan2.1":
        n_prompt = config.get("sample_neg_prompt", "")
        context = text_encoders[0].infer([text], config)
        context_null = text_encoders[0].infer([n_prompt if n_prompt else ""], config)
        text_encoder_output["context"] = context
        text_encoder_output["context_null"] = context_null

    else:
        raise NotImplementedError(f"Unsupported model type: {config.model_cls}")

    return text_encoder_output


def init_scheduler(config, image_encoder_output):
    if config.model_cls == "hunyuan":
        if config.feature_caching == "NoCaching":
            scheduler = HunyuanScheduler(config, image_encoder_output)
        elif config.feature_caching == "Tea":
            scheduler = HunyuanSchedulerTeaCaching(config, image_encoder_output)
        elif config.feature_caching == "TaylorSeer":
            scheduler = HunyuanSchedulerTaylorCaching(config, image_encoder_output)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {config.feature_caching}")

    elif config.model_cls == "wan2.1":
        if config.feature_caching == "NoCaching":
            scheduler = WanScheduler(config)
        elif config.feature_caching == "Tea":
            scheduler = WanSchedulerTeaCaching(config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {config.feature_caching}")

    else:
        raise NotImplementedError(f"Unsupported model class: {config.model_cls}")
    return scheduler


def run_vae(latents, generator, config):
    images = vae_model.decode(latents, generator=generator, config=config)
    return images




def collect_image_prompt_pairs(input_path) -> list[tuple[str, str]]:
    '''
    batch image and prompt pairs
    :param input_path: path to the directory containing images and prompts
    :return: list of tuples containing image paths and corresponding prompt texts

    example:
    - input_path: 
        |- image0.png
        |- image0.txt
        |- image1.png
        |- image1.txt
    '''
    input_pairs = []
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            image_path = os.path.join(input_path, filename)
            
            prompt_filename = os.path.splitext(filename)[0] + '.txt'
            prompt_path = os.path.join(input_path, prompt_filename)
            
            if not os.path.exists(prompt_path):
                print(f"Warning: No matching prompt file for {image_path}, skipping.")
                continue
                
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
            
            input_pairs.append((image_path, prompt_text))
    
    return input_pairs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None, help="The path to input image file or path for image-to-video (i2v) task")
    parser.add_argument("--save_video_path", type=str, default="./output_lightx2v.mp4", help="The path to save video path/file")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--infer_steps", type=int, required=True)
    parser.add_argument("--target_video_length", type=int, required=True)
    parser.add_argument("--target_width", type=int, required=True)
    parser.add_argument("--target_height", type=int, required=True)
    parser.add_argument("--attention_type", type=str, required=True)
    parser.add_argument("--sample_neg_prompt", type=str, default="")
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--do_mm_calib", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--feature_caching", choices=["NoCaching", "TaylorSeer", "Tea"], default="NoCaching")
    parser.add_argument("--mm_config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel_attn_type", default=None, choices=["ulysses", "ring"])
    parser.add_argument("--parallel_vae", action="store_true")
    parser.add_argument("--max_area", action="store_true")
    parser.add_argument("--vae_stride", default=(4, 8, 8))
    parser.add_argument("--patch_size", default=(1, 2, 2))
    parser.add_argument("--teacache_thresh", type=float, default=0.26)
    parser.add_argument("--use_ret_steps", action="store_true", default=False)
    parser.add_argument("--use_bfloat16", action="store_true", default=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--strength_model", type=float, default=1.0)
    args = parser.parse_args()

    start_time = time.perf_counter()
    print(f"args: {args}")

    seed_all(args.seed)

    config = set_config(args)

    if config.parallel_attn_type:
        dist.init_process_group(backend="nccl")

    print(f"config: {config}")

    if Path(args.image_path).is_dir():
        inputs_files = collect_image_prompt_pairs(args.image_path)
    else:
        inputs_files = [(args.image_path, args.prompt)]

    print("inputs_files count:", len(inputs_files))

    save_video_path = Path(args.save_video_path)
    if save_video_path.is_file():
        output_dir = save_video_path.parent
    else:
        output_dir = save_video_path
        output_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.image_path).is_file():
        if not args.save_video_path.endswith('.mp4'):
            save_video_path = output_dir / 'output.mp4'
        else:
            save_video_path = Path(args.save_video_path)
    else:
        save_video_path = output_dir
    print(f"Output directory set to parent directory: {output_dir}")
    
    with ProfilingContext("Load models"):
        model, text_encoders, vae_model, image_encoder = load_models(config)

    for image_path, prompt in inputs_files:
        if config["task"] in ["i2v"]:
            image_encoder_output = run_image_encoder(config, image_path, image_encoder, vae_model)
        else:
            image_encoder_output = {"clip_encoder_out": None, "vae_encode_out": None}

        with ProfilingContext("Run Text Encoder"):
            text_encoder_output = run_text_encoder(prompt, text_encoders, config, image_encoder_output)

        inputs = {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}

        set_target_shape(config, image_encoder_output)
        scheduler = init_scheduler(config, image_encoder_output)

        model.set_scheduler(scheduler)

        gc.collect()
        torch.cuda.empty_cache()

        if CHECK_ENABLE_GRAPH_MODE():
            default_runner = DefaultRunner(model, inputs)
            runner = GraphRunner(default_runner)
        else:
            runner = DefaultRunner(model, inputs)

        latents, generator = runner.run()

        if config.cpu_offload:
            scheduler.clear()
            del text_encoder_output, image_encoder_output, model, text_encoders, scheduler
            torch.cuda.empty_cache()

        with ProfilingContext("Run VAE"):
            images = run_vae(latents, generator, config)


        if save_video_path.is_dir():
            save_video_name = str(save_video_path / f"{Path(image_path).stem}.mp4")
        else:
            save_video_name = str(save_video_path)

        if not config.parallel_attn_type or (config.parallel_attn_type and dist.get_rank() == 0):
            with ProfilingContext("Save video"):
                if config.model_cls == "wan2.1":
                    cache_video(tensor=images, save_file=save_video_name, fps=16, nrow=1, normalize=True, value_range=(-1, 1))
                else:
                    save_videos_grid(images, save_video_name, fps=24)

    end_time = time.perf_counter()
    print(f"Total cost: {end_time - start_time}")
