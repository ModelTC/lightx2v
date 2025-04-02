import argparse
import torch
import torch.distributed as dist
import os
import time
import gc
import json
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from lightx2v.text2v.models.text_encoders.hf.llama.model import TextEncoderHFLlamaModel
from lightx2v.text2v.models.text_encoders.hf.clip.model import TextEncoderHFClipModel
from lightx2v.text2v.models.text_encoders.hf.t5.model import T5EncoderModel

from lightx2v.text2v.models.schedulers.hunyuan.scheduler import HunyuanScheduler
from lightx2v.text2v.models.schedulers.hunyuan.feature_caching.scheduler import HunyuanSchedulerFeatureCaching
from lightx2v.text2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.text2v.models.schedulers.wan.feature_caching.scheduler import WanSchedulerFeatureCaching

from lightx2v.text2v.models.networks.hunyuan.model import HunyuanModel
from lightx2v.text2v.models.networks.wan.model import WanModel

from lightx2v.text2v.models.video_encoders.hf.autoencoder_kl_causal_3d.model import VideoEncoderKLCausal3DModel
from lightx2v.text2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.utils.utils import save_videos_grid, seed_all, cache_video
from lightx2v.common.ops import *
from lightx2v.image2v.models.wan.model import CLIPModel


def load_models(args, model_config):
    if model_config["parallel_attn"]:
        cur_rank = dist.get_rank()  # 获取当前进程的 rank
        torch.cuda.set_device(cur_rank)  # 设置当前进程的 CUDA 设备
    image_encoder = None
    if args.cpu_offload:
        init_device = torch.device("cpu")
    else:
        init_device = torch.device("cuda")

    if args.model_cls == "hunyuan":
        text_encoder_1 = TextEncoderHFLlamaModel(os.path.join(args.model_path, "text_encoder"), init_device)
        text_encoder_2 = TextEncoderHFClipModel(os.path.join(args.model_path, "text_encoder_2"), init_device)
        text_encoders = [text_encoder_1, text_encoder_2]
        model = HunyuanModel(args.model_path, model_config)
        vae_model = VideoEncoderKLCausal3DModel(args.model_path, dtype=torch.float16, device=init_device)

    elif args.model_cls == "wan2.1":
        text_encoder = T5EncoderModel(
            text_len=model_config["text_len"],
            dtype=torch.bfloat16,
            device=init_device,
            checkpoint_path=os.path.join(args.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(args.model_path, "google/umt5-xxl"),
            shard_fn=None,
        )
        text_encoders = [text_encoder]
        model = WanModel(args.model_path, model_config)
        vae_model = WanVAE(vae_pth=os.path.join(args.model_path, "Wan2.1_VAE.pth"), device=init_device, parallel=args.parallel_vae)
        if args.task == "i2v":
            image_encoder = CLIPModel(
                dtype=torch.float16,
                device=init_device,
                checkpoint_path=os.path.join(args.model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                tokenizer_path=os.path.join(args.model_path, "xlm-roberta-large"),
            )
    else:
        raise NotImplementedError(f"Unsupported model class: {args.model_cls}")

    return model, text_encoders, vae_model, image_encoder


def set_target_shape(args):
    if args.model_cls == "hunyuan":
        vae_scale_factor = 2 ** (4 - 1)
        args.target_shape = (
            1,
            16,
            (args.target_video_length - 1) // 4 + 1,
            int(args.target_height) // vae_scale_factor,
            int(args.target_width) // vae_scale_factor,
        )
    elif args.model_cls == "wan2.1":
        if args.task == "i2v":
            args.target_shape = (16, 21, args.lat_h, args.lat_w)
        elif args.task == "t2v":
            args.target_shape = (
                16,
                (args.target_video_length - 1) // 4 + 1,
                int(args.target_height) // args.vae_stride[1],
                int(args.target_width) // args.vae_stride[2],
            )


def run_image_encoder(args, image_encoder, vae_model):
    if args.model_cls == "hunyuan":
        return None
    elif args.model_cls == "wan2.1":
        img = Image.open(args.image_path).convert("RGB")
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        clip_encoder_out = image_encoder.visual([img[:, None, :, :]]).squeeze(0).to(torch.bfloat16)

        h, w = img.shape[1:]
        aspect_ratio = h / w
        max_area = args.target_height * args.target_width
        lat_h = round(np.sqrt(max_area * aspect_ratio) // args.vae_stride[1] // args.patch_size[1] * args.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // args.vae_stride[2] // args.patch_size[2] * args.patch_size[2])
        h = lat_h * args.vae_stride[1]
        w = lat_w * args.vae_stride[2]

        args.lat_h = lat_h
        args.lat_w = lat_w

        msk = torch.ones(1, 81, lat_h, lat_w, device=torch.device("cuda"))
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        vae_encode_out = vae_model.encode([torch.concat([torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1), torch.zeros(3, 80, h, w)], dim=1).cuda()])[0]
        vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)
        return {"clip_encoder_out": clip_encoder_out, "vae_encode_out": vae_encode_out}

    else:
        raise NotImplementedError(f"Unsupported model class: {model_cls}")


def run_text_encoder(args, text, text_encoders, model_config):
    text_encoder_output = {}
    if args.model_cls == "hunyuan":
        for i, encoder in enumerate(text_encoders):
            text_state, attention_mask = encoder.infer(text, args)
            text_encoder_output[f"text_encoder_{i + 1}_text_states"] = text_state.to(dtype=torch.bfloat16)
            text_encoder_output[f"text_encoder_{i + 1}_attention_mask"] = attention_mask

    elif args.model_cls == "wan2.1":
        n_prompt = model_config.get("sample_neg_prompt", "")
        context = text_encoders[0].infer([text], args)
        context_null = text_encoders[0].infer([n_prompt if n_prompt else ""], args)
        text_encoder_output["context"] = context
        text_encoder_output["context_null"] = context_null

    else:
        raise NotImplementedError(f"Unsupported model type: {args.model_cls}")

    return text_encoder_output


def init_scheduler(args):
    if args.model_cls == "hunyuan":
        if args.feature_caching == "NoCaching":
            scheduler = HunyuanScheduler(args)
        elif args.feature_caching == "TaylorSeer":
            scheduler = HunyuanSchedulerFeatureCaching(args)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {args.feature_caching}")

    elif args.model_cls == "wan2.1":
        if args.feature_caching == "NoCaching":
            scheduler = WanScheduler(args)
        elif args.feature_caching == "Tea":
            scheduler = WanSchedulerFeatureCaching(args)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {args.feature_caching}")

    else:
        raise NotImplementedError(f"Unsupported model class: {args.model_cls}")
    return scheduler


def run_main_inference(args, model, text_encoder_output, image_encoder_output):
    for step_index in range(model.scheduler.infer_steps):
        torch.cuda.synchronize()
        time1 = time.time()

        model.scheduler.step_pre(step_index=step_index)

        torch.cuda.synchronize()
        time2 = time.time()

        model.infer(text_encoder_output, image_encoder_output, args)

        torch.cuda.synchronize()
        time3 = time.time()

        model.scheduler.step_post()

        torch.cuda.synchronize()
        time4 = time.time()

        print(f"step {step_index} infer time: {time3 - time2}")
        print(f"step {step_index} all time: {time4 - time1}")
        print("*" * 10)

    return model.scheduler.latents, model.scheduler.generator


def run_vae(latents, generator, args):
    images = vae_model.decode(latents, generator=generator, args=args)
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--save_video_path", type=str, default="./output_ligthx2v.mp4")
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
    parser.add_argument("--parallel_attn", action="store_true")
    parser.add_argument("--parallel_vae", action="store_true")
    parser.add_argument("--max_area", action="store_true")
    parser.add_argument("--vae_stride", default=(4, 8, 8))
    parser.add_argument("--patch_size", default=(1, 2, 2))
    parser.add_argument("--teacache_thresh", type=float, default=0.26)
    parser.add_argument("--use_ret_steps", action="store_true", default=False)
    args = parser.parse_args()

    start_time = time.time()
    print(f"args: {args}")

    seed_all(args.seed)

    if args.parallel_attn:
        dist.init_process_group(backend="nccl")

    if args.mm_config:
        mm_config = json.loads(args.mm_config)
    else:
        mm_config = None

    model_config = {
        "task": args.task,
        "attention_type": args.attention_type,
        "sample_neg_prompt": args.sample_neg_prompt,
        "mm_config": mm_config,
        "do_mm_calib": args.do_mm_calib,
        "cpu_offload": args.cpu_offload,
        "feature_caching": args.feature_caching,
        "parallel_attn": args.parallel_attn,
        "parallel_vae": args.parallel_vae,
    }

    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            config = json.load(f)
        model_config.update(config)

    print(f"model_config: {model_config}")

    model, text_encoders, vae_model, image_encoder = load_models(args, model_config)

    if args.task in ["i2v"]:
        image_encoder_output = run_image_encoder(args, image_encoder, vae_model)
    else:
        image_encoder_output = {"clip_encoder_out": None, "vae_encode_out": None}

    text_encoder_output = run_text_encoder(args, args.prompt, text_encoders, model_config)

    set_target_shape(args)
    scheduler = init_scheduler(args)

    model.set_scheduler(scheduler)

    gc.collect()
    torch.cuda.empty_cache()

    if args.cpu_offload:
        model.to_cuda()

    latents, generator = run_main_inference(args, model, text_encoder_output, image_encoder_output)

    if args.cpu_offload:
        model.to_cpu()
        gc.collect()
        torch.cuda.empty_cache()

    images = run_vae(latents, generator, args)

    if not args.parallel_attn or (args.parallel_attn and dist.get_rank() == 0):
        if args.model_cls == "wan2.1":
            cache_video(tensor=images, save_file=args.save_video_path, fps=16, nrow=1, normalize=True, value_range=(-1, 1))
        else:
            save_videos_grid(images, args.save_video_path, fps=24)

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
