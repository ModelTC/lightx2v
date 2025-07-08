import asyncio
import gc
import aiohttp
import requests
from requests.exceptions import RequestException
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from lightx2v.utils.profiler import ProfilingContext4Debug, ProfilingContext
from lightx2v.utils.utils import save_videos_grid, cache_video
from lightx2v.utils.generate_task_id import generate_task_id
from lightx2v.utils.envs import *
from lightx2v.utils.service_utils import TensorTransporter, ImageTransporter
from loguru import logger


class DefaultRunner:
    def __init__(self, config):
        self.config = config
        self.has_prompt_enhancer = False
        if self.config["task"] == "t2v" and self.config.get("sub_servers", {}).get("prompt_enhancer") is not None:
            self.has_prompt_enhancer = True
            if not self.check_sub_servers("prompt_enhancer"):
                self.has_prompt_enhancer = False
                logger.warning("No prompt enhancer server available, disable prompt enhancer.")
        if not self.has_prompt_enhancer:
            self.config["use_prompt_enhancer"] = False
        self.set_init_device()

    def init_modules(self):
        logger.info("Initializing runner modules...")
        if self.config["mode"] == "split_server":
            self.tensor_transporter = TensorTransporter()
            self.image_transporter = ImageTransporter()
            if not self.check_sub_servers("dit"):
                raise ValueError("No dit server available")
            if not self.check_sub_servers("text_encoders"):
                raise ValueError("No text encoder server available")
            if self.config["task"] == "i2v":
                if not self.check_sub_servers("image_encoder"):
                    raise ValueError("No image encoder server available")
            if not self.check_sub_servers("vae_model"):
                raise ValueError("No vae server available")
            self.run_dit = self.run_dit_server
            self.run_vae_decoder = self.run_vae_decoder_server
            if self.config["task"] == "i2v":
                self.run_input_encoder = self.run_input_encoder_server_i2v
            else:
                self.run_input_encoder = self.run_input_encoder_server_t2v
        else:
            if not self.config.get("lazy_load", False):
                self.load_model()
            self.run_dit = self.run_dit_local
            self.run_vae_decoder = self.run_vae_decoder_local
            if self.config["task"] == "i2v":
                self.run_input_encoder = self.run_input_encoder_local_i2v
            else:
                self.run_input_encoder = self.run_input_encoder_local_t2v

    def set_init_device(self):
        if self.config["parallel_attn_type"]:
            cur_rank = dist.get_rank()
            torch.cuda.set_device(cur_rank)
        if self.config.cpu_offload:
            self.init_device = torch.device("cpu")
        else:
            self.init_device = torch.device("cuda")

    @ProfilingContext("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.image_encoder = self.load_image_encoder()
        self.vae_encoder, self.vae_decoder = self.load_vae()

    def check_sub_servers(self, task_type):
        urls = self.config.get("sub_servers", {}).get(task_type, [])
        available_servers = []
        for url in urls:
            try:
                status_url = f"{url}/v1/local/{task_type}/generate/service_status"
                response = requests.get(status_url, timeout=2)
                if response.status_code == 200:
                    available_servers.append(url)
                else:
                    logger.warning(f"Service {url} returned status code {response.status_code}")

            except RequestException as e:
                logger.warning(f"Failed to connect to {url}: {str(e)}")
                continue
        logger.info(f"{task_type} available servers: {available_servers}")
        self.config["sub_servers"][task_type] = available_servers
        return len(available_servers) > 0

    def set_inputs(self, inputs):
        self.config["prompt"] = inputs.get("prompt", "")
        self.config["use_prompt_enhancer"] = False
        if self.has_prompt_enhancer:
            self.config["use_prompt_enhancer"] = inputs.get("use_prompt_enhancer", False)  # Reset use_prompt_enhancer from clinet side.
        self.config["negative_prompt"] = inputs.get("negative_prompt", "")
        self.config["image_path"] = inputs.get("image_path", "")
        self.config["save_video_path"] = inputs.get("save_video_path", "")
        self.config["infer_steps"] = inputs.get("infer_steps", self.config.get("infer_steps", 5))
        self.config["target_video_length"] = inputs.get("target_video_length", self.config.get("target_video_length", 81))
        self.config["seed"] = inputs.get("seed", self.config.get("seed", 42))
        # self.config["sample_shift"] = inputs.get("sample_shift", self.config.get("sample_shift", 5))
        # self.config["sample_guide_scale"] = inputs.get("sample_guide_scale", self.config.get("sample_guide_scale", 5))

    def run(self):
        for step_index in range(self.model.scheduler.infer_steps):
            logger.info(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

            with ProfilingContext4Debug("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            with ProfilingContext4Debug("infer"):
                self.model.infer(self.inputs)

            with ProfilingContext4Debug("step_post"):
                self.model.scheduler.step_post()

        return self.model.scheduler.latents, self.model.scheduler.generator

    async def run_step(self, step_index=0):
        self.init_scheduler()
        await self.run_input_encoder()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        self.model.scheduler.step_pre(step_index=step_index)
        self.model.infer(self.inputs)
        self.model.scheduler.step_post()

    def end_run(self):
        self.model.scheduler.clear()
        del self.inputs, self.model.scheduler
        if self.config.get("lazy_load", False):
            self.model.transformer_infer.weights_stream_mgr.clear()
            del self.model
        torch.cuda.empty_cache()
        gc.collect()

    @ProfilingContext("Run Encoders")
    async def run_input_encoder_local_i2v(self):
        prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
        img = Image.open(self.config["image_path"]).convert("RGB")
        clip_encoder_out = self.run_image_encoder(img)
        vae_encode_out, kwargs = self.run_vae_encoder(img)
        text_encoder_output = self.run_text_encoder(prompt, img)
        torch.cuda.empty_cache()
        gc.collect()
        return self.get_encoder_output_i2v(clip_encoder_out, vae_encode_out, text_encoder_output, img)

    @ProfilingContext("Run Encoders")
    async def run_input_encoder_local_t2v(self):
        prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
        text_encoder_output = self.run_text_encoder(prompt, None)
        torch.cuda.empty_cache()
        gc.collect()
        return {"text_encoder_output": text_encoder_output, "image_encoder_output": None}

    @ProfilingContext("Run DiT")
    async def run_dit_local(self, kwargs):
        if self.config.get("lazy_load", False):
            self.model = self.load_transformer()
        self.init_scheduler()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        latents, generator = self.run()
        self.end_run()
        return latents, generator

    @ProfilingContext("Run VAE Decoder")
    async def run_vae_decoder_local(self, latents, generator):
        if self.config.get("lazy_load", False):
            self.vae_decoder = self.load_vae_decoder()
        images = self.vae_decoder.decode(latents, generator=generator, config=self.config)
        if self.config.get("lazy_load", False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()
        return images

    @ProfilingContext("Save video")
    def save_video(self, images):
        if not self.config.parallel_attn_type or (self.config.parallel_attn_type and dist.get_rank() == 0):
            self.save_video_func(images)

    async def post_task(self, task_type, urls, message, device="cuda"):
        while True:
            for url in urls:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/v1/local/{task_type}/generate/service_status") as response:
                        status = await response.json()
                    if status["service_status"] == "idle":
                        async with session.post(f"{url}/v1/local/{task_type}/generate", json=message) as response:
                            result = await response.json()
                            if result["kwargs"] is not None:
                                for k, v in result["kwargs"].items():
                                    setattr(self.config, k, v)
                            return self.tensor_transporter.load_tensor(result["output"], device)
            await asyncio.sleep(0.1)

    def post_prompt_enhancer(self):
        while True:
            for url in self.config["sub_servers"]["prompt_enhancer"]:
                response = requests.get(f"{url}/v1/local/prompt_enhancer/generate/service_status").json()
                if response["service_status"] == "idle":
                    response = requests.post(f"{url}/v1/local/prompt_enhancer/generate", json={"task_id": generate_task_id(), "prompt": self.config["prompt"]})
                    enhanced_prompt = response.json()["output"]
                    logger.info(f"Enhanced prompt: {enhanced_prompt}")
                    return enhanced_prompt

    async def post_encoders_i2v(self, prompt, img=None, n_prompt=None, i2v=False):
        tasks = []
        img_byte = self.image_transporter.prepare_image(img)
        tasks.append(
            asyncio.create_task(self.post_task(task_type="image_encoder", urls=self.config["sub_servers"]["image_encoder"], message={"task_id": generate_task_id(), "img": img_byte}, device="cuda"))
        )
        tasks.append(
            asyncio.create_task(self.post_task(task_type="vae_model/encoder", urls=self.config["sub_servers"]["vae_model"], message={"task_id": generate_task_id(), "img": img_byte}, device="cuda"))
        )
        tasks.append(
            asyncio.create_task(
                self.post_task(
                    task_type="text_encoders",
                    urls=self.config["sub_servers"]["text_encoders"],
                    message={"task_id": generate_task_id(), "text": prompt, "img": img_byte, "n_prompt": n_prompt},
                    device="cuda",
                )
            )
        )
        results = await asyncio.gather(*tasks)
        # clip_encoder, vae_encoder, text_encoders
        return results[0], results[1], results[2]

    async def post_encoders_t2v(self, prompt, n_prompt=None):
        tasks = []
        tasks.append(
            asyncio.create_task(
                self.post_task(
                    task_type="text_encoders",
                    urls=self.config["sub_servers"]["text_encoders"],
                    message={"task_id": generate_task_id(), "text": prompt, "img": None, "n_prompt": n_prompt},
                    device="cuda",
                )
            )
        )
        results = await asyncio.gather(*tasks)
        # text_encoders
        return results[0]

    async def run_input_encoder_server_i2v(self):
        prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
        n_prompt = self.config.get("negative_prompt", "")
        img = Image.open(self.config["image_path"]).convert("RGB")
        clip_encoder_out, vae_encode_out, text_encoder_output = await self.post_encoders_i2v(prompt, img, n_prompt)
        torch.cuda.empty_cache()
        gc.collect()
        return self.get_encoder_output_i2v(clip_encoder_out, vae_encode_out, text_encoder_output, img)

    async def run_input_encoder_server_t2v(self):
        prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
        n_prompt = self.config.get("negative_prompt", "")
        text_encoder_output = await self.post_encoders_t2v(prompt, n_prompt)
        torch.cuda.empty_cache()
        gc.collect()
        return {"text_encoder_output": text_encoder_output, "image_encoder_output": None}

    async def run_dit_server(self, kwargs):
        if self.inputs.get("image_encoder_output", None) is not None:
            self.inputs["image_encoder_output"].pop("img", None)
        dit_output = await self.post_task(
            task_type="dit",
            urls=self.config["sub_servers"]["dit"],
            message={"task_id": generate_task_id(), "inputs": self.tensor_transporter.prepare_tensor(self.inputs), "kwargs": self.tensor_transporter.prepare_tensor(kwargs)},
            device="cuda",
        )
        return dit_output, None

    async def run_vae_decoder_server(self, latents, generator):
        images = await self.post_task(
            task_type="vae_model/decoder",
            urls=self.config["sub_servers"]["vae_model"],
            message={"task_id": generate_task_id(), "latents": self.tensor_transporter.prepare_tensor(latents)},
            device="cpu",
        )
        return images

    async def run_pipeline(self):
        if self.config["use_prompt_enhancer"]:
            self.config["prompt_enhanced"] = self.post_prompt_enhancer()
        self.inputs = await self.run_input_encoder()
        kwargs = self.set_target_shape()
        latents, generator = await self.run_dit(kwargs)
        images = await self.run_vae_decoder(latents, generator)
        self.save_video(images)
        del latents, generator, images
        torch.cuda.empty_cache()
        gc.collect()
