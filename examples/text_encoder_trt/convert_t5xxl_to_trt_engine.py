from pathlib import Path
import os
import argparse

import torch
from loguru import logger

from lightx2v.text2v.models.text_encoders.hf.t5.model import T5EncoderModel
from lightx2v.text2v.models.text_encoders.trt.t5.trt_t5_infer import T5TrtModelInfer


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", help="", type=str, default="models/Wan2.1-T2V-1.3B")
    args.add_argument("--dtype", default=torch.float16)
    args.add_argument("--device", default="cuda", type=str)
    return args.parse_args()


def convert_trt_engine(args):
    t5_checkpoint_path = os.path.join(args.model_path, "models_t5_umt5-xxl-enc-bf16.pth")
    t5_tokenizer_path  = os.path.join(args.model_path, "google/umt5-xxl")
    assert Path(t5_checkpoint_path).exists(), f"{t5_checkpoint_path} not exists."
    model = T5EncoderModel(
        text_len=512,
        dtype=args.dtype,
        device=args.device,
        checkpoint_path=t5_checkpoint_path,
        tokenizer_path=t5_tokenizer_path,
        shard_fn=None
    )
    texts = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ids, mask = model.tokenizer(texts, return_mask=True, add_special_tokens=True)
    ids  = ids.to(args.device)
    mask = mask.to(args.device)
    onnx_path = T5TrtModelInfer.export_to_onnx(model.model, model_dir=args.model_path, ids=ids, mask=mask)
    del model
    torch.cuda.empty_cache()
    engine_path = onnx_path.replace(".onnx", ".engine")
    T5TrtModelInfer.convert_to_trt_engine(onnx_path, engine_path)
    logger.info(f"ONNX: {onnx_path}")
    logger.info(f"TRT Engine: {engine_path}")
    return


def main():
    args = parse_args()
    convert_trt_engine(args)


if __name__ == "__main__":
    main()
