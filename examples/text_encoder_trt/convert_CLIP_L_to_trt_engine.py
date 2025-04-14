import os
import argparse

import torch
from loguru import logger

from lightx2v.text2v.models.text_encoders.hf.clip.model import TextEncoderHFClipModel
from lightx2v.text2v.models.text_encoders.trt.clip.trt_clip_infer import CLIPTrtModelInfer


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", help="", type=str, default="/mtc/yongyang/models/x2v_models/hunyuan/lightx2v_format/t2v")
    args.add_argument("--dtype", default=torch.float32)
    args.add_argument("--device", default="cuda", type=str)
    return args.parse_args()


def convert_trt_engine(args):
    init_device = torch.device(args.device)
    text_encoder_2 = TextEncoderHFClipModel(os.path.join(args.model_path, "text_encoder_2"), init_device)
    texts = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    tokens = text_encoder_2.tokenizer(
                    texts,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_attention_mask=True,
                    truncation=True,
                    max_length=text_encoder_2.max_length,
                    padding="max_length",
                    return_tensors="pt",
                ).to(init_device)
    input_ids=tokens["input_ids"].to(init_device)
    attention_mask=tokens["attention_mask"].to(init_device)
    onnx_path = CLIPTrtModelInfer.export_to_onnx(text_encoder_2.model, model_dir=args.model_path, input_ids=input_ids, attention_mask=attention_mask)
    del text_encoder_2
    torch.cuda.empty_cache()
    engine_path = onnx_path.replace(".onnx", ".engine")
    CLIPTrtModelInfer.convert_to_trt_engine(onnx_path, engine_path)
    logger.info(f"ONNX: {onnx_path}")
    logger.info(f"TRT Engine: {engine_path}")
    return


def main():
    args = parse_args()
    convert_trt_engine(args)


if __name__ == "__main__":
    main()
