import torch
from tqdm import tqdm
import gc
import argparse
import os
import logging
from typing import Dict, List, Tuple
import safetensors
import safetensors.torch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def quantize_model(
    weights: Dict[str, torch.Tensor], w_bit: int = 8, target_keys: List[str] = ["attn", "ffn"], min_params: int = 1e6, key_idx: int = 2, ignore_key: str = None
) -> Dict[str, torch.Tensor]:
    """
    Quantize model weights in-place

    Args:
        weights: Model state dictionary
        w_bit: Quantization bit width
        target_keys: List of module names to quantize
        min_params: Minimum parameter count to process tensor

    Returns:
        Modified state dictionary with quantized weights and scales
    """
    total_quantized = 0
    total_size = 0
    keys = list(weights.keys())

    for key in tqdm(keys, desc="Quantizing weights"):
        if ignore_key is not None and ignore_key in key:
            continue

        tensor = weights[key]

        # Skip non-tensors, small tensors, and non-2D tensors
        if not isinstance(tensor, torch.Tensor) or tensor.numel() < min_params or tensor.dim() != 2:
            continue

        # Check if key matches target modules
        parts = key.split(".")
        if len(parts) < key_idx + 1 or parts[key_idx] not in target_keys:
            continue

        try:
            # Quantize tensor and store results
            w_q, scales = quantize_tensor(tensor, w_bit)

            # Replace original tensor and store scales
            weights[key] = w_q
            weights[key + "_scale"] = scales

            total_quantized += 1
            total_size += tensor.numel() * tensor.element_size() / (1024**2)  # MB
            del w_q, scales

        except Exception as e:
            logger.error(f"Error quantizing {key}: {str(e)}")

        gc.collect()

    logger.info(f"Quantized {total_quantized} tensors, reduced size by {total_size:.2f} MB")
    return weights


def main():
    parser = argparse.ArgumentParser(description="Model Quantization Tool")
    parser.add_argument("--input", type=str, required=True, help="Input model file (.pth or .safetensors)")
    parser.add_argument("--output", type=str, required=True, help="Output quantized model file")
    parser.add_argument("--bits", type=int, default=8, choices=[8], help="Quantization bit width")
    parser.add_argument("--target_keys", nargs="+", default=["attn", "ffn"], help="Module keys to quantize (e.g., attn ffn)")
    parser.add_argument("--min_params", type=int, default=1000000, help="Minimum parameters to consider for quantization")
    parser.add_argument("--ignore_key", type=str, default=None)
    parser.add_argument("--key_idx", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for quantization (cpu/cuda)")
    args = parser.parse_args()

    # Validate input/output formats
    input_ext = os.path.splitext(args.input)[1].lower()
    output_ext = ".safetensors"

    if input_ext not in [".pth", ".safetensors"]:
        raise ValueError(f"Unsupported input format: {input_ext}")
    if output_ext not in [".pth", ".safetensors"]:
        raise ValueError(f"Unsupported output format: {output_ext}")

    logger.info(f"Quantization config: {args.bits}-bit, target keys: {args.target_keys}, min params: {args.min_params:,}")

    # Load model weights
    logger.info(f"Loading weights from {args.input}")
    if input_ext == ".safetensors":
        weights = safetensors.torch.load_file(args.input, device=args.device)
    else:
        weights = torch.load(args.input, map_location=args.device, weights_only=True)

    # Quantize model
    quantized_weights = quantize_model(weights, w_bit=args.bits, target_keys=args.target_keys, min_params=args.min_params, key_idx=args.key_idx)

    # Save quantized model
    logger.info(f"Saving quantized model to {args.output}")
    if output_ext == ".safetensors":
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        safetensors.torch.save_file(quantized_weights, args.output)
    else:
        torch.save(quantized_weights, args.output)

    logger.info("Quantization completed successfully")


if __name__ == "__main__":
    main()
