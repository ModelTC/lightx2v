# 模型转换工具

A powerful utility for converting model weights between different formats and performing quantization tasks.

## Diffusers
Facilitates mutual conversion between diffusers architecture and lightx2v architecture

### Lightx2v->Diffusers
```bash
python converter.py \
       --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P \
       --output /Path/To/Wan2.1-I2V-14B-480P-Diffusers \
       --direction forward
```

### Diffusers->Lightx2v
```bash
python converter.py \
       --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
       --output /Path/To/Wan2.1-I2V-14B-480P \
       --direction backward
```


## Quantization
This tool supports converting fp32/fp16/bf16 model weights to INT8、FP8 type. In this project, it is mainly used for offline conversion of models such as **Dit**, **CLIPModel** and **T5EncoderModel** of **Wan2.1** to 8-bit weights, significantly reducing the model size while maintaining performance.


### DIT

```bash
python converter.py \
    --quantized \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/ \
    --output /Path/To/output \
    --output_ext .pth\
    --output_name wan_int8 \
    --key_idx 2 \
    --target_keys self_attn cross_attn ffn \
    --dtype torch.int8
```

```bash
python converter.py \
    --quantized \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/ \
    --output /Path/To/output \
    --output_ext .pth\
    --output_name wan_fp8 \
    --key_idx 2 \
    --target_keys self_attn cross_attn ffn \
    --dtype torch.float8_e4m3_fn
```


### T5EncoderModel

```bash
python converter.py \
    --quantized \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
    --output /Path/To/output \
    --output_ext .pth\
    --output_name models_t5_umt5-xxl-enc-int8 \
    --key_idx 2 \
    --target_keys attn ffn \
    --dtype torch.int8
```

```bash
python converter.py \
    --quantized \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
    --output /Path/To/output \
    --output_ext .pth\
    --output_name models_t5_umt5-xxl-enc-fp8 \
    --key_idx 2 \
    --target_keys attn ffn \
    --dtype torch.float8_e4m3fn
```


### CLIPModel

```bash
python converter.py \
  --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --quantized \
  --output /Path/To/output \
  --output_ext .pth \
  --output_name clip_int8.pth \
  --key_idx 3 \
  --target_keys attn mlp \
  --ignore_key textual \
  --dtype torch.int8
```
```bash
python converter.py \
  --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --quantized \
  --output /Path/To/output \
  --output_ext .pth \
  --output_name clip_fp8.pth \
  --key_idx 3 \
  --target_keys attn mlp \
  --ignore_key textual \
  --dtype torch.float8_e4m3fn
```
