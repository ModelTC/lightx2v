# 模型转换工具

一款功能强大的实用工具，可在不同格式之间转换模型权重并执行量化任务。

## Diffusers
支持 Diffusers 架构与 LightX2V 架构之间的相互转换

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


## 量化

该工具支持将 **FP32/FP16/BF16** 模型权重转换为 **INT8、FP8** 类型。在本项目中，主要用于将**Wan2.1** 的 **Dit**, **CLIPModel**和 **T5EncoderModel** 等模型离线转换为 8 位权重，在保持性能的同时显著减小模型体积。

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
