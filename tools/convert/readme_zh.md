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

该工具支持将 **FP32/FP16/BF16** 模型权重转换为 **INT8、FP8** 类型。

### Wan DIT

```bash
python converter.py \
    --quantized \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/ \
    --output /Path/To/output \
    --output_ext .safetensors \
    --output_name wan_int8 \
    --dtype torch.int8 \
    --model_type wan_dit
```

```bash
python converter.py \
    --quantized \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/ \
    --output /Path/To/output \
    --output_ext .safetensors \
    --output_name wan_fp8 \
    --dtype torch.float8_e4m3fn \
    --model_type wan_dit
```

### Hunyuan DIT

```bash
python converter.py \
    --quantized \
    --source /Path/To/hunyuan/lightx2v_format/i2v/ \
    --output /Path/To/output \
    --output_ext ..safetensors \
    --output_name hunyuan_int8 \
    --dtype torch.int8 \
    --model_type hunyuan_dit
```

```bash
python converter.py \
    --quantized \
    --source /Path/To/hunyuan/lightx2v_format/i2v/ \
    --output /Path/To/output \
    --output_ext .safetensors \
    --output_name hunyuan_fp8 \
    --dtype torch.float8_e4m3fn \
    --model_type hunyuan_dit
```


### Wan T5EncoderModel

```bash
python converter.py \
    --quantized \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
    --output /Path/To/output \
    --output_ext .pth\
    --output_name models_t5_umt5-xxl-enc-int8 \
    --dtype torch.int8 \
    --model_type wan_t5
```

```bash
python converter.py \
    --quantized \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
    --output /Path/To/output \
    --output_ext .pth\
    --output_name models_t5_umt5-xxl-enc-fp8 \
    --dtype torch.float8_e4m3fn \
    --model_type wan_t5
```


### Wan CLIPModel

```bash
python converter.py \
  --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --quantized \
  --output /Path/To/output \
  --output_ext .pth \
  --output_name clip_int8.pth \
  --dtype torch.int8 \
  --model_type wan_clip

```
```bash
python converter.py \
  --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --quantized \
  --output /Path/To/output \
  --output_ext .pth \
  --output_name clip_fp8.pth \
  --dtype torch.float8_e4m3fn \
  --model_type wan_clip
```
