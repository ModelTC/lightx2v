# 模型转换工具

A powerful utility for converting model weights between different formats and performing quantization tasks.

## Diffusers
Facilitates mutual conversion between diffusers architecture and lightx2v architecture

### Lightx2v->Diffusers
```bash
python converter.py \
       --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P \
       --output /Path/To/Wan2.1-I2V-14B-480P-Diffusers \
       --direction forward \
       --save_by_block
```

### Diffusers->Lightx2v
```bash
python converter.py \
       --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
       --output /Path/To/Wan2.1-I2V-14B-480P \
       --direction backward \
       --save_by_block
```


## Quantization
This tool supports converting fp32/fp16/bf16 model weights to INT8、FP8 type.


### Wan DIT

```bash
python converter.py \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/ \
    --output /Path/To/output \
    --output_ext .safetensors \
    --output_name wan_int8 \
    --dtype torch.int8 \
    --model_type wan_dit \
    --quantized \
    --save_by_block
```

```bash
python converter.py \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/ \
    --output /Path/To/output \
    --output_ext .safetensors \
    --output_name wan_fp8 \
    --dtype torch.float8_e4m3fn \
    --model_type wan_dit \
    --quantized \
    --save_by_block
```

### Wan DiT + LoRA

```bash
python converter.py \
    --source /Path/To/Wan-AI/Wan2.1-T2V-14B/ \
    --output /Path/To/output \
    --output_ext .safetensors \
    --output_name wan_int8 \
    --dtype torch.int8 \
    --model_type wan_dit \
    --lora_path /Path/To/LoRA1/ /Path/To/LoRA2/ \
    --lora_alpha 1.0 1.0 \
    --quantized \
    --save_by_block
```

### Hunyuan DIT

```bash
python converter.py \
    --source /Path/To/hunyuan/lightx2v_format/i2v/ \
    --output /Path/To/output \
    --output_ext ..safetensors \
    --output_name hunyuan_int8 \
    --dtype torch.int8 \
    --model_type hunyuan_dit \
    --quantized
```

```bash
python converter.py \
    --source /Path/To/hunyuan/lightx2v_format/i2v/ \
    --output /Path/To/output \
    --output_ext .safetensors \
    --output_name hunyuan_fp8 \
    --dtype torch.float8_e4m3fn \
    --model_type hunyuan_dit \
    --quantized
```


### Wan T5EncoderModel

```bash
python converter.py \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
    --output /Path/To/output \
    --output_ext .pth\
    --output_name models_t5_umt5-xxl-enc-int8 \
    --dtype torch.int8 \
    --model_type wan_t5 \
    --quantized
```

```bash
python converter.py \
    --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
    --output /Path/To/output \
    --output_ext .pth\
    --output_name models_t5_umt5-xxl-enc-fp8 \
    --dtype torch.float8_e4m3fn \
    --model_type wan_t5 \
    --quantized
```


### Wan CLIPModel

```bash
python converter.py \
  --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --output /Path/To/output \
  --output_ext .pth \
  --output_name clip-int8 \
  --dtype torch.int8 \
  --model_type wan_clip \
  --quantized

```
```bash
python converter.py \
  --source /Path/To/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --output /Path/To/output \
  --output_ext .pth \
  --output_name clip-fp8 \
  --dtype torch.float8_e4m3fn \
  --model_type wan_clip \
  --quantized
```
