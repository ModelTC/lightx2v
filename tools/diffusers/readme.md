

# Model Converter Tool


接受pth或者safetensor格式，输出pth或者safetensor格式


## 模型结构转换，diffusers格式和lightx2v格式互相转换







## Quantization
This tool supports converting floating-point model weights to INT8、FP8 type

In this project, it is mainly used for offline conversion of models such as **CLIPModel** and **T5EncoderModel** of **Wan2.1** to 8-bit weights, significantly reducing the model size while maintaining performance.


### DIT

```bash
python converter.py \
    --quantized \
    --source /path/to/wan/models_t5_umt5-xxl-enc-bf16.pth \
    --output /path/to/output \
    --output_ext .pth\
    --output_name models_t5_umt5-xxl-enc-int8 \
    --key_idx 2 \
    --target_keys attn ffn \
    --dtype torch.int8
```


### T5EncoderModel

```bash
python converter.py \
    --quantized \
    --source /path/to/wan/models_t5_umt5-xxl-enc-bf16.pth \
    --output /path/to/output \
    --output_ext .pth\
    --output_name models_t5_umt5-xxl-enc-int8 \
    --key_idx 2 \
    --target_keys attn ffn \
    --dtype torch.int8
```

### CLIPModel

```bash
python converter.py \
  --source /path/to/wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --quantized \
  --output /path/to/output \
  --output_ext .pth \
  --output_name clip_int8.pth \
  --key_idx 3 \
  --target_keys attn mlp \
  --ignore_key textual \
  --dtype torch.int8
```
