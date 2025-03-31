#!/bin/bash

# model_path=/mnt/nvme1/yongyang/models/hy/ckpts # H800-13
# model_path=/mnt/nvme0/yongyang/projects/hy/HunyuanVideo/ckpts # H800-14
model_path=/workspace/ckpts_link # H800-14

export CUDA_VISIBLE_DEVICES=2
python main.py \
--model_cls hunyuan \
--model_path $model_path \
--prompt "A cat walks on the grass, realistic style." \
--infer_steps 20 \
--target_video_length 33 \
--target_height 720 \
--target_width 1280 \
--attention_type flash_attn3 \
--save_video_path ./output_lightx2v_int8.mp4 \
--mm_config '{"mm_type": "W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm", "weight_auto_quant": true}'
