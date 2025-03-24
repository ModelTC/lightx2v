#!/bin/bash

# model_path=/mnt/nvme1/yongyang/models/hy/ckpts # H800-13
model_path=/mnt/nvme0/yongyang/projects/hy/HunyuanVideo/ckpts # H800-14
# model_path=/workspace/ckpts_link # H800-14

# export CUDA_VISIBLE_DEVICES=2
# python main.py \
# --model_cls hunyuan \
# --model_path $model_path \
# --prompt "A cat walks on the grass, realistic style." \
# --infer_steps 20 \
# --target_video_length 33 \
# --target_height 720 \
# --target_width 1280 \
# --attention_type flash_attn3 \
# --save_video_path ./output_lightx2v_int8.mp4 \
# --mm_config '{"mm_type": "W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm", "weight_auto_quant": true}' 


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node=4 main.py \
# --model_cls hunyuan \
# --model_path $model_path \
# --prompt "A cat walks on the grass, realistic style." \
# --infer_steps 20 \
# --target_video_length 33 \
# --target_height 720 \
# --target_width 1280 \
# --attention_type flash_attn2 \
# --mm_config '{"mm_type": "W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm", "weight_auto_quant": true}' \
# --parallel_attn


export CUDA_VISIBLE_DEVICES=2
python main.py \
--model_cls hunyuan \
--model_path $model_path \
--prompt "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting." \
--infer_steps 50 \
--target_video_length 65 \
--target_height 480 \
--target_width 640 \
--attention_type flash_attn3 \
--cpu_offload \
--feature_caching TaylorSeer \
--save_video_path ./output_lightx2v_offload_TaylorSeer.mp4 \
# --mm_config '{"mm_type": "W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm", "weight_auto_quant": true}' 

