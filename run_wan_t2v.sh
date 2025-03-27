#!/bin/bash

# model_path=/mnt/nvme1/yongyang/models/hy/ckpts # H800-13
# model_path=/workspace/wan/Wan2.1-T2V-1.3B # H800-14
# config_path=/workspace/wan/Wan2.1-T2V-1.3B/config.json
model_path=/mnt/nvme0/yongyang/projects/wan/Wan2.1-T2V-1.3B # H800-14
config_path=/mnt/nvme0/yongyang/projects/wan/Wan2.1-T2V-1.3B/config.json

export CUDA_VISIBLE_DEVICES=0
python main.py \
--model_cls wan2.1 \
--task t2v \
--model_path $model_path \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--infer_steps 50 \
--target_video_length 81 \
--target_width  832 \
--target_height 480 \
--attention_type flash_attn2 \
--seed 42 \
--sample_neg_promp 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--config_path $config_path \
--save_video_path ./output_lightx2v_seed42.mp4 \
--sample_guide_scale 6 \
--sample_shift 8
# --mm_config '{"mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm", "weight_auto_quant": true}' 