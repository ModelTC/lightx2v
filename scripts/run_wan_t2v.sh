#!/bin/bash

lightx2v_path=/mtc/yongyang/projects/lightx2v
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=2

model_path=/mtc/yongyang/models/x2v_models/wan/Wan2.1-T2V-1.3B
config_path=/mtc/yongyang/models/x2v_models/wan/Wan2.1-T2V-1.3B/config.json


python ${lightx2v_path}/lightx2v/__main__.py \
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
--save_video_path ./output_lightx2v_seed42_q8f1_teacache.mp4 \
--sample_guide_scale 6 \
--sample_shift 8 \
# --mm_config '{"mm_type": "W-int8-channel-sym-A-int8-channel-sym-dynamic-Q8F", "weight_auto_quant": true}' \
# --feature_caching Tea \
# --use_ret_steps \
# --teacache_thresh 0.2
