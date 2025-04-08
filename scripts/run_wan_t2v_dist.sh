#!/bin/bash

# set path and first
lightx2v_path=""
model_path=""
config_path=""

# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using defalt value: 0,1,2,3 , change at shell script or set env variable."
    cuda_devices="0,1,2,3"
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

if [ -z "${lightx2v_path}" ]; then
    echo "Error: lightx2v_path is not set. Please set this variable first."
    exit 1
fi

if [ -z "${model_path}" ]; then
    echo "Error: model_path is not set. Please set this variable first."
    exit 1
fi

if [ -z "${config_path}" ]; then
    echo "Error: config_path is not set. Please set this variable first."
    exit 1
fi

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH


torchrun --nproc_per_node=4 ${lightx2v_path}/lightx2v/__main__.py \
--model_cls wan2.1 \
--task t2v \
--model_path $model_path \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--infer_steps 50 \
--target_video_length 84 \
--target_width  832 \
--target_height 480 \
--attention_type flash_attn2 \
--seed 42 \
--sample_neg_promp 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--config_path $config_path \
--save_video_path ./output_lightx2v_seed42.mp4 \
--sample_guide_scale 6 \
--sample_shift 8 \
--parallel_attn_type ring \
--parallel_vae \
--save_video_path ./output_lightx2v_wan_t2v_dist_ring.mp4

torchrun --nproc_per_node=4 ${lightx2v_path}/lightx2v/__main__.py \
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
--sample_shift 8 \
--parallel_attn_type ulysses \
--parallel_vae \
--save_video_path ./output_lightx2v_wan_t2v_dist_ulysses.mp4
