#!/bin/bash

# set path and first
lightx2v_path=""
model_path=""

# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using defalt value: 0, change at shell script or set env variable."
    cuda_devices="0"
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

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH

python ${lightx2v_path}/lightx2v/__main__.py \
--model_cls hunyuan \
--model_path $model_path \
--task i2v \
--prompt "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick." \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_1.jpg \
--infer_steps 20 \
--target_video_length 33 \
--target_height 720 \
--target_width 1280 \
--attention_type flash_attn2 \
--save_video_path ./output_lightx2v_hy_i2v.mp4 \
--seed 0
