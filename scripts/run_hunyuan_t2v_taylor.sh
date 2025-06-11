#!/bin/bash

# set path and first
lightx2v_path="/home/yangrongjin/lightx2v"
model_path="/mtc/yongyang/models/x2v_models/hunyuan/lightx2v_format/t2v"

# check section
export CUDA_VISIBLE_DEVICES=3
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using default value: ${cuda_devices}, change at shell script or set env variable."
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

export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH

export ENABLE_PROFILING_DEBUG=true

python -m lightx2v.infer \
--model_cls hunyuan \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/caching/hunyuan_t2v_taylor.json \
--prompt "A cat walks on the grass, realistic style." \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_hy_t2v_taylor.mp4
