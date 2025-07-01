#!/bin/bash

# set path and first
lightx2v_path=
model_path=

# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0,1,2,3
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
export DTYPE=BF16
export ENABLE_PROFILING_DEBUG=true

torchrun --nproc_per_node=4 ${lightx2v_path}/lightx2v/infer.py \
--model_cls hunyuan \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/hunyuan/hunyuan_t2v_dist_ulysses.json \
--prompt "A cat walks on the grass, realistic style." \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_hunyuan_t2v_dist_ulysses.mp4

torchrun --nproc_per_node=4 ${lightx2v_path}/lightx2v/infer.py \
--model_cls hunyuan \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/hunyuan/hunyuan_t2v_dist_ring.json \
--prompt "A cat walks on the grass, realistic style." \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_hunyuan_t2v_dist_ring.mp4
