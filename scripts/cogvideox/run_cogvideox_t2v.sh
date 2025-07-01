#!/bin/bash

# set path and first
lightx2v_path=
model_path=

# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using defalt value: ${cuda_devices}, change at shell script or set env variable."
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
export ENABLE_GRAPH_MODE=false
export DTYPE=BF16
export PYTHONPATH=/mtc/wushuo/VideoGen/diffusers:$PYTHONPATH

python -m lightx2v.infer \
--model_cls cogvideox \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/cogvideox/cogvideox_t2v.json \
--prompt "A little girl smile." \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_cogvideox_t2v.mp4
