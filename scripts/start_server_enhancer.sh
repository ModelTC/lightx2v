#!/bin/bash

# set path and first
lightx2v_path=
model_path=
prompt_enhancer_path=

# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0,1
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

python -m lightx2v.api_server \
--model_cls wan2.1_causal \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan_t2v_causal.json \
--prompt_enhancer ${prompt_enhancer_path} \
--port 8000
