#!/bin/bash

# 设置路径
lightx2v_path=/mnt/aigc/users/lijiaqi2/ComfyUI/custom_nodes/ComfyUI-Lightx2vWrapper/lightx2v
model_path=/mnt/aigc/users/lijiaqi2/wan_model/Wan2.1-I2V-14B-720P-cfg

# 检查参数
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=2,3
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using default value: ${cuda_devices}"
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

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export ENABLE_PROFILING_DEBUG=true
export ENABLE_GRAPH_MODE=false
export DTYPE=BF16

echo "=========================================="
echo "启动分布式推理API服务器"
echo "模型路径: $model_path"
echo "CUDA设备: $CUDA_VISIBLE_DEVICES"
echo "API端口: 8000"
echo "=========================================="

# 启动API服务器，同时启动分布式推理服务
python -m lightx2v.api_server_dist \
--model_cls wan2.1 \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan_i2v_dist.json \
--port 8000 \
--start_inference \
--nproc_per_node 2

echo "服务已停止"
