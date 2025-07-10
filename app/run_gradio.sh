#!/bin/bash

lightx2v_path=/mtc/gushiqiao/llmc_workspace/lightx2v_new/lightx2v
model_path=/data/nvme0/gushiqiao/models/I2V/Wan2.1-I2V-14B-720P-Lightx2v-Step-Distill

export CUDA_VISIBLE_DEVICES=7
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export ENABLE_PROFILING_DEBUG=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python gradio_demo.py \
    --model_path $model_path \
    --server_name 0.0.0.0 \
    --server_port 8005

# python gradio_demo_zh.py \
#     --model_path $model_path \
#     --server_name 0.0.0.0 \
#     --server_port 8005
