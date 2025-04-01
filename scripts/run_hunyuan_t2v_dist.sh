#!/bin/bash

lightx2v_path=/mtc/yongyang/projects/lightx2v
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,1,2,3

model_path=/mtc/yongyang/models/x2v_models/hunyuan/lightx2v_format/t2v


torchrun --nproc_per_node=4 ${lightx2v_path}/lightx2v/__main__.py \
--model_cls hunyuan \
--model_path $model_path \
--prompt "A cat walks on the grass, realistic style." \
--infer_steps 20 \
--target_video_length 33 \
--target_height 720 \
--target_width 1280 \
--attention_type flash_attn2 \
--mm_config '{"mm_type": "W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm", "weight_auto_quant": true}' \
--parallel_attn