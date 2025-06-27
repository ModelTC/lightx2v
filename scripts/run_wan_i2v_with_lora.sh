#!/bin/bash

# set path and first
lightx2v_path=
model_path=
lora_path=

# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using default value: ${cuda_devices}, change at shell script or set env variable."
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

if [ -z "${model_path}" ]; then
    echo "Error: model_path is not set. Please set this variable first."
    exit 1
fi

if [ -z "${lora_path}" ]; then
    echo "Error: lora_path is not set. Please set this variable first."
    exit 1
fi

export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export DTYPE=BF16
export ENABLE_PROFILING_DEBUG=true

python -m lightx2v.infer \
--model_cls wan2.1 \
--task i2v \
--model_path $model_path \
--prompt "画面中的物体轻轻向上跃起，变成了外貌相似的毛绒玩具。毛绒玩具有着一双眼睛，它的颜色和之前的一样。然后，它开始跳跃起来。背景保持一致，气氛显得格外俏皮。" \
--infer_steps 40 \
--target_video_length 81 \
--target_width  832 \
--target_height 480 \
--attention_type flash_attn3 \
--seed 42 \
--negative_prompt "画面过曝，模糊，文字，字幕" \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_wan_i2v.mp4 \
--sample_guide_scale 5 \
--sample_shift 5 \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
--lora_path ${lora_path} \
--feature_caching Tea \
--mm_config '{"mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm", "weight_auto_quant": true}' \
# --mm_config '{"mm_type": "Default", "weight_auto_quant": true}' \
# --use_ret_steps \
