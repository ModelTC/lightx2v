#!/bin/bash

# set path and first
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="$(dirname "$script_dir")"

model_path=/mnt/aigc/shared_data/cache/huggingface/hub/Wan2.1-I2V-14B-480P
config_path=$model_path/config.json
lora_path=/mnt/aigc/shared_data/wan_quant/lora/toy_zoe_epoch_324.safetensors
# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using defalt value: ${cuda_devices}, change at shell script or set env variable."
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
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

python -m lightx2v \
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
--sample_neg_promp "画面过曝，模糊，文字，字幕" \
--config_path $config_path \
--save_video_path ./output_lightx2v_wan_i2v.mp4 \
--sample_guide_scale 5 \
--sample_shift 5 \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
--lora_path ${lora_path} \
--feature_caching Tea \
--mm_config '{"mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm", "weight_auto_quant": true}' \
# --mm_config '{"mm_type": "Default", "weight_auto_quant": true}' \
# --use_ret_steps \
