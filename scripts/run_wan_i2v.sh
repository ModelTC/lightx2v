#!/bin/bash

# set path and first
lightx2v_path=""
model_path=""

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

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH

export ENABLE_PROFILING_DEBUG=true
export ENABLE_GRAPH_MODE=false

python ${lightx2v_path}/lightx2v/__main__.py \
--model_cls wan2.1 \
--task i2v \
--model_path $model_path \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--infer_steps 40 \
--target_video_length 81 \
--target_width  832 \
--target_height 480 \
--attention_type flash_attn3 \
--seed 42 \
--sample_neg_promp 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--save_video_path ./output_lightx2v_wan_i2v.mp4 \
--sample_guide_scale 5 \
--sample_shift 5 \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
--mm_config '{"mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm", "weight_auto_quant": true}' \
# --feature_caching Tea \
# --use_ret_steps \
