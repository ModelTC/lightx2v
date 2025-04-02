#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# model_path=/mnt/nvme1/yongyang/models/hy/ckpts # H800-13
model_path=/mtc/yongyang/models/x2v_models/wan/Wan2.1-I2V-14B-480P # H800-14
config_path=/mtc/yongyang/models/x2v_models/wan/Wan2.1-I2V-14B-480P/config.json

python main.py \
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
--config_path $config_path \
--save_video_path ./output_lightx2v_seed42_fp8_base.mp4 \
--sample_guide_scale 5 \
--sample_shift 5 \
--image_path ./i2v_input.JPG \
--mm_config '{"mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm", "weight_auto_quant": true}' \
# --feature_caching Tea \
# --use_ret_steps \