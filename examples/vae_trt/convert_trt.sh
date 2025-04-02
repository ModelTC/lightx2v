#!/bin/bash
export PYTHONPATH="./":$PYTHONPATH
# trtexec \
#     --onnx="/mnt/nvme0/wq/project/sd/code/lightx2v/vae_decoder_hf_sim.onnx" \
#     --saveEngine="./vae_decoder_hf_sim.engine" \
#     --allowWeightStreaming \
#     --stronglyTyped \
#     --fp16 \
#     --weightStreamingBudget=100 \
#     --minShapes=inp:1x16x9x18x16 \
#     --optShapes=inp:1x16x17x32x16 \
#     --maxShapes=inp:1x16x17x32x32

python examples/vae_trt/convert_vae_trt_engine.py --model_path "/mnt/nvme1/yongyang/models/hy/ckpts"
