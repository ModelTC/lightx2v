#!/bin/bash

# Default values
num_gpus=1
lightx2v_path=
model_path=

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            num_gpus="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--num_gpus <number_of_gpus>]"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$lightx2v_path" ]; then
    echo "Error: lightx2v_path not set"
    exit 1
fi

if [ -z "$model_path" ]; then
    echo "Error: model_path not set"
    exit 1
fi

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export ENABLE_PROFILING_DEBUG=true
export ENABLE_GRAPH_MODE=false

# Start multiple servers
python -m lightx2v.api_multi_server \
    --num_gpus $num_gpus \
    --start_port 8000 \
    --model_cls wan2.1 \
    --task t2v \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/wan_t2v.json
