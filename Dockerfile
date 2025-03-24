FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y vim tmux zip unzip wget git cmake build-essential software-properties-common curl libibverbs-dev ca-certificates iproute2 ffmpeg libsm6 libxext6 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN pip install packaging ninja

RUN pip install vllm

RUN pip install torch torchvision

# FROM tmp-image AS base

WORKDIR /workspace

# download flash-attention source code
# git clone https://github.com/Dao-AILab/flash-attention.git --recursive
COPY flash-attention /workspace/flash-attention

RUN cd flash-attention && pip install --no-cache-dir -v -e .

RUN cd flash-attention/hopper && pip install --no-cache-dir -v -e .

RUN pip install diffusers transformers tokenizers accelerate safetensors opencv-python numpy imageio imageio-ffmpeg einops loguru

RUN pip install sgl-kernel


# FROM registry.cn-sh-01.sensecore.cn/devsft-ccr-2/video-gen:25030702 AS base

RUN pip install qtorch ftfy
