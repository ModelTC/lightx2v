FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS base

WORKDIR /workspace

COPY . /workspace/lightx2v/

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# use tsinghua source
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && apt install -y software-properties-common  \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y vim tmux zip unzip wget git cmake build-essential \
     curl libibverbs-dev ca-certificates iproute2 \
     ffmpeg libsm6 libxext6 \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && pip install packaging ninja vllm torch torchvision diffusers transformers \
     tokenizers accelerate safetensors opencv-python numpy imageio imageio-ffmpeg \
     einops loguru sgl-kernel qtorch ftfy

# install flash-attention 2
RUN cd lightx2v/3rd/flash-attention && pip install --no-cache-dir -v -e .

# install flash-attention 3, only if hopper
RUN cd lightx2v/3rd/flash-attention/hopper && pip install --no-cache-dir -v -e .
