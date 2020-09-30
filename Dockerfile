#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ARG PYTORCH="1.4.0"
ARG TORCHVISION="0.5.0"
ARG CUDA="9.2"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*