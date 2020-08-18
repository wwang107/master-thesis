#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ARG PYTORCH="1.4"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/miniconda3/bin:$PATH
ENV CPLUS_INCLUDE_PATH /opt/miniconda3/include

RUN echo "hey"
RUN apt-get update
RUN apt-get install -y apt-file
RUN apt-get update
RUN apt-get install -y build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    gfortran \
    libjpeg8-dev libpng-dev \
    libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev \
    libxine2-dev libv4l-dev \
    liblmdb-dev libleveldb-dev libsnappy-dev \
    mesa-utils and libgl1-mesa-glx x11-apps eog \
    vim tmux curl

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install Cython
RUN pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"

RUN conda install -c conda-forge opencv
RUN conda install -c conda-forge json_tricks

RUN conda install -c conda-forge yacs
RUN conda install -c conda-forge tensorboard 

RUN adduser --disabled-password --gecos -u $UID user
USER user