#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ARG PYTORCH="1.4.0"
ARG TORCHVISION="0.5.0"
ARG CUDA="9.2"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev &&\
    rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
    /opt/conda/bin/conda install -y -c pytorch magma-cuda90 && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available

RUN pip install Cython
RUN pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"

RUN conda install pytorch==${PYTORCH} torchvision==${TORCHVISION} cudatoolkit=${CUDA} -c pytorch
RUN conda install -c conda-forge opencv
RUN conda install -c conda-forge json_tricks

RUN conda install -c conda-forge yacs
RUN conda install -c conda-forge tensorboard 

RUN adduser --disabled-password --gecos -u $UID user
USER user