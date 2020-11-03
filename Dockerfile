FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
RUN conda install -c conda-forge tensorboard
RUN conda install -c conda-forge pytorch-lightning
RUN conda install -c conda-forge yacs
RUN conda install -c conda-forge opencv