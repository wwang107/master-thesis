FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
RUN conda install -c conda-forge tensorboard
RUN conda install -c conda-forge pytorch-lightning
RUN conda install -c conda-forge yacs
RUN conda install -c conda-forge opencv
RUN conda install -c conda-forge json_tricks
RUN conda install -c conda-forge pycocotools
RUN conda install -c conda-forge gdown
RUN conda install -c anaconda scipy
RUN conda install -c anaconda scikit-image
RUN conda install -c numba numba