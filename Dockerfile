FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"] 

ENV PATH ~/anaconda3/bin:$PATH

RUN cd /etc/apt/sources.list.d && \
    mv cuda.list cuda.list.disabled && \
    mv nvidia-ml.list nvidia-ml.list.disabled

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        bash \
        sudo \
        wget

RUN wget -q https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O anaconda.sh && \
    bash anaconda.sh -b -p ~/anaconda3 && \
    rm anaconda.sh

RUN source ~/anaconda3/bin/activate
RUN conda init
