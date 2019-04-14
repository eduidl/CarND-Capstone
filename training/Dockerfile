FROM nvidia/cuda:10.0-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      libgtk2.0-dev \
      libprotobuf-dev \
      libprotoc-dev \
      protobuf-compiler \
      libhdf5-dev \
      python-dev \
      python-pip \
      python-setuptools \
      python-wheel \
      python-pil \
      python-lxml \
      python-tk \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    Cython \
    contextlib2 \
    jupyter \
    matplotlib \
    tensorflow-gpu==1.13.1 \
    tensorboard

RUN mkdir /workspace
WORKDIR /workspace
RUN cd /workspace && \
    git clone -b v1.13.0 https://github.com/tensorflow/models.git && \
    cd models/research && \
    protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH "/workspace/models/research:/workspace/models/research/slim:/workspace/models/research/object_detection:$PYTHONPATH"
