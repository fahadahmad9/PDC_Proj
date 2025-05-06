FROM ubuntu:22.04

RUN apt update && apt install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    ocl-icd-opencl-dev \
    clinfo \
    nano \
    git

WORKDIR /workspace
CMD ["/bin/bash"]
