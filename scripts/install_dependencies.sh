#!/bin/bash -e

################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

# This script installs all the dependencies required to run the CVCUDA samples.
# It uses the /tmp folder to download temporary data and libraries.

# SCRIPT_DIR is the directory where this script is located.
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

set -e  # Exit script if any command fails

# Move to parent directory
if [ ! -d "assets" ]; then
    echo "Moving to parent directory."
    cd ..
fi

# Install basic packages first.
cd /tmp
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    yasm \
    unzip \
    cmake \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add repositories and install g++
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update && apt-get install -y --no-install-recommends \
    gcc-11 g++-11 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
update-alternatives --set gcc /usr/bin/gcc-11
update-alternatives --set g++ /usr/bin/g++-11

# Install Python and gtest
apt-get update && apt-get install -y --no-install-recommends \
    libgtest-dev \
    libgmock-dev \
    python3-pip \
    ninja-build ccache \
    mlocate && updatedb \
    && rm -rf /var/lib/apt/lists/*

# Install ffmpeg and other libraries needed for VPF.
# Note: We are not installing either libnv-encode or decode libraries here.
apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libavutil-dev\
    && rm -rf /var/lib/apt/lists/*

# Install libssl 1.1.1
cd /tmp
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb
dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb

# Install tao-converter which parses the .etlt model file, and generates an optimized TensorRT engine
wget 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v4.0.0_trt8.5.1.7_x86/files/tao-converter' --directory-prefix=/usr/local/bin
chmod a+x /usr/local/bin/tao-converter

# Install NVIDIA NSIGHT 2023.2.1
cd /tmp
wget https://developer.download.nvidia.com/devtools/nsight-systems/nsight-systems-2023.2.1_2023.2.1.122-1_amd64.deb
apt-get update && apt-get install -y \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxext6 \
    libx11-dev \
    libxkbfile-dev \
    /tmp/nsight-systems-2023.2.1_2023.2.1.122-1_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

echo "export PATH=$PATH:/opt/tensorrt/bin" >> ~/.bashrc
export CPATH=$CPATH:/usr/local/cuda-12.2/targets/x86_64-linux/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-12.2/targets/x86_64-linux/lib
export PATH=/usr/local/cuda/bin:$PATH

# Upgrade pip and install all required Python packages.
pip3 install --upgrade pip
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# Install VPF
cd /tmp
[ ! -d 'VideoProcessingFramework' ] && git clone https://github.com/NVIDIA/VideoProcessingFramework.git
# HotFix: Must change the PyTorch version used by PytorchNvCodec to match the one we are using.
# # Since we are using 2.2.0 we must use that.
# sed -i 's/torch/torch==2.2.0/g' /tmp/VideoProcessingFramework/src/PytorchNvCodec/pyproject.toml
# sed -i 's/"torch"/"torch==2.2.0"/g' /tmp/VideoProcessingFramework/src/PytorchNvCodec/setup.py
pip3 install /tmp/VideoProcessingFramework
pip3 install /tmp/VideoProcessingFramework/src/PytorchNvCodec

# Install NvImageCodec
# pip3 install nvidia-nvimgcodec-cu${CUDA_MAJOR_VERSION}
pip3 install nvidia-pyindex
# pip3 install nvidia-nvjpeg-cu${CUDA_MAJOR_VERSION}

# Install NvPyVideoCodec
# cd /tmp
# wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/py_nvvideocodec/versions/0.0.9/zip -O py_nvvideocodec_0.0.9.zip
# pip3 install py_nvvideocodec_0.0.9.zip

# Done
