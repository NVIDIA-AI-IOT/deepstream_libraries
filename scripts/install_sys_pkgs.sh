#!/bin/bash -e

################################################################################
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
    gcc-12 g++-12 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
update-alternatives --set gcc /usr/bin/gcc-12
update-alternatives --set g++ /usr/bin/g++-12

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

# Install NVIDIA NSIGHT 2024.6.1
cd /tmp
wget https://developer.download.nvidia.com/devtools/nsight-systems/nsight-systems-2024.6.1_2024.6.1.90-1_amd64.deb
apt-get update && apt-get install -y \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxext6 \
    libx11-dev \
    libxkbfile-dev \
    ./nsight-systems-2024.6.1_2024.6.1.90-1_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

# Install venv
apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*