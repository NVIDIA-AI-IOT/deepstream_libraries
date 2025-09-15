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

echo "export PATH=$PATH:/opt/tensorrt/bin" >> ~/.bashrc
export CPATH=$CPATH:/usr/local/cuda-12.8/targets/x86_64-linux/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-12.8/targets/x86_64-linux/lib
export PATH=/usr/local/cuda/bin:$PATH

# Upgrade pip and install all required Python packages.
pip3 install --upgrade pip
python3 -m pip install --upgrade --force-reinstall pip
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# Install VPF
cd /tmp
[ ! -d 'VideoProcessingFramework' ] && git clone https://github.com/NVIDIA/VideoProcessingFramework.git
# HotFix: Must change the PyTorch version used by PytorchNvCodec to match the one we are using.
# # Since we are using 2.7.0 we must use that.
sed -i 's/"torch[^"]*"/"torch==2.7.0"/g' /tmp/VideoProcessingFramework/src/PytorchNvCodec/pyproject.toml
sed -i 's/"torch[^"]*"/"torch==2.7.0"/g' /tmp/VideoProcessingFramework/src/PytorchNvCodec/setup.py
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip3 install /tmp/VideoProcessingFramework
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip3 install /tmp/VideoProcessingFramework/src/PytorchNvCodec

pip3 install nvidia-pyindex