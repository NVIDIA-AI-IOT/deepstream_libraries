#!/bin/bash -e

################################################################################
# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
CUDA_PATH=$(find /usr/local -maxdepth 1 -type d -name 'cuda*' 2>/dev/null | head -1)
[ -z "$CUDA_PATH" ] && CUDA_PATH="/usr/local/cuda"
export CPATH=$CPATH:$CUDA_PATH/targets/x86_64-linux/include
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_PATH/targets/x86_64-linux/lib
export PATH=$CUDA_PATH/bin:$PATH

# Set flags for deprecated CUDA functions (for pycuda with CUDA 13.1)
export CFLAGS="-Wno-deprecated-declarations"
export CXXFLAGS="-Wno-deprecated-declarations"

# Upgrade pip and install all required Python packages.
pip3 install --upgrade pip
python3 -m pip install --upgrade --force-reinstall pip
pip3 install setuptools wheel appdirs

# Install nvidia-pyindex first (required for tensorrt to find NVIDIA PyPI index)
pip3 install nvidia-pyindex --no-build-isolation || {
    echo "⚠ nvidia-pyindex installation failed, continuing (tensorrt may still work)..."
}
# Install Cython before av (required for building av)
pip3 install Cython==3.0.10 numpy==1.26.4
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# Install VPF
cd /tmp
if [ ! -d 'VideoProcessingFramework' ]; then
    git clone https://github.com/NVIDIA/VideoProcessingFramework.git || {
        echo "⚠ Failed to clone VPF, continuing..."
    }
fi

if [ -d 'VideoProcessingFramework' ]; then
    # HotFix: Must change the PyTorch version used by PytorchNvCodec to match the one we are using.
    # Since we are using 2.7.0 we must use that.
    sed -i 's/"torch[^"]*"/"torch==2.7.0"/g' /tmp/VideoProcessingFramework/src/PytorchNvCodec/pyproject.toml 2>/dev/null || true
    sed -i 's/"torch[^"]*"/"torch==2.7.0"/g' /tmp/VideoProcessingFramework/src/PytorchNvCodec/setup.py 2>/dev/null || true
    
    # Install VPF main package (deprecated, may fail due to pkg_resources or CUDA 13.1 issues)
    # PyNvVideoCodec from wheel is the modern replacement and already installed
    CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip3 install /tmp/VideoProcessingFramework || {
        echo "⚠ VPF main package installation failed (this is OK, PyNvVideoCodec from wheel is the modern replacement)"
    }
    
    # Install PytorchNvCodec extension (optional, for PyTorch support)
    CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip3 install /tmp/VideoProcessingFramework/src/PytorchNvCodec || {
        echo "⚠ PytorchNvCodec installation failed, continuing (optional component)..."
    }
else
    echo "⚠ VPF directory not found, skipping VPF installation..."
fi