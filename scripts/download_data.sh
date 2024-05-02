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

# Move to parent directory
if [ ! -d "assets" ]; then
    echo "Moving to parent directory."
    cd ..
fi
## CVCUDA Image Samples
wget -P assets/images/ https://github.com/CVCUDA/CV-CUDA/raw/main/samples/assets/images/Weimaraner.jpg
wget -P assets/images/ https://github.com/CVCUDA/CV-CUDA/raw/main/samples/assets/images/peoplenet.jpg
wget -P assets/images/ https://github.com/CVCUDA/CV-CUDA/raw/main/samples/assets/images/tabby_tiger_cat.jpg

## CVCUDA Video Samples
wget -P assets/videos/ https://github.com/CVCUDA/CV-CUDA/raw/main/samples/assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4
wget -P assets/videos/ https://github.com/CVCUDA/CV-CUDA/raw/main/samples/assets/videos/pexels-ilimdar-avgezer-7081456.mp4
