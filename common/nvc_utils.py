# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import logging

from struct import pack
import sys
import os
import argparse
from pathlib import Path
from enum import Enum

import pycuda.driver as cuda
import PyNvVideoCodec as nvc
import numpy as np
from pycuda.compiler import SourceModule
import ctypes as C

import pycuda.autoinit as context
import json

from contextlib import contextmanager

import io
import tempfile

SERVICE_LOGGING_FORMAT = (
        "[{filename:s}][{funcName:s}:{lineno:d}]" + "[{levelname:s}] {message:s}"
)
SERVICE_LOGGING_STREAM = sys.stdout


def get_logger(logger_name, log_level="info"):
    SERVICE_LOGGING_LEVEL = getattr(logging, log_level.upper(), None)

    logger = logging.getLogger(logger_name)
    logger.setLevel(SERVICE_LOGGING_LEVEL)
    ch = logging.StreamHandler(SERVICE_LOGGING_STREAM)
    formatter = logging.Formatter(SERVICE_LOGGING_FORMAT, style="{")
    ch.setFormatter(formatter)
    ch.setLevel(SERVICE_LOGGING_LEVEL)
    logger.addHandler(ch)
    logger.propagate = False

    return logger


logger = get_logger(__file__)


def cast_address_to_1d_bytearray(base_address, size):
    return np.ctypeslib.as_array(C.cast(base_address, C.POINTER(C.c_uint8)),
                                 shape=(size,))


from contextlib import contextmanager
import ctypes
import io
import os, sys
import tempfile

class AppCAI:
    def __init__(self, shape, stride, typestr, gpualloc):
        shape_int = tuple([int(x) for x in shape])
        stride_int = tuple([int(x) for x in stride])
        self.__cuda_array_interface__ = {"shape": shape_int, "strides": stride_int, "data": (int(gpualloc), False),
                                         "typestr": typestr, "version": 3}


class AppFrame:
    def __init__(self, width, height, format):
        if format == "NV12":
            nv12_frame_size = int(width * height * 3 / 2)
            self.gpuAlloc = cuda.mem_alloc(nv12_frame_size)
            self.cai = []
            self.cai.append(AppCAI((height, width, 1), (width, 1, 1), "|u1", self.gpuAlloc))
            chroma_alloc = int(self.gpuAlloc) + width * height
            self.cai.append(AppCAI((int(height / 2), int(width / 2), 2), (width, 2, 1), "|u1", chroma_alloc))
            self.frameSize = nv12_frame_size
        if format == "ARGB" or format == "ABGR":
            self.frameSize = width * height * 4
            self.gpuAlloc = cuda.mem_alloc(self.frameSize)
            self.cai = AppCAI((height, width, 4), (4 * width, 4, 1), "|u1", self.gpuAlloc)
        if format == "YUV444":
            self.frameSize = width * height * 3
            self.gpuAlloc = cuda.mem_alloc(self.frameSize)
            self.cai = []
            self.cai.append(AppCAI((height, width, 1), (width, 1, 1), "|u1", self.gpuAlloc))
            self.cai.append(AppCAI((height, width, 1), (width, 1, 1), "|u1", int(self.gpuAlloc) + width * height))
            self.cai.append(AppCAI((height, width, 1), (width, 1, 1), "|u1", int(self.gpuAlloc) + 2 * width * height))
        if format == "YUV420":
            self.frameSize = int(width * height * 3 / 2)
            self.gpuAlloc = cuda.mem_alloc(self.frameSize)
            self.cai = []
            self.cai.append(AppCAI((height, width, 1), (width, 1, 1), "|u1", self.gpuAlloc))
            self.cai.append(
                AppCAI((height / 2, width / 2, 1), (width / 2, 1, 1), "|u1", int(self.gpuAlloc) + width * height))
            self.cai.append(AppCAI((height / 2, width / 2, 1), (width / 2, 1, 1), "|u1",
                                   int(self.gpuAlloc) + width * height + width / 2 * height / 2))
        if format == "P010":
            self.frameSize = int(width * height * 3 / 2 * 2)
            self.gpuAlloc = cuda.mem_alloc(self.frameSize)
            self.cai = []
            self.cai.append(AppCAI((height, width, 1), (width * 2, 2, 1), "|u1", self.gpuAlloc))
            self.cai.append(
                AppCAI((height / 2, width / 2, 2), (width * 2, 2, 1), "|u1", int(self.gpuAlloc) + width * height * 2))
        if format == "YUV444_16BIT":
            self.frameSize = int(width * height * 3 * 2)
            self.gpuAlloc = cuda.mem_alloc(self.frameSize)
            self.cai = []
            self.cai.append(AppCAI((height, width, 1), (width * 2, 2, 1), "|u1", self.gpuAlloc))
            self.cai.append(
                AppCAI((height, width, 1), (width * 2, 2, 1), "|u1", int(self.gpuAlloc) + width * height * 2))
            self.cai.append(
                AppCAI((height, width, 1), (width * 2, 2, 1), "|u1", int(self.gpuAlloc) + width * height * 4))

    def cuda(self):
        return self.cai


class AppFramePerf:
    def __init__(self, width, height, format, dataptr, frame_idx):

        if format == "NV12":
            nv12_frame_size = int(width * height * 3 / 2)
            self.gpuAlloc = int(dataptr) + (frame_idx * nv12_frame_size)
            self.cai = []
            self.cai.append(AppCAI((height, width, 1), (width, 1, 1), "|u1", self.gpuAlloc))
            chroma_alloc = int(self.gpuAlloc) + width * height
            self.cai.append(AppCAI((int(height / 2), int(width / 2), 2), (width, 2, 1), "|u1", chroma_alloc))
            self.frameSize = nv12_frame_size
        if format == "ARGB" or format == "ABGR":
            self.frameSize = width * height * 4
            self.gpuAlloc = int(dataptr) + (frame_idx * self.frameSize)
            self.cai = AppCAI((height, width, 4), (4 * width, 4, 1), "|u1", self.gpuAlloc)
        if format == "YUV444":
            self.frameSize = width * height * 3
            self.gpuAlloc = int(dataptr) + (frame_idx * self.frameSize)
            self.cai = []
            self.cai.append(AppCAI((height, width, 1), (width, 1, 1), "|u1", self.gpuAlloc))
            self.cai.append(AppCAI((height, width, 1), (width, 1, 1), "|u1", int(self.gpuAlloc) + width * height))
            self.cai.append(AppCAI((height, width, 1), (width, 1, 1), "|u1", int(self.gpuAlloc) + 2 * width * height))
        if format == "YUV420":
            self.frameSize = int(width * height * 3 / 2)
            self.gpuAlloc = int(dataptr) + (frame_idx * self.frameSize)
            self.cai = []
            self.cai.append(AppCAI((height, width, 1), (width, 1, 1), "|u1", self.gpuAlloc))
            self.cai.append(
                AppCAI((height / 2, width / 2, 1), (width / 2, 1, 1), "|u1", int(self.gpuAlloc) + width * height))
            self.cai.append(AppCAI((height / 2, width / 2, 1), (width / 2, 1, 1), "|u1",
                                   int(self.gpuAlloc) + width * height + width / 2 * height / 2))
        if format == "P010":
            self.frameSize = int(width * height * 3 / 2 * 2)
            self.gpuAlloc = int(dataptr) + (frame_idx * self.frameSize)
            self.cai = []
            self.cai.append(AppCAI((height, width, 1), (width * 2, 2, 1), "|u1", self.gpuAlloc))
            self.cai.append(
                AppCAI((height / 2, width / 2, 2), (width * 2, 2, 1), "|u1", int(self.gpuAlloc) + width * height * 2))
        if format == "YUV444_16BIT":
            self.frameSize = int(width * height * 3 * 2)
            self.gpuAlloc = int(dataptr) + (frame_idx * self.frameSize)
            self.cai = []
            self.cai.append(AppCAI((height, width, 1), (width * 2, 2, 1), "|u1", self.gpuAlloc))
            self.cai.append(
                AppCAI((height, width, 1), (width * 2, 2, 1), "|u1", int(self.gpuAlloc) + width * height * 2))
            self.cai.append(
                AppCAI((height, width, 1), (width * 2, 2, 1), "|u1", int(self.gpuAlloc) + width * height * 4))

    def cuda(self):
        return self.cai


def FetchGPUFrame(input_frame_list, GetCPUFrameFunc, num_frames):
    for i in range(num_frames):
        n = i % len(input_frame_list)
        raw_frame = GetCPUFrameFunc()
        if not raw_frame.size:
            return
        cuda.memcpy_htod(input_frame_list[n].gpuAlloc, raw_frame)
        yield input_frame_list[n]


def FetchCPUFrame(dec_file, frame_size):
    def InnerFunc():
        return np.fromfile(dec_file, np.uint8, count=frame_size)

    return InnerFunc
