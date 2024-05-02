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
import json
import argparse
import numpy as np
sys.path.append('../')
from pathlib import Path
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
from common.nvc_utils import AppFrame


def transcode(gpu_id, in_file_path, out_file_path, config_params):
    """
                    This function demonstrates transcoding of an input video stream.

                    Parameters:
                        - gpu_id (int): Ordinal of GPU to use [Parameter not in use]
                        - in_file_path (str): Path to
                    file to be decoded
                        - out_file_path (str): Path to output file into which raw frames are stored
                        - config_params(key value pairs) : key value pairs providing fine-grained control on encoding

                    Returns: - None.

                    Example:
                    >>> transcode(0, "path/to/input/video/file","path/to/output/elementary/bitstream")
    """
    nv_dmx = nvc.CreateDemuxer(filename=in_file_path)
    nv_dec = nvc.CreateDecoder(gpuid=0,
                               codec=nv_dmx.GetNvCodecId(),
                               cudacontext=0,
                               cudastream=0,
                               usedevicememory=True)
    
    width = 0
    height = 0
    nv12_frame_size = 0
    input_frame_list = 0
    encoder_created = False
    with open(out_file_path, "wb") as dec_file:
        frame_cnt = 0

        for packet in nv_dmx:
            for decoded_frame in nv_dec.Decode(packet):
                if not encoder_created:
                    width = nv_dec.GetWidth()
                    height = nv_dec.GetHeight()
                    print("width=", width)
                    print("height=", height)
                    nv12_frame_size = nv_dec.GetFrameSize()
                    input_frame_list = list([AppFrame(width, height, "NV12") for x in range(0, 5)])
                    nvenc = nvc.CreateEncoder(width, height, "NV12", False, **config_params)
                    encoder_created = True
                frame_cnt = frame_cnt + 1
                luma_base_addr = decoded_frame.GetPtrToPlane(0)
                cuda.memcpy_dtod(input_frame_list[frame_cnt % 5].gpuAlloc, luma_base_addr, nv12_frame_size)
                bitstream = nvenc.Encode(input_frame_list[frame_cnt % 5])
                bitstream = bytearray(bitstream)
                dec_file.write(bitstream)

        bitstream = nvenc.EndEncode()
        bitstream = bytearray(bitstream)
        dec_file.write(bitstream)
        print("frame count : ", frame_cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'This sample application demonstrates transcoding of an input video stream.'
    )
    parser.add_argument(
        "-g", "--gpu_id", type=int, help="GPU id, check nvidia-smi. Do not use", )
    parser.add_argument(
        "-i", "--in_file_path", required=True, type=Path, help="Encoded video file (read from)", )
    parser.add_argument(
        "-c", "--codec", type=str, required=True, help="h264, hevc, av1")
    parser.add_argument(
        "-p", "--preset", type=str, default='', help="P1,P2,P3,P4,P5,P6,P7")
    parser.add_argument(
        "-o", "--out_file_path", required=True, type=Path, help="Encoded video file (write to)", )
    parser.add_argument(
        "-json", "--config_file", type=str, default='', help="path of json config file", )

    args = parser.parse_args()
    config = {}
    if len(args.config_file):
        with open(args.config_file) as jsonFile:
            json_content = jsonFile.read()
        config = json.loads(json_content)
    config = {"preset": args.preset.upper(), "codec": args.codec.lower()}

    transcode(args.gpu_id,
              args.in_file_path.as_posix(),
              args.out_file_path.as_posix(),
              config)
