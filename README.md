# DeepStream Libraries
DeepStream Libraries provide [CVCUDA](https://github.com/CVCUDA/CV-CUDA), [NvImageCodec](https://github.com/NVIDIA/nvImageCodec), and [PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/) modules as Python APIs to easily integrate into custom frameworks.
Developers can build complete Python applications with fully accelerated components leveraging intuitive Python APIs.
Most of the DeepStream Libraries building blocks and their Python APIs are available today as standalone packages. DeepStream Libraries provide a way for Python developers to install these packages with a single installer.
All these packages are built against the same CUDA version and validated with the specified driver version. Reference applications are provided to demonstrate the usage of Python APIs.


## System Requirements

- **Operating System:**
  - [Ubuntu 22.04](https://releases.ubuntu.com/22.04/?_gl=1*19ip6hm*_gcl_au*MTE4NTIyOTI0MS4xNzA3MTMxMDQx&_ga=2.149898549.2084151835.1707729318-1126754318.1683186906)

- **Python:**
  - Python-3.10 (Should be pre-installed with Ubuntu 22.04)

- **CUDA:**
  - [CUDA Toolkit 12.2](https://developer.nvidia.com/cuda-12-2-0-download-archive)

- **NVIDIA Driver:**
  - [Nvidia Driver-535.161.08](https://www.nvidia.cn/Download/driverResults.aspx/222416/en-us/)

- **TensorRT:**
  - [TensorRT-8.6.1.6](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading)

## DeepStream Libraries Installation
1. Download DeepStream Libraries wheel file from NGC.
    - Download wheel file from this NGC [link](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream)

2. Install DeepStream Libraries package.
    ```bash
    pip3 install deepstream_libraries-1.0-cp310-cp310-linux_x86_64.whl
    ```

## DeepStream Libraries Repository Setup
To run [sample apps](https://github.com/NVIDIA-AI-IOT/deepstream_libraries), follow below steps:
1. Clone DeepStream Libraries repo.
    ```
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_libraries.git
    cd deepstream_libraries
    ```
2. Install dependencies.

    Install all the dependent packages required by sample apps:
    ```
    sudo sh scripts/install_dependencies.sh
    ```

3. Download test files

    Download images/videos to run sample apps:
    ```
    sh scripts/download_data.sh
    ```

## Getting Started with DeepStream Libraries APIs
We can use DeepStream Libraries API's to create an application.

Consider the below reference example:
- Read an image from the given file path using NvImageCodec
- Resize the image with specified dimensions and Cubic interpolation method using CVCUDA
- Save the resized image using NvImageCodec

```python
# Import necessary libraries
import cvcuda
from nvidia import nvimgcodec

# Create Decoder
decoder = nvimgcodec.Decoder()

# Read image with nvImageCodec
inputImage = decoder.read("path/to/image.jpg")

# Pass it to cvcuda using as_tensor
nvcvInputTensor = cvcuda.as_tensor(inputImage, "HWC")

# Resize with cvcuda to 320x240
cvcuda_stream = cvcuda.Stream()
with cvcuda_stream:
    nvcvResizeTensor = cvcuda.resize(nvcvInputTensor, (320, 240, 3), cvcuda.Interp.CUBIC)
    nvcvResizeTensor.cuda().__cuda_array_interface__

# Write with nvImageCodec
encoder = nvimgcodec.Encoder()
output_image_path = "output.jpg"
encoder.write(output_image_path, nvimgcodec.as_image(nvcvResizeTensor.cuda(), cuda_stream = cvcuda_stream.handle))
```

## Sample Applications
| Application               | Description                                                                                                                                                      |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Classification    | A CUDA-accelerated image and video classification pipeline integrating PyTorch or TensorRT for efficient processing on NVIDIA GPUs                               |
| Object-Detection  | GPU accelerated Object detection using CV-CUDA library with TensorFlow or TensorRT                                                 |
| Segmentation      | GPU accelerated Semantic segmentation by utilizing the CV-CUDA library with PyTorch or TensorRT                                               |
| Resize-Image      | A sample app that decodes, resizes, and encodes images using the CVCUDA and NvImageCodec Python API's                                                            |
| Decode-Video      | Decodes encoded bitstreams using PyNvVideoCodec decode APIs                                                               |
| Encode-Video      | Encodes a raw YUV file using PyNvVideoCodec encode APIs                                                                |
| Transcode-Video   | Transcodes the video files using PyNvVideoCodec API's                                                                        |

## Additional References and Applications
For more references and application please refer to the below link:
- [CVCUDA](https://github.com/CVCUDA/CV-CUDA)
- [NvImageCodec](https://github.com/NVIDIA/nvImageCodec)
- [PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/)
- [PyNvVIdeoCodec Online Documents](https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/index.html)
- [Deepstream Libraries](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Libraries.html)