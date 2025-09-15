# DeepStream Libraries
DeepStream Libraries provide [CVCUDA](https://github.com/CVCUDA/CV-CUDA), [NvImageCodec](https://github.com/NVIDIA/nvImageCodec), and [PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/) modules as Python APIs to easily integrate into custom frameworks.
Developers can build complete Python applications with fully accelerated components leveraging intuitive Python APIs.
Most of the DeepStream Libraries building blocks and their Python APIs are available today as standalone packages. DeepStream Libraries provide a way for Python developers to install these packages with a single installer.
All these packages are built against the same CUDA version and validated with the specified driver version. Reference applications are provided to demonstrate the usage of Python APIs.


## System Requirements

- **Operating System:**
  - [Ubuntu 24.04](https://releases.ubuntu.com/noble/)

- **Python:**
  - Python-3.12 (Should be pre-installed with Ubuntu 24.04)

- **CUDA:**
  - [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)

- **NVIDIA Driver:**
  - [Nvidia Driver-570.133.20](https://www.nvidia.cn/drivers/details/242548/)

- **TensorRT:**
  - [TensorRT-10.9](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/installing-tensorrt/overview.html)

## DeepStream Libraries Repository Setup

Follow these steps to set up your environment for running [sample applications](https://github.com/NVIDIA-AI-IOT/deepstream_libraries):

### 1. Clone Repository
```bash
git clone https://github.com/NVIDIA-AI-IOT/deepstream_libraries.git
cd deepstream_libraries
```

### 2. Install System Dependencies
```bash
sudo sh scripts/install_sys_pkgs.sh
```

### 3. Download Sample Data
```bash
sh scripts/download_data.sh
```

### 4. Setup Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv deepstream_libraries_env

# Activate virtual environment
source deepstream_libraries_env/bin/activate

# Verify activation
which python3  # Should point to virtual environment
```
**Note:** Activate the virtual environment for Python dependencies and wheel installation, and in each new terminal session.

### 5. Install Python Dependencies
```bash
sh scripts/install_python_pkgs.sh
```

## DeepStream Libraries Installation
1. Download DeepStream Libraries wheel file from NGC.
    - Download wheel file from this NGC [link](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream_libraries)

2. Install DeepStream Libraries package.
    ```bash
    pip3 install deepstream_libraries-1.2-cp312-cp312-linux_x86_64.whl
    ```

## Getting Started with DeepStream Libraries APIs
We can use DeepStream Libraries API's to create an application.

Consider the below reference example:
- Read an image from the given file path using NvImageCodec
- Resize the image with specified dimensions and Cubic interpolation method using CVCUDA
- Align output dimensions to ensure compatibility with nvImageCodec
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

# Align output dimensions to 32-byte boundaries for nvImageCodec compatibility
output_width, output_height, alignment = 320, 240 , 32
aligned_width, aligned_height = ((output_width + alignment - 1) // alignment) * alignment , ((output_height + alignment - 1) // alignment) * alignment

# Resize with cvcuda using aligned dimensions
cvcuda_stream = cvcuda.Stream()
with cvcuda_stream:
    nvcvResizeTensor = cvcuda.resize(nvcvInputTensor,(aligned_height, aligned_width, 3), cvcuda.Interp.CUBIC)

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
- [CVCUDA](https://github.com/CVCUDA/CV-CUDA/releases/tag/v0.15.0-beta)
- [NvImageCodec](https://github.com/NVIDIA/nvImageCodec/releases/tag/v0.6.0)
- [PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/2.0.0/)
- [Deepstream Libraries](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Libraries.html)
