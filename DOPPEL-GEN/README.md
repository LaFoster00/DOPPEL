# DOPPEL-GENERATOR
Tool for generating images of the same person in different conditions

## Requirements
### DOPPEL-GEN
* Linux is supported.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
* 64-bit Python 3.8 (Should be part of the conda environment) and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later (Should be part of the conda environment).
* GCC 7 or later (Linux) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 or Anaconda to create and activate your FACE-GAN Python environment:
  - `conda env create -f generator_environment.yml`
  - `conda activate DOPPEL-GEN`

### Stylegan3
You will also need stylegan3 to generate the images.
1. Init, update all submodules. Stylegan3 will be downloaded now.
2. Export stylegans dnnlib and torch_utils directories to the python environment using this command
   * `conda develop third_party/stylegan3`