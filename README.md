# DOPPEL
D.O.P.P.E.L. - Detection of People through Evaluated Likelihood

## Requirements
### DOPPEL
* Linux is supported.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
* 64-bit Python 3.9 (Should be part of the conda environment) and tensorflow 2.16.1. See https://www.tensorflow.org/install for Tensorflow install instructions.
* Python libraries: see [generator_cuda.yml](./environment-cuda.yml) for exact library dependencies.  You can use the following commands with Miniconda3 or Anaconda to create and activate your DOPPEL Python environment:
  - `conda env create -f generator_cuda.yml`
  - `conda activate DOPPEL`