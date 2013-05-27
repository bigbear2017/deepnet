#!/bin/sh

export CUDA_SDK_PATH=$HOME/NVIDIA_GPU_Computing_SDK
export CUDA_INSTALL_PATH=/usr/local/cuda/
export PYTHON_INCLUDE_PATH=/usr/include/python2.6/
export NUMPY_INCLUDE_PATH=/usr/lib64/python2.6/site-packages/numpy/core/include/numpy/
export ATLAS_LIB_PATH=/usr/lib64/atlas
#export ATLAS_LIB_PATH=/usr/lib/atlas-base/atlas
make $*

