#!/usr/bin/env bash

TF_INC="/home/threed-detection/anaconda3/envs/pc-nbv/lib/python3.6/site-packages/tensorflow_core/include"
TF_LIB_PATH="/home/threed-detection/anaconda3/envs/pc-nbv/lib/python3.6/site-packages/tensorflow_core/libtensorflow_framework.so.1"

/usr/local/cuda/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC \
    -I "$TF_INC" \
    -I /usr/local/cuda/include \
    -I "$TF_INC/external/nsync/public" \
    -lcudart -L /usr/local/cuda/lib64/ \
    "$TF_LIB_PATH" -O2 -D_GLIBCXX_USE_CXX11_ABI=0

if [ $? -eq 0 ]; then
    echo "Compilation successful"
else
    echo "Compilation failed"
fi