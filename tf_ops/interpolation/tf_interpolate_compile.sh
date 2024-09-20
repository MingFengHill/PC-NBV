#!/usr/bin/env bash

# Set correct TensorFlow paths
TF_INC="/home/threed-detection/anaconda3/envs/pc-nbv/lib/python3.6/site-packages/tensorflow_core/include"
TF_LIB_PATH="/home/threed-detection/anaconda3/envs/pc-nbv/lib/python3.6/site-packages/tensorflow_core/libtensorflow_framework.so.1"

# Compile and link C++ code
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC \
    -I "$TF_INC" \
    -I /usr/local/cuda/include \
    -I "$TF_INC/external/nsync/public" \
    -lcudart -L /usr/local/cuda/lib64/ \
    "$TF_LIB_PATH" -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful"
else
    echo "Compilation failed"
fi