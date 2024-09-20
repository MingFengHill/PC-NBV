#!/usr/bin/env bash

TF_PATH=$(python -c "import tensorflow as tf; import os; print(os.path.dirname(tf.__file__))")
TF_CORE_PATH=$(dirname "$TF_PATH")/tensorflow_core

TF_INC="$TF_CORE_PATH/include"
TF_LIB_PATH="$TF_CORE_PATH/libtensorflow_framework.so.1"

/usr/local/cuda/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
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