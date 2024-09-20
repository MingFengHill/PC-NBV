TF_PATH=$(python -c "import tensorflow as tf; import os; print(os.path.dirname(tf.__file__))")
TF_INC="$TF_PATH/include"
TF_LIB="$TF_PATH"