# PC-NBV: A Point Cloud Based Deep Network for Efficient Next Best View Planning

### Introduction 

This repository is for our IROS 2020 paper "PC-NBV: A Point Cloud Based Deep Network for Efficient Next Best View Planning". The code is modified from [pcn](https://github.com/wentaoyuan/pcn) and [PU-GAN](https://github.com/liruihui/PU-GAN). 

### Installation
This repository is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators. 

For installing tensorflow, please follow the official instructions in [here](https://www.tensorflow.org/install/install_linux). The code is tested under:
- Tensorflow 1.12, Python 3.7, Ubuntu 16.04.
- Tensorflow 1.15(with gpu), Python 3.6, Ubuntu 18.04.

For compiling TF operators, please check `tf_xxx_compile.sh` under each op subfolder in `tf_ops` folder. Note that you need to update `nvcc`, `python` and `tensoflow include library` if necessary. 

### Note
When running the code, if you have `undefined symbol: _ZTIN10tensorflow8OpKernelE` error, you need to compile the TF operators. If you have already added the `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` but still have ` cannot find -ltensorflow_framework` error. Please use 'locate tensorflow_framework' to locate the tensorflow_framework library and make sure this path is in `$TF_LIB`.

### Usage

0. Setting up the Environment
   ``` python
   conda create -n "pc-nbv" python=3.6 ipython
   conda activate pc-nbv

   pip install -r requirements.txt
   ```

1. Clone the repository:

   ```shell
   https://github.com/Smile2020/PC-NBV.git
   cd PC-NBV
   ```
   
2. Compile the TF operators:

   Follow the above information to compile the TF operators. 
   
3. Generate the data:

   To generate your own data from ShapeNet, first Download [ShapeNetCore.v1](https://shapenet.org). Then, create partial point clouds from depth images (see instructions in `render`) and corresponding ground truths by sampling from CAD models ([sample code](https://github.com/hexygen/sample_mesh)). 

   You can generate networks' inputs and supervision npy data using generate_nbv_data.py, then use lmdb_write_shapenet.py to make lmdb data.

4. Train the model:
   ```shell
   python train.py 
   ```

5. Evaluate the model:

   To test your trained model, you can run:
   ```shell
   python test.py --checkpoint model_path
   ```

### Questions

Please contact 'zengr17@mails.tsinghua.edu.cn'

