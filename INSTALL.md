# Installation steps
This repository is built using Restomer framework (PyTorch 2.0.1, Python3.9, CUDA11.8) and tested on Windows environment. 

Follow these instructions:
### 1. Clone our Repository
```bash 
git clone https://github.com/Aishwarya-S-Murthy/UDVSR-Net
cd UDVSR-Net 
```

### 2. Create a conda environment
```bash
conda create --name UDVSRNet python=3.9 -y
conda activate UDVSRNet
```

### 3. Install Dependencies
```bash
conda install -c conda-forge ninja -y
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install setuptools==59.5.0
pip install numpy opencv-python scipy tqdm pyyaml tensorboard einops lmdb scikit-image addict future
pip install "opencv-python<4.10"
pip install "numpy<2"
```

### 4. Install BasicSR
```bash
python setup.py develop --no_cuda_ext
```
