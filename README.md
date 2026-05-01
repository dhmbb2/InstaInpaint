
<div align="center">

# InstaInpaint: Instant 3D-Scene Inpainting </br> with Masked Large Reconstruction Model



<a href="https://arxiv.org/abs/2506.10980"><img src="https://img.shields.io/badge/arXiv-2506.10980-b31b1b.svg"></a>
<a href="https://huggingface.co/dhmbb/instainpaint/tree/main"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow"></a>
<a href="https://dhmbb2.github.io/InstaInpaint_page/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green"></a>

**Junqi You**<sup>1</sup>, **Chieh Hubert Lin**<sup>2</sup>, **Weijie Lyu**<sup>2</sup>, **Zhengbo Zhang**<sup>3</sup>, **Ming-Hsuan Yang**<sup>2</sup>


<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>UC Merced, <sup>3</sup>Singapore University of Technology and Design
</div>

## Abstract

We propose InstaInpaint, a reference-based feed-forward framework that produces 3D-scene inpainting from a 2D inpainting proposal within 0.4 seconds. InstaInpaint achieves a 1000× speed-up from prior methods while maintaining a state-of-the-art performance across two standard benchmarks.

![InstaInpaint overview](assets/teaser.png)



## 🚀 Get Started

### 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/dhmbb2/InstaInpaint.git
cd InstaInpaint

# 2. Create conda environment
export CUDA_HOME=/usr/local/cuda-12.4/
conda create -n instainpaint python=3.10
conda activate instainpaint

# 3. Install PyTorch 
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Install dependencies
pip install -r requirements.txt
```
Note: If you are using a different CUDA version, update the pre-built flash-attn wheel in `requirements.txt` accordingly.

## Inference

### 1. Preparation
Download the preprocessed Spin-NeRF dataset and checkpoint from Hugging Face.
```bash
mkdir -p data checkpoints

wget -O data/spinnerf_dataset.zip https://huggingface.co/dhmbb/instainpaint/resolve/main/spinnerf_dataset.zip?download=true

unzip -q data/spinnerf_dataset.zip -d data/

wget -O checkpoints/exp_ins+random+3d_121_multimask.pth https://huggingface.co/dhmbb/instainpaint/resolve/main/exp_ins%2Brandom%2B3d_121_multimask.pth?download=true
```

### 2. Run inference scripts

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test_inpaint_spinnerf.sh
```
The inference results are written to `outputs/spinnerf_eval`.


## Training

### 1. Dataset Preparation

Download DL3DV_960P from [the official repository](https://github.com/DL3DV-10K/Dataset) and put it in `data/dl3dv_960`. 

### 2. Pretrained Checkpoints 
```bash
mkdir -p checkpoints

wget -O checkpoints/pretraining_ckpt.pth https://huggingface.co/dhmbb/instainpaint/resolve/main/pretraining_ckpt.pth?download=true
```

Next, we need to use SAM2 to get instance masks and cache them for training efficiency.
```bash
# create another environment for sam2
conda create -n sam python=3.10
conda activate sam
cd third-party/sam2
pip install -e .
cd ../../

wget -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/precalculate_mask.sh 
```
We also need to cache image depth from the pretrained LRM in order to calculate regional masks during training.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/precalculate_depth.sh
``` 

### Start Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WORLD_SIZE=8 bash scripts/train_inpaint_4v_clip15.sh instainpaint_train
```


## 📝 Citation 

If you find this work useful in your research, please consider citing:

```bibtex
@article{you2025instainpaint,
  title={Instainpaint: Instant 3d-scene inpainting with masked large reconstruction model},
  author={You, Junqi and Lin, Chieh Hubert and Lyu, Weijie and Zhang, Zhengbo and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2506.10980},
  year={2025}
}
```
