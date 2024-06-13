# FFNet for Semantic Segmentation

This folder contains the implementation of the FFNet for semantic segmentation. 

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

### Install

- Clone this repo:

```bash
git clone https://github.com/ysj9909/FFNet.git
cd FFNet
```

- Create a conda virtual environment and activate it:

```bash
conda create -n ffnet python=3.7 -y
conda activate ffnet
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install torch==1.11 with CUDA==11.3 and nvcc:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge cudatoolkit-dev=11.3 -y # to install nvcc
```

- Install other requirements:

  note: conda opencv will break torchvision as not to support GPU, so we need to install opencv using pip. 	  

```bash
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
pip install opencv-python
```

- Install `timm` and `mmcv-full` and `mmsegmentation':

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
```


### Data Preparation

Prepare datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.


### Results and Fine-tuned Models

| Variant | Dataset | Pretrained Model | Method | Lr Schd | mIoU | mIoU (ms+flip) | #params | FLOPs | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:|:---:|
| FFNet-2 | ADE20K | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/ffnet_2_distillation.pth.tar) | UPerNet | 160K | 47.1 | 47.8 | 58M | 942G | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/upernet_ffnet_2_512_160k_ade20k.pth) |
| FFNet-3 | ADE20K | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/ffnet_3_distillation.pth.tar) | UPerNet | 160K | 49.6 | 50.2 | 80M | 1010G | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/upernet_ffnet_3_512_160k_ade20k.pth) |
| FFNet-4 | ADE20K | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/ffnet_4_384.pth.tar) | UPerNet | 160K | 50.7 | 51.7 | 113M | 1158G | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/upernet_ffnet_4_512_160k_ade20k.pth) |
| FFNet<sub>seg</sub> | ADE20K | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/ffnet_seg.pth.tar) | FFNet | 160K | 50.1 | 51.2 | 68M | 74G | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/ffnet_seg_sys_512_160k_ade20k.pth) |
| FFNet<sub>seg</sub> | Cityscapes | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/ffnet_seg.pth.tar) | FFNet | 160K | 83.2 | 84.1 | 68M | 577G | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/ffnet_seg_sys_1024_160k_cityscapes.pth) |


### Evaluation

To evaluate FFNets on the ADE20K val set, run
```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU --cfg-options model.backbone.init_cfg.checkpoint=None
```
Note that we use ```--cfg-options model.backbone.init_cfg.checkpoint=None``` to overwrite the initialization config of the backbone so that its initialization with the ImageNet-pretrained weights will be skipped. This is because we will load its weights together with the UperNet heads from the checkpoint file.

You may also 1) change ```checkpoint``` to ```None``` in the config file to realize the same effect or 2) simply ignore it if you have downloaded the ImageNet-pretrained weights (initializing the backbone twice does no harm except for wasting time).

For example, to evaluate the `FFNet-2` with a single GPU:
```bash
python test.py configs/ade20k/upernet_ffnet_2_512_160k_ade20k.py upernet_ffnet_2_512_160k_ade20k.pth --eval mIoU --cfg-options model.backbone.init_cfg.checkpoint=None
```

For example, to evaluate the `FFNet-4` with a single node with 8 GPUs:
```bash
sh dist_test.sh configs/ade20k/upernet_ffnet_4_512_160k_ade20k.py upernet_ffnet_4_512_160k_ade20k.pth 8 --eval mIoU --cfg-options model.backbone.init_cfg.checkpoint=None
```

**For system-level semantic segmentation experiments, we conducted our experiments based on the [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt/tree/main) codebase for a fair comparison.
We applied the model code (`mmseg_custom/models/backbones/ffnet_sys.py` and `mmseg_custom/models/decode_heads/ffnet_head.py`) and configuration files to the SegNeXt code, enabling us to train and evaluate FFNet<sub>seg</sub>.
Please follow the instructions of SegNeXt to train/evaluate our models.**


### Training

To train a `FFNet` on ADE20K, 1) ensure that the ```init_cfg.checkpoint``` in the config file refers to the downloaded pretrained weights, and 2) run

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `FFNet-2` with 8 GPUs on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/ade20k/upernet_ffnet_2_512_160k_ade20k.py 8
```

## Bibtex
```
@article{yun2024metamixer,
  title={MetaMixer Is All You Need},
  author={Yun, Seokju and Lee, Dongheon and Ro, Youngmin},
  journal={arXiv preprint arXiv:2406.02021},
  year={2024}
}
```


### Acknowledgements 

Our code is heavily based on the codes from internimage and segnext. We are very grateful for their amazing work.
