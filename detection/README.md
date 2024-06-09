# FFNet for Object Detection

This folder contains the implementation of FFNet for object detection.

Our detection code is developed on top of [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/tree/v2.28.1).

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

For examples, to install torch==1.11 with CUDA==11.3:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
```

- Install `timm==0.6.11` and `mmcv-full==1.5.0`:

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

### Data Preparation

Prepare COCO according to the guidelines in [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).


### Results and Fine-tuned Models
| Variant | Pretrained Model | Method | Lr Schd | box mAP | mask mAP | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| FFNet-2 | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/ffnet_2_distillation.pth.tar) | Cascade Mask R-CNN | 3x | 51.8 | 44.9 | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/casc_mask_rcnn_ffnet_2_fpn_3x_coco.pth) |
| FFNet-3 | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/ffnet_3_distillation.pth.tar) | Cascade Mask R-CNN | 3x | 52.8 | 45.6 | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/casc_mask_rcnn_ffnet_3_fpn_3x_coco.pth) |
| FFNet-4 | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/ffnet_4_384.pth.tar) | Cascade Mask R-CNN | 3x | 53.4 | 45.9 | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/casc_mask_rcnn_ffnet_4_fpn_3x_coco.zip) |


### Evaluation

To evaluate FFNets on COCO val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval bbox segm --cfg-options model.backbone.init_cfg.checkpoint=None
```
Note that we use ```--cfg-options model.backbone.init_cfg.checkpoint=None``` to overwrite the initialization config of the backbone so that its initialization with the ImageNet-pretrained weights will be skipped. This is because we will load its weights together with the Cascade Mask RCNN heads from the checkpoint file.

You may also 1) change ```checkpoint``` to ```None``` in the config file to realize the same effect or 2) simply ignore it if you have downloaded the ImageNet-pretrained weights (initializing the backbone twice does no harm except for wasting time).

For example, to evaluate the `FFNet-2` with a single GPU:

```bash
python test.py configs/coco/casc_mask_rcnn_ffnet_2_fpn_3x_coco.py casc_mask_rcnn_ffnet_2_fpn_3x_coco.pth --eval bbox segm --cfg-options model.backbone.init_cfg.checkpoint=None
```

For example, to evaluate the `FFNet-4` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/coco/casc_mask_rcnn_ffnet_4_fpn_3x_coco.py casc_mask_rcnn_ffnet_4_fpn_3x_coco.pth 8 --eval bbox segm --cfg-options model.backbone.init_cfg.checkpoint=None
```

### Training on COCO

To train a `FFNet` on COCO, 1) ensure that the ```init_cfg.checkpoint``` in the config file refers to the downloaded pretrained weights, and 2) run

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `FFNet-2` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/coco/casc_mask_rcnn_ffnet_2_fpn_3x_coco.py 8
```


### Acknowledgements 

Our code is heavily based on the code from internimage. We are very grateful for their amazing work.
