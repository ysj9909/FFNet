# FFNet for 3D Semantic Segmentation

This folder contains the implementation of the FFNet for 3D semantic segmentation. 

Our segmentation code is developed on top of [Pointcept](https://github.com/Pointcept/Pointcept).


## Installation

### Requirements
- Ubuntu: 18.04 and above.
- CUDA: 11.3 and above.
- PyTorch: 1.10.0 and above.

### Conda Environment

```bash
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d
```


### Training
First, prepare the dataset according to the instructions in [Pointcept](https://github.com/Pointcept/Pointcept).

**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard, and checkpoints will also be saved into the experiment folder during the training process.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
```

For example:
```bash
# By script (Recommended)
# -p is default set as python and can be ignored
sh scripts/train.sh -g 4 -d scannet -c scannet-ffnet -n ffnet
sh scripts/train.sh -g 4 -d scannet200 -c scannet200-ffnet -n ffnet
sh scripts/train.sh -g 4 -d s3dis -c s3dis-ffnet -n ffnet
```

### Testing
```bash
# By script (Based on experiment folder created by training script)
sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
```
For example:
```bash
# By script (Based on experiment folder created by training script)
# -p is default set as python and can be ignored
# -w is default set as model_best and can be ignored
sh scripts/test.sh -p python -d scannet -n ffnet -w model_best
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

## Acknowledgment
We sincerely appreciate [Pointcept](https://github.com/Pointcept/Pointcept), [PTv3](https://github.com/Pointcept/PointTransformerV3), and [spconv](https://github.com/traveller59/spconv) for their wonderful implementations.
