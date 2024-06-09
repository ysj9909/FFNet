# FFNet for Super-Resolution

<!-- figs/Architecture.png here? -->

### Setup
```bash
conda create -n ffnetsr python=3.10
conda activate ffnetsr
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
python setup.py develop
```

### Training
Run the following commands for training:
```bash
python ffnetsr/train.py --config $CONFIG_FILE
```

### Evaluation
Download the pre-trained models and test sets. Run the following commands:
```bash
python ffnetsr/test.py --config $CONFIG_FILE
```

## Results

### Pre-trained Models and Results
|  Variant   | Settings | model |
|  ----  | ----  | --- |
| FFNet<sub>sr</sub>-light  | DIV2K $\times 2$ | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/FFNetSR_light_DIV2K.pth) |
| FFNet<sub>sr</sub>  | DF2K $\times 2$ | [model](https://github.com/ysj9909/FFNet/releases/download/v1.0/FFNetSR_DF2K.pth) | 

![image](https://github.com/ysj9909/FFNet/blob/main/super_resolution/figs/Quantitative.png)


### Visualization
![image](https://github.com/ysj9909/FFNet/blob/main/super_resolution/figs/Visual.png)


## Bibtex
```
@article{yun2024metamixer,
  title={MetaMixer Is All You Need},
  author={Yun, Seokju and Lee, Dongheon and Ro, Youngmin},
  journal={arXiv preprint arXiv:2406.02021},
  year={2024}
}
```


### Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.
  
