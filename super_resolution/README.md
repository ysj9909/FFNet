# FFNetSR

<!-- figs/Architecture.png here? -->

## Installation
```bash
conda create -n ffnetsr python=3.10
conda activate ffnetsr
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
python setup.py develop
```

## Usage

### Train
```bash
python ffnetsr/train.py --config $CONFIG_FILE
```

### Test
```bash
python ffnetsr/test.py --config $CONFIG_FILE
```

## Results

### Quantitative

<!-- figs/Quantitative.png here? -->

### Visual

<!-- figs/Visual.png here? -->