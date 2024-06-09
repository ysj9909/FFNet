# FFNet for Time Series Long-Term Forecasting

This folder contains the implementation of the FFNet for time series forecasting.

To demonstrate the generality of the MetaMixer framework, we apply our convolutional mixer design not only to image recognition problems but also to time series analysis tasks. Recent MLP- and Transformer-based models, which excel in forecasting tasks, have shown the importance of a large receptive field. Therefore, by utilizing large kernels (e.g., 51) in MetaMixer, we achieve impressive performance. **These results highlight that the key is not any specific module but equipping the MetaMixer framework with the appropriate functionalities for each task.**


### Usage

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

- Setup

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

- Prepare Data. You can obtain all datasets from [Times-series-library](https://github.com/thuml/Time-Series-Library).

- Train and evaluate model. We provide the experiment scripts for all datasets under the folder `./scripts/`. You can reproduce the experiment results as the following examples:
```
sh ./scripts/ETTm1.sh
```


### Results
![forecasting_results](https://github.com/ysj9909/FFNet/blob/main/docs/forecasting_results.png)

## Bibtex
```
@article{yun2024metamixer,
  title={MetaMixer Is All You Need},
  author={Yun, Seokju and Lee, Dongheon and Ro, Youngmin},
  journal={arXiv preprint arXiv:2406.02021},
  year={2024}
}
```


## Acknowledgement

We are very grateful to the following GitHub repositories for their valuable codebase and datasets:

[RevIN](https://github.com/ts-kim/RevIN), [PatchTST](https://github.com/PatchTST/PatchTST), [ModernTCN](https://github.com/luodhhh/ModernTCN), and [Times-series-library](https://github.com/thuml/Time-Series-Library).

