# MetaMixer Is All You Need

Official PyTorch implementation of FFNet, from the following paper "[MetaMixer Is All You Need](https://arxiv.org/abs/2406.02021)".

*Seokju Yun, Dongheon Lee, Youngmin Ro.*

![first metamixer fig](https://github.com/ysj9909/FFNet/blob/main/docs/metamixer.png)
Figure: Overview of MetaMixer. (a) MetaMixer is derived by not specifying sub-operations within the query-key-value framework. We assert that the competence of Transformers primarily originates from MetaMixer, which we deem as the true **backbone** of Transformer. (b) To demonstrate this and propose a FFN-like efficient token mixer, we replace the inefficient sub-operations of self-attention with those from FFN while retaining MetaMixer structure. (c) Our MetaMixer-based pure ConvNets outperform domain-specialized competitors in various tasks, confirming the superiority of the MetaMixer framework.

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Transformer, composed of self-attention and Feed-Forward Network (FFN), has revolutionized the landscape of network design across various vision tasks.
While self-attention is extensively explored as a key factor in performance, FFN has received little attention.
FFN is a versatile operator seamlessly integrated into nearly all AI models to effectively harness rich representations.
Recent works also show that FFN functions like key-value memories.
Thus, akin to the query-key-value mechanism within self-attention, FFN can be viewed as a memory network, where the input serves as query and the two projection weights operate as keys and values, respectively.
Based on these observations, we hypothesize that the importance lies in query-key-value framework itself rather than in self-attention.
To verify this, we propose converting self-attention into a more FFN-like efficient token mixer with only convolutions while retaining query-key-value framework, namely \textit{FFNification}.
Specifically, FFNification replaces query-key and attention coefficient-value interactions with large kernel convolutions and adopts GELU activation function instead of softmax.
The derived token mixer, \textit{FFNified attention}, serves as key-value memories for detecting locally distributed spatial patterns, and operates in the opposite dimension to the ConvNeXt block within each corresponding sub-operation of the query-key-value framework.
Building upon the above two modules, we present a family of Fast-Forward Networks (FFNet).
Our FFNet achieves remarkable performance improvements over previous state-of-the-art methods across a wide range of tasks.
The strong and general performance of our proposed method validates our hypothesis and leads us to introduce “MetaMixer”, a general mixer architecture that does not specify sub-operations within the query-key-value framework.
We show that using only simple operations like convolution and GELU in the MetaMixer can achieve superior performance.
We hope that this intuition will catalyze a paradigm shift in the battle of network structures, sparking a wave of new research.
</details>


## Pre-trained Models

ImageNet-1K
| Variant | Resolution | Top-1 Acc. | #params | FLOPs | Latency | model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| FFNet-1 | 256x256 | 81.3 | 13.7M | 2.9G | 1.8 |[model](-) |
| FFNet-2 | 256x256 | 82.9 | 26.9M | 6.0G | 3.1 | [model](-) |
| FFNet-3 | 256x256 | 83.9 | 48.3M | 10.1G | 4.5 | [model](-) |
| FFNet-3 | 384x384 | 84.5 | 48.3M | 22.8G | 9.1 | [model](-) |
| FFNet-4 | 384x384 | 85.3 | 79.2M | 43.1G | 15.2 | [model](-) |

Models trained on ImageNet-1K with knowledge distillation.
| Variant | Resolution | Top-1 Acc. | #params | FLOPs | Latency | model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| FFNet-1 | 256x256 | 82.1 | 13.7M | 2.9G | 1.8 |[model](-) |
| FFNet-2 | 256x256 | 83.7 | 26.9M | 6.0G | 3.1 | [model](-) |
| FFNet-3 | 256x256 | 84.5 | 48.3M | 10.1G | 4.5 | [model](-) |

## Training
### Image Classification

#### Setup
```bash
conda create -n ffnet python=3.9
conda activate ffnet
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

#### Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  validation/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

To train FFNet models, follow the respective command below:
<details>
<summary>
FFNet-1
</summary>

```
# Without Distillation
python -m torch.distributed.launch --nproc_per_node=8 train.py \
/path/to/ImageNet/dataset --model ffnet_1 -b 128 --lr 1e-3 \
--native-amp --mixup 0.2 --output /path/to/save/results \
--input-size 3 256 256 --drop-path 0.1

# With Distillation
python -m torch.distributed.launch --nproc_per_node=8 train.py \
/path/to/ImageNet/dataset --model ffnet_1 -b 128 --lr 1e-3 \ 
--native-amp --output /path/to/save/results \
--input-size 3 256 256 --drop-path 0.02 \
--distillation-type "hard"
```
</details>


<details>
<summary>
FFNet-2
</summary>

```
# Without Distillation
python -m torch.distributed.launch --nproc_per_node=8 train.py \
/path/to/ImageNet/dataset --model ffnet_2 -b 128 --lr 1e-3 \
--native-amp --mixup 0.2 --output /path/to/save/results \
--input-size 3 256 256 --drop-path 0.15

# With Distillation
python -m torch.distributed.launch --nproc_per_node=8 train.py \
/path/to/ImageNet/dataset --model ffnet_2 -b 128 --lr 1e-3 \ 
--native-amp --output /path/to/save/results \
--input-size 3 256 256 --drop-path 0.08 \
--distillation-type "hard"
```
</details>


<details>
<summary>
FFNet-3
</summary>

```
# Without Distillation
python -m torch.distributed.launch --nproc_per_node=8 train.py \
/path/to/ImageNet/dataset --model ffnet_3 -b 128 --lr 1e-3 \
--native-amp --output /path/to/save/results \
--input-size 3 256 256 --drop-path 0.35

# With Distillation
python -m torch.distributed.launch --nproc_per_node=8 train.py \
/path/to/ImageNet/dataset --model ffnet_3 -b 128 --lr 1e-3 \ 
--native-amp --output /path/to/save/results \
--input-size 3 256 256 --drop-path 0.2 \
--distillation-type "hard"
```
</details>

We finetune models pre-trained on ImageNet-1K at a resolution of 384x384.
<details>
<summary>
FFNet-3
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 train.py \
/path/to/ImageNet/dataset --model ffnet_3 -b 64 --lr 5e-6 \
--native-amp --output /path/to/save/results --input-size 3 384 384 \
--drop-path 0.4 --epochs 30 --warmup-epochs 0 --weight-decay 1e-8 \
--sched none --finetune --resume /path/to/checkpoint.pth
```
</details>


<details>
<summary>
FFNet-4
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 train.py \
/path/to/ImageNet/dataset --model ffnet_4 -b 64 --lr 5e-6 \
--native-amp --output /path/to/save/results --input-size 3 384 384 \
--drop-path 0.5 --epochs 30 --warmup-epochs 0 --weight-decay 1e-8 \
--sched none --finetune --resume /path/to/checkpoint.pth
```
</details>


## Evaluation
Run the following command to evaluate a pre-trained FFNet-2 on ImageNet-1K validation set with a single GPU:
```bash
python -m torch.distributed.launch --nproc_per_node=1 train.py \
/path/to/ImageNet/dataset --model ffnet_2 -b 128 \
--native-amp --input-size 3 256 256 --resume /path/to/checkpoint.pth --eval
```

## Latency Measurement
The mobile latency reported in our work for iPhone 12 uses the deployment tool from [XCode 14](https://developer.apple.com/videos/play/wwdc2022/10027/).

export the model to Core ML model

```
python export_model.py --variant ffnet_1 --output-dir /path/to/save/exported_model \
--checkpoint /path/to/pretrained_checkpoints/ffnet_1.pth.tar
```

## Bibtex
-

## Acknowledgements
We sincerely appreciate [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [PyTorch](https://github.com/pytorch/pytorch), [DeiT](https://github.com/facebookresearch/deit), [RepVGG](https://github.com/DingXiaoH/RepVGG), and [FastViT](https://github.com/apple/ml-fastvit) for their wonderful implementations.
