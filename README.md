# MetaMixer Is All You Need

Official PyTorch implementation of FFNet, from the following paper "[MetaMixer Is All You Need](https://arxiv.org/abs/2406.02021)".

*Seokju Yun, Dongheon Lee, Youngmin Ro.*

![first metamixer fig](https://github.com/ysj9909/FFNet/blob/main/docs/metamixer.png)

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
