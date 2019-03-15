

#### 1. CNN 网络的 FLOPs 是如何计算的？

> FLOPs：注意s小写，是floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。

> FLOPS：注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。

对于卷积层，假设卷积核大小是`K`,输入通道数是`Ci`,输出的特征图尺寸是`H*W`，输出通道为`Co`，那么卷积层的`FLOPs`的计算如下所示：

```
FLOPs = 2×(Ci×K×K+1)×H×W×Co
```
其中 2 是因为 MAC 操作（即累乘和累加操作）。

对于全连接层，假设输入维度是`I`, 输出维度`O`，则有：

```
FLOPs = 2×(I-1)×O
```

当然，以上计算都是对于单个输入，没有考虑`batch size`的。


参考：

- [cnn模型所需的计算力（flops）是怎么计算的？](https://www.zhihu.com/question/65305385)
- [PRUNING CONVOLUTIONAL NEURAL NETWORKS FOR RESOURCE EFFICIENT INFERENCE](https://arxiv.org/pdf/1611.06440v2.pdf)



