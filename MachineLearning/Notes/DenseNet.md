# DenseNet

参考：

- [DenseNet算法详解](https://blog.csdn.net/u014380165/article/details/75142664)
- [DenseNet详解](https://zhuanlan.zhihu.com/p/43057737)



论文：Densely Connected Convolutional Networks
论文链接：https://arxiv.org/pdf/1608.06993.pdf
代码的github链接：https://github.com/liuzhuang13/DenseNet
MXNet版本代码（有ImageNet预训练模型）: https://github.com/miraclewkf/DenseNet



## 简介

DenseNet 是 CVPR2017 的 Oral，作者从 feature 入手，通过对特征的极致利用达到更好的效果和更少的参数。

**DenseNet**  其目的是避免梯度消失。和 residual 模块不同，dense **模块中任意两层之间均有短路连接**。也就是说，每一层的输入通过级联(concatenation) 包含了之前所有层的结果，即包含由低到高所有层次的特征。和之前方法不同的是，DenseNet 中**卷积层的滤波器数很少**。DenseNet 只用 ResNet 一半的参数即可达到 ResNet 的性能。

实现方面，作者在大会报告指出，直接将输出级联会占用很大 GPU 存储。后来，通过**共享存储**，可以在相同的GPU存储资源下训练更深的DenseNet。但由于有些中间结果需要重复计算，该实现会增加训练时间。 

下图展示了dense block 的结构图：

![preview](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/DenseNet_fig1.png)

DenseNet 具有这几个优点：

1. 减轻了 vanishing-gradient（梯度消失）和网络退化的问题；
2. short path 加强了feature的传递和重用
3. 相比 ResNet 拥有更少的参数数量
4. 网络更易于训练，并具有一定的正则效果

而缺点有：

1. 有些中间结果需要重复计算，该实现会增加训练时间
2. 直接级联会占用很大的 GPU 显存，后续通过共享内存解决了这个问题；



DenseNet 的性能很好的一个原因是这种 dense connection 相当于每一层都直接连接 input 和 loss，因此就可以减轻梯度消失现象，这样更深网络不是问题，此外由于参数减少了，也是有正则化的效果。



## 模型原理

在上图 dense block 中，第 i 层的输入不仅和第 i-1 层的输出相关，还和之前所有层的输出相关，即有：
$$
X_l = H_l([X_0, X_1,\ldots,X_{l-1}])
$$
其中这里的符号 [] 表示 concatenation 操作，即将之前的所有层输出的 feature map 按 channel 维度拼接在一起，另外这里用的非线性变换 H 是 BN+ReLU+Conv(3×3)的组合。

**Pooling 层**

由于在 DenseNet 中需要对不同层的 feature map 进行拼接操作，所以需要不同层的 feature map 保持相同的 feature size，这就限制了网络中 Down sampling 的实现。为了使用 Down sampling，作者将 DenseNet 分为多个 Denseblock，如下图所示:

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/dense_fig2.jpeg)

在同一个 Denseblock 中要求 feature size 保持相同大小，在不同 Denseblock 之间设置 transition layers 实现 Down sampling，在作者的实验中 transition layer 由 BN + Conv(1×1) ＋2×2 average-pooling 组成。

**Growth rate**

在 Denseblock 中，假设每一个非线性变换 H 的输出为 K 个 feature map，那么第i层网络的输入便为 $K*0*+(i-1)×K$, 这里我们可以看到 DenseNet 和现有网络的一个主要的不同点：**DenseNet可以接受较少的特征图数量作为网络层的输出**，如下图所示

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/dense_fig3.jpeg)

原因就是在同一个 Denseblock 中的每一层都与之前所有层相关联，如果我们把 feature 看作是一个 Denseblock 的全局状态，那么每一层的训练目标便是通过现有的全局状态，判断需要添加给全局状态的更新值。

因而**每个网络层输出的特征图数量 K 又称为 Growth rate**，同样决定着每一层需要给全局状态更新的信息的多少。在作者的实验中只需要较小的 K 便足以实现 state-of-art 的性能。

**Bottleneck Layers**

虽然 DenseNet 接受较少的 k，也就是 feature map 的数量作为输出，但由于不同层 feature map 之间由 cat 操作组合在一起，最终仍然会是 feature map 的 channel 较大而成为网络的负担。

在 Dense block 的 3×3 卷积前面都包含了一个 1×1的卷积操作，也就是 Bottleneck layer， 使用 **1×1 卷积作为特征降维的方法来降低 channel 数量**，以提高计算效率，同时还可以融合不同通道的特征。

经过改善后的非线性变换变为 BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)，使用 Bottleneck layers 的 DenseNet 被作者称为 DenseNet-B。在实验中，作者使用 1×1 卷积生成channel 数量为 4k 的 feature map.

**Compression**

为了进一步优化模型的简洁性，同样可以在 transition layer 中降低 feature map 的数量。若一个 Denseblock 中包含 m 个 feature maps，那么我们使其输出连接的 transition layer 层生成 ⌊θm⌋ 个输出 feature map。其中 θ 为 Compression factor, 当 θ=1时，transition layer 将保留原 feature 维度不变.

作者将使用 compression 且 θ=0.5 的 DenseNet 命名为 DenseNet-C， 将使用 Bottleneck 和 compression 且 θ=0.5 的 DenseNet 命名为DenseNet-BC。



通过以上的结构，DenseNet 是拥有这些特性：

1. 模型压缩：即通过加入了 bottleneck 和 transition layer，可以大大减少参数量；
2. 隐式的深度监督（Implicit Deep Supervision）：这是该网络模型性能很好的原因，那就是每个网络中每一层不仅接受了来自 loss 的监督，还由于存在多个 bypass 和 shortcut，所以监督是多样的；
3. 特征复用：作者设计了一个实验，对于任意卷积层，计算之前某层 feature map 在该层权重的绝对值平均数，这一平均数表明了这一层对于之前某一层 feature 的利用率，如下图所示：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/dense_fig4.jpeg)

从图中可以得到这些结论：

1. 一些较早层提取出的特征仍可能被较深层直接使用
2. 即使是 Transition layer 也会使用到之前 Denseblock 中所有层的特征
3. 第 2-3 个 Denseblock 中的层对之前 Transition layer 利用率很低，说明 transition layer 输出大量冗余特征。这也为 DenseNet-BC 提供了证据支持，既压缩的必要性。
4. 最后的分类层虽然使用了之前 Denseblock 中的多层信息，但更偏向于使用最后几个 feature map 的特征，说明在网络的最后几层，某些高级的特征可能被产生。





















