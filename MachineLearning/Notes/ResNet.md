# ResNet

参考资料：

1. [你必须要知道CNN模型：ResNet](https://zhuanlan.zhihu.com/p/31852747)





## 模型介绍

论文: 《Deep residual learning for image recognition》

论文地址：https://arxiv.org/abs/1512.03385



VGG 证明更深的网络层数是提高精度的有效手段，但是更深的网络极易导致梯度弥散，从而导致网络无法收敛。经测试，20 层以上会随着层数增加收敛效果越来越差。

**ResNet 可以很好的解决梯度消失的问题（其实是缓解，并不能真正解决），ResNet 增加了 shortcut 连边**。



### 简介

ResNet 是 2015 年提出的并在 ILSVRC 和 COCO2015 上共 5 个竞赛中都获得第一名的成绩，包括了 ImageNet 的分类、检测和定位，COCO 的检测和分割比赛，并且在 ImageNet 上刷新了成绩，将之前 2014 年 GoogleNet 的 top-5 误差从 6.7% 降低到了 3.57%，并且网络层数提高到了 152 层。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/resnet_1.png)



### 深度网络的退化问题

定义：在深度网络中，随着网络深度增加，网络准确率出现饱和，甚至下降。

具体如下图所示，56 层的网络反而比 20 层的网络效果更差，这不是过拟合问题，因为在训练集也存在这个问题，训练集的误差同样也是 56 层网络更糟糕。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/resnet_%E7%BD%91%E7%BB%9C%E9%80%80%E5%8C%96%E9%97%AE%E9%A2%98.png)



### 残差学习

深度网络存在退化问题表明随着网络层数的增加，训练难度也随之增加。

这里有一个假设：

> 现在你有一个浅层网络，你想通过向上堆积新层来建立深层网络，一个极端情况是这些增加的层什么也不学习，**仅仅复制浅层网络的特征，即这样的新层是恒等映射（Identity mapping）**。在这种情况下，深层网络应该至少和浅层网络性能一样，也不应该出现退化现象。

基于这样的思想，也就有了残差学习，残差学习单元如下图所示。假设输入是 x，输出是 H(x)，则有 H(x) = F(x) + x，这样其实原始的学习特征是 ![[公式]](https://www.zhihu.com/equation?tex=F%28x%29%2Bx) 。之所以这样是因为残差学习相比原始特征直接学习更容易。

当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。

这里的残差学习单元结构也是一种短路连接(shortcut connection)。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/resnet_%E6%AE%8B%E5%B7%AE%E5%AD%A6%E4%B9%A0%E5%8D%95%E5%85%83.png)

为什么残差学习相对更容易呢？

直观上来看残差学习需要学习的内容少，因为一般残差会比较小，学习难度也小点。

从数学上分析，则有，首先残差单元可以表示为：
$$
y_l = h(x_l) + F(x_l, W_l) \\
x_{l+1} = f(y_l)
$$
其中 $x_l$ 和 $x_{l+1}$ 分别是第 l 个残差单元的输入和输出，而 F 是残差函数，表示学习到的残差，f 是激活函数。根据上述公式，从浅层 l 到深层 L 的学习特征是：
$$
x_L = x_l + \sum_{i=l}^L F(x_i, W_i)
$$
利用链式规则，可以求得反向过程的梯度：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/resnet_loss.svg)





## 模型结构

ResNet网络是参考了VGG19网络，在其基础上进行了修改，并通过短路机制加入了残差单元，如下图所示。

变化主要体现在 ResNet 直接使用stride=2的卷积做下采样，并且用global average pool层替换了全连接层。ResNet的一个重要设计原则是：**当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度**。

从图中可以看到，ResNet相比普通网络每两层间增加了短路机制，这就形成了残差学习，其中虚线表示feature map数量发生了改变。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/resnet_2.jpg" style="zoom:80%;" />

图5 展示的34-layer的ResNet，还可以构建更深的网络如表所示。

从表中可以看到，对于18-layer和34-layer的ResNet，其进行的两层间的残差学习，当网络更深时，其进行的是三层间的残差学习，三层卷积核分别是1x1，3x3和1x1，一个值得注意的是隐含层的feature map数量是比较小的，并且是输出feature map数量的1/4。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/resnet_3.jpg)



下面我们再分析一下残差单元，ResNet使用两种残差单元，如下图所示。左图对应的是浅层网络，而右图对应的是深层网络。

对于短路连接，当输入和输出维度一致时，可以直接将输入加到输出上。但是当维度不一致时（对应的是维度增加一倍），这就不能直接相加。有两种策略：

1. 采用zero-padding增加维度，此时一般要先做一个downsamp，可以采用strde=2的pooling，这样不会增加参数；
2. 采用新的映射（projection shortcut），一般采用1x1的卷积，这样会增加参数，也会增加计算量。短路连接除了直接使用恒等映射，当然都可以采用 projection shortcut。



![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/resnet_4.png)

resnet 的作者在另一篇论文 [Identity Mappings in Deep Residual Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1603.05027) 中又对不同残差单元进行了研究，发现如下图所示的结构是最优的，其变化就是先采用 pre-activation，以及 BN 和 ReLU 都提前，而且作者推荐短路连接采用恒等变换，这样保证短路连接不会有阻碍。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/resnet_5.jpg)





## 模型特点

1. 采用残差学习单元结构，避免了深度网络的退化问题；
2. 直接使用stride=2的卷积做下采样，并且用 global average pool 层替换了全连接层；
3. 当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度；
4. 使用**短路连接**，使训练深层网络更容易，并且**重复堆叠**相同的模块组合。
5. ResNet大量使用了**批量归一层**。
6. 对于很深的网络(超过50层)，ResNet使用了更高效的**瓶颈(bottleneck)**结构。





