# Inception

介绍下 Incepton 目前 4 个版本的网络模型。

1. Inception[**V1**]: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
2. Inception[**V2**]: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
3. Inception[**V3**]: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
4. Inception[**V4**]: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)



inception 的改进，主要是围绕着CNN 这几个限制其性能的问题来做优化的：

1. 参数空间大，容易过拟合，而且训练数据集有限；
2. 网络结构复制，计算资源不足，导致难以应用；
3. 深层次网络结构容易出现梯度弥散，模型性能下降



对于四种版本的改进总结：

- **Inception V1**——构建了 1x1、3x3、5x5 的 conv 和 3x3 的 pooling 的**分支网络**，同时使用 **MLPConv** 和**全局平均池化**，扩宽卷积层网络宽度，增加了网络对尺度的适应。
- **Inception V2**——提出了 **Batch Normalization**，代替 **Dropout** 和 **LRN**，其正则化的效果让大型卷积网络的训练速度加快很多倍，同时收敛后的分类准确率也可以得到大幅提高，同时学习 **VGG** 使用两个 $3*3$ 的卷积核代替 $5*5$ 的卷积核，在降低参数量同时提高网络学习能力；
- **Inception V3**——引入了 **Factorization**，将一个较大的二维卷积拆成两个较小的一维卷积，比如将 $3*3$ 卷积拆成 $1*3$ 卷积和 $3*1$ 卷积，一方面节约了大量参数，加速运算并减轻了过拟合，同时增加了一层非线性扩展模型表达能力，除了在 **Inception Module** 中使用分支，还在分支中使用了分支（**Network In Network In Network**）；
- **Inception V4**——研究了 **Inception Module** 结合 **Residual Connection**，结合 **ResNet** 可以极大地加速训练，同时极大提升性能，在构建 Inception-ResNet 网络同时，还设计了一个更深更优化的 Inception v4 模型，能达到相媲美的性能。





## v1

### 模型介绍

v1 也就是 GoogleNet。

**Inception module** 的提出主要考虑多个不同 size 的卷积核能够增强网络的适应力，paper 中分别使用$1*1、3*3、5*5$卷积核，同时加入$3*3$ max pooling。


随后文章指出这种 naive 结构存在着问题：每一层 Inception module 的 filters 参数量为所有分支上的总数和，多层 Inception **最终将导致 model 的参数数量庞大，对计算资源有更大的依赖**。

在 NIN 模型中与$1*1$卷积层等效的 MLPConv 既能跨通道组织信息，提高网络的表达能力，同时可以对输出有效进行**降维**，因此文章提出了**Inception module with dimension reduction**，在不损失模型特征表示能力的前提下，尽量减少 filters 的数量，达到降低模型复杂度的目的：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v1_module.jpg)

这里 b 图对比 a 图是减少了权重参数的数量和输出 feature map 的维度，其计算如下所示：

> 假设previous layer的大小为 $28*28*192$，则，
>
> a的weights大小，$1*1*192*64+3*3*192*128+5*5*192*32=387072$
>
> a的输出featuremap大小，$28*28*64+28*28*128+28*28*32+28*28*192=28*28*416$
>
> b的weights大小，$1*1*192*64+(1*1*192*96+3*3*96*128)+(1*1*192*16+5*5*16*32)+1*1*192*32=163328$
>
> b的输出feature map大小，$28*28*64+28*28*128+28*28*32+28*28*32=28*28*256$

Inception module中，所有层的步长均为1，并且使用了SAME填充，因此，各个层的输出具有相同的尺寸(深度可以不同)。由于各个输出具有相同的尺寸，可以将其沿深度方向叠加，构成深度叠加层(depth concat layer)，该层可以通过TensorFlow中的concat()函数实现，其中axis=3(axis 3为深度方向)。

Inception Module 的4个分支在最后通过一个聚合操作合并（在输出通道数这个维度上聚合，在 TensorFlow 中使用 tf.concat(3, [], []) 函数可以实现合并）。



完整的 GoogLeNet 结构在传统的卷积层和池化层后面引入了 Inception 结构，对比 AlexNet 虽然网络层数增加，但是参数数量减少的原因是绝大部分的参数集中在全连接层，最终取得了 ImageNet 上 **6.67%** 的成绩。



### 模型结构

由9个Inception module组成，如下如，图中Inception module模块有一个图钉标记，每个Inception module分为三层。卷积层、池化层的结构为“特征图(卷积核)数，卷积核尺寸+步长(填充标记)”。图中Inception module模块的6个数字为每个卷积层输出的特征图数目，数字位置和 Inception module中各卷积层一一对应。所有卷积层都使用ReLU激活函数。



<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v1_network_2.jpg" style="zoom:80%;" />

参数表格如下：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v1_table.jpg" style="zoom:80%;" />

按照结构：

- 第一层将输入图像的边长减半，第二层再次将输入边长减半，即此层输出为原始输入图像的1/4，降低了计算量。
- LRN 层将前两层的输出进行**归一化**，使每个特征层更专一化，能够获取更广泛的特征，具有更高的泛化度。
- 接下来的第一个卷积层为瓶颈层，与其后一个卷积层构成了一个类似inception module中$1*1，3*3$的结构，其作用也一样。
- LRN层，作用不变。
- 最大池化层，步长为2，降低计算量。
- 9个 Inception module，在第二个 Inception module 之后，第七个 Inception module 之后，各有两个步长为2的最大池化层，为了降低计算量、为网络提速。
- 平均池化层使用了 VALID 填充的核，核的数目为前一个 Inception module 第二层特征图个数(4个卷积层输出的特征图个数)的总和，输出1*1尺寸的特征图(从上图avg pool层的输入可得)，这种方式称为global average pooling。该方式将前面层的输入转化为特征图，该特征图实际上是每个类的置信度图，因为在求均值时去除了其他类别的特征。因为每个特征图表示了一个类的置信度，所以仅需一个全连接层即可，这样减少了参数个数、降低了过拟合的可能性。
- 最后是用于正则化的dropout层，使用softmax激活函数的全连接层，输出预测结果的概率分布。

上述结构是一个简化的GoogLeNet，原始的结构在第三、第六个Inception module上，有两个**辅助分类器**。均由平均池化层、卷积层、两个全连接层、一个softmax激活层组成。**在训练时，其损失的30%添加到整体的损失中。目的是解决梯度消失问题、归一化网络**。实际上，其作用不大。

辅助分类器结构如下如，具体细节：

- 均值池化层核尺寸为5x5，步长为3，(4a)的输出为4x4x512，(4d)的输出为4x4x528。
- 1x1的卷积用于降维，拥有128个滤波器，采用ReLU激活函数。
- 全连接层有1024个神经元，采用ReLU激活函数。
- dropout层的dropped的输出比率为70%。
- softmax激活函数用来分类，和主分类器一样预测1000个类，但在推理时移除。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v1_network_3.jpg" style="zoom:80%;" />



完整的网络结构如下：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v1_network.jpg" style="zoom:50%;" />



### 模型特点


特点如下：

1. 卷积层共有的一个功能，可以实现通道方向的降维和增维，至于是降还是增，取决于卷积层的通道数（滤波器个数），在Inception v1中 $1*1$卷积用于降维，减少 weights 大小和 feature map 维度。
2. $1*1$ 卷积特有的功能，由于$1*1$卷积只有一个参数，相当于对原始 feature map 做了一个scale，并且这个 scale 还是训练学出来的，无疑会对识别精度有提升。
3. 增加了网络的深度和宽度
4. 同时使用了$1*1，3*3，5*5$的卷积，增加了网络对尺度的适应性
5. 整个网络为了收敛，采用了 3 个 loss
6. 最后一个全连接层前采用的是 global average pooling
7. 另外增加了两个辅助的softmax分支，作用有两点，**一是为了避免梯度消失**，用于向前传导梯度。反向传播时如果有一层求导为0，链式求导结果则为0。**二是将中间某一层输出用作分类，起到模型融合作用**。最后的 $loss=loss_2 + 0.3 * loss_1 + 0.3 * loss_0$。实际测试时，这两个辅助softmax分支会被去掉。








## v2

### 模型介绍

Inception v2 学习 VGG 使用 2 个$ 3*3$替代 1 个 $5*5$ 卷积。和 v1 版本的对比如下所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v2_compare.jpg" style="zoom:75%;" />

并且，提出了著名的 **Batch Normalization**（以下简称BN）方法。BN 是一个非常有效的正则化方法，可以让大型卷积网络的训练速度加快很多倍，同时收敛后的分类准确率也可以得到大幅提高。

BN 在用于神经网络某层时，会对每一个 mini-batch 数据的内部进行标准化（**normalization**）处理，使输出规范化到 N(0,1) 的正态分布，减少了 **Internal Covariate Shift**（内部神经元分布的改变）。

BN 的论文指出，**传统的深度神经网络在训练时，每一层的输入的分布都在变化，导致训练变得困难**，我们只能使用一个很小的学习速率解决这个问题。

而对每一层使用 BN 之后，我们就可以有效地解决这个问题，学习速率可以增大很多倍，达到之前的准确率所需要的迭代次数只有1/14，训练时间大大缩短。而达到之前的准确率后，可以继续训练，并最终取得远超于 Inception V1 模型的性能—— top-5 错误率 ***\*4.8%\****，已经优于人眼水平。因为 BN 某种意义上还起到了**正则化的作用**，所以可以减少或者取消 **Dropout** 和 **LRN**，简化网络结构。







### 模型结构





参数表如下：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v2_table.jpg" style="zoom:67%;" />

- $7*7$ 卷积分解为 3 个 $3*3$ 卷积——大卷积核分解为小卷积核；
- 3个传统Inception modules，卷积核 $35*35$，特征数 288，降维得到 $17*17*768$——高效的降维方法；
- 5 个应用了小卷积核分解的Inception modules，如下图，将上一层的输出降维至 $8*8*1280$——高效的降维方法；

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v2_module.jpg" style="zoom:80%;" />










### 模型特点

1. 加入了BN层，减少了 InternalCovariate Shift（内部neuron的数据分布发生变化），使每一层的输出都规范化到一个N(0, 1)的高斯，从而增加了模型的鲁棒性，可以以更大的学习速率训练，收敛更快，初始化操作更加随意，同时作为一种正则化技术，可以减少dropout层的使用。
2. 用2个连续的$ 3*3$ conv替代 inception 模块中的 $5*5$，**从而实现网络深度的增加**，网络整体深度增加了9层，**缺点就是增加了25%的weights和30%的计算消耗**。







## v3

### 模型介绍

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v3_module2.png)

Inception v3网络，主要在v2的基础上，提出了**卷积分解（Factorization）**，即  **Factorization into small convolutions** 的思想，将一个较大的二维卷积拆成两个较小的一维卷积，比如将 $7 * 7$ 卷积拆成 $1*7$ 卷积和 $7*1$ 卷积，或者将 $3*3$ 卷积拆成 $1*3$ 卷积和 $3*1$ 卷积，如上图所示。

一方面**节约了大量参数**，加速运算并减轻了过拟合（比将 $7*7$ 卷积拆成 $1*7$ 卷积和 $7*1$ 卷积，比拆成 3 个 $3*3$ 卷积更节约参数），同时增加了**一层非线性扩展模型表达能力**。

论文中指出，这种非对称的卷积结构拆分，其结果比对称地拆为几个相同的小卷积核效果更明显，可以处理更多、更丰富的空间特征，增加特征多样性。

另一方面，**Inception V3 优化了 Inception Module 的结构**，现在 Inception Module 有 $35*35、17*17和8*8$ 三种不同结构。这些 Inception Module **只在网络的后部出现**，前部还是普通的卷积层。并且 Inception V3 除了在 Inception Module 中使用分支，**还在分支中使用了分支**（$8*8$ 的结构中），可以说是Network In Network In Network。最终取得 top-5 错误率 **3.5%**。





### 模型结构



将粗糙的 $8*8$ 层通过2个Inception modules来降维(高效的降维方法)，如下图，每个连接的输出堆叠在一起，构成2048个特征图。同时，也对应了更高维度的表征更容易在网络的局部中处理的规则。即顶层的特征具有一定的关联性，降维没有太大影响，然后通过卷积获得关联性强的特征的组合，所以先降维，后卷积；

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v2_module2.jpg" style="zoom:100%;" />





### 模型特点

1. 将$7*7$ 分解成两个一维的卷积$（1*7,7*1）$，$3*3$ 也是一样$（1*3,3*1）$，这样的好处，既可以**加速计算**（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，**使得网络深度进一步增加，增加了网络的非线性**，更加精细设计了 $35*35/17*17/8*8$ 的模块。
2. **增加网络宽度**，网络输入从 $224*224$ 变为了 $299*299$。





## v4

### 模型介绍

Inception v4主要利用**残差连接**（Residual Connection）来改进v3结构，代表作为，Inception-ResNet-v1，Inception-ResNet-v2，Inception-v4

resnet中的残差结构如下，这个结构设计的就很巧妙，简直神来之笔，使用原始层和经过2个卷基层或者3个卷积层的 feature map 做Eltwise。

首先介绍几个概念，左边的$3*3+3*3$(ResNet18，ResNet34)和 $1*1+3*3+1*1$（ResNet50，ResNet101，ResNet152）称为**瓶颈单元**（bootlenect，因为输入为256，中间为64，输出为256，宽窄宽的结构，像瓶子的颈部）。右面的直线，有的实现是直线中有 $1*1$ 卷积，称为shortcut。

整个 bootlenect+shortcut 称为Residual uint。几个 Residual uint 的叠加称为 Residual block。Resnet结构都是由4个 Residual block 组成的。

Inception-ResNet的改进就是使用上文的 Inception module 来替换 resnet shortcut 中的bootlenect。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v4_resnet.png" style="zoom:80%;" />

作者尝试了多种残差Inception，文中仅详细介绍两种：Inception-ResNet-v1，计算量与Inception-v3相似；另一种是Inception-ResNet-v2，计算量与新提出的Inception-v4的主体计算量相似。**实际中，Inception-v4的单步时间比较慢，可能是拥有大量的层导致**。



在残差版Inception网络中，**使用了比原始Inception模块计算开销更低的Inception模块**。每个Inception模块后面都添加一个过滤器扩展层(没有激活函数的1*1卷积层)，在与输入相加之前，用来增加过滤器组的维度(深度)，使其与输入的深度一致。这么做是为了补偿Inception模块产生的降维。

残差Inception与非残差Inception的另一个技术差异：**Inception-ResNet中，仅在传统层上使用BN，并未在完成输入与输出相加的层使用BN**。在所有层都使用BN是合理的、有益的，但是为了使每个模型副本能够在单个GPU上训练，并未这么做。事实证明，拥有较大核(激活尺寸/卷积核)的层消耗的内存，与整个GPU内存相比是不成比例的，明显较高。通过去掉这些层的BN操作，能够大幅提高Inception模块的数目。作者希望能够有更好的计算资源利用方法，从而省去Inception模块数目和层数之间的权衡。

作者发现，**如果过滤器数目超过1000，残差网络将变得不稳定，并且网络在训练的早期就‘死亡’了**，即迭代上万次之后，在平均池化层之前的层只输出0。即使降低学习率、添加额外的BN层也无法避免。

作者发现，**在将残差与其上一层的激活值相加之前，将残差缩放，这样可以使训练过程稳定**。通常采用0.1至0.3之间的缩放因子来缩放残差层，然后再将其添加到累加层的激活层。如下图：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_resnet_reduction.jpg" style="zoom:80%;" />

缩放模块仅作用于最后的线性激活值，缩放因子为常数，通常为0.1

其他深度网络也出现了类似问题。在一个非常深的残差网络中，研究者提出了两阶段训练。第一阶段，称为预热阶段，以较低的学习率训练，然后在第二阶段采用较高的学习率。

本文作者发现，如果过滤器数目非常高，即使学习率极低(如0.00001)也无法解决不稳定现象，并且第二阶段较高的学习率很可能降低第一阶段的学习效果，降低模型性能。**作者认为使用缩放更为可靠**。

尽管缩放在某些地方不是必须的，但是并未发现缩放会降低最终的准确性，而且缩放在一定程度上会使训练变得稳定。







### 模型结构

inception-v4 完整结构：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v4_network.jpg" style="zoom:80%;" />

 Inception-ResNet-v1 和 Inception-ResNet-v2的总体结构：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_resnet_network.jpg" style="zoom:80%;" />



Inception-resnet 模块结构如下所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_v4_resnet_module.png" style="zoom:80%;" />







### 模型特点

1. 将 Inception 模块和 Residual Connection 结合，提出了 Inception-ResNet-v1，Inception-ResNet-v2，使得训练加速收敛更快，精度更高。
2. 设计了更深的 Inception-v4 版本，效果和 Inception-ResNet-v2 相当。
3. 网络输入大小和V3一样，还是 $299*299$







## 参考

1. [从Inception v1,v2,v3,v4,RexNeXt到Xception再到MobileNets,ShuffleNet,MobileNetV2,ShuffleNetV2,MobileNetV3](https://blog.csdn.net/qq_14845119/article/details/73648100)
2. [GoogLeNet 之 Inception(V1-V4)](https://blog.csdn.net/hejin_some/article/details/78636586)
3. [Inception V1,V2,V3,V4 模型总结](https://zhuanlan.zhihu.com/p/52802896)

