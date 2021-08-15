参考资料

1. 深度学习 500 问--经典网络模型

2. [CNN网络结构的发展：从LeNet到EfficientNet](https://mp.weixin.qq.com/s/ooK2aAC_TAPFmK9dPLF-Fw)



# AlexNet

## 模型介绍

论文：《ImageNet Classification with Deep Convolutional Neural Networks》

论文地址：

AlexNet是由$Alex$ $Krizhevsky $提出的首个应用于图像分类的深层卷积神经网络，该网络在2012年ILSVRC（ImageNet Large Scale Visual Recognition Competition）图像分类竞赛中以15.3%的top-5测试错误率赢得第一名。

AlexNet使用GPU代替CPU进行运算，使得在可接受的时间范围内模型结构能够更加复杂，**它的出现证明了深层卷积神经网络在复杂模型下的有效性**，使CNN在计算机视觉中流行开来，直接或间接地引发了深度学习的热潮。





## 模型结构

AlexNet的参数和结构图如下：

- 卷积层：5层
- 全连接层：3层
- 深度：8层
- 参数个数：60M
- 神经元个数：650k
- 分类数目：1000类



![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/alexnet.png)

如图所示，除去下采样（池化层）和局部响应规范化操作（Local Responsible Normalization, LRN），AlexNet一共包含8层，前5层由卷积层组成，而剩下的3层为全连接层。

网络结构分为上下两层，分别对应两个GPU的操作过程，除了中间某些层（$C_3$卷积层和$F_{6-8}$全连接层会有GPU间的交互），其他层两个GPU分别计算结 果。最后一层全连接层的输出作为$softmax$的输入，得到1000个图像分类标签对应的概率值。

除去GPU并行结构的设计，AlexNet网络结构与LeNet十分相似，其网络的参数配置如表所示。

|        网络层         |               输入尺寸               |                  核尺寸                  |               输出尺寸               |              可训练参数量               |
| :-------------------: | :----------------------------------: | :--------------------------------------: | :----------------------------------: | :-------------------------------------: |
|   卷积层$C_1$ $^*$    |        $224\times224\times3$         | $11\times11\times3/4,48(\times2_{GPU})$  | $55\times55\times48(\times2_{GPU})$  | $(11\times11\times3+1)\times48\times2$  |
| 下采样层$S_{max}$$^*$ | $55\times55\times48(\times2_{GPU})$  |       $3\times3/2(\times2_{GPU})$        | $27\times27\times48(\times2_{GPU})$  |                    0                    |
|      卷积层$C_2$      | $27\times27\times48(\times2_{GPU})$  | $5\times5\times48/1,128(\times2_{GPU})$  | $27\times27\times128(\times2_{GPU})$ | $(5\times5\times48+1)\times128\times2$  |
|   下采样层$S_{max}$   | $27\times27\times128(\times2_{GPU})$ |       $3\times3/2(\times2_{GPU})$        | $13\times13\times128(\times2_{GPU})$ |                    0                    |
|   卷积层$C_3$ $^*$    |  $13\times13\times128\times2_{GPU}$  | $3\times3\times256/1,192(\times2_{GPU})$ | $13\times13\times192(\times2_{GPU})$ | $(3\times3\times256+1)\times192\times2$ |
|      卷积层$C_4$      | $13\times13\times192(\times2_{GPU})$ | $3\times3\times192/1,192(\times2_{GPU})$ | $13\times13\times192(\times2_{GPU})$ | $(3\times3\times192+1)\times192\times2$ |
|      卷积层$C_5$      | $13\times13\times192(\times2_{GPU})$ | $3\times3\times192/1,128(\times2_{GPU})$ | $13\times13\times128(\times2_{GPU})$ | $(3\times3\times192+1)\times128\times2$ |
|   下采样层$S_{max}$   | $13\times13\times128(\times2_{GPU})$ |       $3\times3/2(\times2_{GPU})$        |  $6\times6\times128(\times2_{GPU})$  |                    0                    |
|  全连接层$F_6$  $^*$  |   $6\times6\times128\times2_{GPU}$   |     $9216\times2048(\times2_{GPU})$      | $1\times1\times2048(\times2_{GPU})$  |       $(9216+1)\times2048\times2$       |
|     全连接层$F_7$     |  $1\times1\times2048\times2_{GPU}$   |     $4096\times2048(\times2_{GPU})$      | $1\times1\times2048(\times2_{GPU})$  |       $(4096+1)\times2048\times2$       |
|     全连接层$F_8$     |  $1\times1\times2048\times2_{GPU}$   |             $4096\times1000$             |         $1\times1\times1000$         |       $(4096+1)\times1000\times2$       |

>卷积层$C_1$输入为$224\times224\times3$的图片数据，分别在两个GPU中经过核为$11\times11\times3$、步长（stride）为4的卷积卷积后，分别得到两条独立的$55\times55\times48$的输出数据。
>
>下采样层$S_{max}$实际上是嵌套在卷积中的最大池化操作，但是为了区分没有采用最大池化的卷积层单独列出来。在$C_{1-2}$卷积层中的池化操作之后（ReLU激活操作之前），还有一个LRN操作，用作对相邻特征点的归一化处理。
>
>卷积层$C_3$ 的输入与其他卷积层不同，$13\times13\times192\times2_{GPU}$表示汇聚了上一层网络在两个GPU上的输出结果作为输入，所以在进行卷积操作时通道上的卷积核维度为384。
>
>全连接层$F_{6-8}$中输入数据尺寸也和$C_3$类似，都是融合了两个GPU流向的输出结果作为输入。





## 模型特点

- 所有卷积层都使用ReLU 激活函数，降低了Sigmoid类函数的计算量，加快了模型收敛速度；
- 在多个GPU上进行模型的训练，不但可以提高模型的训练速度，还能提升数据的使用规模；
- 使用LRN对局部的特征进行归一化，结果作为ReLU激活函数的输入能有效降低错误率，提高精度；
- 重叠最大池化（overlapping max pooling），即池化范围z与步长s存在关系$z>s$（如$S_{max}$中核尺度为$3\times3/2$），避免平均池化（average pooling）的平均效应
- 使用随机丢弃技术（dropout）选择性地忽略训练中的单个神经元，避免模型的过拟合



------

# VGG

## 模型介绍

论文：《Very Deep Convolutional Networks for Large-Scale Image Recognition》

论文地址：https://arxiv.org/abs/1409.1556



VGGNet是由牛津大学视觉几何小组（Visual Geometry Group, VGG）提出的一种深层卷积网络结构，他们以7.32%的错误率赢得了2014年ILSVRC分类任务的亚军（冠军由GoogLeNet以6.65%的错误率夺得）和25.32%的错误率夺得定位任务（Localization）的第一名（GoogLeNet错误率为26.44%），网络名称VGGNet取自该小组名缩写。

VGGNet是首批把图像分类的错误率降低到10%以内模型，同时该网络所采用的 $3\times3$卷积核的思想是后来许多模型的基础，该模型发表在2015年国际学习表征会议（International Conference On Learning Representations, ICLR）后至今被引用的次数已经超过1万4千余次。



## 模型结构

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/vgg16.png)

在原论文中的 VGGNet **包含了6个版本的演进**，分别对应 VGG11、VGG11-LRN、VGG13、VGG16-1、VGG16-3和VGG19，不同的后缀数值表示不同的网络层数（VGG11-LRN表示在第一层中采用了LRN的VGG11，VGG16-1表示后三组卷积块中最后一层卷积采用卷积核尺寸为$1\times1$，相应的VGG16-3表示卷积核尺寸为$3\times3$），本节介绍的 VGG16为VGG16-3。

上图中的VGG16体现了VGGNet的核心思路，使用$3\times3$的卷积组合代替大尺寸的卷积（2个$3\times3卷积即可与$$5\times5$卷积拥有相同的感受视野），网络参数设置如表所示。



|       网络层       |        输入尺寸         |             核尺寸              |        输出尺寸         |             参数个数              |
| :----------------: | :---------------------: | :-----------------------------: | :---------------------: | :-------------------------------: |
|   卷积层$C_{11}$   |  $224\times224\times3$  |      $3\times3\times64/1$       | $224\times224\times64$  |   $(3\times3\times3+1)\times64$   |
|   卷积层$C_{12}$   | $224\times224\times64$  |      $3\times3\times64/1$       | $224\times224\times64$  |  $(3\times3\times64+1)\times64$   |
| 下采样层$S_{max1}$ | $224\times224\times64$  |          $2\times2/2$           | $112\times112\times64$  |                $0$                |
|   卷积层$C_{21}$   | $112\times112\times64$  |      $3\times3\times128/1$      | $112\times112\times128$ |  $(3\times3\times64+1)\times128$  |
|   卷积层$C_{22}$   | $112\times112\times128$ |      $3\times3\times128/1$      | $112\times112\times128$ | $(3\times3\times128+1)\times128$  |
| 下采样层$S_{max2}$ | $112\times112\times128$ |          $2\times2/2$           |  $56\times56\times128$  |                $0$                |
|   卷积层$C_{31}$   |  $56\times56\times128$  |      $3\times3\times256/1$      |  $56\times56\times256$  | $(3\times3\times128+1)\times256$  |
|   卷积层$C_{32}$   |  $56\times56\times256$  |      $3\times3\times256/1$      |  $56\times56\times256$  | $(3\times3\times256+1)\times256$  |
|   卷积层$C_{33}$   |  $56\times56\times256$  |      $3\times3\times256/1$      |  $56\times56\times256$  | $(3\times3\times256+1)\times256$  |
| 下采样层$S_{max3}$ |  $56\times56\times256$  |          $2\times2/2$           |  $28\times28\times256$  |                $0$                |
|   卷积层$C_{41}$   |  $28\times28\times256$  |      $3\times3\times512/1$      |  $28\times28\times512$  | $(3\times3\times256+1)\times512$  |
|   卷积层$C_{42}$   |  $28\times28\times512$  |      $3\times3\times512/1$      |  $28\times28\times512$  | $(3\times3\times512+1)\times512$  |
|   卷积层$C_{43}$   |  $28\times28\times512$  |      $3\times3\times512/1$      |  $28\times28\times512$  | $(3\times3\times512+1)\times512$  |
| 下采样层$S_{max4}$ |  $28\times28\times512$  |          $2\times2/2$           |  $14\times14\times512$  |                $0$                |
|   卷积层$C_{51}$   |  $14\times14\times512$  |      $3\times3\times512/1$      |  $14\times14\times512$  | $(3\times3\times512+1)\times512$  |
|   卷积层$C_{52}$   |  $14\times14\times512$  |      $3\times3\times512/1$      |  $14\times14\times512$  | $(3\times3\times512+1)\times512$  |
|   卷积层$C_{53}$   |  $14\times14\times512$  |      $3\times3\times512/1$      |  $14\times14\times512$  | $(3\times3\times512+1)\times512$  |
| 下采样层$S_{max5}$ |  $14\times14\times512$  |          $2\times2/2$           |   $7\times7\times512$   |                $0$                |
|  全连接层$FC_{1}$  |   $7\times7\times512$   | $(7\times7\times512)\times4096$ |      $1\times4096$      | $(7\times7\times512+1)\times4096$ |
|  全连接层$FC_{2}$  |      $1\times4096$      |        $4096\times4096$         |      $1\times4096$      |       $(4096+1)\times4096$        |
|  全连接层$FC_{3}$  |      $1\times4096$      |        $4096\times1000$         |      $1\times1000$      |       $(4096+1)\times1000$        |



## 模型特点

- 整个网络都使用了同样大小的卷积核尺寸 $3\times3$ 和最大池化尺寸 $2\times2$。
- $1\times1$ 卷积的意义主要在于线性变换，而输入通道数和输出通道数不变，没有发生降维。
- 两个 $3\times3$ 的卷积层串联相当于  1个 $5\times5$ 的卷积层，感受野大小为 $5\times5$。同样地，3 个 $3\times3$ 的卷积层串联的效果则相当于 1  个 $7\times7$ 的卷积层。这样的连接方式使得网络参数量更小，而且多层的激活函数令网络对特征的学习能力更强。
- VGGNet 在训练时有一个小技巧，先**训练浅层的的简单网络 VGG11**，再复用 VGG11 的权重来初始化 VGG13，如此反复训练并初始化 VGG19，能够使训练时收敛的速度更快。
- 在训练过程中使用**多尺度的变换**对原始数据做数据增强，使得模型不易过拟合。





