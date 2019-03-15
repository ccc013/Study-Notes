

原文是：

http://bamos.github.io/2016/08/09/deep-completion/

---

本文的基本目录如下：

- 介绍
- 第一步：将图像解释为概率分布中的样本
    - 如何填充缺失的信息？
    - 对于图片在哪里适配这些统计数据？
    - 我们如何修复图片呢？
- 第二步：快速生成假的图片
    - 从未知的概率分布中学习生成新的样本
    - [ML-Heavy] 建立 GAN 模型
    - 采用 G(z) 生成假的图片
    - [ML-Heavy] 训练 DCGAN
    - 目前的 GAN 和 DCGAN 实现
    - [ML-Heavy] TensorFlow 实现 DCGAN
    - 在你的数据集上运行 DCGAN 模型
- 第三步：为图像修复寻找最佳的假图片
    - 利用 DCGANs 实现图像修复
    - [ML-Heavy] 损失函数
    - [ML-Heavy] TensorFlow 实现 DCGANs 模型来实现图像修复
    - 修复你的图片
- 结论
- 对本文/项目的引用
- 供进一步阅读的部分参考书目
- 一些未实现的对于 TensorFlow 和 Torch 的想法

---

### 介绍

设计师和摄像师习惯使用一个非常强有力的工具--内容感知填充，来修复图片中不需要或者缺失的部分。图像修复是指用于修复图像中缺失或者毁坏的部分区域。实现图像的修复有很多种方法。在本文中，介绍的是在 2016年7月26日发表在 arXiv 上的论文[“Semantic Image Inpainting with Perceptual and Contextual Losses”](https://arxiv.org/abs/1607.07539)，这篇论文介绍如何采用 [DCGAN](https://arxiv.org/abs/1511.06434) 来实现图像修复。这篇文章会即兼顾非机器学习背景和有机器学习背景的读者，带有 [ML-Heavy] 标签的标题内容表示可以跳过这部分细节内容。我们只考虑有限制的修复带有缺失像素的人脸图片的例子。TensorFlow 实现的源代码可以在下面的 Github 地址上查看：

https://github.com/bamos/dcgan-completion.tensorflow

我们将从以下三个步骤来完成图片修复工作：

1. 首先将图像解释为概率分布中的样本
2. 这样的解释步骤可以让我们学习如何生成假的图片
3. 为修复图片寻找最佳的生成图片

下面是两张修复前和修复后的图片例子：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/content-aware-1.jpg)

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/content-aware-2.jpg)

下面是本文将用到的带有缺失区域的人脸例子：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/missing_faces.png)

### 第一步：将图像解释为概率分布中的样本

#### 如何填充缺失的信息？

对于上述几张图片例子，假设你正在设计一个系列来填充这些缺失的区域，你会选择如何做？你认为人脑会怎么处理它呢？你需要使用哪些信息来实现这个修复工作呢？

本文会主要关注下面两种信息：

1. **上下文信息(Contextual information)**：利用缺失像素区域周围像素提供的信息来填充
2. **感知信息(Perceptual information)**：将填充的部分解释为“正常”，如同现实生活或者其他图片中看到的一样。

这两种信息都非常重要。没有上下文信息，你怎么知道填充什么信息呢？没有感知信息，对于一个上下文来说会有很多种有效的填充方式。比如一些对于机器学习系统来说看上去是“正常”的填充信息，但对于我们人类来说其实就是非常奇怪的填充内容。

因此，有一个即精确又直观的捕获这两种属性，并且可以解释说明如何一步步实现图像修复的算法是再好不过了。创造出这样的算法可能只会适用于特殊的例子，但通常都没有人知道如何创造这样的算法。现在最佳的做法是使用统计数据和机器学习方法来实现一种近似的技术。

#### 对于图片在哪里适配这些统计数据？

为了解释这个问题，首先介绍一个非常好理解而且能简明表示的[概率分布](https://en.wikipedia.org/wiki/Probability_distribution)：[正态分布](https://en.wikipedia.org/wiki/Normal_distribution)。下面是一个正态分布的[概率密度函数(probability density function, PDF)](https://en.wikipedia.org/wiki/Probability_density_function)的图示。你可以这么理解 PDF，它是水平方向表示输入空间的数值，在垂直方向上表示默写数值发生的概率。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/normal-pdf.png)

上面这张图的绘制代码如下：

```
# !/usr/bin/env python3

import numpy as np
from scipy.stats import norm

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('bmh')
import matplotlib.mlab as mlab

np.random.seed(0)
### 绘制一个正态分布的概率密度函数图###
# 生成数据 X范围是(-3,3),步进为0.001, Y的范围是(0,1)
X = np.arange(-3, 3, 0.001)
Y = norm.pdf(X, 0, 1)
# 绘制
fig = plt.figure()
plt.plot(X, Y)
plt.tight_layout()
plt.savefig("./images/normal-pdf.png")
```

接着可以从上述分布中采样得到一些样本数据，如下图所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/normal-samples.png)

绘制代码如下：


```
### 绘制从正态分布采样的 1D 散点图例子 ###
nSamples = 35
# np.random.normal 是从正态分布中随机采样指定数量的样本,这里指定 35个
X = np.random.normal(0, 1, nSamples)
Y = np.zeros(nSamples)
fig = plt.figure(figsize=(7, 3))
# 绘制散点图
plt.scatter(X, Y, color='k')
plt.xlim((-3, 3))
frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
plt.savefig("./images/normal-samples.png")
```

这是 1 维概率分布的例子，因为输入数据就只是一维数据，我们也可以实现二维的例子，如下图所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/normal-2d.png)

绘制代码如下：


```
### 绘制从正态分布采样的 2D 散点图例子###

delta = 0.025
# 设置 X,Y 的数值范围和步长值，分别生成 240个数
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-3.0, 3.0, delta)
print('x shape', x.shape)
# 根据坐标向量来生成坐标矩阵
X, Y = np.meshgrid(x, y)  # X, Y shape: (240, 240)

print('X shape', X.shape)
print('Y shape', Y.shape)
# Bivariate Gaussian distribution for equal shape *X*, *Y*
# 等形状的双变量高斯分布
Z = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)  # Z shape (240, 240)
print('Z shape', Z.shape)

plt.figure()
# 绘制环形图轮廓
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)

nSamples = 200
mean = [0, 0]
cov = [[1, 0], [0, 1]]
# 从多元正态分布中采样，得到结果图中的黑点例子
X, Y = np.random.multivariate_normal(mean, cov, nSamples).T
plt.scatter(X, Y, color='k')

plt.savefig("./images/normal-2d.png")
```

绘制上述三张图的完整代码如下所示，代码地址为：

https://github.com/bamos/dcgan-completion.tensorflow/blob/master/simple-distributions.py

图片和统计学之间的关键关系就是**我们可以将图片解释为高维概率分布的样本**。概率分布就体现在图片的像素上。假设你正采用你的相机进行拍照，照片的像素数量是有限的，当你用相机拍下一张照片的时候，就相当于从这个复杂的概率分布中进行采样的操作。而这个分布也是我们用来定义一张图片是否正常。和正态分布不同的是，只有图片，我们是不知道真实的概率分布，只是在收集样本而已。

在本文中，我们采用 [RGB 颜色模型](https://en.wikipedia.org/wiki/RGB_color_model)表示的彩色图片。我们采用的是宽和高都是 64 像素的图片，所以概率分布的维度应该是 64×64×3≈12k。

##### 我们如何修复图片呢？

首先为了更加直观，我们先考虑之前介绍的多元正态分布。给定`x=1`时，`y`最有可能的取值是什么呢？这可以通过固定`x=1`，然后最大化 PDF 的值来找到所有可能的`y`的取值。如下图所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/normal-2d-max.png)

上图中垂直的黑色直线经过的黑点就是符合要求的`y`值。

这个概念可以延伸到我们的图像概率分布中，当我们知道某些数值，然后想填补完成缺失的数值的时候。只需要将它当做寻找所有可能缺失数值的最大问题，那么找到的结果就是最有可能的图片。

从视觉上观察由正态分布采样得到的样本，仅凭它们就找到概率密度函数是一件似乎很合理的事情。我们只需要选择最喜欢的[统计模型](https://en.wikipedia.org/wiki/Statistical_model)并将其与数据相适应即可。

然而，我们并不会应用这个方法。虽然从简单分布中恢复概率密度函数是很简单，但这对于图像的复杂分布是非常困难和棘手的事情。其复杂性一定程度上是来自于复杂的[条件独立性](https://en.wikipedia.org/wiki/Conditional_dependence)：图像中的每个像素值之间都是相互依赖的。因此，最大化一个通用的概率密度函数是一个极其困难而且往往难以解决的非凸优化问题。

---
### 第二步：快速生成假的图片

#### 从未知的概率分布中学习生成新的样本

与其考虑如何计算概率密度函数，现在在统计学中更好的方法是采用一个[生成模型](https://en.wikipedia.org/wiki/Generative_model)来学习如何生成新的、随机的样本。过去生成模型一直是很难训练或者非常难以实现，但最近在这个领域已经有了一些让人惊讶的进展。[Yann LeCun](http://yann.lecun.com/)在这篇 Quora 上的问题[“最近在深度学习有什么潜在的突破的领域”]( https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning/answer/Yann-LeCun?srid=nZuy
)中给出了一种训练生成模型（对抗训练）方法的介绍，并将其描述为过去十年内机器学习最有趣的想法：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/lecun-quora.png)

Yann LeCun 在回答中简单介绍了 GAN 的基本原理，也就是两个网络相互博弈的过程。

实际上，深度学习还有其他方法来训练生成模型，比如 [Variational Autoencoders(VAEs)](http://arxiv.org/abs/1312.6114)。但在本文，主要介绍对抗生成网络（GANs）

#### [ML-Heavy] 建立 GAN 模型

GANs 这个想法是 Ian Goodfellow 在其带有里程碑意义的论文[“Generative Adversarial Nets” (GANs)](http://papers.nips.cc/paper/5423-generative-adversarial)发表在 2014 年的  [Neural Information Processing Systems (NIPS)](https://nips.cc/) 会议上后开始火遍整个深度学习领域的。这个想法就是我们首先定义一个简单并众所周知的概率分布，并表示为$p_z$，在本文后面，我们用 $p_z$ 表示在[-1,1)（包含-1，但不包含1）范围的均匀分布。用$z \thicksim p_z$表示从这个分布中采样，如果$p_z$是一个五维的，我们可以利用下面一行的 Python 代码来进行采样得到，这里用到 [numpy](http://www.numpy.org/)这个库：


```
z = np.random.uniform(-1, 1, 5)
array([ 0.77356483,  0.95258473, -0.18345086,  0.69224724, -0.34718733])
```

现在我们有一个简单的分布来进行采样，接下来可以定义一个函数`G(z)`来从原始的概率分布中生成样本，代码例子如下所示：

```
def G(z):
   ...
   return imageSample

z = np.random.uniform(-1, 1, 5)
imageSample = G(z)
```

那么问题来了，怎么定义这个`G(Z)`函数，让它实现输入一个向量然后返回一张图片呢？答案就是采用一个深度神经网络。对于深度神经网络基础，网络上有很多的介绍，本文就不再重复介绍了。这里推荐的一些参考有斯坦福大学的 [CS231n 课程](http://cs231n.github.io/)、Ian Goodfellow 等人编著的[《深度学习》书籍](http://www.deeplearningbook.org/)、[形象解释图像的核心](http://setosa.io/ev/image-kernels/)以及论文["A guide to convolution arithmetic for deep learning"](https://arxiv.org/abs/1603.07285)。

通过深度学习可以有多种方法来实现`G(z)`函数。在原始的 GAN 论文中提出一种训练方法并给出初步的实验结果，这个方法得到了极大的发展和改进。其中一种想法就是在论文[“Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”](https://arxiv.org/abs/1511.06434)中提出的，这篇论文的作者是 Alec Radford, Luke Metz, and Soumith Chintala，发表在 2016 年的 [International Conference on Learning Representations (ICLR)](http://www.iclr.cc/)会议上，**这个方法因为提出采用深度卷积神经网络，被称为 DCGANs，它主要采用小步长卷积（ fractionally-strided convolution）方法来上采样图像**。

那么什么是小步长卷积以及如何实现对图片的上采样呢？ Vincent Dumoulin and Francesco Visin’s 在论文["A guide to convolution arithmetic for deep learning"](https://arxiv.org/abs/1603.07285)以及 Github 项目都给出了这种卷积算术的详细介绍，Github 地址如下：

https://github.com/vdumoulin/conv_arithmetic

上述 Github 项目给出了非常直观的可视化，如下图所示，这让我们可以很直观了解小步长卷积是如何工作的。

首先，你要知道一个正常的卷积操作是一个卷积核划过输入区域（下图中蓝色区域）后生成一个输出区域（下图的绿色区域）。这里，输出区域的尺寸是小于输入区域的。（当然，如果你还不知道，可以先看下斯坦福大学的[CS231n 课程](http://cs231n.github.io/)或者论文["A guide to convolution arithmetic for deep learning"](https://arxiv.org/abs/1603.07285)。）

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/padding_strides.gif)

接下来，假设输入是 3x3。我们的目标是通过上采样让输出尺寸变大。你可以认为小步长卷积就是在像素之间填充 0 值来拓展输入区域的方法，然后再对输入区域进行卷积操作，正如下图所示，得到一个 5x5 的输出。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/padding_strides_transposed.gif)

注意，对于作为上采样的卷积层有很多不同的名字，比如[全卷积(full convolution)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
), 网络内上采样（in-network upsampling）, 小步长卷积（fractionally-strided convolution）, 反向卷积（backwards convolution）, 反卷积（deconvolution）, 上卷积（upconvolution）, 转置卷积（transposed convolution）。这里并不鼓励使用反卷积（deconvolution）这个词语，因为在[数学运算](https://en.wikipedia.org/wiki/Deconvolution)或者[计算机视觉的其他应用](http://www.matthewzeiler.com/pubs/iccv2011/iccv2011.pdf
)中，这个词语有着其他完全不同的意思，这是一个非常频繁使用的词语。

现在利用小步长卷积作为基础，我们可以实现`G(z)`函数，让它接收一个$z \thicksim p_z$的向量输入，然后输出一张尺寸是 64x64x3 的彩色图片，其网络结构如下图所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/gen-architecture.png)

在 DCGAN 这篇论文中还提出了其他的一些技巧和改进来训练 DCGANs，比如采用批归一化(batch normalization)或者是 leaky ReLUs 激活函数。

#### 采用 G(z) 生成假的图片
   
现在先让我们暂停并欣赏下这种`G(z)`网络结构的强大，在 DCGAN 论文中给出了如何采用一个卧室图片数据集训练 一个 DCGAN 模型，然后采用`G(z)`生成如下的图片，它们都是生成器网络 G 认为的卧室图片，注意，**下面这些图片都是原始训练数据集没有的！**

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/generated-bedrooms.png)
   
此外，你还可以对 `z` 输入实现一个向量算术操作，下图就是一个例子：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/face-arithmetic.png)

#### [ML-Heavy] 训练 DCGAN

现在我们定义好了`G(z)`，也知道它的能力有多强大，问题来了，怎么训练呢？我们需要确定很多隐变量（或者说参数），这也是采用对抗网络的原因了。

首先，我们先定义几个符号。$p_data$表示训练数据，但概率分布未知，$p_z$表示从已知的概率分布采样的样本，一般从高斯分布或者均匀分布采样，`z`也被称为随机噪声，最后一个，$p_g$就是 G 网络生成的数据，也可以说是生成概率分布。

接着介绍下判别器（discriminator，D）网络，它是输入一批图片`x`，然后返回该图片来自训练数据$p_{data}$的概率。如果来自训练数据，D 应该返回一个接近 1 的数值，否则应该是一个接近 0 的值来表示图片是假的，来自 G 网络生成的。在 DCGANs 中，D 网络是一个传统的卷积神经网络，如下图所示，一个包含4层卷积层和1层全连接层的卷积神经网络结构。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/discrim-architecture.png)

因此，训练 D 网络的目标有以下两个：

1. 如果`x`来自训练数据集，最大化`D(x)`；
2. 如果`x`是来自 G 生成的数据，最小化`D(x)`。

对应的 G 网络的目标就是要欺骗 D 网络，生成以假乱真的图片。它生成的图片也是 D 的输入，**所以 G 的目标就是最大化`D(G(z))`，也等价于最小化`1-D(G(z))`，因为 D 其实是一个概率估计，且输出范围是在 0 到 1 之间。**

正如论文提到的，训练对抗网络就如同在实现一个最小化最大化游戏(minimax game)。如下面的公式所示，第一项是对真实数据分布的期望，第二项是对生成数据的期望值。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/gan_maths.png)

训练的步骤如下图所示，具体可以看下我之前写的文章[[GAN学习系列2] GAN的起源](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483732&idx=1&sn=99cb91edf6fb6da3c7d62132c40b0f62&chksm=fe3b0f21c94c8637a8335998c3fc9d0adf1ac7dea332c2bd45e63707eac6acad8d84c1b3d16d&token=985117826&lang=zh_CN#rd)有简单介绍了这个训练过程，或者是看下 GAN 论文[5]的介绍

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/gan-training.png)

#### 目前的 GAN 和 DCGAN 实现

目前在 Github 上有许多 GAN 和 DCGAN 的实现（原文是写于2016年八月份，现在的话代码就更多了）：

- https://github.com/goodfeli/adversarial
- https://github.com/tqchen/mxnet-gan
- https://github.com/Newmu/dcgan_code
- https://github.com/soumith/dcgan.torch
- https://github.com/carpedm20/DCGAN-tensorflow
- https://github.com/openai/improved-gan
- https://github.com/mattya/chainer-DCGAN
- https://github.com/jacobgil/keras-dcgan

本文实现的代码是基于 https://github.com/carpedm20/DCGAN-tensorflow

#### [ML-Heavy] TensorFlow 实现 DCGAN

这部分的实现的源代码可以在如下 Github 地址：

https://github.com/bamos/dcgan-completion.tensorflow

当然，主要实现部分代码是来自 https://github.com/carpedm20/DCGAN-tensorflow 。但采用这个项目主要是方便实现下一部分的图像修复工作。

主要实现代码是在`model.py`中的类`DCGAN`。采用类来实现模型是有助于训练后保存中间层的状态以及后续的加载使用。

首先，我们需要定义生成器和判别器网络结构。在`ops.py`会定义网络结构用到的函数，如`linear`,`conv2d_transpose`, `conv2d`以及 `lrelu`。代码如下所示


```
def generator(self, z):
    self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4,
                                           'g_h0_lin', with_w=True)

    self.h0 = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
    h0 = tf.nn.relu(self.g_bn0(self.h0))

    self.h1, self.h1_w, self.h1_b = conv2d_transpose(h0,
        [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
    h1 = tf.nn.relu(self.g_bn1(self.h1))

    h2, self.h2_w, self.h2_b = conv2d_transpose(h1,
        [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
    h2 = tf.nn.relu(self.g_bn2(h2))

    h3, self.h3_w, self.h3_b = conv2d_transpose(h2,
        [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3', with_w=True)
    h3 = tf.nn.relu(self.g_bn3(h3))

    h4, self.h4_w, self.h4_b = conv2d_transpose(h3,
        [self.batch_size, 64, 64, 3], name='g_h4', with_w=True)

    return tf.nn.tanh(h4)

def discriminator(self, image, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
    h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
    h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin')

    return tf.nn.sigmoid(h4), h4
```

当初始化这个类的时候，就相当于用上述函数来构建了这个模型。我们需要创建两个 D 网络来共享参数，一个的输入是真实数据，另一个是来自 G 网络的生成数据。


```
self.G = self.generator(self.z)
self.D, self.D_logits = self.discriminator(self.images)
self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
```

接下来是定义损失函数。这里采用的是 D 的输出之间的交叉熵函数，并且它的效果也不错。D 是期望对真实数据的预测都是 1，对生成的假数据预测都是 0，相反，生成器 G 希望 D 的预测都是 1。代码的实现如下：


```
self.d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,
                                            tf.ones_like(self.D)))
self.d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                            tf.zeros_like(self.D_)))
self.d_loss = self.d_loss_real + self.d_loss_fake

self.g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                            tf.ones_like(self.D_)))
```
接着是分别对 G 和 D 的参数聚集到一起，方便后续的梯度计算：

```
t_vars = tf.trainable_variables()

self.d_vars = [var for var in t_vars if 'd_' in var.name]
self.g_vars = [var for var in t_vars if 'g_' in var.name]
```

现在才有 ADAM 作为优化器来计算梯度，ADAM 是一个深度学习中常用的自适应非凸优化方法，它相比于随机梯度下降方法，不需要手动调整学习率、动量（momentum)以及其他的超参数。

```
d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.d_loss, var_list=self.d_vars)
g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.g_loss, var_list=self.g_vars)
```
定义好模型和训练策略后，接下来就是开始输入数据进行训练了。在每个 epoch 中，先采样一个 mini-batch 的图片，然后运行优化器来更新网络。有趣的是如果 G 只更新一次，D 的 loss 是不会变为0的。此外，在后面额外调用`d_loss_fake`和`d_loss_real`会增加不必要的计算量，并且也是多余的，因为它们的数值在`d_optim`和`g_optim`计算的时候已经计算到了。这里你可以尝试优化这部分代码，然后发送一个 PR 到原始的 Github 项目中。

```
for epoch in xrange(config.epoch):
    ...
    for idx in xrange(0, batch_idxs):
        batch_images = ...
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.images: batch_images, self.z: batch_z })

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })

        # Run g_optim twice to make sure that d_loss does not go to zero
        # (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })

        errD_fake = self.d_loss_fake.eval({self.z: batch_z})
        errD_real = self.d_loss_real.eval({self.images: batch_images})
        errG = self.g_loss.eval({self.z: batch_z})
```

完整的代码可以在 https://github.com/bamos/dcgan-completion.tensorflow/blob/master/model.py 中查看


#### 在你的数据集上运行 DCGAN 模型

如果你跳过上一小节，但希望运行一些代码：这部分的实现的源代码可以在如下 Github 地址：

https://github.com/bamos/dcgan-completion.tensorflow

当然，主要实现部分代码是来自 https://github.com/carpedm20/DCGAN-tensorflow 。但采用这个项目主要是方便实现下一部分的图像修复工作。但必须注意的是，如果你没有一个可以使用 CUDA 的 GPU 显卡，那么训练网络将会非常慢。

首先需要克隆两份项目代码，地址分别如下：

https://github.com/bamos/dcgan-completion.tensorflow

http://cmusatyalab.github.io/openface

第一份就是作者的项目代码，第二份是采用 OpenFace 的预处理图片的 Python 代码，并不需要安装它的 Torch 依赖包。先创建一个新的工作文件夹，然后开始克隆，如下所示：

```
git clone https://github.com/cmusatyalab/openface.git
git clone https://github.com/bamos/dcgan-completion.tensorflow.git
```

接着是安装 Python2 版本的 [OpenCV](http://opencv.org/)和 [dlib](http://dlib.net/)（采用 Python2 版本是因为 OpenFace 采用这个版本，当然你也可以尝试修改为适应 Python3 版本）。对于 OpenFace 的 Python 库安装，可以查看其安装指导教程，链接如下：

http://cmusatyalab.github.io/openface/setup/

此外，如果你没有采用一个虚拟环境，那么需要加入`sudo`命令来运行`setup.py`实现全局的安装 OpenFace，当然如果安装这部分有问题，也可以采用 OpenFace 的 docker 镜像安装。安装的命令如下所示

```
cd openface
pip2 install -r requirements.txt
python2 setup.py install
models/get-models.sh
cd ..
```

接着就是下载一些人脸图片数据集了，这里并不要求它们是否带有标签，因为不需要。目前开源可选的数据集包括 
- MS-Celeb-1M--https://www.microsoft.com/en-us/research/project/msr-image-recognition-challenge-irc/
- CelebA--http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- CASIA-WebFace--http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
- FaceScrub--http://vintage.winklerbros.net/facescrub.html
- LFW--http://vis-www.cs.umass.edu/lfw/
- MegaFace--http://megaface.cs.washington.edu/

然后将数据集放到目录`dcgan-completion.tensorflow/data/your-dataset/raw`下表示其是原始的图片。

接着采用 OpenFace 的对齐工具来预处理图片并调整成`64x64`的尺寸：

```
./openface/util/align-dlib.py data/dcgan-completion.tensorflow/data/your-dataset/raw align innerEyesAndBottomLip data/dcgan-completion.tensorflow/data/your-dataset/aligned --size 64
```
最后是整理下保存对齐图片的目录，保证只包含图片而没有其他的子文件夹：

```
cd dcgan-completion.tensorflow/data/your-dataset/aligned
find . -name '*.png' -exec mv {} . \;
find . -type d -empty -delete
cd ../../..
```

然后确保已经安装了 TensorFlow，那么可以开始训练 DCGAN了：

```
./train-dcgan.py --dataset ./data/your-dataset/aligned --epoch 20
```
在`samples`文件夹中可以查看保存的由 G 生成的图片。这里作者是采用手上有的两个数据集 CASIA-WebFace 和 FaceScrub 进行训练，并在训练 14 个 epochs 后，生成的结果如下图所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/dcgan-results.png)

还可以通过 TensorBoard 来查看 loss 的变化：


```
tensorboard --logdir ./logs
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/dcgan-tensorboard-results.png)



这是本文的最后一部分内容了，前两部分内容的文章：

1. [[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(上）](https://mp.weixin.qq.com/s/S_uiSe74Ti6N_u4Y5Fd6Fw)
2. [[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(中）](https://mp.weixin.qq.com/s/nYDZA75JcfsADYyNdXjmJQ)

以及原文的地址：

http://bamos.github.io/2016/08/09/deep-completion/


最后一部分的目录如下：

- 第三步：为图像修复寻找最佳的假图片
    - 利用 DCGANs 实现图像修复
    - [ML-Heavy] 损失函数
    - [ML-Heavy] TensorFlow 实现 DCGANs 模型来实现图像修复
    - 修复你的图片

---
### 第三步：为图像修复寻找最佳的假图片

#### 利用 DCGANs 实现图像修复

在第二步中，我们定义并训练了判别器`D(x)`和生成器`G(z)`，那接下来就是如何利用`DCGAN`网络模型来完成图片的修复工作了。

在这部分，作者会参考论文["Semantic Image Inpainting with Perceptual and Contextual Losses"](https://arxiv.org/abs/1607.07539) 提出的方法。

对于部分图片`y`，对于缺失的像素部分采用最大化`D(y)`这种看起来合理的做法并不成功，它会导致生成一些既不属于真实数据分布，也属于生成数据分布的像素值。如下图所示，我们需要一种合理的将`y`映射到生成数据分布上。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/inpainting-projection.png)

#### [ML-Heavy] 损失函数

首先我们先定义几个符号来用于图像修复。用`M`表示一个二值的掩码(Mask)，即只有 0 或者是 1 的数值。其中 1 数值表示图片中要保留的部分，而 0 表示图片中需要修复的区域。定义好这个 Mask 后，接下来就是定义如何通过给定一个 Mask 来修复一张图片`y`，具体的方法就是让`y`和`M`的像素对应相乘，这种两个矩阵对应像素的方法叫做[**哈大马乘积**](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))，并且表示为 `M ⊙ y ` ，它们的乘积结果会得到图片中原始部分，如下图所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/mask-example.png)

接下来，假设我们从生成器`G`的生成结果找到一张图片，如下图公式所示，第二项表示的是`DCGAN`生成的修复部分：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/math_1.png)

根据上述公式，我们知道最重要的就是第二项生成部分，也就是需要实现很好修复图片缺失区域的做法。为了实现这个目的，这就需要回顾在第一步提出的两个重要的信息，上下文和感知信息。而这两个信息的获取主要是通过损失函数来实现。损失函数越小，表示生成的`G(z)`越适合待修复的区域。

##### Contextual Loss

为了保证输入图片相同的上下文信息，需要让输入图片`y`（可以理解为标签）中已知的像素和对应在`G(z)`中的像素尽可能相似，因此需要对产生不相似像素的`G(z)`做出惩罚。该损失函数如下所示，采用的是 L1 正则化方法：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/math_2.png)

这里还可以选择采用 L2 正则化方法，但论文中通过实验证明了 L1 正则化的效果更好。

理想的情况是`y`和`G(z)`的所有像素值都是相同的，也就是说它们是完全相同的图片，这也就让上述损失函数值为0

##### Perceptual Loss

为了让修复后的图片看起来非常逼真，我们需要让判别器`D`具备正确分辨出真实图片的能力。对应的损失函数如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/math_3.png)

因此，最终的损失函数如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/math_4.png)

这里 λ 是一个超参数，用于控制两个函数的各自重要性。

另外，论文还采用[泊松混合(poisson blending)](http://dl.acm.org/citation.cfm?id=882269) 方法来平滑重构后的图片。

#### [ML-Heavy] TensorFlow 实现 DCGANs 模型来实现图像修复

代码实现的项目地址如下：

https://github.com/bamos/dcgan-completion.tensorflow

首先需要新添加的变量是表示用于修复的 mask，如下所示，其大小和输入图片一样

```
self.mask = tf.placeholder(tf.float32, [None] + self.image_shape, name='mask')
```
对于最小化损失函数的方法是采用常用的梯度下降方法，而在 TensorFlow 中已经实现了[自动微分](https://en.wikipedia.org/wiki/Automatic_differentiation)的方法，因此只需要添加待实现的损失函数代码即可。添加的代码如下所示：

```
self.contextual_loss = tf.reduce_sum(
    tf.contrib.layers.flatten(
        tf.abs(tf.mul(self.mask, self.G) - tf.mul(self.mask, self.images))), 1)
self.perceptual_loss = self.g_loss
self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)
```
接着，就是定义一个 mask。这里作者实现的是位置在图片中心部分的 mask，可以根据需求来添加需要的任意随机位置的 mask，实际上代码中实现了多种 mask

```
if config.maskType == 'center':
    scale = 0.25
    assert(scale <= 0.5)
    mask = np.ones(self.image_shape)
    l = int(self.image_size*scale)
    u = int(self.image_size*(1.0-scale))
    mask[l:u, l:u, :] = 0.0
```
因为采用梯度下降，所以采用一个 mini-batch 的带有动量的映射梯度下降方法，将`z`映射到`[-1,1]`的范围。代码如下：

```
for idx in xrange(0, batch_idxs):
    batch_images = ...
    batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)
    zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

    v = 0
    for i in xrange(config.nIter):
        fd = {
            self.z: zhats,
            self.mask: batch_mask,
            self.images: batch_images,
        }
        run = [self.complete_loss, self.grad_complete_loss, self.G]
        loss, g, G_imgs = self.sess.run(run, feed_dict=fd)
        # 映射梯度下降方法
        v_prev = np.copy(v)
        v = config.momentum*v - config.lr*g[0]
        zhats += -config.momentum * v_prev + (1+config.momentum)*v
        zhats = np.clip(zhats, -1, 1)
```

#### 修复你的图片

选择需要进行修复的图片，并放在文件夹`dcgan-completion.tensorflow/your-test-data/raw`下面，然后根据之前第二步的做法来对人脸图片进行对齐操作，然后将操作后的图片放到文件夹`dcgan-completion.tensorflow/your-test-data/aligned`。作者随机从数据集`LFW`中挑选图片进行测试，并且保证其`DCGAN`模型的训练集没有包含`LFW`中的人脸图片。

接着可以运行下列命令来进行修复工作了：

```
./complete.py ./data/your-test-data/aligned/* --outDir outputImages
```

上面的代码会将修复图片结果保存在`--outDir`参数设置的输出文件夹下，接着可以采用`ImageMagick`工具来生成动图。这里因为动图太大，就只展示修复后的结果图片：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/completion.png)

而原始的输入待修复图片如下：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/missing_faces.png)

---
### 小结

当然这个图片修复方法由于也是2016年提出的方法了，所以效果不算特别好，这两年其实已经新出了好多篇新的图片修复方法的论文，比如：

1. 2016CVPR [Context encoders: Feature learning by inpainting](https://arxiv.org/abs/1604.07379)
 
2. Deepfill 2018--[Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892)

3. Deepfillv2--[Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)

4. 2017CVPR--[High-resolution image inpainting using multi-scale neural patch synthesis](https://arxiv.org/abs/1611.09969)

5. 2018年的 NIPrus收录论文--[Image Inpainting via Generative Multi-column Convolutional Neural Networks](https://arxiv.org/abs/1810.08771)







