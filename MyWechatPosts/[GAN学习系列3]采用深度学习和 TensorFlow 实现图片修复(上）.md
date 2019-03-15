

在之前的两篇 GAN 系列文章--[[GAN学习系列1]初识GAN](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483711&idx=1&sn=ead88d5b21e08d9df853b72f31d4b5f4&chksm=fe3b0f4ac94c865cfc243123eb4815539ef2d5babdc8346f79a29b681e55eee5f964bdc61d71&token=1760252914&lang=zh_CN#rd)以及[[GAN学习系列2] GAN的起源](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483732&idx=1&sn=99cb91edf6fb6da3c7d62132c40b0f62&chksm=fe3b0f21c94c8637a8335998c3fc9d0adf1ac7dea332c2bd45e63707eac6acad8d84c1b3d16d&token=985117826&lang=zh_CN#rd)中简单介绍了 GAN 的基本思想和原理，这次就介绍利用 GAN 来做一个图片修复的应用，主要采用的也是 GAN 在网络结构上的升级版--DCGAN，最初始的 GAN 采用的还是神经网络，即全连接网络，而 DCGAN 则是换成卷积神经网络（CNNs）了，这可以很好利用 CNN 强大的特征提取能力，更好的生成质量更好的图片。

原文是：

http://bamos.github.io/2016/08/09/deep-completion/

由于原文比较长，所以会分为 3 篇来介绍。

---

这篇文章的目录如下：

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

本文会先讲述背景和第一步的工作内容。

### 介绍

设计师和摄像师习惯使用一个非常强有力的工具--内容感知填充，来修复图片中不需要或者缺失的部分。图像修复是指用于修复图像中缺失或者毁坏的部分区域。实现图像的修复有很多种方法。在本文中，介绍的是在 2016年7月26日发表在 arXiv 上的论文“Semantic Image Inpainting with Perceptual and Contextual Losses”[1]，这篇论文介绍如何采用 DCGAN[2] 来实现图像修复。这篇文章会即兼顾非机器学习背景和有机器学习背景的读者，带有 [ML-Heavy] 标签的标题内容表示可以跳过这部分细节内容。我们只考虑有限制的修复带有缺失像素的人脸图片的例子。TensorFlow 实现的源代码可以在下面的 Github 地址上查看：

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

为了解释这个问题，首先介绍一个非常好理解而且能简明表示的概率分布[3]：正态分布[4]。下面是一个正态分布的概率密度函数(probability density function, PDF)[5]的图示。你可以这么理解 PDF，它是水平方向表示输入空间的数值，在垂直方向上表示默写数值发生的概率。

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

在本文中，我们采用 RGB 颜色模型[6]表示的彩色图片。我们采用的是宽和高都是 64 像素的图片，所以概率分布的维度应该是 64×64×3≈12k。

##### 我们如何修复图片呢？

首先为了更加直观，我们先考虑之前介绍的多元正态分布。给定`x=1`时，`y`最有可能的取值是什么呢？这可以通过固定`x=1`，然后最大化 PDF 的值来找到所有可能的`y`的取值。如下图所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/normal-2d-max.png)

上图中垂直的黑色直线经过的黑点就是符合要求的`y`值。

这个概念可以延伸到我们的图像概率分布中，当我们知道某些数值，然后想填补完成缺失的数值的时候。只需要将它当做寻找所有可能缺失数值的最大问题，那么找到的结果就是最有可能的图片。

从视觉上观察由正态分布采样得到的样本，仅凭它们就找到概率密度函数是一件似乎很合理的事情。我们只需要选择最喜欢的统计模型[7]并将其与数据相适应即可。

然而，我们并不会应用这个方法。虽然从简单分布中恢复概率密度函数是很简单，但这对于图像的复杂分布是非常困难和棘手的事情。其复杂性一定程度上是来自于复杂的条件独立性[8]：图像中的每个像素值之间都是相互依赖的。因此，最大化一个通用的概率密度函数是一个极其困难而且往往难以解决的非凸优化问题。


---
### 小结

第一篇主要介绍了图像修复的简单背景，然后就是开始实现的第一步，也是比较偏理论，将我们待处理的图片数据作为一个概率分布的样本，并简单用代码实现了一维和二维的正态分布函数图。

在下一篇将介绍第二步内容，也就是快速生成假数据的工作。

欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，在后台留言，和我分享你的建议和看法，指正文章中可能存在的错误，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)

由于暂时没有留言功能，也可以到我的个人博客和 CSDN 博客进行留言：

http://ccc013.github.io/

https://blog.csdn.net/lc013/article/details/84845439

---

文中的一些链接：

1. https://arxiv.org/abs/1607.07539
2. https://arxiv.org/abs/1511.06434
3. https://en.wikipedia.org/wiki/Probability_distribution
4. https://en.wikipedia.org/wiki/Normal_distribution
5. https://en.wikipedia.org/wiki/Probability_density_function
6. https://en.wikipedia.org/wiki/RGB_color_model
7. https://en.wikipedia.org/wiki/Statistical_model
8. https://en.wikipedia.org/wiki/Conditional_dependence


**推荐阅读**

1.[机器学习入门系列(1)--机器学习概览(上)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483667&idx=1&sn=c6b6feb241897ede16bd745d595cef92&chksm=fe3b0f66c94c86701e9b071e62750d189c254fd3ebe9bb6251505162139efefdf866093b38c3&token=2134085567&lang=zh_CN#rd)

2.[机器学习入门系列(2)--机器学习概览(下)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483672&idx=1&sn=34b6687030db92fd3e04dcdebd09fffc&chksm=fe3b0f6dc94c867b2a72c427ebb90e2a683e6ad97ea2c5fbdc3a3bb86a8b159b8e5f107d2dcc&token=2134085567&lang=zh_CN#rd)

3.[[GAN学习系列] 初识GAN](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483711&idx=1&sn=ead88d5b21e08d9df853b72f31d4b5f4&chksm=fe3b0f4ac94c865cfc243123eb4815539ef2d5babdc8346f79a29b681e55eee5f964bdc61d71&token=1493836032&lang=zh_CN#rd)

4.[[GAN学习系列2] GAN的起源](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483732&idx=1&sn=99cb91edf6fb6da3c7d62132c40b0f62&chksm=fe3b0f21c94c8637a8335998c3fc9d0adf1ac7dea332c2bd45e63707eac6acad8d84c1b3d16d&token=985117826&lang=zh_CN#rd)

5.[[资源分享] TensorFlow 官方中文版教程来了](https://mp.weixin.qq.com/s/Si1YaYLfhL1upbjQkvireQ)

如果你觉得我写得还不错，可以给我点个赞！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/0.jpg)

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/02.gif)



