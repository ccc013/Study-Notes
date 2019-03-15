
之前推荐过一个基于 TensorFlow 的 GAN 框架--[谷歌开源的 GAN 库--TFGAN](https://mp.weixin.qq.com/s/Kd_nsit-JMaEjT5o8rEkKQ)。

而最近也有一个新的 GAN 框架工具，并且是基于 Pytorch 实现的，项目地址如下：

https://github.com/torchgan/torchgan

对于习惯使用 Pytorch 框架的同学，现在可以采用这个开源项目快速搭建一个 GAN 网络模型了！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/torchgan.png)

目前该开源项目有 400+ 星，它给出了安装的教程、API 文档以及使用教程，文档的地址如下：

https://torchgan.readthedocs.io/en/latest/

#### 安装

对于 TorchGAN 的安装，官网给出 3 种方法，但实际上目前仅支持两种安装方式，分别是`pip`方式安装以及源码安装，采用`conda`安装的方法目前还不支持。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/torchgan_install.png)

##### Pip 安装方法

安装最新的发布版本的命令如下：

```
$ pip3 install torchgan
```

而如果是最新版本：

```
$ pip3 install git+https://github.com/torchgan/torchgan.git
```

##### Conda 安装

这是目前版本还不支持的安装方式，将会在`v0.1`版本实现这种安装方法。

##### 源码方式安装

按照下列命令的顺序执行来进行从源码安装

```
$ git clone https://github.com/torchgan/torchgan
$ cd torchgan
$ python setup.py install
```

##### 依赖库

**必须按照的依赖库**：

- Numpy
- Pytorch 0.4.1
- Torchvision

**可选**

- TensorboardX：主要是为了采用`Tensorboard`来观察和记录实验结果。安装通过命令`pip install tensorboardX`
- Visdom：为了采用`Xisdom`进行记录。安装通过命令`pip install visdom`

#### API 文档

API 的文档目录如下：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/torchgan_api.png)

从目录主要分为以下几个大类：

- torchgan.layers：包含当前常用的用于构建 GAN 结构的一些网络层，包括残差块，Self-Attention，谱归一化(Spectral Normalization)等等
- torchgan.logging：提供了很强的可视化工具接口，包括对损失函数、梯度、测量标准以及生成图片的可视化等
- torchgan.losses：常见的训练 GANs 模型的损失函数，包括原始的对抗损失、最小二乘损失、WGAN的损失函数等；
- torchgan.metrics：主要是提供了不同的评判测量标准
- torchgan.models：包含常见的 GAN 网络结构，可以直接使用并且也可以进行拓展，包括 DCGAN、cGAN等
- torchgan.trainer：主要是提供训练模型的函数接口

#### 教程

教程部分如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/torchgan_tutorials.png)

教程给出了几个例子，包括 DCGAN、Self-Attention GAN、CycleGAN 例子，以及如何自定义损伤的方法。

对于 Self-Attention GAN，还提供了一个在谷歌的 Colab 运行的例子，查看链接：

https://torchgan.readthedocs.io/en/latest/tutorials/sagan.html


---
### 小结

最后，再给出 Github 项目的链接和文档的对应链接地址：

https://github.com/torchgan/torchgan

https://torchgan.readthedocs.io/en/latest/index.html

另外大家如果有想要的有关机器学习、深度学习、python方面或者是编程方面，比如数据结构等方面的教程或者电子书资源，也可以在后台回复，如果我有的话，也会免费分享给你的！

欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)

之前分享的资源和教程文章有：

- [推荐几本数据结构算法书籍和课程](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483683&idx=1&sn=3a75e0eb3f2c897bf14777a311017c9a&chksm=fe3b0f56c94c8640f7bf90f0cbdbf5ebab838c6a90b24d43984b8fbdb94405552fada4946fc4&token=985117826&lang=zh_CN#rd)
- [[资源分享] Github上八千Star的深度学习500问教程](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483737&idx=1&sn=5e9a27bd2b88a608a49685213cc0d481&chksm=fe3b0f2cc94c863a0f86a062d4bab98d333332be4b546101fd15f0dd5269f2407ca5f3618e2d&token=985117826&lang=zh_CN#rd)
- [[资源分享] 吴恩达最新《机器学习训练秘籍》中文版可以免费下载了！](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483716&idx=1&sn=0dc336f5ef002dd0dd703908288cf6aa&chksm=fe3b0f31c94c8627ad8329cb4688fe08118d79cceb3c27f96a48543253978688d1786cb7a79e&token=985117826&lang=zh_CN#rd)
- [[资源分享] TensorFlow 官方中文版教程来了](https://mp.weixin.qq.com/s/Si1YaYLfhL1upbjQkvireQ)
- [必读的AI和深度学习博客](https://mp.weixin.qq.com/s/0J2raJqiYsYPqwAV1MALaw)
- [[教程]一份简单易懂的 TensorFlow 教程](https://mp.weixin.qq.com/s/vXIM6Ttw37yzhVB_CvXmCA)
- [谷歌开源的 GAN 库--TFGAN](https://mp.weixin.qq.com/s/Kd_nsit-JMaEjT5o8rEkKQ)



