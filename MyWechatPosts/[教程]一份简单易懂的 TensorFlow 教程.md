
上周分享了一份 TensorFlow 官方的中文版教程，这次分享的是在 Github 上的一份简单易懂的教程，项目地址是：

https://github.com/open-source-for-science/TensorFlow-Course#why-use-tensorflow

如下图所示，已经有超过7000的 Star了

![image](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/TF%E6%95%99%E7%A8%8B.png)

这个仓库的目标是提供一份简单且容易上手的 TensorFlow 教程，每个教程都包含**源代码**，并且大部分都包含有一份**文档**。

### 目录

- 什么是 TensorFlow？
    - 动机
- 为什么要使用 TensorFlow？
-  TensorFlow 的安装和环境配置
-  TensorFlow 教程
- 热身
- 基础知识
- 机器学习基础
- 神经网络
    - 一些有用的教程

#### 什么是 TensorFlow？

TensorFlow 是一个用于多任务数据流编程的开源软件库。它是一个符号数学库，同时也能应用在如神经网络方面的机器学习应用。它在谷歌可以同时应用在研究和工程中。

TensorFlow 是谷歌大脑团队开发出来作为谷歌内部使用的。它在2015年9月份公布出来，并采用 Apache 2.0 开源协议。

目前最新的稳定版本是 2018年9月27日的1.11.0版本。

##### 动机

开始这个开源项目的动机有很多。TensorFlow 是目前可用的最好的深度学习框架之一，所以应该问的是现在网上能找到这么多关于 TensorFlow 教程，为什么还需要创建这个开源项目呢？

#### 为什么要使用 TensorFlow？

深度学习现在是非常的火，并且现在也有快速和优化实现算法和网络结构的需求。而 TensorFlow 就是为了帮助实现这个目标而设计出来的。

TensorFlow 的亮点就在于它可以非常灵活的设计模块化的模型，但是这对于初学者是一个缺点，因为这意味着需要考虑很多东西才能建立一个模型。

当然，上述问题因为有很多高级的 API 接口，如 Keras(https://keras.io/) 和 Slim(https://github.com/tensorflow/models/blob/031a5a4ab41170d555bc3e8f8545cf9c8e3f1b28/research/inception/inception/slim/README.md) 等通过抽象机器学习算法中的许多模块的软件库而得到较好的解决。

对于 TensorFlow 来说，一件非常有趣的事情就是现在到处都可以找到它的身影。大量的研究者和开发者都在使用它，而且它的社区正以光速的速度发展起来。所以很多问题都可以轻松解决，因为在它的社区中有非常多的人都在使用，大部分人都会遇到相同的问题。

#### TensorFlow 的安装和环境配置

TensorFlow 的安装和环境配置可以如下面动图所示，按照这个教程：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/docs/tutorials/installation 操作即可。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/installation.gif)

这里主要推荐的是采用虚拟环境安装的方式，一是可以避免安装库冲突的问题，特别是因为 python 的版本问题；第二个是可以自定义工作环境，针对 python 的 2.x 版本 和 3.x 版本分别设置不同的虚拟环境，安装不同的软件库。

---

#### TensorFlow 教程

接下来就是本教程的主要内容了，大部分的教程都包含了文档的说明，所有的教程都有代码和用 Jupyter notebook 编写的代码，也就是 Ipython。

##### 热身

入门的代码：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/python/0-welcome

IPython 形式：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/ipython/0-welcome/code/0-welcome.ipynb

文档介绍：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/docs/tutorials/0-welcome

---

##### 基础

![image](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/basics.gif)

###### 基础的数学运算

文档介绍：https://github.com/open-source-for-science/TensorFlow-Course/tree/master/docs/tutorials/1-basics/basic_math_operations

代码：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/python/1-basics/basic_math_operations

Ipython：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/ipython/1-basics/basic_math_operations/code/basic_math_operation.ipynb

###### TensorFlow 变量介绍

文档介绍：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/docs/tutorials/1-basics/variables

代码：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/python/1-basics/variables/README.rst

Ipython：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/ipython/1-basics/variables/code/variables.ipynb

---

##### 机器学习基础

![image](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/basicmodels.gif)

###### 线性回归

文档介绍：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/docs/tutorials/2-basics_in_machine_learning/linear_regression

代码：https://github.com/open-source-for-science/TensorFlow-Course/tree/master/codes/python/2-basics_in_machine_learning/linear_regression

Ipython：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/ipython/2-basics_in_machine_learning/linear_regression/code/linear_regression.ipynb

###### 逻辑回归

文档说明：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/docs/tutorials/2-basics_in_machine_learning/logistic_regression

代码：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/python/2-basics_in_machine_learning/logistic_regression

Ipython：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/ipython/2-basics_in_machine_learning/logistic_regression/code/logistic_regression.ipynb

###### 线性支持向量机

代码：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/python/2-basics_in_machine_learning/linear_svm

Ipython：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/ipython/2-basics_in_machine_learning/linear_svm/code/linear_svm.ipynb

###### 多类核支持向量机

代码：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/python/2-basics_in_machine_learning/multiclass_svm

Ipython：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/ipython/2-basics_in_machine_learning/multiclass_svm/code/multiclass_svm.ipynb

---

##### 神经网络

![image](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/CNNs.png)

###### 多层感知器

代码：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/python/3-neural_networks/multi-layer-perceptron

Ipython：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/ipython/3-neural_networks/multi-layer-perceptron/code/train_mlp.ipynb

###### 卷积神经网络

文档介绍：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/docs/tutorials/3-neural_network/convolutiona_neural_network

代码：https://github.com/open-source-for-science/TensorFlow-Course/tree/master/codes/python/3-neural_networks/convolutional-neural-network

###### 循环神经网络

代码：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/python/3-neural_networks/recurrent-neural-networks/code/rnn.py

Ipython：https://github.com/open-source-for-science/TensorFlow-Course/blob/master/codes/ipython/3-neural_networks/recurrent-neural-networks/code/rnn.ipynb

##### 其他有用的教程

- TensorFlow Examples--适合初学者的教程和代码例子
  https://github.com/aymericdamien/TensorFlow-Examples
- Sungjoon's TensorFlow-101--采用 Jupyter Notebook 编写的教程
  https://github.com/sjchoi86/Tensorflow-101
- Terry Um’s TensorFlow Exercises--根据其他 TensorFlow 例子重新编写的代码
  https://github.com/terryum/TensorFlow_Exercises
- Classification on time series--采用 TensorFlow 实现的 LSTM 的循环神经网络分类代码
  https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition


---

这就是本次分享的 TensorFlow 教程，后面我也会继续分享对这个教程的学习笔记和翻译。

欢迎关注我的微信公众号--机器学习与计算机视觉或者扫描下方的二维码，在后台留言，和我分享你的建议和看法，指正文章中可能存在的错误，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)

由于没有留言功能，也可以到我的 CSDN 博客进行留言，我的 CSDN 博客网址是：

https://blog.csdn.net/lc013/article/details/84845439



**推荐阅读**

1.[机器学习入门系列(1)--机器学习概览(上)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483667&idx=1&sn=c6b6feb241897ede16bd745d595cef92&chksm=fe3b0f66c94c86701e9b071e62750d189c254fd3ebe9bb6251505162139efefdf866093b38c3&token=2134085567&lang=zh_CN#rd)

2.[机器学习入门系列(2)--机器学习概览(下)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483672&idx=1&sn=34b6687030db92fd3e04dcdebd09fffc&chksm=fe3b0f6dc94c867b2a72c427ebb90e2a683e6ad97ea2c5fbdc3a3bb86a8b159b8e5f107d2dcc&token=2134085567&lang=zh_CN#rd)

3.[[GAN学习系列] 初识GAN](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483711&idx=1&sn=ead88d5b21e08d9df853b72f31d4b5f4&chksm=fe3b0f4ac94c865cfc243123eb4815539ef2d5babdc8346f79a29b681e55eee5f964bdc61d71&token=1493836032&lang=zh_CN#rd)

4.[[GAN学习系列2] GAN的起源](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483732&idx=1&sn=99cb91edf6fb6da3c7d62132c40b0f62&chksm=fe3b0f21c94c8637a8335998c3fc9d0adf1ac7dea332c2bd45e63707eac6acad8d84c1b3d16d&token=985117826&lang=zh_CN#rd)

5.[[资源分享] TensorFlow 官方中文版教程来了](https://mp.weixin.qq.com/s/Si1YaYLfhL1upbjQkvireQ)

如果你觉得我写得还不错，可以给我点个赞！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/0.jpg)

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/02.gif)

