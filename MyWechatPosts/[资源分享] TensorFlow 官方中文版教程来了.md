
最近，TensorFlow 提供了中文版的教程（Tutorials）和指南（Guide）。

其中，教程是介绍了一些基本的机器学习模型，包括分类、回归等，也包括一些深度学习方面的模型，包括常用的卷积神经网络、生成对抗网络、循环神经网络等等，并且主要使用高阶的 Keras 等 API 来实现代码。

而指南则是深入介绍了 TensorFlow 的工作原理，包括高阶 API、Estimator、加速器、低阶 API 和 TensorBoard 等等。

项目地址是：

https://tensorflow.google.cn/tutorials/?hl=zh-cn

#### 教程

TensorFlow 是一个用于研究和生产的开放源代码机器学习库。TensorFlow 提供了各种 API，可供初学者和专家在桌面、移动、网络和云端环境下进行开发。中文版教程是为了让初学者可以快速上手 TensorFlow，所以也采用高阶的 keras 等 API 来展示不同模型的例子，包括基础的分类回归模型，更深入点的 CNN、GAN、RNN 等。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/TF%E4%B8%AD%E6%96%87%E7%89%88%E5%AE%98%E6%96%B9%E6%95%99%E7%A8%8B.png)

如上图所示，首先介绍的是机器学习方面的基本模型，分类和回归，其中分类是分别基于图像和文本来介绍，给出两个例子。基于图像的是采用 Fashion Mnist 这个数据集，如下图所示，

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/TF%E4%B8%AD%E6%96%87%E7%89%88%E5%AE%98%E6%96%B9%E6%95%99%E7%A8%8B3.png)

而基于文本的是采用 IMDB 的数据集，包含来自互联网电影数据库的 50000 条的影评文本。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/TF%E4%B8%AD%E6%96%87%E7%89%88%E5%AE%98%E6%96%B9%E6%95%99%E7%A8%8B4.png)

此外，应用在研究和实验方面的 Eager Execution 和分布式大规模训练的 Estimator 接口也有给出教程介绍使用。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/TF%E4%B8%AD%E6%96%87%E7%89%88%E5%AE%98%E6%96%B9%E6%95%99%E7%A8%8B2.png)

然后就是介绍其他的深度学习方面的模型，包括视觉方面的 CNN 和 GAN，序列模型 RNN 等等，最后就是给出后续的学习计划了，包括推荐 CS20(http://web.stanford.edu/class/cs20si/)、CS231n(http://cs231n.stanford.edu/)课程，书籍《使用Python进行深度学习》、《深度学习》等进行后续的学习和提升。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/%E5%90%8E%E7%BB%AD%E5%AD%A6%E4%B9%A0%E8%AE%A1%E5%88%92.png)

#### 指南

指南主要是深入介绍了 TensorFlow 的工作原理，包括以下的部分。

##### 高阶 API

- Keras，用于构建和训练深度学习模型的 TensorFlow 高阶 API。
- Eager Execution，一个以命令方式编写 TensorFlow 代码的 API，就像使用 NumPy 一样。
- Estimator，一个高阶 API，可以提供已准备好执行大规模训练和生产的完全打包的模型。
- 导入数据，简单的输入管道，用于将您的数据导入 TensorFlow 程序。

##### Estimator

- Estimator，了解如何将 Estimator 用于机器学习。
- 预创建的 Estimator，预创建的 Estimator 的基础知识。
- 检查点，保存训练进度并从您停下的地方继续。
- 特征列，在不对模型做出更改的情况下处理各种类型的输入数据。
- Estimator 的数据集，使用 tf.data 输入数据。
- 创建自定义 Estimator，编写自己的 Estimator。

##### 加速器

- 使用 GPU - 介绍了 TensorFlow 如何将操作分配给设备，以及如何手动更改此类分配。
- 使用 TPU - 介绍了如何修改 Estimator 程序以便在 TPU 上运行。

##### 低阶 API

- 简介 - 介绍了如何使用高阶 API 之外的低阶 TensorFlow API 的基础知识。
- 张量 - 介绍了如何创建、操作和访问张量（TensorFlow 中的基本对象）。
- 变量 - 详细介绍了如何在程序中表示共享持久状态。
- 图和会话 - 介绍了以下内容：
    - 数据流图：这是 TensorFlow 将计算表示为操作之间的依赖关系的一种表示法。
    - 会话：TensorFlow 跨一个或多个本地或远程设备运行数据流图的机制。如果您使用低阶 TensorFlow API 编程，请务必阅读并理解本单元的内容。如果您使用高阶 TensorFlow API（例如 Estimator 或 Keras）编程，则高阶 API 会为您创建和管理图和会话，但是理解图和会话依然对您有所帮助。
- 保存和恢复 - 介绍了如何保存和恢复变量及模型。

##### TensorBoard

TensorBoard 是一款实用工具，能够直观地展示机器学习的各个不同方面。以下指南介绍了如何使用 TensorBoard：

- TensorBoard：可视化学习过程 - 介绍了 TensorBoard。
- TensorBoard：图的可视化 - 介绍了如何可视化计算图。
- TensorBoard 直方图信息中心 - 演示了如何使用 TensorBoard 的直方图信息中心。

##### 其他

- TensorFlow 版本兼容性 - 介绍了向后兼容性保证及无保证内容。
- 常见问题解答 - 包含关于 TensorFlow 的常见问题解答。


---

欢迎关注我的微信公众号--机器学习与计算机视觉或者扫描下方的二维码，在后台留言，和我分享你的建议和看法，指正文章中可能存在的错误，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)

**推荐阅读**

1.[机器学习入门系列(1)--机器学习概览(上)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483667&idx=1&sn=c6b6feb241897ede16bd745d595cef92&chksm=fe3b0f66c94c86701e9b071e62750d189c254fd3ebe9bb6251505162139efefdf866093b38c3&token=2134085567&lang=zh_CN#rd)

2.[机器学习入门系列(2)--机器学习概览(下)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483672&idx=1&sn=34b6687030db92fd3e04dcdebd09fffc&chksm=fe3b0f6dc94c867b2a72c427ebb90e2a683e6ad97ea2c5fbdc3a3bb86a8b159b8e5f107d2dcc&token=2134085567&lang=zh_CN#rd)

3.[[GAN学习系列] 初识GAN](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483711&idx=1&sn=ead88d5b21e08d9df853b72f31d4b5f4&chksm=fe3b0f4ac94c865cfc243123eb4815539ef2d5babdc8346f79a29b681e55eee5f964bdc61d71&token=1493836032&lang=zh_CN#rd)

4.[[GAN学习系列2] GAN的起源](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483732&idx=1&sn=99cb91edf6fb6da3c7d62132c40b0f62&chksm=fe3b0f21c94c8637a8335998c3fc9d0adf1ac7dea332c2bd45e63707eac6acad8d84c1b3d16d&token=985117826&lang=zh_CN#rd)

5.[谷歌开源的 GAN 库--TFGAN](https://mp.weixin.qq.com/s/Kd_nsit-JMaEjT5o8rEkKQ)

如果你觉得我写得还不错，可以给我点个赞！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/0.jpg)


