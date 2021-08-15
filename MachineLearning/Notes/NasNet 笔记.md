# NasNet 笔记

论文：《Learning Transferable Architectures for Scalable Image Recognition》

论文地址：https://arxiv.org/abs/1707.07012

代码地址：

- https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/nasnet.py
- https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py



CVPR2018的论文，作者是Google Brain的[Barret Zoph](https://link.zhihu.com/?target=https%3A//arxiv.org/search/cs%3Fsearchtype%3Dauthor%26query%3DZoph%2C%2BB),[Vijay Vasudevan](https://link.zhihu.com/?target=https%3A//arxiv.org/search/cs%3Fsearchtype%3Dauthor%26query%3DVasudevan%2C%2BV),[Jonathon Shlens](https://link.zhihu.com/?target=https%3A//arxiv.org/search/cs%3Fsearchtype%3Dauthor%26query%3DShlens%2C%2BJ),[Quoc V. Le](https://link.zhihu.com/?target=https%3A//arxiv.org/search/cs%3Fsearchtype%3Dauthor%26query%3DLe%2C%2BQ%2BV). 

它是论文[Neural Architecture Search With Reinforcement Learning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1611.01578)的升级版，主要的不同在于作者设计了一个新的search space（称为NASNet search space）使得训练出来的结构具有**可转移性**，作者在CIFAR-10数据集上搜索出最佳结构，然后堆叠多次再运用在ImageNet数据集上。另外，作者还提出了一种新的正则化技术-ScheduleDropPath。



## 简介

### 1. NASNet 控制器

在NASNet中，完整的网络的结构还是需要手动设计的，NASNet学习的是完整网络中被堆叠、被重复使用的网络单元。为了便于将网络迁移到不同的数据集上，我们需要学习两种类型的网络块：

1. **Normal Cell**：输出Feature Map和输入Feature Map的尺寸相同；
2. **Reduction Cell**：输出Feature Map对输入Feature Map进行了一次降采样，在Reduction Cell中，对使用Input Feature作为输入的操作（卷积或者池化）会默认步长为2。

NASNet 的控制器的结构如图1所示，每个网络单元由 ![[公式]](https://www.zhihu.com/equation?tex=B) 个网络块（block）组成，在实验中 ![[公式]](https://www.zhihu.com/equation?tex=B%3D5) 。每个块的具体形式如图1右侧部分，每个块有并行的两个卷积组成，它们会由控制器决定选择哪些 Feature Map 作为输入（灰色部分）以及使用哪些运算（黄色部分）来计算输入的 Feature Map。最后它们会由控制器决定如何合并这两个 Feature Map。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/nasnet_1.png" style="zoom:50%;" />

更精确的讲，NASNet网络单元的计算分为5步：

1. 从第 ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bi-1%7D) 个Feature Map或者第 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 个Feature Map或者之前已经生成的网络块中选择一个Feature Map作为hidden layer A的输入，图2是学习到的网络单元，从中可以看到三种不同输入Feature Map的情况；
2. 采用和1类似的方法为Hidden Layer B选择一个输入；
3. 为1的Feature Map选择一个运算；
4. 为2的Feature Map选择一个元素；
5. 选择一个合并3，4得到的Feature Map的运算。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/nasnet_2.png" style="zoom:50%;" />

在3，4中我们可以选择的操作有：

- 直接映射
- ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes1) 卷积；
- ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积；
- ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 深度可分离卷积；
- ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 空洞卷积；
- ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 平均池化；
- ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 最大池化；
- ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes3) 卷积 + ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes1) 卷积；
- ![[公式]](https://www.zhihu.com/equation?tex=5%5Ctimes5) 深度可分离卷积；
- ![[公式]](https://www.zhihu.com/equation?tex=5%5Ctimes5) 最大池化；
- ![[公式]](https://www.zhihu.com/equation?tex=5%5Ctimes5) 深度可分离卷积；
- ![[公式]](https://www.zhihu.com/equation?tex=5%5Ctimes5) 最大池化；
- ![[公式]](https://www.zhihu.com/equation?tex=7%5Ctimes7) 深度可分离卷积；
- ![[公式]](https://www.zhihu.com/equation?tex=7%5Ctimes7) 最大池化；
- ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes7) 卷积 + ![[公式]](https://www.zhihu.com/equation?tex=7%5Ctimes1) 卷积；

在5中可以选择的合并操作有（1）单位加；（2）拼接。

最后所有生成的Feature Map通过拼接操作合成一个完整的Feature Map。

为了能让控制器同时预测Normal Cell和Reduction Cell，RNN会有 ![[公式]](https://www.zhihu.com/equation?tex=2%5Ctimes5%5Ctimes+B) 个输出，其中前 ![[公式]](https://www.zhihu.com/equation?tex=5%5Ctimes+B) 个输出预测 Normal Cell的 ![[公式]](https://www.zhihu.com/equation?tex=B) 个块（如图1每个块有5个输出），后 ![[公式]](https://www.zhihu.com/equation?tex=5%5Ctimes+B) 个输出预测 Reduction Cell的 ![[公式]](https://www.zhihu.com/equation?tex=B) 个块。RNN使用的是单层100个隐层节点的 [LSTM](https://zhuanlan.zhihu.com/p/42717426)。



### 2. NASNet的强化学习

NASNet的强化学习思路和NAS相同，有几个技术细节这里说明一下：

1. NASNet进行迁移学习时使用的优化策略是Proximal Policy Optimization（PPO）；
2. 作者尝试了均匀分布的搜索策略，效果略差于策略搜索。



### 3. Scheduled Drop Path

在优化类似于Inception的多分支结构时，**以一定概率随机丢弃掉部分分支是避免过拟合的一种非常有效的策略**，例如DropPath。但是DropPath对NASNet不是非常有效。在NASNet的Scheduled Drop Path中，**丢弃的概率会随着训练时间的增加线性增加**。这么做的动机很好理解：**训练的次数越多，模型越容易过拟合，DropPath的避免过拟合的作用才能发挥的越有效**。



### 4. 其它超参

在NASNet中，强化学习的搜索空间大大减小，很多超参数已经由算法写死或者人为调整。这里介绍一下NASNet需要人为设定的超参数。

1. 激活函数统一使用ReLU，实验结果表明ELU nonlinearity 效果略优于ReLU；
2. 全部使用Valid卷积，padding值由卷积核大小决定；
3. Reduction Cell的Feature Map的数量需要乘以2，Normal Cell数量不变。初始数量人为设定，一般来说数量越多，计算越慢，效果越好；
4. Normal Cell的重复次数（下图中的 ![[公式]](https://www.zhihu.com/equation?tex=N) ）人为设定；
5. 深度可分离卷积在深度卷积和单位卷积中间不使用BN或ReLU;
6. 使用深度可分离卷积时，该算法执行两次；
7. 所有卷积遵循ReLU->卷积->BN的计算顺序；
8. 为了保持Feature Map的数量的一致性，必要的时候添加 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes1) 卷积。

堆叠Cell得到的CIFAR_10和ImageNet的实验结果如图所示

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/nasnet_3.png" style="zoom:50%;" />



## 总结

NASNet最大的贡献是解决了NAS无法应用到大数据集上的问题，它使用的策略是先在小数据集上学一个网络单元，然后在大数据集上堆叠更多的单元的形式来完成模型迁移的。

NASNet已经不再是一个dataset interest的网络了，因为其中大量的参数都是人为设定的，网络的搜索空间更倾向于**密集连接的方式**。这种人为设定参数的一个正面影响就是减小了强化学习的搜索空间，从而提高运算速度，在相同的硬件环境下，NASNet的速度要比NAS快7倍。

NASNet的网络单元本质上是一个更复杂的Inception，可以通过堆叠 网络单元的形式将其迁移到任意分类任务，乃至任意类型的任务中。论文中使用NASNet进行的物体检测也要优于其它网络。

本文使用CIFAR-10得到的网络单元其实并不是非常具有代表性，理想的数据集应该是ImageNet。但是现在由于硬件的计算能力受限，无法在ImageNet上完成网络单元的学习，随着硬件性能提升，基于ImageNet的NASNet一定会出现。或者我们也可以期待某个土豪团队多费电电费帮我们训练出这样一个架构来。







## 参考文献

[1] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition[J]. arXiv preprint arXiv:1707.07012, 2017, 2(6).

[2] Zoph B, Le Q V. Neural architecture search with reinforcement learning[J]. arXiv preprint arXiv:1611.01578, 2016.

[3] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[4] G. Larsson, M. Maire, and G. Shakhnarovich. Fractalnet: Ultra-deep neural networks without residuals. arXiv preprint arXiv:1605.07648, 2016.

[5] D.-A. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). In International Conference on Learning Representa- tions, 2016.





## 参考文章

1. [论文笔记-NASNet](https://zhuanlan.zhihu.com/p/47246311)
2. [NASNet详解](https://zhuanlan.zhihu.com/p/52616166)
3. [NASNet学习笔记](https://blog.csdn.net/xjz18298268521/article/details/79079008)









