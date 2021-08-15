

# 深度学习中的 normalization

参考：

- [详解深度学习中的Normalization，BN/LN/WN](https://zhuanlan.zhihu.com/p/33173246)
- [Batch normalization和Instance normalization的对比？](https://www.zhihu.com/question/68730628)
- [BatchNormalization、LayerNormalization、InstanceNorm、GroupNorm、SwitchableNorm总结](https://blog.csdn.net/liuxiao214/article/details/81037416)





## 1. 为什么需要 normalization

对于机器学习来说，**独立同分布的数据可以简化常规机器学习模型的训练、提升模型的性能**，这是一个常识，当然并非所有机器学习模型都需要满足这个条件，比如朴素贝叶斯的模型建立在特征彼此独立的基础，而逻辑回归和神经网络在非独立的特征数据上也能有很好的表现。

但无论如何，让数据变成独立同分布还是有好处的，所以在训练模型之前，都会对数据进行一个预处理步骤--**白化(whitening)**，其目的是两个：

1. 独立：去除特征之间的相关性
2. 同分布：使得所有特征具有相同的均值和方差

最常用的方法就是 PCA 了， 可以参考阅读[PCAWhitening](https://link.zhihu.com/?target=http%3A//ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/)



深度神经网络模型的训练为什么会很困难？其中一个重要的原因是，**深度神经网络涉及到很多层的叠加**，而每一层的参数更新会导致上层的输入数据分布发生变化，通过层层叠加，高层的输入分布变化会非常剧烈，**这就使得高层需要不断去重新适应底层的参数更新**。为了训好模型，我们需要非常谨慎地去设定学习率、初始化权重、以及尽可能细致的参数更新策略。



Google 将这一现象总结为 **Internal Covariate Shift**，简称 ICS. 什么是 ICS 呢？

[@魏秀参](https://www.zhihu.com/people/b716bc76c2990cd06dae2f9c1f984e6d) 在[一个回答](https://www.zhihu.com/question/38102762/answer/85238569)中做出了一个很好的解释：

> 大家都知道在统计机器学习中的一个经典假设是“源空间（source domain）和目标空间（target domain）的数据分布（distribution）是一致的”。如果不一致，那么就出现了新的机器学习问题，如 transfer learning / domain adaptation 等。而 covariate shift 就是分布不一致假设之下的一个分支问题，它是指源空间和目标空间的条件概率是一致的，但是其边缘概率不同，即：对所有$x\in X$, $P_s(Y|X=x)=P_t(Y|X=x), P_s(X)\neq P_t(X)$
> 但是大家细想便会发现，的确，对于神经网络的各层输出，由于它们经过了层内操作作用，其分布显然与各层对应的输入信号分布不同，而且差异会随着网络深度增大而增大，可是它们所能“指示”的样本标记（label）仍然是不变的，这便符合了covariate shift的定义。由于是对层间信号的分析，也即是“internal”的来由。

ICS 会导致什么问题呢？

简而言之，每个神经元的输入数据不再是“独立同分布”。

其一，上层参数需要不断适应新的输入数据分布，降低学习速度。

其二，下层输入的变化可能趋向于变大或者变小，导致上层落入饱和区，使得学习过早停止。

其三，每层的更新都会影响到其它层，因此每层的参数更新策略需要尽可能的谨慎。



## 2. **Normalization 的通用框架与基本思想**

这里以一个普通的神经网为例，它接收一组输入向量：
$$
x = (x_1, x_2, \cdots,x_d)
$$
经过运算后，输出一个标量值：
$$
y=f(x)
$$
由于 ICS 问题的存在，x 的分布可能相差很大。要解决独立同分布的问题，“理论正确”的方法就是对每一层的数据都进行白化操作。**然而标准的白化操作代价高昂，特别是我们还希望白化操作是可微的，保证白化操作可以通过反向传播来更新梯度**。

因此，以 BN 为代表的 Normalization 方法退而求其次，进行了简化的白化操作。基本思想是：在将x送给神经元之前，先对其做**平移和伸缩变换**， 将 x 的分布规范化成在固定区间范围的标准分布。

因此，通用变换框架就如下所示：
$$
h=f(g\cdot \frac{x-\mu}{\sigma}+b)
$$
看下上述公式的各个参数：

1. $\mu$ 是平移参数，$\sigma$ 是缩放参数，通过这两个参数进行 shift 和 scala 变换：$\hat x = \frac{x-\mu}{\sigma}$，得到的数据就符合均值为 0，方差为 1 的标准分布；
2. b 是再平移参数，g 是再缩放参数，再次进行变换，公式可以变为如下所示：

$$
y=g\cdot \hat x+b
$$

最终得到的数据符合均为为 b，方差是 $g^2$ 的分布。

所以，这里第一步已经得到了标准分布，为什么需要再做第二步变换呢？

答案是**为了保证模型的表达能力不因为规范化而下降**。

因为第一步的变换将输入数据限制到了一个全局统一的确定范围（均值为 0、方差为 1）。下层神经元可能很努力地在学习，但不论其如何变化，其输出的结果在交给上层神经元进行处理之前，将被粗暴地重新调整到这一固定范围。

所以，为了尊重底层神经网络的学习结果，我们将规范化后的数据进行再平移和再缩放，使得每个神经元对应的输入范围是**针对该神经元量身定制的一个确定范围**（均值为 b 、方差为 $g^2$ ）。rescale 和 reshift 的参数都是可学习的，这就使得 Normalization 层可以学习如何去尊重底层的学习结果。

除了充分利用底层学习的能力，**另一方面的重要意义在于保证获得非线性的表达能力**。Sigmoid 等激活函数在神经网络中有着重要作用，通过区分饱和区和非饱和区，使得神经网络的数据变换具有了非线性计算能力。**而第一步的规范化会将几乎所有数据映射到激活函数的非饱和区（线性区），仅利用到了线性变化能力，从而降低了神经网络的表达能力。而进行再变换，则可以将数据从线性区变换到非线性区，恢复模型的表达能力**。

那么问题又来了——

**经过这么的变回来再变过去，会不会跟没变一样？**

不会。因为，再变换引入的两个新参数 g 和 b，可以表示旧参数作为输入的同一族函数，但是新参数有不同的学习动态。在旧参数中， x 的均值取决于下层神经网络的复杂关联；但在新参数中，  $y=g\cdot \hat x+b$ 仅由 b 来确定，去除了与下层计算的密切耦合。新参数很容易通过梯度下降来学习，简化了神经网络的训练。

那么还有一个问题——

**这样的 Normalization 离标准的白化还有多远？**

标准白化操作的目的是“独立同分布”。独立就不说了，暂不考虑。变换为均值为 b 、方差为 $g^2$ 的分布，也并不是严格的同分布，只是映射到了一个确定的区间范围而已。（所以，这个坑还有得研究呢！）



## 3. **主流 Normalization 方法梳理**

这里梳理主流的几种规范化方法。

将输入的图像shape记为[N, C, H, W]，这几个方法主要的区别如下，如下图所示：

- batchNorm 是在batch上，对NHW做归一化，对小batchsize效果不好；
- layerNorm 在通道方向上，对CHW归一化，主要对RNN作用明显；
- instanceNorm在图像像素上，对HW做归一化，用在风格化迁移；
- GroupNorm将channel分组，然后再做归一化；
- SwitchableNorm是将BN、LN、IN结合，赋予权重，让网络自己去学习归一化层应该使用什么方法。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/BN_LN_GN_SN.png)



### 3.1 Batch Normalization——纵向规范化

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/bn.png)

论文：https://arxiv.org/pdf/1502.03167.pdf

#### BN 原理

Batch Normalization 于2015年由 Google 提出，开 Normalization 之先河。其规范化针对单个神经元进行，利用网络训练时一个 mini-batch 的数据来计算该神经元 $x_i$ 的均值和方差,因而称为 Batch Normalization。
$$
\mu_i=\frac{1}{M}\sum x_i\\
\sigma_i=\sqrt {\frac{1}{M}\sum(x_i-\mu_i)^2+\epsilon}
$$
其中 M 是 mini-batch 的大小。

按上图所示，相对于一层神经元的水平排列，BN 可以看做一种纵向的规范化。由于 BN 是针对单个维度定义的，因此标准公式中的计算均为 element-wise 的。

算法的过程：

- 沿着通道计算每个batch的均值 $\mu$
- 沿着通道计算每个batch的方差 $σ^2$
- 对x做归一化，$\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$
- 加入缩放和平移变量 γ 和 β ,归一化后的值，$y=γ\hat x+β$

加入缩放平移变量的原因是：**保证每一次数据经过归一化后还保留原有学习来的特征，同时又能完成归一化操作，加速训练**。 这两个参数是用来学习的参数。

因为对某一层的输入数据做归一化，然后送入网络的下一层，这样是会影响到本层网络所学习的特征的，比如网络中学习到的数据本来大部分分布在0的右边，经过RELU激活函数以后大部分会被激活，如果直接强制归一化，那么就会有大多数的数据无法激活了，这样学习到的特征不就被破坏掉了么？论文中对上面的方法做了一些改进：**变换重构**，引入了可以学习的参数 $\gamma$ 和 $\beta$，这就是算法的关键之处（这两个希腊字母就是要学习的）。

每个batch的每个通道都有这样的一对参数：（看完后面应该就可以理解这句话了）
$$
\gamma = \sqrt{\sigma_B^2} \quad, \quad  \beta = \mu_B
$$
这样的时候可以恢复出原始的某一层学习到的特征的，因此我们引入这个可以学习的参数使得我们的网络可以恢复出原始网络所要学习的特征分布。

**我们在一些源码中，可以看到带有BN的卷积层，bias设置为False，就是因为即便卷积之后加上了Bias，在BN中也是要减去的，所以加Bias带来的非线性就被BN一定程度上抵消了。**



#### BN中的均值与方差通过哪些维度计算得到

神经网络中传递的张量数据，其维度通常记为[N, H, W, C]，其中N是 batch_size，H、W是行、列，C是通道数。那么上式中BN的输入集合  $B$  就是下图中蓝色的部分。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/bn_compute.jpg" alt="img" style="zoom:33%;" />

**均值的计算，就是在一个批次内，将每个通道中的数字单独加起来，再除以 $N*H*W$** 。举个栗子：该批次内有十张图片，每张图片有三个通道RGB，每张图片的高宽是 $H、W$　那么R通道的均值就是计算这十张图片R通道的像素数值总和再除以 $10*H*W$ ，其他通道类似，方差的计算也类似。

**可训练参数$\gamma$ 和 $\beta$ 的维度等于张量的通道数**，在上述栗子中，RGB三个通道分别需要一个$\gamma$ 和 $\beta$，所以他们的维度为３。



#### 训练与推理时BN中的均值和方差分别是多少

正确的答案是：

**训练时**：均值、方差分别是**该批次内数据相应维度的均值与方差**。

**推理时**：均值来说直接计算所有训练时batch的 $\mu_B$ 的平均值，而方差采用训练时每个batch的 $\sigma_B^2$ 的无偏估计，公式如下：
$$
E[x] \leftarrow E_B[\mu_B] \\
Var[x] \leftarrow \frac{m}{m-1}E_B[\sigma_B^2]
$$
但在实际实现中，如果训练几百万个Batch，那么是不是要将其均值方差全部储存，最后推理时再计算他们的均值作为推理时的均值和方差？这样显然太过笨拙，占用内存随着训练次数不断上升。为了避免该问题，后面代码实现部分使用了**滑动平均**，储存固定个数Batch的均值和方差，不断迭代更新推理时需要的 $E[x]$ 和 $Var[x]$  。

为了证明准确性，贴上原论文中的公式：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/3hysDxtZJmdkzac.jpg)



如上图第11行所示：最后测试阶段，BN采用的公式是：
$$
y = \frac{\gamma}{\sqrt{Var[x] + \varepsilon}}*x+(\beta - \frac{\gamma E[x]}{\sqrt{Var[x]+\varepsilon}})
$$
测试阶段的 $\gamma$ 和 $\beta$ 是在网络训练阶段已经学习好了的，直接加载进来计算即可。

#### BN的好处

1. **防止网络梯度消失**：这个要结合sigmoid函数进行理解
2. **加速训练，也允许更大的学习率**：输出分布向着激活函数的上下限偏移，带来的问题就是梯度的降低，（比如说激活函数是sigmoid），通过normalization，数据在一个合适的分布空间，经过激活函数，仍然得到不错的梯度。梯度好了自然加速训练。
3. **降低参数初始化敏感**：以往模型需要设置一个不错的初始化才适合训练，加了BN就不用管这些了，现在初始化方法中随便选择一个用，训练得到的模型就能收敛。
4. **提高网络泛化能力防止过拟合**：所以有了BN层，可以不再使用L2正则化和dropout。可以理解为在训练中，BN的使用使得一个mini-batch中的所有样本都被关联在了一起，因此网络不会从某一个训练样本中生成确定的结果。
5. **可以把训练数据彻底打乱**（防止每批训练的时候，某一个样本都经常被挑选到，文献说这个可以提高1%的精度）。
6. **可以不用考虑过拟合中 dropout、L2 正则参数的选择问题**。采用BN算法后，你可以移除这两项了参数，或者可以选择更小的L2正则约束参数了，因为BN具有提高网络泛化能力的特性。
7. **再也不需要使用使用局部响应归一化层**了（局部响应归一化是Alexnet网络用到的方法，搞视觉的估计比较熟悉），因为BN本身就是一个归一化网络层；



#### 代码实现

```python
import numpy as np

def Batchnorm(x, gamma, beta, bn_param):

    # x_shape:[B, C, H, W]
    running_mean = bn_param['running_mean']
    running_var = bn_param['running_var']
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(0, 2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta

    # 因为在测试时是单个图片测试，这里保留训练时的均值和方差，用在后面测试时用
    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return results, bn_param
```

基于 PyTorch 实现 BN 层

```python
def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是推理模式
    if not is_training:
        # 如果是在推理模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持X的形状以便后面可以做广播运算
            # torch.Tensor 高维矩阵的表示： （nSample）x C x H x W，所以对C维度外的维度求均值
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移（变换重构）
    return Y, moving_mean, moving_var

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):　# num_features就是通道数
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training,
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```



#### 特点和用途

BN 独立地规范化每一个输入维度 $x_i$ ，但规范化的参数是一个 mini-batch 的一阶统计量和二阶统计量。这就要求 每一个 mini-batch 的统计量是整体统计量的近似估计，或者说每一个 mini-batch 彼此之间，以及和整体数据，都应该是近似同分布的。**分布差距较小的 mini-batch 可以看做是为规范化操作和模型训练引入了噪声，可以增加模型的鲁棒性**；

但如果**每个 mini-batch的原始分布差别很大，那么不同 mini-batch 的数据将会进行不一样的数据变换，这就增加了模型训练的难度**。

因此，BN 比较适用的场景是：**每个 mini-batch 比较大，数据分布比较接近。在进行训练之前，要做好充分的shuffle. 否则效果会差很多**。

其缺点有：

- 对 batch size 大小比较敏感，因为是对每个 batch 的样本计算均值和方差，如果 batch 太小，计算得到的均值和方差不足以代表整个数据分布；
- 另外，由于 BN 需要在运行过程中统计每个 mini-batch 的一阶统计量和二阶统计量，**因此不适用于动态的网络结构 和 RNN 网络**。主要是 RNN 处理的输入序列长度不是一致的，即深度是不一致的，另外每个 time step 都需要保存均值和方差，影响训练和计算；



#### 防止过拟合的原因

参考：

- [batch normalization为什么可以防止过拟合?](https://www.zhihu.com/question/275788133/answer/386749776)
- [Batch Normalization原理与实战](https://zhuanlan.zhihu.com/p/34879333)

BN的核心思想并不是防止梯度消失或者过拟合，**而是通过对系统参数搜索空间进行约束来增加系统鲁棒性**，这种约束压缩了搜索空间，也改善了系统的结构合理性，这可以加速收敛、保证梯度，缓解过拟合等；

在Batch Normalization中，由于我们使用mini-batch的均值与方差作为对整体训练样本均值与方差的估计，尽管每一个batch中的数据都是从总体样本中抽样得到，但**不同mini-batch的均值与方差会有所不同，这就为网络的学习过程中增加了随机噪音**，与Dropout通过关闭神经元给网络训练带来噪音类似，在一定程度上对模型起到了正则化的效果。







### 3.2 Layer Normalization——横向规范化

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/ln.png)

论文：https://arxiv.org/pdf/1607.06450v1.pdf

层规范化就是针对 BN 的上述不足而提出的。与 BN 不同，LN 是一种横向的规范化，如图所示。**它综合考虑一层所有维度的输入，计算该层的平均输入值和输入方差，然后用同一个规范化操作来转换各个维度的输入**。
$$
\mu=\sum_i x_i\\
\sigma=\sqrt {\sum_i(x_i-\mu_i)^2+\epsilon}
$$
其中 i 枚举了该层所有的输入神经元。对应到标准公式中，四大参数 $\mu,\sigma,g,b$ 均为标量（BN中是向量），所有输入共享一个规范化变换。

**LN 针对单个训练样本进行，不依赖于其他数据**，因此可以避免 BN 中受 mini-batch 数据分布影响的问题，可以用于 小mini-batch场景、动态网络场景和 RNN，特别是自然语言处理领域。此外，LN 不需要保存 mini-batch 的均值和方差，**节省了额外的存储空间**。

但是，BN 的转换是针对单个神经元可训练的——不同神经元的输入经过再平移和再缩放后分布在不同的区间，而 LN 对于一整层的神经元训练得到同一个转换——所有的输入都在同一个区间范围内。**如果不同输入特征不属于相似的类别（比如颜色和大小），那么 LN 的处理可能会降低模型的表达能力**。

BN 和 LN 的区别：

- LN中同层神经元输入拥有相同的均值和方差，不同的输入样本有不同的均值和方差；
- BN中则针对不同神经元输入计算均值和方差，同一个batch中的输入拥有相同的均值和方差。

因此，LN 可以用于 batch 为 1 和 RNN 中，在 RNN 的效果更明显，在 CNN 上则不如 BN。

实现如下：

```python
def ln(x, b, s):
    _eps = 1e-5
    output = (x - x.mean(1)[:,None]) / tensor.sqrt((x.var(1)[:,None] + _eps))
    output = s[None, :] * output + b[None,:]
    return output
```

用在四维图像上：

```python
def Layernorm(x, gamma, beta):

    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(1, 2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```



### 3.3 Weight Normalization——参数规范化

BN 和 LN 均将规范化应用于输入的特征数据 x ，而 WN 则另辟蹊径，将规范化应用于线性变换函数的权重 w ，这就是 WN 名称的来源。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/wn.png)

论文：

具体而言，WN 提出的方案是，将权重向量 w 分解为向量方向$\hat v$ 和向量模 $\hat g$ 两部分：
$$
w= g\cdot \hat v = g \cdot \frac{v}{||v||}
$$
其中 v 是和 w 同纬度的向量，||v|| 是欧式范数，因此 $\hat v$ 是单位向量，决定了 w 的方向，g 是标量，决定了 w 的长度。由于 ||w|| = |g| ，因此这一权重分解的方式将权重向量的欧式范数进行了固定，从而实现了正则化的效果。

对比之前讲的通用框架公式，这里进行一个推导：
$$
f_w(WN(x)) = w\cdot WN(x)=g\cdot \frac{v}{||v||}\cdot x\\
=v\cdot g \cdot \frac{x}{||v||} = f_v(g\cdot \frac{x}{||v||})
$$
对比之前的通用框架公式：
$$
h=f(g\cdot \frac{x-\mu}{\sigma}+b)
$$
只需要令：
$$
\sigma=||v||,\mu=0,b=0
$$
就可以完美对号入座。



回忆一下，BN 和 LN 是用输入的特征数据的方差对输入数据进行 scale，而 WN 则是用 神经元的权重的欧氏范式对输入数据进行 scale。**虽然在原始方法中分别进行的是特征数据规范化和参数的规范化，但本质上都实现了对数据的规范化，只是用于 scale 的参数来源不同。**

另外，我们看到这里的规范化只是对数据进行了 scale，而没有进行 shift，因为我们简单地令 $\mu=0$. 但事实上，这里留下了与 BN 或者 LN 相结合的余地——那就是利用 BN 或者 LN 的方法来计算输入数据的均值 $\mu$ 。

WN 的规范化不直接使用输入数据的统计量，因此避免了 BN 过于依赖 mini-batch 的不足，以及 LN 每层唯一转换器的限制，同时也可以用于动态网络结构。





### 3.4 **Cosine Normalization—— 余弦规范化**

论文：

有学者认为，之所以要对数据进行规范化的原因是，数据经过神经网络的计算之后可能会变得很大，导致数据分布的方差爆炸，这一个问题的根源是计算方式——点积，权重向量 w 和特征数据向量 x 的点积，而向量点积是无界的。

向量点积是衡量两个向量相似度的方法之一。哪还有没有其他的相似度衡量方法呢？有啊，很多啊！夹角余弦就是其中之一啊！而且关键的是，夹角余弦是有确定界的啊，[-1, 1] 的取值范围。

所以也就有了 Cosine Normalization：
$$
f_w(x) = cos\theta = \frac{w\cdot x}{||w|| \cdot ||x||}
$$
其中$\theta$ 是 w 和 x 的夹角，通过这个计算，所以的数据都限制在 [-1, 1] 的范围内了。

对比 WN 的操作：
$$
f_w(WN(x)) = f_v(g\cdot \frac{x}{||v||})
$$
WN是通过权重向量 w 的模||v|| 对输入向量进行 scale，而 CN 则在此基础加上输入向量的模 ||x|| 对输入向量进行了进一步的 scale。

CN 通过用余弦计算代替内积计算实现了规范化，但成也萧何败萧何。**原始的内积计算，其几何意义是 输入向量在权重向量上的投影，既包含 二者的夹角信息，也包含 两个向量的scale信息**。

**去掉scale信息，可能导致表达能力的下降，因此也引起了一些争议和讨论**。具体效果如何，可能需要在特定的场景下深入实验。



### 3.5 Group Normalization——组规范化

论文：https://arxiv.org/pdf/1803.08494.pdf

代码：https://github.com/facebookresearch/Detectron/blob/master/projects/GN

主要是针对Batch Normalization对小batchsize效果差，GN将channel方向分group，然后每个group内做归一化，算 $(C//G)*H*W$ 的均值，这样与batchsize无关，不受其约束。





代码实现：

```python
def GroupNorm(x, gamma, beta, G=16):

    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5
    x = np.reshape(x, (x.shape[0], G, x.shape[1]/16, x.shape[2], x.shape[3]))

    x_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    x_var = np.var(x, axis=(2, 3, 4), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```





### 3.6 Instance Normalization——实例规范化

论文：https://arxiv.org/pdf/1607.08022.pdf

代码：https://github.com/DmitryUlyanov/texture_nets

BN注重对每个batch进行归一化，保证数据分布一致，因为判别模型中结果取决于数据整体分布。

但是图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/In_formula.png)

代码实现：

```python
def Instancenorm(x, gamma, beta):

    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(2, 3), keepdims=True)
    x_var = np.var(x, axis=(2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```





### 3.7 Switchable Normalization——自适配规范化

论文：https://arxiv.org/pdf/1806.10779.pdf

代码：https://github.com/switchablenorms/Switchable-Normalization

本篇论文作者认为，

- 第一，归一化虽然提高模型泛化能力，然而归一化层的操作是人工设计的。在实际应用中，解决不同的问题原则上需要设计不同的归一化操作，并没有一个通用的归一化方法能够解决所有应用问题；
- 第二，一个深度神经网络往往包含几十个归一化层，通常这些归一化层都使用同样的归一化操作，因为手工为每一个归一化层设计操作需要进行大量的实验。


因此作者提出自适配归一化方法——Switchable Normalization（SN）来解决上述问题。与强化学习不同，SN使用可微分学习，为一个深度网络中的每一个归一化层确定合适的归一化操作。





代码实现：

```python
def SwitchableNorm(x, gamma, beta, w_mean, w_var):
    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5

    mean_in = np.mean(x, axis=(2, 3), keepdims=True)
    var_in = np.var(x, axis=(2, 3), keepdims=True)

    mean_ln = np.mean(x, axis=(1, 2, 3), keepdims=True)
    var_ln = np.var(x, axis=(1, 2, 3), keepdims=True)

    mean_bn = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var_bn = np.var(x, axis=(0, 2, 3), keepdims=True)

    mean = w_mean[0] * mean_in + w_mean[1] * mean_ln + w_mean[2] * mean_bn
    var = w_var[0] * var_in + w_var[1] * var_ln + w_var[2] * var_bn

    x_normalized = (x - mean) / np.sqrt(var + eps)
    results = gamma * x_normalized + beta
    return results
```










## 4. **Normalization 为什么会有效？**

下面以这个简化的神经网络为例进行分析：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/bn_example.png)

### 4.1 Normalization 的权重伸缩不变性

**权重伸缩不变性（weight scale invariance）**指的是，当权重 W 按照常量 $\lambda$ 进行伸缩时，得到的规范化后的值保持不变，即：
$$
Norm(\hat W x) = Norm(Wx)
$$
其中 $\hat W = \lambda W$

**上述规范化方法均有这一性质**，这是因为，当权重 W 伸缩时，对应的均值和标准差均等比例伸缩，分子分母相抵。
$$
Norm(\hat Wx)=Norm(g\cdot \frac{\hat Wx-\hat \mu}{\hat \sigma}+b)\\
=Norm(g\cdot \frac{\lambda Wx-\lambda \mu}{\lambda \sigma}+b)\\
=Norm(g\cdot \frac{Wx-\mu}{\sigma}+b)\\
=Norm(Wx)
$$
**权重伸缩不变性可以有效地提高反向传播的效率**。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/normalization_equation.png)

因此，**权重的伸缩变化不会影响反向梯度的 Jacobian 矩阵**，因此也就对反向传播没有影响，避免了反向传播时因为权重过大或过小导致的梯度消失或梯度爆炸问题，从而加速了神经网络的训练。

**权重伸缩不变性还具有参数正则化的效果，可以使用更高的学习率。**

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/normalization_equation2.png)

因此，下层的权重值越大，其梯度就越小。这样，参数的变化就越稳定，相当于实现了参数正则化的效果，避免参数的大幅震荡，提高网络的泛化性能。

### 4.2 Normalization的数据伸缩不变性

**数据伸缩不变性（data scale invariance）**指的是，当数据 x 按照常量 $\lambda$ 进行伸缩时，得到的规范化后的值保持不变，即：
$$
Norm(W\hat x) = Norm(Wx)
$$
其中 $\hat x= \lambda x$ 。

**数据伸缩不变性仅对 BN、LN 和 CN 成立。**因为这三者对输入数据进行规范化，因此当数据进行常量伸缩时，其均值和方差都会相应变化，分子分母互相抵消。而 WN 不具有这一性质。

**数据伸缩不变性可以有效地减少梯度弥散，简化对学习率的选择**。

对于某一层神经元 $h_l=f_{W_l}(x_l)$ 而言，展开可得

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/normalization_equation3.png)

每一层神经元的输出依赖于底下各层的计算结果。如果没有正则化，当下层输入发生伸缩变化时，经过层层传递，可能会导致数据发生剧烈的膨胀或者弥散，从而也导致了反向计算时的梯度爆炸或梯度弥散。

加入 Normalization 之后，不论底层的数据如何变化，对于某一层神经元 $h_l=f_{W_l}(x_l)$ 而言，其输入 $x_l$ 永远保持标准的分布，这就使得高层的训练更加简单。从梯度的计算公式来看：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/normalization_equation4.png)

数据的伸缩变化也不会影响到对该层的权重参数更新，使得训练过程更加鲁棒，简化了对学习率的选择。



## 5. 相关问题

### 1. Batch normalization和Instance normalization的对比？

#### 适用场景

来自 https://www.zhihu.com/question/68730628/answer/277339783 的回答：

> BN和IN其实本质上是同一个东西，只是IN是作用于单张图片，但是BN作用于一个batch。但是为什么IN还会被单独提出，而且在Style Transfer的这个任务中大放异彩呢？简言之，这背后的逻辑链是这样的：
>
> 1. 通过调整BN统计量，或学习的参数beta和gamma，BN可以用来做domain adaptation。
> 2. Style Transfer是一个把每张图片当成一个domain的domain adaptation问题

来自 https://www.zhihu.com/question/68730628/answer/607608890 的回答：

> BN 适用于判别模型中，比如图片分类模型。因为BN注重对每个batch进行归一化，从而保证数据分布的一致性，而判别模型的结果正是取决于数据整体分布。但是BN对batchsize的大小比较敏感，由于每次计算均值和方差是在一个batch上，所以如果batchsize太小，则计算的均值、方差不足以代表整个数据分布；
>
> IN 适用于生成模型中，比如图片风格迁移。因为图片生成的结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，在风格迁移中使用Instance Normalization不仅可以加速模型收敛，并且可以保持每个图像实例之间的独立。

BN 是对每个样本的同一个通道进行 Normalization 操作，而 IN 是对单个样本的单个通道单独进行 Normalization 操作，示例如下所示，图片来自论文 https://arxiv.org/pdf/1803.08494.pdf

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/gn_fig.png)

#### 计算公式

BN 的计算公式：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/bn_formula.png" style="zoom:80%;" />

IN 的计算公式：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/In_formula.png)

其中 t 代表图片的 index，i 代表的是 feature map 的 index

#### 算法过程

##### BN

- 沿着通道计算每个batch的均值 $\mu$
- 沿着通道计算每个batch的方差 $σ^2$
- 对x做归一化，$\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$
- 加入缩放和平移变量 γ 和 β ,归一化后的值，$y=γ\hat x+β$

##### IN

- 沿着通道计算每张图的均值 $\mu$
- 沿着通道计算每张图的方差 $σ^2$
- 对 x 做归一化，$\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$
- 加入缩放和平移变量γ和β ,归一化后的值，$y=γ\hat x+β$



### 2. Weight Normalization 相比batch Normalization 有什么优点呢？

参考：[Weight Normalization 相比batch Normalization 有什么优点呢？](https://www.zhihu.com/question/55132852/answer/171250929)

WN 相比 BN 的优势如下：

1. WN 是通过改写网络的权重参数w，没有引入 batch 的依赖，所以可以适用于 RNN（LSTM）网络，RNN 网络不能用 BN 的原因有这几个：
   - RNN 处理的序列是变长的；
   - RNN 是基于 time step 计算的，如果用 BN 处理，需要保存每个 time step 下 mini batch 的均值和方差，效率低且占内存；
2. BN 是基于一个 batch 的数据计算均值和方差，**相当于进行梯度计算引入噪声**，所以不适合对噪声敏感的强化学习、生成模型（如 GAN,VAE)。而 WN 是通过标量 g 和向量 v 对权重 W 进行重写，重写向量 v 是固定的，因此 WN 的操作比 BN 引入的噪声会更少；
3. 不用额外的空间保存 batch 的均值和方差，另外 WN 的计算开销会更小；

当然，WN 需要注意的是参数初始值的选择，它不具备 BN 的将网络每一层的输出 y 固定在一个变化范围的作用。

