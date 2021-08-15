

AI_ops 的学习笔记。

相关资源的 github：

1. https://github.com/linjinjin123/awesome-AIOps
2. https://github.com/visenger/awesome-mlops

------

# 异常检测

参考文章：

1. [异常检测：百度是这样做的](https://mp.weixin.qq.com/s/AXhjawsINKl6cLDV1yf6fw)
2. 



## 简介

自动异常检测的目标是发现复杂业务指标(请求量、收入等)的异常波动，是智能监控系统的重要环节。

异常检测主要面临的问题包括：

1. **数据规模大**。对于百度，搜索、广告、地图等会有上百万量级的指标，而如果是电商平台，在大促阶段，那就是更大的量级；
2. **曲线的特征差异明显**，监控难度大。比如下图中展示的三种曲线，表示三种情况：
   - 第一幅图是有蓝、绿两个曲线，分别代表当前时刻数据和上周同一时刻的数据，蓝色几乎完全覆盖了绿色曲线，**说明数据有规整的周期特性**。
   - 第二幅图，紫色曲线是当前时刻数据，蓝色曲线是上一周的数据，**可以看出来数据有一定的周期性**，但又不如第一幅图那么规整；
   - 第三幅图中的数据大致平整，**但在某些时段出现了异常上涨**。

![](/Users/luocai/Nutstore Files/Study-Notes/MachineLearning/Notes/images/异常检测.png)





## 通用场景的异常检测算法

大部分的曲线可以分为这三种场景：

1. **数据无规律波动，但正常基本在一个较小的波动范围内**，典型的场景就是**拒绝数监控**。通常的做法就是按照拒绝数的常态波动范围设定一个或者多个恒定阈值，超过阈值即报警；
2. **数据的长期波动幅度较大，但正常情况下短期内的波动范围较小**，体现在图像上是一根比较光滑的曲线，不应有突然性的上涨或者下跌。这类场景监控的主要思想就是**环比附近的数据**，检查是否存在突然的大幅上涨或者下跌；
3. **数据有规律地周期性波动**。检测方法就是**与历史数据作同比**，判断是否有异常。



针对上述三种场景，这里有相应的几种检测算法。

### 恒定阈值类算法

恒定阈值类算法，分两种情况，单点恒定阈值和累计恒定阈值。

#### 单点恒定阈值

最简单的就是设置一个阈值，超过该阈值即报警，缺点是实际使用会出现**单点毛刺**的问题，也就是可能出现数据来回抖动，导致产生大量无效报警。

对应的解决方法是**采用 filter 来解决**，比如设置连续 5 个时刻都超过阈值才报警，但这种方法也比较僵硬，中间只要有一个时间点回到阈值范围就不会报警。



#### 累计恒定阈值

比较柔性的方法是累积法：

> 一段时间内数据的均值超过阈值才报警。

这种做法即可以**滤除毛刺**，也考虑了原始数据的**累计效应**。

计算公式如下：
$$
s(t) = \frac{x_t+x_{t-1}+\dots+x_{t-w+1}}{w}
$$


### 突升突降类算法

该算法是要解决场景二的问题，也就是突升突降，也就是如下左图所示的情况。

![](/Users/luocai/Nutstore Files/Study-Notes/MachineLearning/Notes/images/异常检测2.png)

突变的含义是发生了**均值漂移**，所以这里是求取数据最近两个窗口的均值变化比例，计算公式如下：
$$
r(t) = \frac{x_t+x_{t-1}+\cdots+x_{t-w+1}}{x_{t-w}+x_{t-w-1}+\cdots+x_{x-2w+1}}
$$
通过这样计算，将原始数据转换到了**变化比例空间**(r 空间)，也就是上面右图所示，在 r 空间上设置阈值就可以检测出数据的突升或者突降。



### 同比类算法

对于场景三有显著周期性的数据，可以采用同比类算法。先计算历史上相同时间窗口内数据的均值和标准差，然后计算当前点的 **z-score 值**，即当前点的值减去均值再除以标准差。
$$
z-score: z(t) = \frac{x_t-mean(x_{t-kT-w}:x_{x-kT+w})}{std(x_{t-kT-w}:x_{t-kT+w})}
$$


逐点计算 z 值可以把原始数据转换到另外一个空间--z 空间，在 z 空间设置阈值就可以发现这类异常，如下图所示，左图是原始数据，右图是转换到 z 空间的数据，阈值是红线，取值位于红线下方即可报警。



![](/Users/luocai/Nutstore Files/Study-Notes/MachineLearning/Notes/images/异常检测3.png)





## 自动选择算法和配置参数

基本的异常检测算法是上述三种算法，在实际业务中，主要是面临这些问题的：

1. 当有众多曲线的时候，不同曲线要用不同算法，比如周期性明显的适用同比类算法，比较平稳的可以采用恒定阈值算法；
2. 同个算法，在不同时间段的参数也不一样，比如工作日和休假日的参数，白天和晚上的参数都不同，**参数配置成本非常高**；
3. 曲线会随着业务系统的架构调整发生相应的变化，**即算法和参数需要定期维护**。



所以，我们许也可以自动选择算法和配置参数，这里会介绍算法选择决策树和参数自动配置算法。

### 算法选择决策树

曲线配置算法本质上是**建立数据特点和算法本身的映射**。正如上述介绍的三个场景，一般数据特点和算法的映射是这样的：

- **周期性数据**，选择配置**同比算法**；
- 非周期性数据根据波动范围来界定：
  - 数据的全局波动（长期波动）远大于局部波动（短期波动）的时候，倾向于选择**突升突降**；
  - 数据的全局波动近似等于局部波动，即**比较平稳**，更适合选择**恒定阈值算法**；

如下图所示：

![](/Users/luocai/Nutstore Files/Study-Notes/MachineLearning/Notes/images/异常检测 4.png)



所以接下来就是对数据进行这两个方面的判断：

- 周期性的判断
- 数据的全局波动和局部波动情况



#### 周期数据判断方法

这里采用一种**基于差分**的数据周期特征判断方法：

1. **先将临近两天的数据做差分**，如果是周期数据，差分后就可以消除掉原有数据的全局波动；
2. 再结合**方差的阈值**判断，就可以确定数据是否有周期性。
3. 因为不同天的数据会有一定的上下浮动，所以**差分之前先对数据进行归一化**。



#### 数据的全局波动和局部波动范围

如何度量数据的全局波动和局部波动的相对大小呢？

- **数据方差**可以直接表达全局波动范围；
- 对数据采用**小尺度的小波变换**可以得到局部波动，局部波动的方差反应了局部波动的大小



### 参数自动配置算法

当选择好算法好，下一步就是实现自动给算法配置参数。

首先对于**恒定阈值的自动参数配置**。如下左图所示，在红色区域的数据，因为数值罕见，一般就会被认为是异常的数据，通过估算这些罕见数据出现的概率，从而确定曲线的阈值。将数据看作是一组独立同分布的随机变量的数值，可以采用 **ECDF（经验累积概率分布曲线）**来估计随机变量的分布，即如下右图所示的曲线，其横轴是数据值，而纵轴是概率，**表示小于等于某数值的样本比例**。用户给定**经验故障概率**（ECDF 的纵轴），即可找到数值的阈值，也就是 ECDF 的横轴。**这样就通过 ECDF 把配置阈值转换成了配置经验故障概率**。

![](/Users/luocai/Nutstore Files/Study-Notes/MachineLearning/Notes/images/异常检测5.png)

实际使用中，因为历史数据样本有限，ECDF 和真实的 CDF 有一定差距，直接采用会有较多误报，所以可以用**补偿系数**来解决问题。

对于突升突降算法，其实也是一样的，将数据转换到 r 空间，在 r 空间上配置恒定阈值即可，当然也需要设置窗口大小 w，但不同曲线一般不会有太大区别，这里就不需要自动设置了；

而同比类算法，将数据转换到z 空间后，也是在 z 空间上配置恒定阈值，同比天数 k 和窗口大小 w 一般也可以全局设置。







------

## 基于 SLS 的智能巡检算法实战

参考资料：

- [基于SLS的智能巡检算法实战](https://www.zhihu.com/zvideo/1308768801656328192)



### 简介

SLS 是阿里自研的日志数据平台，国内公有云 Top1 的日志分析产品。

SLS为 AIOps 提供的基础能力，整体结构如下图所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/SLS_for_AIOps.png" style="zoom:67%;" />



### 时序异常检测实践

常见的异常形态：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/%E5%B8%B8%E8%A7%81%E5%BC%82%E5%B8%B8%E5%BD%A2%E6%80%81.png)



需要注意的是，故障不等于异常，异常有时候可能只是一个小抖动。

对于检测对象：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/%E6%97%B6%E5%BA%8F%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B_%E6%A3%80%E6%B5%8B%E5%AF%B9%E8%B1%A1.png" style="zoom:67%;" />

检测方法有：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/%E6%97%B6%E5%BA%8F%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B_%E6%A3%80%E6%B5%8B%E6%96%B9%E6%B3%95.png" style="zoom:67%;" />

算法设计说明：

- 针对数据流进行处理，采用流式更新，而非定时更新的做法；
- 每个算法任务背后都是多模型的组合使用，提高检测的准确率；
- 对用户提供的参数尽可能少，让算法自主去学习更新；
- 后台会定期对算法进行升级，保持算法兼容性。



几种算法适应性的说明：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/%E6%97%B6%E5%BA%8F%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B_%E7%AE%97%E6%B3%95%E9%80%82%E5%BA%94%E6%80%A7.png)



### ADFlow 算法

算法设计说明：

1. 纯流式统计算法，指标统计都是增量式计算；
2. 为了兼容准确性和资源消耗，内存中会驻留最近一段时间的数据；
3. 将 KMeans、KDE 等算法改写成增量更新的策略，使用 LOF/KDE/KNN-CAD/Dynamic Jump State 等多模型的综合判断



<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/%E6%97%B6%E5%BA%8F%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B_ADFlow.png" style="zoom:67%;" />



逐渐搭建完善的 AIOps 平台

<img src="images/AIOps%20%E5%B9%B3%E5%8F%B0.png" style="zoom:50%;" />



------

## 异常检测算法

参考：

- [基于数据流的异常检测： Random Cut Forest](https://zhuanlan.zhihu.com/p/88267557)
- [RRCF：基于随机割森林的数据流异常检测模型](https://dreamhomes.top/posts/202104081613.html)



### 1. 无监督树形异常检测算法发展历程

- 2008 - Isolation Forest Published
- 2013 - Survey on outlier detection
- 2016 - RRCF published in JMLR
- 2016 - RRCF available on Amazon Kinesis
- 2018 - RCCF available on Hydroserving



### 2. 2008年Isolate Forest原理

#### 1. 单棵树的构建过程

iTree是一种随机二叉树，每个节点要么有两个孩子，要么就是叶子节点，每个节点包含一个属性q和一个划分值p。

具体的构建过程如下：

1. 从原始训练集合X种无放回的抽取样本子集
2. 划分样本子集，每次划分都从样本中随机选择一个属性q，再从该属性中随机选择一个划分的值p，该p值介于属性q的最大与最小值之间
3. 根据2中的属性q与值p，划分当前样本子集，小于值p的样本作为当前节点的左孩子，大于p值的样本作为当前的右孩子
4. 重复上述2，3步骤，递归构建每个节点的左、右孩子节点，知道满足终止条件为止，通常终止条件为所有节点均只包含一个样本或者多个相同的样本，或者树的深度达到了限定的高度



#### 2. 构建参数的说明

有两个参数控制着模型的复杂度：

- **每棵树的样本子集大小**，它控制着训练数据的大小，论文中的实验发现，当该值增加到一定程度后，IForest的辨识能力会变得可靠，但没有必要继续增加下去，因为这并不会带来准确率的提升，反而会影响计算时间和内存容量，**论文实现发现该参数取256对于很多数据集已经足够**；
- **集成的树的数量**，也可以理解为对原始数据的采样的次数，它控制着模型的集成度，论文中描述取值100就可以了；



#### 3. 如何衡量样本的异常值

计算样本的异常值的过程如下：将测试数据在iTree树上沿着对应的条件分支往下走，直到达到叶子节点，并记录这过程中经过的路径长度h(x)，利用如下公式进行异常分数的计算：
$$
s(x,n)=2^{(-\frac{E(h(x))}{c(n)})}
$$
其中， $c(n)=2*H(n-1)-\frac{2*(n-1)}{n}$ 是二叉搜索树的平均路径长度，用来对结果进行归一化处理，其中的 $H(k)=ln(k)+\epsilon$ 来估计，其中 $\epsilon$ 是欧拉常数，其值大约为0.5772156649。

#### 4. 一些缺点

- Contains all points
- Every leaf contains one distinct point
- Each node separates bounding box of it's points in two halves



### 3. RRCF 算法

#### 0. 为何会引入RRCF算法？

- 数据是持续产生的，数据中的时间戳是一个重要因素，而这个维度却经常被大家忽略
- 数据的结构和形态是未知的，需要设计一个鲁棒性的算法来应对各种复杂的场景需求
- iTree是针对候选数据，进行N次无放回的采样，通过对静态数据集进行划分而得到。若针对流式数据，每次都要针对最新的数据进行采样，再去构造数据集，运行算法，得到相应的结果；
- 在针对流式数据的异常检测场景中，缺少对序列中时序的关系的考虑，算法仅仅把当前的点当做孤立的点进行建模；



#### 1. 针对数据流进行采样建模

针对第一个上述的第一个问题：

- 可以采用一些采样策略（**蓄水池采样**）能准确的当前的数据点是否参与异常建模；
- 同时指定一个时间窗口长度，当建模的数据过期后，应该从模型中剔除掉；



#### 2. 算法中核心的几个操作

给定数据点集 S 随机割树 **RRCT** 的定义如下：

1. 随机选择一个特征维度，概率正比于$\frac{\ell_{i}}{\sum_{j} \ell_{j}}$，其中$\ell_{i}=max⁡_{x∈S}x_i−min_{x∈S}x_i$.
2. 选择样本点 $X_i∼Uniform[min⁡_{x∈S}x_i,max⁡_{x∈S}x_i]$符合均匀分布。
3. 划分集合 $S_1=\{x∣x∈S,x_i≤X_i\}，S_2=S−S_1$。
4. 在 S1 和 S2 中重复以上步骤。

RRCF就是 RRCT 的集合；

维护 RRCF 主要包含两种操作：添加节点和删除节点。

给定历史数据点 S 得到的森林为 RRCF(S)，插入节点即为 RRCF(S) 中的 T 和新的数据点 p 生成新的树 T′，删除节点则反之。

- 构建 RRCF

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/construce_rrcf.png" style="zoom:50%;" />

- 从一个Tree中删除某个样本

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/delete_point.png" style="zoom:50%;" />

- 插入一个新的样本到树结构中

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/insert_point.png" style="zoom:50%;" />

- 计算异常得分

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/compute_score.png" style="zoom:50%;" />





#### 3. 如何衡量样本的异常值

- 引用论文中的一段话

> Let 0 be a left decision and 1 be a right decision. Thus, for x ∈ S, we can write its path as a binary string in the same manner of the isolation forest, m(x). We can repeat this for all x ∈ S and arrive at a complete model description: M(S) = ⊕x∈Sm(x), the appending of all our m(x) together. We will consider the anomaly score of a point x to be the degree to which including it causes our model description to change if we include it or not,

- 形象化的描述如下所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/rrcf_fig2.png" style="zoom:50%;" />



上图左侧表示构造出来的树的结构，其中x是我们待处理的样本点，有图表示将该样本点删除后，动态调整树结构的形态。其中 $q_0,...,q_r$ 表示从树的根节点编码到  a 节点的描述串。

- 每个样本的异常分数的含义：**将点x的异常得分视为包含或不包含该点，而导致模型的描述发生改变的程度**

$$
E_T[|M(T|]-E_T[|M(Delete(x,T|]
$$

论文中通过对上式的变换，得到对应的公式：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/rrcf_fig3.png" style="zoom:50%;" />

- 利用上述公式的描述，可以得到具体的衡量分数，但是如果将上述分数直接转换为异常值，还需要算法同学根据自己的场景进行合理的转换





#### 3.3 流式改造 - 蓄水池采样算法

算法大致描述：给定一个数据流，数据流长度N很大，且N直到处理完所有数据之前都不可知，请问如何在只遍历一遍数据（O(N)）的情况下，能够随机选取出m个不重复的数据。

具体的伪代码描述：

```cpp
int[] reservoir = new int[m];

// init
for (int i = 0; i < reservoir.length; i++)
{
    reservoir[i] = dataStream[i];
}

for (int i = m; i < dataStream.length; i++)
{
    // 随机获得一个[0, i]内的随机整数
    int d = rand.nextInt(i + 1);
    // 如果随机整数落在[0, m-1]范围内，则替换蓄水池中的元素
    if (d < m)
    {
        reservoir[d] = dataStream[i];
    }
}
```

通过对数据流进行采样，可以较好的从数据流中等概率的进行采样，通过RRCF中提供的DELETE方法，可以将置换出模型的数据动态的删除掉，将新选择的样本数据动态的加入到已经有的树中去，进而得到对应的CODISP值。



#### 3.4 并行调用的改造

该算法同Isolation Forest算法一样，非常适合并行构建，在此不做太多的赘述，推荐读者使用Python一个并行的软件包Joblib，能非常方便的帮助用户开发。

传送门：[Joblib: running Python functions as pipeline jobs](https://link.zhihu.com/?target=https%3A//joblib.readthedocs.io/en/latest/index.html)





## 时间序列预测算法

参考：

- [Facebook 时间序列预测算法 Prophet 的研究](https://zhuanlan.zhihu.com/p/52330017)
- [时间序列预测（一）：趋势分解法+ARIMA](https://zhuanlan.zhihu.com/p/50741970)



常用的时间序列预测算法包括了：

1. Prophet
2. ARIMA, ARMA
3. STL
4. LSTM,RNN,Seq2Seq



### 1. Prophet

#### 简介

Prophet是 Facebook 在 2017 年开源的一个时间序列预测算法，其代码、论文和官网地址如下：

- Github：https://github.com/facebook/prophet
- 官网：https://facebook.github.io/prophet/
- 论文：[Forecasting at scale](https://peerj.com/preprints/3190/)

从官网的介绍来看，Facebook 所提供的 prophet 算法不仅可以处理时间序列存在一些异常值的情况，也可以处理部分缺失值的情形，还能够几乎全自动地预测时间序列未来的走势。

从论文上的描述来看，这个 prophet 算法是**基于时间序列分解和机器学习的拟合**来做的，其中在拟合模型的时候使用了 pyStan 这个开源工具，因此能够在较快的时间内得到需要预测的结果。

除此之外，为了方便统计学家，机器学习从业者等人群的使用，prophet 同时提供了 R 语言和 Python 语言的接口。从整体的介绍来看，如果是一般的商业分析或者数据分析的需求，都可以尝试使用这个开源算法来预测未来时间序列的走势。



#### 算法原理

##### 输入和输出

prophet 的输入和输出分别是：

1. 输入：
   - 已知的时间序列的时间戳和对应数值
   - 需要预测的时间序列长度
2. 输出：
   - 预测到的未来时间序列的走势，还可以包括一些必要的统计指标，比如拟合曲线、上下界等

##### 算法实现

在时间序列分析领域，有一种常见的分析方法叫做时间序列的分解（Decomposition of Time Series），它把时间序列 $y_t$ 分成几个部分，分别是季节项 $S_t$ ，趋势项 $T_t$ ，剩余项 $R_t$ 。也就是说对所有的 $t\ge0$ ，都有
$$
y_t = S_t+T_t+R_t
$$
除了加法的形式，还有乘法的形式，也就是：
$$
y_t = S_t\times T_t \times R_t
$$
以上式子等价于 $ln y_t=lnS_t+lnT_t+lnR_t$ 。所以，有的时候在预测模型的时候，会先取对数，然后再进行时间序列的分解，就能得到乘法的形式。在 fbprophet 算法中，作者们基于这种方法进行了必要的改进和优化。

一般来说，在实际生活和生产环节中，除了季节项，趋势项，剩余项之外，通常还有节假日的效应。所以，在 prophet 算法里面，作者同时考虑了以上四项，也就是：
$$
y(t)=g(t)+s(t)+h(t)+\epsilon_t
$$
其中 g(t) 表示趋势项，它表示时间序列在非周期上面的变化趋势； s(t) 表示周期项，或者称为季节项，一般来说是以周或者年为单位； h(t) 表示节假日项，表示在当天是否存在节假日；$\epsilon_t$ 表示误差项或者称为剩余项。Prophet 算法就是通过拟合这几项，然后最后把它们累加起来就得到了时间序列的预测值。





### 2. ARIMA





------

# 参考资料

1. [异常检测：百度是这样做的](https://mp.weixin.qq.com/s/AXhjawsINKl6cLDV1yf6fw)
2. [腾讯运维的AI实践](https://myslide.cn/slides/8935)
3. [基于SLS的智能巡检算法实战](https://www.zhihu.com/zvideo/1308768801656328192)

















