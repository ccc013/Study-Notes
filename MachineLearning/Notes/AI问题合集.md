# AI 问题合集

收集 AI 相关的一些问题，进行查漏补缺



------

## 机器学习基础

### 1. 常用的损失函数，原理以及特点



### 2. L0、L1、L2

参考：

- l1正则与l2正则的特点是什么，各有什么优势？ - Andy Yang的回答 - 知乎 https://www.zhihu.com/question/26485586/answer/616029832
- [机器学习中的范数规则化之（一）L0、L1与L2范数](https://blog.csdn.net/zouxy09/article/details/24971995)
- [L1,L2,L0区别，为什么可以防止过拟合](https://www.jianshu.com/p/475d2c3197d2)



#### 原理和公式

##### L0范数

L0是指向量中非0的元素的个数。如果我们用L0范数来规则化一个参数矩阵W的话，就是希望W的大部分元素都是0。换句话说，让参数W是稀疏的。

但不幸的是，L0范数的最优化问题是一个NP hard问题，而且理论上有证明，L1范数是L0范数的最优凸近似，因此通常使用L1范数来代替。

公式如下：
$$
||x||_0=\#(i)with\ x_i \neq 0
$$


##### L1范数

L1范数是指向量中各个元素绝对值之和，也有个美称叫“稀疏规则算子”（Lasso regularization）。

是L0的最优凸近似（L1是L0的最紧的凸放松），**且具有特征自动选择和使得模型更具解释性的优点**。公式如下：
$$
||x||_1=\sum^n_{i=1}|x_i|
$$
L1正则化之所以可以防止过拟合，是因为L1范数就是各个参数的绝对值相加得到的，我们前面讨论了，参数值大小和模型复杂度是成正比的。因此复杂的模型，其L1范数就大，最终导致损失函数就大，说明这个模型就不够好。



##### L2范数

也叫“岭回归”（Ridge Regression），也叫它“权值衰减weight decay”，公式如下，也是欧式距离
$$
||x||_2=\sqrt{\sum_{i=1}^nx_i^2}
$$


但与L1范数不一样的是，它不会是每个元素为0，而只是接近于0。越小的参数说明模型越简单，越简单的模型越不容易产生过拟合现象。

使得权重接近于0但是不等于0，有利于处理条件数不好情况下矩阵求逆问题（条件数用来衡量病态问题的可信度，也就是当输入发生微小变化的时候，输出会发生多大变化，即系统对微小变动的敏感度，条件数小的就是well-conditioned的，大的就是ill-conditioned的），对于线性回归来说，如果加上L2规则项，原有对XTX（转置）求逆就变为可能，而目标函数收敛速率的上界实际上是和矩阵XTX的条件数有关，XTX的 condition number 越小，上界就越小，也就是收敛速度会越快；另外从优化的角度来看，加入规则项实际上是将目标函数变成λ强凸，这样可以保证函数在任意一点都存在一个非常漂亮的二次函数下界，从而能通过梯度更快找到近似解。总结就是：**L2范数不但可以防止过拟合，还可以让我们的优化求解变得稳定和快速**。



#### 作用

首先范数规则化有两个作用：

1）保证模型尽可能的简单，避免过拟合。

2）约束模型特性，加入一些先验知识，例如稀疏、低秩等。



1. L1 作为损失函数，具有鲁棒性更强，对异常值更不敏感的优点；

2. L2 计算更加方便，可以直接求导获得取最小值时，各个参数的取值；

   

#### 各自的问题

1. L1 在 0 处不可导，解决办法是可以使用Proximal Algorithms或者ADMM来解决；



#### 区别

1. L1会产生稀疏的特征，而 L2 会产生更多的特征，但是都会接近于 0；因此，L1 更适合在特征选择的时候使用，而 L2 只是一种正则化方法；
2. L1对应**拉普拉斯分布，是不完全可微的**，表现在图像上就是有很多角出现，这些角和目标函数的接触机会远大于其他部分。就会造成最优值出现在坐标轴上，因此就会导致某一维的权重为0 ，产生稀疏权重矩阵，进而防止过拟合。
3. L2对应**高斯分布，是完全可微的**。和L1相比，图像上的棱角被圆滑了很多。一般最优值不会在坐标轴上出现。在最小化正则项时，可以是参数不断趋向于0.最后活的很小的参数。
4. L2 计算更方便，而 L1 在非稀疏向量上的计算效率很低；
5. L2 对大数、outlier 更敏感



为什么 L1 会产生稀疏的特征呢，可以根据下面 L1 和 L2 的函数图和各自的导数图：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/L1_L2_fig.png" style="zoom:50%;" />

L1 和 L2 的导数如下公式所示：
$$
L_1=|w_1|+|w_2|+\ldots+|w_n|,\frac{\partial L_1}{\partial w_i}=sign(w_i)=1\ or \ -1\\
L_2=\frac{1}{2}w_1^2+w_2^2+\ldots+w_n^2, \frac{\partial L_2}{\partial w_i}=w_i
$$


所以(不失一般性，我们假定：wi等于不为0的某个正的浮点数，学习速率η 为0.5)：

L1的权值更新公式为 $w_i= w_i- η * 1  = w_i- 0.5 * 1$，也就是说权值每次更新都固定减少一个特定的值(比如0.5)，再根据上图所示，经过若干次迭代之后，权值就有可能减少到0。

L2的权值更新公式为$w_i= w_i- η * w_i= w_i- 0.5 * w_i$，也就是说权值每次都等于上一次的1/2，那么，虽然权值不断变小，但是因为每次都等于上一次的一半，所以很快会收敛到较小的值但不为0。

所以可以解释为什么 L1 稀疏，而 L2 平滑：

- L1 能产生等于0的权值，即能够剔除某些特征在模型中的作用（特征选择），**即产生稀疏的效果**。

- L2 可以得迅速得到比较小的权值，但是难以收敛到0，**所以产生的不是稀疏而是平滑的效果**。



因此，L1 经过一定的步数更新后是很可能变为 0 的，而 L2 则几乎不可能；





#### 应用场景

主要的应用两个：

- 作为损失函数使用；
- 作为正则化项



#### L1 为什么不用于 CNN



#### 为什么不用 L0

1. L0 范数很难优化求解（NP 难问题）；
2. L1 是 L0 的最优凸近似，而且比 L0 更容易优化求解；



#### 实现参数的稀疏有什么好处吗？

一个好处是可以简化模型，避免过拟合。因为一个模型中真正重要的参数可能并不多，如果考虑所有的参数起作用，那么可以对训练数据可以预测的很好，但是对测试数据的预测效果就很糟糕了。

另一个好处是参数变少可以使整个模型获得更好的可解释性。



#### 参数值越小代表模型越简单吗？

是的。

为什么参数越小，说明模型越简单呢，这是因为越复杂的模型，**越是会尝试对所有的样本进行拟合**，甚至包括一些异常样本点，**这就容易造成在较小的区间里预测值产生较大的波动**，这种较大的波动也反映了在这个区间里的导数很大，而只有较大的参数值才能产生较大的导数。因此复杂的模型，其参数值会比较大。





------

### 3. 手推交叉熵公式



### 4. 手推Softmax公式



### 5. 数据不平衡问题



### 6. 关于归一化

#### 6.1 为什么机器学习中要进行特征归一化？为什么要对输入图像做归一化呢？

参考：

- [09_为什么输入网络前要对图像做归一化](https://github.com/GYee/CV_interviews_Q-A/blob/master/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/09_%E4%B8%BA%E4%BB%80%E4%B9%88%E8%BE%93%E5%85%A5%E7%BD%91%E7%BB%9C%E5%89%8D%E8%A6%81%E5%AF%B9%E5%9B%BE%E5%83%8F%E5%81%9A%E5%BD%92%E4%B8%80%E5%8C%96.md)
- [为什么要对数据进行归一化处理？ ](https://zhuanlan.zhihu.com/p/27627299)
- [深度学习中图像为什么要归一化？](https://www.zhihu.com/question/293640354)
- [深度学习中，为什么需要对数据进行归一化](https://blog.csdn.net/qq_32172681/article/details/100876348)
- [深度学习的输入数据集为什么要做均值化处理](https://blog.csdn.net/hai008007/article/details/79718251)

对于特征归一化的原因，主要是这几个原因：

1. **消除特征之间量纲的影响**，使得不同特征之间具有可比性
2. 在使用随机梯度下降求解的模型中，**能加快模型收敛速度**
3. **归一化还有可能提高精度**：一些分类器需要计算样本之间的距离（如欧氏距离），例如KNN。如果一个特征值域范围非常大，那么距离计算就主要取决于这个特征，从而与实际情况相悖（比如这时实际情况是值域范围小的特征更重要）。
4. **避免神经元饱和**。当神经元的激活接近 1 或者 0 的时候，就是饱和，在这种取值下，其梯度几乎是 0，那么反向传播的时候局部梯度就会接近 0，从而”杀死“梯度；
5. 保证输出数据中数值小的不被吞食



对于输入图像做归一化，原因可能是这几种：

1. 灰度数据表示有两种方法：一种是uint8类型、另一种是double类型。其中uint8类型数据的取值范围为 [0,255]，而double类型数据的取值范围为[0,1]，两者正好相差255倍。对于double类型数据，其取值大于1时，就会表示为白色，不能显示图像的信息，**故当运算数据类型为double时，为了显示图像要除255**。

2. 图像深度学习网络也是使用gradient descent来训练模型的，使用gradient descent的都要在数据预处理步骤进行数据归一化，主要原因是，根据反向传播公式：
   $$
   \frac{\partial J}{\omega_{11}} = x_1*后面层梯度的乘积
   $$
   如果输入层 x  很大，在反向传播时候传递到输入层的梯度就会变得很大。梯度大，学习率就得非常小，否则会越过最优。在这种情况下，**学习率的选择需要参考输入层数值大小**，而直接将数据归一化操作，能很方便的选择学习率。**在未归一化时，输入的分布差异大，所以各个参数的梯度数量级不相同，因此，它们需要的学习率数量级也就不相同**。对 w1 适合的学习率，可能相对于 w2  来说会太小，如果仍使用适合 w1 的学习率，会导致在 w2 方向上走的非常慢，会消耗非常多的时间，而使用适合 w2 的学习率，对 w1  来说又太大，搜索不到适合 w1 的解。

3. 通过标准化后，**实现了数据中心化**，数据中心化符合数据分布规律，**能增加模型的泛化能力**

那么深度学习中在训练网络之前应该怎么做图像归一化呢？有两种方法：

1. **归一化到 0 - 1**：因为图片像素值的范围都在0~255，图片数据的归一化可以简单地除以255. 。 (注意255要加 . ，因为是要归一化到double型的 0-1 )
2. **归一化到 [-1, 1]**：在深度学习网络的代码中，将图像喂给网络前，会
   - 先统计训练集中图像RGB这3个通道的均值和方差，如：`mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]`，
   - 接着对各通道的像素做减去均值并除以标准差的操作。
   - 不仅在训练的时候要做这个预处理，在测试的时候，同样是使用在训练集中算出来的均值与标准差进行的归一化。

注意两者的区别：归一化到 [-1, 1] 就不会出现输入都为正数的情况，如果输入都为正数，会出现什么情况呢？：根据求导的链式法则，w的局部梯度是X，当X全为正时，由反向传播传下来的梯度乘以X后不会改变方向，要么为正数要么为负数，也就是说 w 权重的更新在一次更新迭代计算中要么同时减小，要么同时增大。

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/%E5%BD%92%E4%B8%80%E5%8C%96%E9%97%AE%E9%A2%98.png)

其中，w的更新方向向量只能在第一和第三象限。假设最佳的w向量如蓝色线所示，由于输入全为正，现在迭代更新只能沿着红色路径做zig-zag运动，**更新的效率很慢**。

基于此，当输入数据减去均值后，就会有负有正，会消除这种影响。



#### 6.2 哪些机器学习算法不需要归一化处理？

**概率模型不需要归一化**，因为它们不关心变量的值，而是关心变量的分布和变量之间的条件概率，如决策树、RF。

而像Adaboost、GBDT、XGBoost、SVM、LR、KNN、KMeans之类的最优化问题就需要归一化。



#### 6.3 标准化和归一化的区别

简单来说，标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下。

归一化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，也就是说都转化为“单位向量”。





### 7. 什么是端到端的学习





### 8.为什么机器学习中解决回归问题的时候一般使用平方损失（即均方误差）？

参考文章：

https://github.com/GYee/CV_interviews_Q-A/blob/master/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/07_ReLU%E5%87%BD%E6%95%B0%E5%9C%A80%E5%A4%84%E4%B8%8D%E5%8F%AF%E5%AF%BC%EF%BC%8C%E4%B8%BA%E4%BB%80%E4%B9%88%E8%BF%98%E8%83%BD%E7%94%A8.md

损失函数是是模型预测值与真实值之间的一种距离度量，我们可以计算出每个样本的预测值与真实值之间的距离，全部加起来就得到了所谓的损失函数。而距离的度量可以采用预测值与真实值之间差的绝对值，或者两者之差的平方，当然更高次的也行，只要你喜欢。正如问题所述，为什么我们一般使用的是两者之差的平方而不是两者只差的绝对值呢？其实这与模型的求解相关，举最简单的线性回归为例，如果采用的距离是两者之差的绝对值，那么求解的目标函数如下：
$$
(\omega^*, b) = arg min_{(\omega, b)}\sum_{i=1}^{m}\left|{f(x_i)-y_i}\right|
$$
如果采用的距离是两者之差的平方，那么求解的目标函数如下：
$$
(\omega^*, b) = arg min_{(\omega, b)}\sum_{i=1}^{m}({f(x_i)}-y_i)^2
$$
其中：$f(x_i) = \omega x_i + b$ 即预测值，$y_i$ 为真实值，$m$ 为样本总数，$\omega$ 和 $b$ 为要求解的参数

要求得使以上损失函数最小化对应的那个 $\omega$ 和 $b$ ，可将损失函数对 $\omega$ 和 $b$ 求导，并令导数为0。但是当采取的距离是两者之差的绝对值时，函数在0处不可导，且还增加的一个工作量是需要判断 $f(x_i)-y_i$ 正负号。而采用的距离是两者之差的平方时就没有这方面的问题，所以解决回归问题的时候一般使用平方损失。但理论上两者都可以使用，只是如果用两者之差的绝对值的话，那么需要判断和处理的东西多点，例如人为设定0处的导数为0等等。





### 9. 欧式距离和余弦相似度的区别是什么？

参考：https://www.zhihu.com/question/19640394

欧式距离：$d=\sqrt{（x_i-x_j）^2-(y_i-y_j)^2}$，它会受到不同单位刻度的影响，所以一般需要进行标准化，距离越大，表示差异越大；

余弦相似度：$d=cos\theta=\frac{x\cdot y}{||x||\cdot ||y||}$，它不受单位刻度影响，余弦值落在 [-1, 1] ，数值越大，表示差异越小。



总体来说，**欧氏距离体现数值上的绝对差异**，**而余弦距离体现方向上的相对差异**。

例如，统计两部剧的用户观看行为，用户A的观看向量为（0，1），用户B为（1，0）；此时二者的余弦距离很大，而欧氏距离很小；我们分析两个用户对于不同视频的偏好，更关注**相对**差异，显然应当使用余弦距离。而当我们分析用户活跃度，以登陆次数（单位:次）和平均观看时长（单位:分钟）作为特征时，余弦距离会认为（1，10）、（10，100）两个用户距离很近；但显然这两个用户活跃度是有着极大差异的，此时我们更关注数值绝对差异，应当使用欧氏距离。



### 10. 维数灾难

- 高维空间训练得到的分类器相当于低维空间的一个复杂非线性分类器，这类分类器容易产生过拟合
- 如果一直增加维度，原有的数据样本会越来越稀疏，要避免过拟合就需要不断增加样本
- 数据的稀疏性使得数据的分布在空间上是不同的，在高维空间的中心比边缘区域具有更大的稀疏性（举例，正方体和内切圆到超立方体和超球面，随着维度趋于无穷，超球面体积趋向于0而超立方体体积永远是1）



### 11. 伪标签技术

- 将test数据集中的数据加入到train数据集中，其对应的标签为基于原有数据集训练好的模型预测得到的
- 伪标签技术在一定程度上起到一种正则化的作用。如果训练开始就直接使用该技术，则网络可能会有过拟合风险，但是如果经过几轮训练迭代后（只是用原有训练集数据）将训练集和未打标签的数据一起进行训练，则会提升网络的泛化能力
- 操作过程中一般每个batch中的1/4到1/3的数据为伪标签数据



### 12. 绘制ROC的标准和快速方法以及ROC对比PR曲线的优势

- 标准方法：横坐标为FPR， 纵坐标为TPR， 设置一个区分正负预测结果的阈值并进行动态调整，从最高得分（实际上是正无穷）开始逐渐下降，描出每个阈值对应的点最后连接
- 快速方法：根据样本标签统计出正负样本数，将坐标轴根据样本数进行单位分割，根据模型预测结果将样本进行排序后，从高到低遍历样本，每遇到一个正样本就沿纵轴方向绘制一个单位长度的曲线，反之则沿横轴，直到遍历完成
- PR曲线的横坐标是Recall，纵坐标是Precision，相对于PR曲线来说，当正负样本发生剧烈变化时，ROC曲线的形状能够基本保持不变，而PR曲线的形状会发生剧烈改变



### 13. 为什么使用F1 score而不是算术平均？

参考：

[为什么要用f1-score而不是平均值](https://www.cnblogs.com/walter-xh/p/11140715.html) https://www.cnblogs.com/walter-xh/p/11140715.html



F1 score是分类问题中常用的评价指标，定义为精确率（Precision）和召回率（Recall）的调和平均数。
$$
F1=\frac{1}{\frac{1}{Precision}+\frac{1}{Recall}}=\frac{2×Precision×Recall}{Precision+Recall}
$$

> 补充一下精确率和召回率的公式：
>
> > TP（ True Positive）：真正例
> >
> > FP（ False Positive）：假正例
> >
> > FN（False Negative）：假反例
> >
> > TN（True Negative）：真反例
>
> **精确率（Precision）:**	$Precision=\frac{TP}{TP+FP}$ 
>
> **召回率（Recall）:**	$Recall=\frac{TP}{TP+FN}$ 
>
> > 精确率，也称为查准率，衡量的是**预测结果为正例的样本中被正确分类的正例样本的比例**。
> >
> > 召回率，也称为查全率，衡量的是**真实情况下的所有正样本中被正确分类的正样本的比例。**



F1 score 综合考虑了精确率和召回率，**其结果更偏向于 Precision 和 Recall 中较小的那个**，即 Precision 和 Recall 中较小的那个对 F1 score 的结果取决定性作用。例如若 $Precision=1,Recall \approx 0$，由F1 score的计算公式可以看出，此时其结果主要受 Recall 影响。

如果对 Precision 和 Recall 取算术平均值（$\frac{Precision+Recall}{2}$），对于 $Precision=1,Recall \approx 0$，其结果约为 0.5，而 F1 score 调和平均的结果约为 0。

**这也是为什么很多应用场景中会选择使用 F1 score 调和平均值而不是算术平均值的原因，因为我们希望这个结果可以更好地反映模型的性能好坏，而不是直接平均模糊化了 Precision 和 Recall 各自对模型的影响。**



> 补充另外两种评价方法：

**加权调和平均：**

上面的 F1 score 中， Precision 和 Recall 是同等重要的，而有的时候可能希望我们的模型更关注其中的某一个指标，这时可以使用加权调和平均：
$$
F_{\beta}=(1+\beta^{2})\frac{1}{\frac{1}{Precision}+\beta^{2}×\frac{1}{Recall}}=(1+\beta^{2})\frac{Precision×Recall}{\beta^{2}×Precision+Recall}
$$
当 $\beta > 1$ 时召回率有更大影响， $\beta < 1$ 时精确率有更大影响， $\beta = 1$ 时退化为 F1 score。



**几何平均数：**
$$
G=\sqrt{Precision×Recall}
$$

### 14. 统计概率学相关问题

#### 14.1 协方差和相关性有什么区别

相关性是协方差的标准化格式。协方差本身很难做比较。例如：如果我们计算工资（$）和年龄（岁）的协方差，因为这两个变量有不同的度量，所以我们会得到不能做比较的不同的协方差。为了解决这个问题，我们计算相关性来得到一个介于-1和1之间的值，就可以忽略它们各自不同的度量。







------

## 传统的机器学习算法

### 1. 提升树的原理，学习的是什么



### 2. SVM 

#### 损失函数

#### 手推原理

#### 核函数公式，为什么使用核函数



#### 对偶式的推导



#### SVM和LR有哪些区别

##### 相同点

1. 都是分类算法，如果不考虑核函数，都是线性分类算法
2. 都是监督学习算法
3. 都是判别模型
4. 都能通过核函数方法针对非线性情况分类
5. 目标都是找一个分类超平面
6. 都能减少离群点的影响



##### 不同点

###### 1.损失函数不同

逻辑回归是 cross entropy loss，svm 是 hinge loss

$$
LR\ Loss:\ L(\omega,b)=\sum_{i=1}^{m}ln(y_{i}p_{1}(x;\beta)+(1-y_{i})p_{0}(x;\beta)) = \sum_{i=1}^{m}(-y_{i}\beta^{T}x_{i}+ln(1+e^{\beta^{T}x_{i}}))
\\其中，\beta=(\omega;b), p_{1}=p(y=1|x;\beta), p_{0}=p(y=0|x;\beta)  \\
\\SVM\ Loss:\ L(\omega,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^{m}\alpha_{i}(1-y_{i}(\omega^{T}x_{i}+b))
$$

LR： 基于概率理论，通过极大似然估计方法估计参数值

SVM：基于几何间隔最大化原理

> 补充：		Logistic Loss：			 $L_{log}(z)=log(1+e^{-z})$                   // 用于 LR
>
> ​					Hinge Loss：		  	 $L_{hinge}(z)=max(0,1-z)$             // SVM
>
> ​					Exponential Loss：	$L_{exp}(z)=e^{-z}$  								 // Adaboost



###### 2. 考虑的样本点不同

SVM只考虑边界线上局部的点（即support vector），而LR考虑了所有点

影响SVM决策分类面的只有少数的点，即边界线上的支持向量，其他样本对分类决策面没有任何影响，即SVM不依赖于数据分布；

而LR则考虑了全部的点（即依赖于数据分布），优化参数时所有样本点都参与了贡献，通过非线性映射，减少远离分类平面的点的权重，即对不平衡的数据要先做balance。这也是为什么逻辑回归不用核函数，它需要计算的样本太多。并且由于逻辑回归受所有样本的影响，当样本不均衡时需要平衡一下每一类的样本个数。



###### 3. 在解决非线性问题时，SVM采用核函数机制，而LR一般很少采用核函数的方法。

SVM使用的是hinge loss，可以方便地转化成对偶问题进行求解，在解决非线性问题时，引入核函数机制可以大大降低计算复杂度。



###### 4. SVM依赖于数据分布的距离测度，所以需对数据先做normalization，而LR不受影响。

normalization的好处：进行梯度下降时，数值更新的速度一致，少走弯路，可以更快地收敛。



###### 5. 逻辑回归是处理经验风险最小化，svm是结构风险最小化

所以SVM的损失函数中自带正则化项($\frac{1}{2}||w||^2$)，而LR需要另外添加。



###### 6. 逻辑回归对概率建模，svm对分类超平面建模

###### 7. 逻辑回归通过非线性变换减弱分离平面较远的点的影响，svm则只取支持向量从而消去较远点的影响

###### 8. 逻辑回归是统计方法，svm是几何方法





### 3. 逻辑回归

#### 损失函数



#### 为什么用log损失而不是均方误差损失（最小二乘）



#### 它不能解决什么问题



#### 梯度的表示



#### 如何解决低维不可分的问题

通过特征变换的方式把低维空间转换到高维空间，而在低维空间不可分的数据，到高维空间中线性可分的几率会高一些。具体方法：核函数，如：高斯核，多项式核等等





### 4. 在k-means或kNN，我们是用欧氏距离来计算最近的邻居之间的距离。为什么不用曼哈顿距离？

曼哈顿距离只计算水平或垂直距离，有维度的限制。另一方面，欧氏距离可用于任何空间的距离计算问题。因为，数据点可以存在于任何空间，欧氏距离是更可行的选择。例如：想象一下国际象棋棋盘，象或车所做的移动是由曼哈顿距离计算的，因为它们是在各自的水平和垂直方向做的运动。



### 5. GMM 原理，和kmeans 的区别，应用场景



### 6. 决策树的原理

#### 回归决策树的输出是什么

#### 决策树使用什么指标进行划分

#### 信息增益的定义是什么，公式是什么

#### 熵的定义什么，公式是什么



### 7. 集成学习相关问题

#### 7.1 Boosting 和 Bagging 的区别

- Boosting主要思想是将一族弱学习器提升为强学习器的方法，具体原理为：先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，来训练下一个基学习器，如此重复进行直至学习器数目达到了事先要求，最终进行学习器的加权结合
- Bagging是并行式集成学习方法最著名的代表，具体做法是对样本进行有放回的采样，然后基于每个采样训练集得到一个基学习器，再将它们进行组合。在预测时，对于分类任务通常使用简单投票法，对回归任务使用简单平均法









### 8. 随机森林相关

#### 8.1 随机森林思想怎么添加的随机性

随机森林 (RF) 是 Bagging 的一个变体。RF在以决策树为基学习器构建 Bagging 集成的基础上，进一步在决策树的训练过程中引入随机性：

> 传统决策树在选择划分属性时，是在当前结点的属性集合（假定有 d 个属性）中选择一个最优属性；而在 RF 中，对基决策树的每一个结点，**先从结点的属性集合中随机选择一个包含 k 个属性的子集**，然后再从这个子集当中选择一个最优属性用于划分。

这里的参数 k 控制了随机性的引入程度。若令 $k=d$ ，则基决策树的构建与传统决策树相同，一般情况下，推荐值为 $k=log_2 d$ 。



#### 8.2 为什么不容易过拟合

因为随机森林中每棵树的**训练样本是随机的**，每棵树中的**每个结点的分裂属性也是随机选择的**。这两个随机性的引入，使得随机森林不容易陷入过拟合。

而且树的数量越多，随机森林通常会收敛到更低的泛化误差。

理论上当树的数目趋于无穷时，随机森林便不会出现过拟合，但是现实当做做不到训练无穷多棵树。



#### 8.3 如何评估特征的重要性

这个问题是决策树的核心问题，而随机森林是以决策树为基学习器的，所以这里大概提提，详细的可以去看看决策树模型。

决策树中，根节点包含样本全集，其他非叶子结点包含的样本集合根据选择的属性被划分到子节点中，叶节点对应于分类结果。决策树的关键是在非叶子结点中怎么选择最优的属性特征以对该结点中的样本进行划分，方法主要有<u>信息增益、增益率以及基尼系数</u>３种，下面分别叙述。

##### 信息增益  (ID3决策树中采用)

**“信息熵”**是度量样本集合纯度最常用的一种指标，假定当前样本结合 $D$ 中第  $k$ 类样本所占的比例为 $p_k(k = 1, 2, ..., c)$ ，则 $D$ 的信息熵定义为：   
$$
Ent(D)= -\sum_{k=1}^{c}p_klog_2 p_k
$$
$Ent(D)$ 的值越小，则 $D$ 的纯度越高。注意因为 $p_k \le 1$ ，因此 $Ent(D)$ 也是一个大于等于０小于１的值。

假定离散属性 $a$ 有 V 个可能的取值 $\{a^1,a^2,...,a^V\}$ ，若使用 $a$ 来对样本集合 $D$ 进行划分的话，则会产生 V 个分支结点，其中第  $v$  个分支结点包含了 $D$ 中所有在属性 $a$ 上取值为 $a^v$ 的样本，记为 $D^v$ 。同样可以根据上式计算出 $D^v$ 的信息熵，再考虑到不同的分支结点所包含的样本数不同，给分支结点赋予权重 $\frac{|D^v|}{|D|}$ ，即样本数越多的分支结点的影响越大，于是可以计算出使用属性 $a$ 对样本集 $D$ 进行划分时所获得的“信息增益”：
$$
Gain(D,a) = Ent(D) - \sum_{v=1}^{V}\frac{|D^v|}{|D|}Ent(D^v)
$$
一般而言，**信息增益越大越好，因为其代表着选择该属性进行划分所带来的纯度提升**，因此全部计算当前样本集合 $D$ 中存在不同取值的那些属性的信息增益后，取信息增益最大的那个所对应的属性作为划分属性即可。

**缺点：**对可取值数目多的属性有所偏好



##### 增益率  (C4.5决策树中采用)

从信息增益的表达式很容易看出，信息增益准则对可取值数目多的属性有所偏好，为减少这种偏好带来的影响，大佬们提出了增益率准则，定义如下：
$$
Gain\_ratio(D,a) = \frac{Gain(D,a)}{IV(a)} \\
IV(a) = \sum_{v=1}^{V}\frac{|D^v|}{|D|}log_2 \frac{|D^v|}{|D|}
$$
$IV(a)$ 称为属性 a 的“固有值”。属性 a 的可能取值数目越多，则 $IV(a)$ 的值通常会越大，因此一定程度上抵消了信息增益对可取值数目多的属性的偏好。

**缺点：**增益率对可取值数目少的属性有所偏好

> 因为增益率存在以上缺点，因此C4.5算法并不是直接选择增益率最大的候选划分属性，而是使用了一个启发式：**先从候选划分属性中找出信息增益高于平均水平的属性，再从中选择增益率最高的。**



##### 基尼指数  (CART决策树中采用)

这里改用基尼值来度量数据集 $D$ 的纯度，而不是上面的信息熵。基尼值定义如下：
$$
Gini(D) = \sum_{k=1}^{c} \sum_{k^, \not =k}p_kp_{k^,} = 1- \sum_{k=1}^{c}p_k^2 = 1-\sum_{k=1}^{c}(\frac{D^k}{D})^2
$$
直观来看，$Gini(D)$ 反映了从数据集 $D$ 中随机抽取两个样本，其类别标记不一致的概率，因此$Gini(D)$ 越小，则数据集 $D$ 的纯度越高。

对于样本D，个数为|D|，根据特征A的某个值a，把D分成|D1|和|D2|，则在特征A的条件下，样本D的基尼系数表达式为：
$$
Gini\_index(D,A) = \frac{|D^1|}{|D|}Gini(D^1)+ \frac{|D^2|}{|D|}Gini(D^2)
$$
于是，我们在候选属性集合A中，选择那个使得划分后基尼系数最小的属性作为最优划分属性即可。



#### 8.4 如何处理缺失值

方法一（na.roughfix）简单粗暴，对于训练集,同一个class下的数据，如果是分类变量缺失，用众数补上，如果是连续型变量缺失，用中位数补。

方法二（rfImpute）这个方法计算量大，至于比方法一好坏？不好判断。先用na.roughfix补上缺失值，然后构建森林并计算proximity matrix，再回头看缺失值，如果是分类变量，则用没有阵进行加权平均的方法补缺失值。然后迭代4-6次，这个补缺失值的思想和KNN有些类似1缺失的观测实例的proximity中的权重进行投票。如果是连续型变量，则用proximity矩2。





### 9. Xgboost 相关

#### 9.1 **为什么XGBoost要用泰勒展开，优势在哪里？**

XGBoost使用了一阶和二阶偏导, **二阶导数有利于梯度下降得更快更准**。

使用泰勒展开取得二阶倒数形式, 可以在不选定损失函数具体形式的情况下用于算法优化分析。本质上也就把损失函数的选取和模型算法优化/参数选择分开了。**这种去耦合增加了XGBoost的适用性**。



#### 9.2 **XGBoost如何寻找最优特征？是又放回还是无放回的呢？**

XGBoost在训练的过程中给出各个特征的评分，从而表明每个特征对模型训练的重要性。

XGBoost利用梯度优化模型算法，样本是不放回的(想象一个样本连续重复抽出,梯度来回踏步会不会高兴)。

但XGBoost支持子采样, 也就是每轮计算可以不使用全部样本。



### 10. 线性分类器与非线性分类器的区别以及优劣

如果模型是参数的线性函数，并且存在线性分类面，那么就是线性分类器，否则不是。

- 常见的线性分类器有：LR,贝叶斯分类，单层感知机、线性回归。
- 常见的非线性分类器：决策树、RF、GBDT、多层感知机。

- SVM两种都有(看线性核还是高斯核)。

区别：

- 线性分类器速度快、编程方便，但是可能拟合效果不会很好。
- 非线性分类器编程复杂，但是效果拟合能力强。





------

## 深度学习基础

### 1. Batch norm 的作用，原理



### 2. 梯度消失、梯度爆炸的原因以及解决方法

参考：

- [激活函数及其作用以及梯度消失、爆炸、神经元节点死亡的解释](https://blog.csdn.net/qq_17130909/article/details/80582226)

#### 梯度消失和梯度爆炸的表现

网络层数越多，模型训练的时候便越容易出现 梯度消失(gradient vanish) 和 梯度爆炸(gradient explod) 这种梯度不稳定的问题。假设现在有一个含有３层隐含层的神经网络：

![这里写图片描述](images/%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E5%92%8C%E7%88%86%E7%82%B8_fig1.png)

**梯度消失发生时的表现是：**靠近输出层的 hidden layer 3 的权值更新正常，但是靠近输入层的 hidden layer 1 的权值更新非常慢，导致其权值几乎不变，仍接近于初始化的权值。这就导致 hidden layer 1 相当于只是一个映射层，对所有的输入做了一个函数映射，这时的深度学习网络的学习等价于只有后几层的隐含层网络在学习。

**梯度爆炸发生时的表现是：**当初始的权值太大，靠近输入层的 hidden layer 1 的权值变化比靠近输出层的 hidden layer 3 的权值变化更快。

**所以梯度消失和梯度爆炸都是出现在靠近输入层的参数中**。



#### 产生梯度消失与梯度爆炸的根本原因

##### 梯度消失分析

下图是我画的一个非常简单的神经网络，每层都只有一个神经元，且神经元所用的激活函数 $\sigma$ 为 sigmoid 函数，$Loss$ 表示损失函数，前一层的输出与后一层的输入关系如下：
$$
y_i = \sigma(z_i) = \sigma(w_i*x_i+b_i), \quad其中x_i = y_{i-1}
$$
![](images/%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E5%92%8C%E7%88%86%E7%82%B8_fig2.png)

因此，根据反向传播的链式法则，损失函数相对于参数 $b_1$ 的梯度计算公式如下：
$$
\frac{\partial Loss}{\partial b_1} = \frac{\partial Loss}{\partial y_4}*\frac{\partial y_4}{\partial z_4}*\frac{\partial z_4}{\partial x_4}*\frac{\partial x_4}{\partial z_3}*\frac{\partial z_3}{\partial x_3}*\frac{\partial x_3}{\partial z_2}*\frac{\partial z_2}{\partial x_2}*\frac{\partial x_2}{\partial z_1}*\frac{\partial z_1}{\partial b_1} \\
= \frac{\partial Loss}{\partial y_4}*\partial{'}(z_4)*w_4*\partial{'}(z_3)*w_3*\partial{'}(z_2)*w_2*\partial{'}(z_1)
$$
而 sigmoid 函数的导数 $\sigma{'}(x)$ 如下图所示：

![这里写图片描述](images/%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E5%92%8C%E7%88%86%E7%82%B8_fig3.png)

即 $\sigma{'}(x)\le \frac{1}{4}$ ，而我们一般会使用标准方法来初始化网络权重，即使用一个均值为 0 标准差为 1 的高斯分布，因此初始化的网络参数 $w_i$ 通常都小于 1 ，从而有 $|\sigma{'}(z_i)*w_i|\le \frac{1}{4}$ 。

根据公式(2)的计算规律，**层数越多，越是前面的层的参数的求导结果越小，于是便导致了梯度消失情况的出现**。

##### 梯度爆炸分析

在分析梯度消失时，我们明白了导致其发生的主要原因是　$|\sigma{'}(z_i)*w_i|\le \frac{1}{4}$ ，经链式法则反向传播后，越靠近输入层的参数的梯度越小。

而导致梯度爆炸的原因是：$|\sigma{'}(z_i)*w_i|>1$，当该表达式大于 1 时，经链式法则的指数倍传播后，前面层的参数的梯度会非常大，从而出现梯度爆炸。

但是要使得$|\sigma{'}(z_i)*w_i|>1$，就得 $|w_i| > 4$才行，按照 $|\sigma{'}(w_i*x_i+b_i)*w_i|>1$，可以计算出 $x_i$ 的数值变化范围很窄，仅在公式(3)的范围内，才会出现梯度爆炸，**因此梯度爆炸问题在使用 sigmoid 激活函数时出现的情况较少，不容易发生。**

![这里写图片描述](images/%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E5%92%8C%E7%88%86%E7%82%B8_fig4.png)

#### 怎么解决

如上分析，**造成梯度消失和梯度爆炸问题是网络太深，网络权值更新不稳定造成的，本质上是因为梯度反向传播中的连乘效应**。另外一个原因是当激活函数使用 sigmoid 时，梯度消失问题更容易发生，因此可以考虑的解决方法如下：

1. 压缩模型层数
2. 改用其他的激活函数如 ReLU
3. 使用 BN 层
4. 使用 ResNet 的短路连接结构
5. 初始化方式采用 Xavier 或者 msra



------

### 3. 网络初始化的方式，各自的公式和初始化过程



### 4. 优化算法 SGD、ADAM 算法过程，动量算法过程



### 5. Dropout 原理

- 可以把dropout看成是 一种ensemble方法，每次做完dropout相当于从原网络中找到一个更瘦的网络
- 强迫神经元和其他随机挑选出来的神经元共同工作，减弱了神经元节点间的联合适应性，增强泛化能力
- 使用dropout得到更多的局部簇，同等数据下，簇变多了，因而区分性变大，稀疏性也更大





### 6. 尺度变换剧烈如何解决





### 7. 神经网络如何加速





### 8. 过拟合和欠拟合，如何解决

参考：[过拟合与欠拟合及方差偏差](https://www.jianshu.com/p/f2489ccc14b4)  



#### 过拟合的原因

从两个角度去分析：

1. **模型的复杂度**：模型过于复杂，把噪声数据的特征也学习到模型中，导致模型泛化性能下降
2. **数据集规模大小**：数据集规模相对模型复杂度来说太小，使得模型过度挖掘数据集中的特征，把一些不具有代表性的特征也学习到了模型中。例如训练集中有一个叶子图片，该叶子的边缘是锯齿状，模型学习了该图片后认为叶子都应该有锯齿状边缘，因此当新数据中的叶子边缘不是锯齿状时，都判断为不是叶子。

#### 过拟合的解法方法

1. **获得更多的训练数据**：使用更多的训练数据是解决过拟合问题最有效的手段，因为更多的样本能够让模型学习到更多更有效的特征，减少噪声的影响。

   当然直接增加实验数据在很多场景下都是没那么容易的，因此可以通过**数据扩充技术**，例如对图像进行平移、旋转和缩放等等。

   除了根据原有数据进行扩充外，还有一种思路是使用非常火热的**生成式对抗网络 GAN **来合成大量的新训练数据。

   还有一种方法是使用**迁移学习技术**，使用已经在更大规模的源域数据集上训练好的模型参数来初始化我们的模型，模型往往可以更快地收敛。但是也有一个问题是，源域数据集中的场景跟我们目标域数据集的场景差异过大时，可能效果会不太好，需要多做实验来判断。

2. **降低模型复杂度**：在深度学习中我们可以减少网络的层数，改用参数量更少的模型；在机器学习的决策树模型中可以降低树的高度、进行剪枝等。

3. **正则化方法**如 L2 将权值大小加入到损失函数中，根据奥卡姆剃刀原理，拟合效果差不多情况下，模型复杂度越低越好。至于为什么正则化可以减轻过拟合这个问题可以看看[这个博客](https://blog.csdn.net/qq_37344125/article/details/104326946)，挺好懂的.。

   **添加BN层**（这个我们专门在BN专题中讨论过了，BN层可以一定程度上提高模型泛化性能）

   使用**dropout技术**（dropout在训练时会随机隐藏一些神经元，导致训练过程中不会每次都更新(**预测时不会发生dropout**)，最终的结果是每个神经元的权重w都不会更新的太大，起到了类似L2正则化的作用来降低过拟合风险。）

4. **Early Stopping**：Early stopping便是**一种迭代次数截断**的方法来防止过拟合的方法，即在模型对训练数据集迭代收敛之前停止迭代来防止过拟合。

   Early stopping方法的具体做法是：**在每一个Epoch结束时（一个Epoch集为对所有的训练数据的一轮遍历）计算validation data的accuracy，当accuracy不再提高时，就停止训练**。这种做法很符合直观感受，因为accurary都不再提高了，在继续训练也是无益的，只会提高训练的时间。那么该做法的一个重点便是怎样才认为validation accurary不再提高了呢？并不是说validation accuracy一降下来便认为不再提高了，因为可能经过这个Epoch后，accuracy降低了，但是随后的Epoch又让accuracy又上去了，所以不能根据一两次的连续降低就判断不再提高。**一般的做法是，在训练的过程中，记录到目前为止最好的validation accuracy，当连续10次Epoch（或者更多次）没达到最佳accuracy时，则可以认为accuracy不再提高了**。

5. **集成学习方法**：集成学习是把多个模型集成在一起，来降低单一模型的过拟合风险，例如Bagging方法。

   如DNN可以用Bagging的思路来正则化。首先我们要对原始的m个训练样本进行有放回随机采样，构建N组m个样本的数据集，然后分别用这N组数据集去训练我们的DNN。即采用我们的前向传播算法和反向传播算法得到N个DNN模型的W,b参数组合，最后对N个DNN模型的输出用加权平均法或者投票法决定最终输出。不过用集成学习Bagging的方法有一个问题，就是我们的DNN模型本来就比较复杂，参数很多。现在又变成了N个DNN模型，这样参数又增加了N倍，从而导致训练这样的网络要花更加多的时间和空间。因此一般N的个数不能太多，比如5-10个就可以了。

6. **交叉检验**，如S折交叉验证，通过交叉检验得到较优的模型参数，其实这个跟上面的Bagging方法比较类似，只不过S折交叉验证是随机将已给数据切分成S个互不相交的大小相同的自己，然后利用S-1个子集的数据训练模型，利用余下的子集测试模型；将这一过程对可能的S种选择重复进行；最后选出S次评测中平均测试误差最小的模型。



#### 欠拟合的原因

同样可以从两个角度去分析：

1. **模型过于简单**：简单模型的学习能力比较差
2. **提取的特征不好**：当特征不足或者现有特征与样本标签的相关性不强时，模型容易出现欠拟合

#### 欠拟合的解决方法

1. **增加模型复杂度**：如线性模型增加高次项改为非线性模型、在神经网络模型中增加网络层数或者神经元个数、深度学习中改为使用参数量更多更先进的模型等等。
2. **增加新特征**：可以考虑特征组合等特征工程工作
3. 如果损失函数中加了正则项，可以考虑**减小正则项的系数** $\lambda$





### 9. Normalization 的相关问题

#### 9.1 Batch normalization和Instance normalization的对比？

参考：[Batch normalization和Instance normalization的对比？](https://www.zhihu.com/question/68730628)

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



#### 9.2 Weight Normalization 相比batch Normalization 有什么优点呢？

参考：[Weight Normalization 相比batch Normalization 有什么优点呢？](https://www.zhihu.com/question/55132852/answer/171250929)

WN 相比 BN 的优势如下：

1. WN 是通过改写网络的权重参数w，没有引入 batch 的依赖，所以可以适用于 RNN（LSTM）网络，RNN 网络不能用 BN 的原因有这几个：
   - RNN 处理的序列是变长的；
   - RNN 是基于 time step 计算的，如果用 BN 处理，需要保存每个 time step 下 mini batch 的均值和方差，效率低且占内存；
2. BN 是基于一个 batch 的数据计算均值和方差，**相当于进行梯度计算引入噪声**，所以不适合对噪声敏感的强化学习、生成模型（如 GAN,VAE)。而 WN 是通过标量 g 和向量 v 对权重 W 进行重写，重写向量 v 是固定的，因此 WN 的操作比 BN 引入的噪声会更少；
3. 不用额外的空间保存 batch 的均值和方差，另外 WN 的计算开销会更小；

当然，WN 需要注意的是参数初始值的选择，它不具备 BN 的将网络每一层的输出 y 固定在一个变化范围的作用。



### 10.激活函数的一些问题

参考文章：

- [最全最详细的常见激活函数总结（sigmoid、Tanh、ReLU等）及激活函数面试常见问题总结](https://blog.csdn.net/neo_lcx/article/details/100122938)
- [在神经网络中，激活函数sigmoid和tanh除了阈值取值外有什么不同吗？](https://www.zhihu.com/question/50396271?from=profile_question_card)
- [RNN 中为什么要采用 tanh，而不是 ReLU 作为激活函数？](https://www.zhihu.com/question/61265076/answer/186347780)
- [为什么LSTM模型中既存在sigmoid又存在tanh两种激活函数？](https://www.zhihu.com/question/46197687/answer/110123951)



#### 10.1 相比于sigmoid函数，tanh激活函数输出关于“零点”对称的好处是什么？

对于sigmoid函数而言，其输出始终为正，这会**导致在深度网络训练中模型的收敛速度变慢**，因为在反向传播链式求导过程中，权重更新的效率会降低（具体推导可以参考[这篇文章](https://www.zhihu.com/question/50396271?from=profile_question_card)）。

此外，sigmoid函数的输出均大于0，作为下层神经元的输入会导致下层输入不是0均值的，随着网络的加深可能会使得原始数据的分布发生改变。而在深度学习的网络训练中，经常需要将数据处理成零均值分布的情况，以提高收敛效率，因此tanh函数更加符合这个要求。

sigmoid函数的输出在[0,1]之间，比较适合用于二分类问题。



#### 10.2 为什么RNN中常用tanh函数作为激活函数而不是ReLU？

详细分析可以参考[这篇文章](https://www.zhihu.com/question/61265076/answer/186347780)。下面简单用自己的话总结一下：

RNN中将 tanh 函数作为激活函数本身就存在梯度消失的问题，而ReLU本就是为了克服梯度消失问题而生的，那为什么不能**直接**（注意：这里说的是直接替代，事实上通过**截断优化**ReLU仍可以在RNN中取得很好的表现）用ReLU来代替RNN中的tanh来作为激活函数呢？**这是因为ReLU的导数只能为0或1，而导数为1的时候在RNN中很容易造成梯度爆炸问题**。

**为什么会出现梯度爆炸的问题呢？**因为在RNN中，每个神经元在不同的时刻都共享一个参数W（这点与CNN不同，CNN中每一层都使用独立的参数$W_i$），因此在前向和反向传播中，每个神经元的输出都会作为下一个时刻本神经元的输入，从某种意义上来讲相当于对其参数矩阵W作了连乘，如果W中有其中一个特征值大于1，则多次累乘之后的结果将非常大，自然就产生了梯度爆炸的问题。

**那为什么ReLU在CNN中不存在连乘的梯度爆炸问题呢？**因为在CNN中，每一层都有不同的参数$W_i$，有的特征值大于1，有的小于1，在某种意义上可以理解为抵消了梯度爆炸的可能。



#### 10.3 如何解决ReLU神经元“死亡”的问题？

1. 采用Leaky ReLU等激活函数  
2. 设置较小的学习率进行训练 
3. 使用momentum优化算法动态调整学习率



#### 10.4 为什么LSTM模型中既存在sigmoid又存在tanh两种激活函数？

二者目的不一样

1. sigmoid 用在了各种gate上，产生0~1之间的值，这个一般只有sigmoid最直接了。
2. tanh 用在了状态和输出上，是对数据的处理，这个用其他激活函数或许也可以。



### 11. 特征融合 concat 和 add 的区别

参考：

- [理解concat和add的不同作用](https://blog.csdn.net/qq_32256033/article/details/89516738)
- [卷积神经网络中的add和concatnate区别](https://blog.csdn.net/weixin_39610043/article/details/87103358)





在网络模型当中，经常要进行不同通道特征图的信息融合相加操作，以整合不同通道的信息，在具体实现方面特征的融合方式一共有两种：

- 一种是 ResNet 和 FPN 等当中采用的 element-wise add 
- 另一种是 DenseNet 等中采用的 concat 。

他们之间有什么区别呢？



#### add

以下是 keras 中对 add 的实现源码：

```python
def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
        output += inputs[i]
    return output
```

其中 inputs 为待融合的特征图，inputs[0]、inputs[1]……等的通道数一样，且特征图宽与高也一样。

从代码中可以很容易地看出，add 方式有以下特点：

1. 做的是对应通道对应位置的值的相加，**通道数不变**
2. 描述图像的特征个数不变，但是**每个特征下的信息却增加**了。



#### concat

阅读下面代码实例帮助理解 concat 的工作原理：

```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat([t3, t4], 0)) ==> [4, 3]
tf.shape(tf.concat([t3, t4], 1)) ==> [2, 6]
```

在模型网路当中，数据通常为 4 个维度，即 num×channels×height×width ，因此默认值 1 表示的是 channels 通道进行拼接。如：

```python
combine = torch.cat([d1, add1, add2, add3, add4], 1)
```

从代码中可以很容易地看出，concat 方式有以下特点：

1. 做的是通道的合并，**通道数变多了**
2. **描述图像的特征个数变多，但是每个特征下的信息却不变**。



#### 小结

add相当于加了一种prior，当两路输入可以具有“对应通道的特征图语义类似”的性质的时候，可以用add来替代concat，这样更节省参数和计算量（concat是add的2倍）



### 12. 神经网络不收敛的原因以及解决办法

参考：

- [如何调整一个不收敛的卷积神经网络？？](https://www.zhihu.com/question/36023110)
- [My Neural Network isn't working! What should I do?](http://theorangeduck.com/page/neural-network-not-working)
- [神经网络不收敛的11个常见问题](https://zhuanlan.zhihu.com/p/36369878)



一般神经网络不收敛的原因有以下几种情况，来自文章[My Neural Network isn't working! What should I do?](http://theorangeduck.com/page/neural-network-not-working)的总结：

1. 没有对数据做归一化。
2. 没有检查过你的结果。这里的结果包括预处理结果和最终的训练测试结果。
3. 忘了做数据预处理。
4. 忘了使用正则化。
5. Batch Size设的太大。
6. 学习率设的不对。
7. 最后一层的激活函数用的不对。
8. 网络存在坏梯度。比如Relu对负值的梯度为0，反向传播时，0梯度就是不传播。
9. 参数初始化错误。
10. 网络太深。
11. 隐藏层神经元数量错误。



所以相应的解决办法就是：

1. 对数据进行归一化处理；
2. 通过可视化或者其他手段来查看输入和输出数据，看是否得到想要的结果；
3. 对数据进行预处理
4. 加入正则化的方法，比如 CNN 中可以采用 dropout
5. 设置小一点的 batch 大小
6. 关掉梯度裁剪。找到不会发生错误的最高学习率，并稍稍降低一些数值
7. 如果目的是回归而不是分类，那么绝大多数时候你不应该在最后一层使用激活函数
8. 如果你发现多个训练周期以后损失函数都没有收敛，那么可能是由于ReLU激活函数造成的。尝试切换到leaky ReLU或ELU。然后再看看问题是否解决
9. 采用如 xavier 的初始化方法
10. 可以先从 3-8 层的神经网络开始实验，当觉得训练正常了，需要提升模型性能后，才开始考虑更深的网络模型；
11. 通常设置的神经元数量是在 256 到 1024 之间，另外也可以参考同个任务或者同个领域的其他研究者使用的神经元数量，如果其他研究者使用的数量比较特别，那其原因可能对你很重要，需要去搞懂它。



### 13. 学习率设置的 trick

参考：

- [炼丹手册——学习率设置](https://zhuanlan.zhihu.com/p/332766013)



对于学习率的设置，一般分为两种，人工调整或策略调整。

**人工调整学习率**一般是根据我们的经验值进行尝试，首先在整个训练过程中学习率肯定不会设为一个固定的值，原因如上图描述的设置大了得不到局部最优值，设置小了收敛太慢也容易过拟合。通常我们会尝试性的将初始学习率设为：0.1，0.01，0.001，0.0001等来观察网络初始阶段epoch的loss情况：

- 如果训练初期loss出现梯度爆炸或NaN这样的情况（暂时排除其他原因引起的loss异常），说明初始学习率偏大，可以将初始学习率降低10倍再次尝试；
- 如果训练初期loss下降缓慢，说明初始学习率偏小，可以将初始学习率增加5倍或10倍再次尝试；
- 如果训练一段时间后loss下降缓慢或者出现震荡现象，可能训练进入到一个局部最小值或者鞍点附近。如果在局部最小值附近，需要降低学习率使训练朝更精细的位置移动；如果处于鞍点附件，需要适当增加学习率使步长更大跳出鞍点。
- 如果网络权重采用随机初始化方式从头学习，有时会因为任务复杂，初始学习率需要设置的比较小，否则很容易梯度飞掉带来模型的不稳定(振荡)。**这种思想也叫做Warmup**，在预热的小学习率下，模型可以慢慢趋于稳定，等模型相对稳定后再选择预先设置的学习率进行训练,使得模型收敛速度变得更快，模型效果更佳。

- 如果网络基于预训练权重做的finetune，由于模型在原数据集上以及收敛，有一个较好的七点，可以将初始学习率设置的小一些进行微调，比如0.0001。



**策略调整学习率**包括固定策略的学习率衰减和自适应学习率衰减，由于学习率如果连续衰减，不同的训练数据就会有不同的学习率。当学习率衰减时，在相似的训练数据下参数更新的速度也会放慢，就相当于减小了训练数据对模型训练结果的影响。为了使训练数据集中的所有数据对模型训练有相等的作用，通常是以epoch为单位衰减学习率。

**固定学习率衰减包括：**

分段减缓：每N轮学习率减半或者在训练过程中不同阶段设置不同的学习率，便于更精细的调参。TF的接口函数为：

```python
global_step_op = tf.train.get_or_create_global_step()
base_learning_rate = 0.01
decay_boundaries = [2000, 4000] # 学习率衰减边界；
learning_rate_value = [base_learning_rate, base_learning_rate/10., base_learning_rate/100.] # 不同阶段对应学习率。
learning_rate = tf.train.piecewise_constant_decay(global_step_op, boundaries=decay_boundaries,values=learning_rate_value)
```


分数减缓：将学习率随着epoch的次数进行衰减，$\alpha=\frac{1}{(1+decayRate*epoch)}* lr$ ，其中  $\alpha$ 表示学习率，decayRate 表示衰减率(可尝试设为0.1，根据数据集/迭代次数调整)，  epoch 表示迭代次数， lr  表示初始学习率。

指数减缓：与分数减缓类似，只是采用指数形式做了表达， $\alpha=\gamma^{epoch}*lr$ ，其中 $\gamma$ 表示指数的底（通常会设置为接近于1的数值，如0.95），随着训练批次epoch的增加，学习率呈指数下降。TF的接口函数为：

```python
global_step_op = tf.train.get_or_create_global_step()
base_learning_rate = 0.01
decay_rate = 0.98
decay_steps = 2000
learning_rate_no_stair = tf.train.exponential_decay(learning_rate=base_learning_rate,
                                                        decay_rate=decay_rate,
                                                        decay_steps=decay_steps,
                                                        staircase=True,
                                                        global_step=global_step_op,
                                                        name="exponential_decay_use_stair")
```

余弦周期减缓：**余弦周期减缓也叫余弦退火学习率**，不同于传统的学习率，随着epoch的增加，学习率先急速下降，再陡然提升，然后不断重复这个过程。其目的在于跳出局部最优点。

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/cosine_learning_rate.png)

之前介绍的几种学习率调节方式在神经网络训练过程中学习率会逐渐减小，所以模型逐渐找到局部最优点。这个过程中，因为一开始的学习率较大，模型不会踏入陡峭的局部最优点，而是快速往平坦的局部最优点移动。随着学习率逐渐减小，模型最终收敛到一个比较好的最优点。如下图所示：

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/learning_rate_schedule.png)

而余弦退火学习率由于急速下降，所以模型会迅速踏入局部最优点（不管是否陡峭），并保存局部最优点的模型。⌈快照集成⌋中⌈快照⌋的指的就是这个意思。保存模型后，学习率重新恢复到一个较大值，逃离当前的局部最优点，并寻找新的最优点。因为不同局部最优点的模型则存到较大的多样性，所以集合之后效果会更好。如下图所示：

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/cosine_learning_rate_2.png)

两种方式比较起来，可以理解为模型训练的“起点”和“终点”是差不多的。不同的是，余弦退火学习率使得模型的训练过程比较“曲折”。TF的接口函数为：

```python
# total_decay_step = 15000          总的学习率衰减步数
# base_learning_rate = 0.01         基学习率
# warmup_learning_rate = 0.0001     warm-up 学习率
# warmup_steps = 2000               warm-up 迭代次数
# hold_base_rate_steps_2000 = 2000  保持基学习率的步数
# hold_base_rate_steps_0 = 0
# alpha = 0.00001                   最小学习率
global_step_op = tf.train.get_or_create_global_step()
learning_rate = cosine_decay_with_warmup(learning_rate_base=base_learning_rate,
                                             total_decay_steps=total_decay_step,
                                             alpha = alpha,
                                             warmup_learning_rate=warmup_learning_rate,
                                             warmup_steps=warmup_steps,
                                             hold_base_rate_steps=hold_base_rate_steps_2000,
                                             global_step=global_step_op)
def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_decay_steps,
                             alpha = 0.0,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
  """Cosine decay schedule with warm up period.
  In this schedule, the learning rate grows linearly from warmup_learning_rate
  to learning_rate_base for warmup_steps, then transitions to a cosine decay
  schedule."""
  def eager_decay_rate():
    """Callable to compute the learning rate."""
    learning_rate = tf.train.cosine_decay(learning_rate=learning_rate_base,
                                          decay_steps=total_decay_steps - warmup_steps - hold_base_rate_steps,
                                          global_step= global_step - warmup_steps - hold_base_rate_steps,
                                          alpha=alpha)
    if hold_base_rate_steps > 0:
      learning_rate = tf.where(
          global_step > warmup_steps + hold_base_rate_steps,
          learning_rate, learning_rate_base)
    if warmup_steps > 0:
      if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger or equal to '
                         'warmup_learning_rate.')
      slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
      warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
      learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                               learning_rate)
    return tf.where(global_step > total_decay_steps, alpha, learning_rate,
                    name='learning_rate')

  if tf.executing_eagerly():
    return eager_decay_rate
  else:
    return eager_decay_rate()
```

**自适应学习率衰减包括：**

AdaGrad、 RMSprop、 AdaDelta等。此处更偏向于优化算法，暂时不在该篇介绍。



### 14. 深度学习的模型参数量计算

#### 全连接层

假设输入层大小i，隐藏层h，输出层o：

则参数量为各层之间的参数+每层的偏差=$(i*h+h*o)+(h+o)$

#### RNN

首先我们定义三个参数：g:门的数量(RNN有1个门，GRU有3个，LSTM有4个)

h:隐藏单元大小 ，i:输出层大小

每个门中的权重实际上是一个输入大小(h + i)（解释:上一个时刻的隐状态和当前输入的拼接）和输出大小为h（解释：当前时刻的隐状态）的FFNN。所以每个门都有$h\times（h + i）+ h$个参数。即在RNN中参数数量为：**g ×[ h（h + i）+ h ]**

**注意：输出我们只关心h，不关心其上层的全连接或者softmax，因为这已经不是当前rnn的参数了。**

举例：

1. 具有2个隐藏单元和输入尺寸3的LSTM：

则参数量为：g ×[ h（h + i）+ h ]= 4 ×[2（2 + 3）+ 2] = 48

2. 具有5个隐藏单元和输入大小为8的堆叠双向GRU +具有50个隐藏单元的LSTM的参数数量为：

第一层参数：2 × g ×[ h(h + i)+ h ] = 2 ×3×[5(5 + 8)+ 5] = 420

第二层参数： g ×[ h(h + i)+ h ]= 4×[50(50 + 10)+ 50]= 12200

则总参数量为： 420 + 12200 = 12620



#### CNN

首先我们定义三个参数：i:输入尺寸，f:卷积核的大小，o:输出大小

则每个滤波器对应的输出映射参数为：**num_params =权重+偏差= [ i×(f×f)×o ] + o**

例如:

带有1 × 1滤波器的灰度图像，输出3个通道

参数数量为： [ i×(f×f)×o ] + o= [1 ×(2 × 2)× 3] + 3= 15



### 15. 什麽样的资料集不适合用深度学习?

1. **数据集太小**，数据样本不足时，深度学习相对其它机器学习算法，没有明显优势。

2. **数据集没有局部相关特性，**目前深度学习表现比较好的领域主要是图像／语音／自然语言处理等领域，这些领域的一个共性是局部相关性。图像中像素组成物体，语音信号中音位组合成单词，文本数据中单词组合成句子，这些特征元素的组合一旦被打乱，表示的含义同时也被改变。对于没有这样的局部相关性的数据集，不适于使用深度学习算法进行处理。举个例子：预测一个人的健康状况，相关的参数会有年龄、职业、收入、家庭状况等各种元素，将这些元素打乱，并不会影响相关的结果。







## 经典网络

### 1. ResNet

#### 残差网络残差的作用

参考：

- [到底ResNet在解决一个什么问题呢？知乎热门回答](https://mp.weixin.qq.com/s/bofPG1Vm0RvnH2KaLXYw-Q)



#### 处理速度

#### 采用 concat 还是逐像素相加



### 2. DenseNet

#### 作用

- 由于前后层之间的Identity function，有效解决了梯度消失问题，并且强化了特征的重用和传播
- 相比ResNet输出通过相加的方式结合从而阻碍信息的传播，DN通过串联方式结合
- 串联要求特征图大小一致，故把pooling操作放在transition layer中
- 为防止靠后的串联层输入通道过多，引入bottleneck layer，即1x1卷积。文中把引入bottleneck layer的网络成为DenseNet-B
- 在transition layer中进一步压缩通道个数的网络成为DN-C（输入m个通道，则输出θm个通道，0<θ≤1）。同时包含bottleneck layer的和压缩过程的网络称为DN-BC





#### 采用 concat 还是逐像素相加





### 3. Xception 

#### 网络参数减少量，为什么可以达到高精度



#### Xception 的处理速度





### 4. Inception

#### Inception网络多层卷积之后是concat还是逐像素相加





### 5. 小型网络有哪些



### 6. 为什么现在的CNN模型都是在GoogleNet、VGGNet或者AlexNet上调整的？

- **评测对比**：为了让自己的结果更有说服力，在发表自己成果的时候会同一个标准的baseline及在baseline上改进而进行比较，常见的比如各种检测分割的问题都会基于VGG或者Resnet101这样的基础网络。
- **时间和精力有限**：在科研压力和工作压力中，时间和精力只允许大家在有限的范围探索。
- **模型创新难度大**：进行基本模型的改进需要大量的实验和尝试，并且需要大量的实验积累和强大灵感，很有可能投入产出比比较小。
- **资源限制**：创造一个新的模型需要大量的时间和计算资源，往往在学校和小型商业团队不可行。
- **在实际的应用场景中**，其实是有大量的非标准模型的配置。



### 7. 不同网络模型的模型大小、推测速度

来自https://github.com/jcjohnson/cnn-benchmarks

采用 16 的 batch，输入图片大小 224 x 224，运行显卡 8GB 显存的 GTX 1080

| Network                                                      | Layers | Top-1 error | Top-5 error | Speed (ms) | Citation                                                     |
| ------------------------------------------------------------ | ------ | ----------- | ----------- | ---------- | ------------------------------------------------------------ |
| [AlexNet](https://github.com/jcjohnson/cnn-benchmarks#alexnet) | 8      | 42.90       | 19.80       | 14.56      | [[1\]](https://github.com/jcjohnson/cnn-benchmarks#alexnet-paper) |
| [Inception-V1](https://github.com/jcjohnson/cnn-benchmarks#inception-v1) | 22     | -           | 10.07       | 39.14      | [[2\]](https://github.com/jcjohnson/cnn-benchmarks#inception-v1-paper) |
| [VGG-16](https://github.com/jcjohnson/cnn-benchmarks#vgg-16) | 16     | 27.00       | 8.80        | 128.62     | [[3\]](https://github.com/jcjohnson/cnn-benchmarks#vgg-paper) |
| [VGG-19](https://github.com/jcjohnson/cnn-benchmarks#vgg-19) | 19     | 27.30       | 9.00        | 147.32     | [[3\]](https://github.com/jcjohnson/cnn-benchmarks#vgg-paper) |
| [ResNet-18](https://github.com/jcjohnson/cnn-benchmarks#resnet-18) | 18     | 30.43       | 10.76       | 31.54      | [[4\]](https://github.com/jcjohnson/cnn-benchmarks#resnet-cvpr) |
| [ResNet-34](https://github.com/jcjohnson/cnn-benchmarks#resnet-34) | 34     | 26.73       | 8.74        | 51.59      | [[4\]](https://github.com/jcjohnson/cnn-benchmarks#resnet-cvpr) |
| [ResNet-50](https://github.com/jcjohnson/cnn-benchmarks#resnet-50) | 50     | 24.01       | 7.02        | 103.58     | [[4\]](https://github.com/jcjohnson/cnn-benchmarks#resnet-cvpr) |
| [ResNet-101](https://github.com/jcjohnson/cnn-benchmarks#resnet-101) | 101    | 22.44       | 6.21        | 156.44     | [[4\]](https://github.com/jcjohnson/cnn-benchmarks#resnet-cvpr) |
| [ResNet-152](https://github.com/jcjohnson/cnn-benchmarks#resnet-152) | 152    | 22.16       | 6.16        | 217.91     | [[4\]](https://github.com/jcjohnson/cnn-benchmarks#resnet-cvpr) |
| [ResNet-200](https://github.com/jcjohnson/cnn-benchmarks#resnet-200) | 200    | 21.66       | 5.79        | 296.51     | [[5\]](https://github.com/jcjohnson/cnn-benchmarks#resnet-eccv) |



来自：https://github.com/Cadene/pretrained-models.pytorch#evaluation-on-imagenet

| Model                                                        | Version                                                      | Acc@1  | Acc@5  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------ | ------ |
| PNASNet-5-Large                                              | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim) | 82.858 | 96.182 |
| [PNASNet-5-Large](https://github.com/Cadene/pretrained-models.pytorch#pnasnet) | Our porting                                                  | 82.736 | 95.992 |
| NASNet-A-Large                                               | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim) | 82.693 | 96.163 |
| [NASNet-A-Large](https://github.com/Cadene/pretrained-models.pytorch#nasnet) | Our porting                                                  | 82.566 | 96.086 |
| SENet154                                                     | [Caffe](https://github.com/hujie-frank/SENet)                | 81.32  | 95.53  |
| [SENet154](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 81.304 | 95.498 |
| PolyNet                                                      | [Caffe](https://github.com/CUHK-MMLAB/polynet)               | 81.29  | 95.75  |
| [PolyNet](https://github.com/Cadene/pretrained-models.pytorch#polynet) | Our porting                                                  | 81.002 | 95.624 |
| InceptionResNetV2                                            | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 80.4   | 95.3   |
| InceptionV4                                                  | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 80.2   | 95.3   |
| [SE-ResNeXt101_32x4d](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 80.236 | 95.028 |
| SE-ResNeXt101_32x4d                                          | [Caffe](https://github.com/hujie-frank/SENet)                | 80.19  | 95.04  |
| [InceptionResNetV2](https://github.com/Cadene/pretrained-models.pytorch#inception) | Our porting                                                  | 80.170 | 95.234 |
| [InceptionV4](https://github.com/Cadene/pretrained-models.pytorch#inception) | Our porting                                                  | 80.062 | 94.926 |
| [DualPathNet107_5k](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 79.746 | 94.684 |
| ResNeXt101_64x4d                                             | [Torch7](https://github.com/facebookresearch/ResNeXt)        | 79.6   | 94.7   |
| [DualPathNet131](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 79.432 | 94.574 |
| [DualPathNet92_5k](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 79.400 | 94.620 |
| [DualPathNet98](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 79.224 | 94.488 |
| [SE-ResNeXt50_32x4d](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 79.076 | 94.434 |
| SE-ResNeXt50_32x4d                                           | [Caffe](https://github.com/hujie-frank/SENet)                | 79.03  | 94.46  |
| [Xception](https://github.com/Cadene/pretrained-models.pytorch#xception) | [Keras](https://github.com/keras-team/keras/blob/master/keras/applications/xception.py) | 79.000 | 94.500 |
| [ResNeXt101_64x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext) | Our porting                                                  | 78.956 | 94.252 |
| [Xception](https://github.com/Cadene/pretrained-models.pytorch#xception) | Our porting                                                  | 78.888 | 94.292 |
| ResNeXt101_32x4d                                             | [Torch7](https://github.com/facebookresearch/ResNeXt)        | 78.8   | 94.4   |
| SE-ResNet152                                                 | [Caffe](https://github.com/hujie-frank/SENet)                | 78.66  | 94.46  |
| [SE-ResNet152](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 78.658 | 94.374 |
| ResNet152                                                    | [Pytorch](https://github.com/pytorch/vision#models)          | 78.428 | 94.110 |
| [SE-ResNet101](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 78.396 | 94.258 |
| SE-ResNet101                                                 | [Caffe](https://github.com/hujie-frank/SENet)                | 78.25  | 94.28  |
| [ResNeXt101_32x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext) | Our porting                                                  | 78.188 | 93.886 |
| FBResNet152                                                  | [Torch7](https://github.com/facebook/fb.resnet.torch)        | 77.84  | 93.84  |
| SE-ResNet50                                                  | [Caffe](https://github.com/hujie-frank/SENet)                | 77.63  | 93.64  |
| [SE-ResNet50](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 77.636 | 93.752 |
| [DenseNet161](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 77.560 | 93.798 |
| [ResNet101](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 77.438 | 93.672 |
| [FBResNet152](https://github.com/Cadene/pretrained-models.pytorch#facebook-resnet) | Our porting                                                  | 77.386 | 93.594 |
| [InceptionV3](https://github.com/Cadene/pretrained-models.pytorch#inception) | [Pytorch](https://github.com/pytorch/vision#models)          | 77.294 | 93.454 |
| [DenseNet201](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 77.152 | 93.548 |
| [DualPathNet68b_5k](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 77.034 | 93.590 |
| [CaffeResnet101](https://github.com/Cadene/pretrained-models.pytorch#caffe-resnet) | [Caffe](https://github.com/KaimingHe/deep-residual-networks) | 76.400 | 92.900 |
| [CaffeResnet101](https://github.com/Cadene/pretrained-models.pytorch#caffe-resnet) | Our porting                                                  | 76.200 | 92.766 |
| [DenseNet169](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 76.026 | 92.992 |
| [ResNet50](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 76.002 | 92.980 |
| [DualPathNet68](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 75.868 | 92.774 |
| [DenseNet121](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 74.646 | 92.136 |
| [VGG19_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 74.266 | 92.066 |
| NASNet-A-Mobile                                              | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim) | 74.0   | 91.6   |
| [NASNet-A-Mobile](https://github.com/veronikayurchuk/pretrained-models.pytorch/blob/master/pretrainedmodels/models/nasnet_mobile.py) | Our porting                                                  | 74.080 | 91.740 |
| [ResNet34](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 73.554 | 91.456 |
| [BNInception](https://github.com/Cadene/pretrained-models.pytorch#bninception) | Our porting                                                  | 73.524 | 91.562 |
| [VGG16_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 73.518 | 91.608 |
| [VGG19](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 72.080 | 90.822 |
| [VGG16](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 71.636 | 90.354 |
| [VGG13_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 71.508 | 90.494 |
| [VGG11_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 70.452 | 89.818 |
| [ResNet18](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 70.142 | 89.274 |
| [VGG13](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 69.662 | 89.264 |
| [VGG11](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 68.970 | 88.746 |
| [SqueezeNet1_1](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 58.250 | 80.800 |
| [SqueezeNet1_0](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 58.108 | 80.428 |
| [Alexnet](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 56.432 | 79.194 |







## 卷积神经网络（CNN）

### 1. 膨胀卷积的感受野如何计算



### 2. 卷积神经网络耗时的地方



### 3. 手推卷积神经网络公式，卷积

传统卷积运算是将卷积核以滑动窗口的方式在输入图上滑动，当前窗口内对应元素相乘然后求和得到结果，一个窗口一个结果。**相乘然后求和恰好也是向量内积的计算方式**，所以可以将每个窗口内的元素拉成向量，通过向量内积进行运算，多个窗口的向量放在一起就成了矩阵，每个卷积核也拉成向量，多个卷积核的向量排在一起也成了矩阵，于是，卷积运算转化成了矩阵乘法运算。下图很好地演示了矩阵乘法的运算过程：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/im2col.jpg" alt="im2col" style="zoom:80%;" />

将卷积运算转化为矩阵乘法，从乘法和加法的运算次数上看，两者没什么差别，但是转化成矩阵后，运算时需要的数据被存在连续的内存上，这样**访问速度大大提升（cache）**，同时，矩阵乘法有很多库提供了高效的实现方法，像BLAS、MKL等，**转化成矩阵运算后可以通过这些库进行加速**。

缺点呢？**这是一种空间换时间的方法，消耗了更多的内存——转化的过程中数据被冗余存储**。



还有两张形象化的图片帮助理解：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/im2col_2.jpg" alt="这里写图片描述" style="zoom:50%;" />

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/im2col_3.png" alt="这里写图片描述" style="zoom:50%;" />

对于输出尺寸的计算：
$$
output = \lfloor \frac{input-kernel + 2*pad}{stride} \rfloor + 1
$$
其中：

input: 输入图像的尺寸，比如宽和高 w，h

output：输出图像的尺寸

kernel：卷积核的尺寸

pad：是否补零

stride：步长

对于不同补零方式的值：

- VALID：pad=0，$output=\frac{input-kernel}{stride}+1$
- SAME：输入和输出尺寸一样，即 input=output，所以 $pad=\frac{(output-1)*stride+kernel-input}{2}$
- FULL：pad=kernel-1，$output=\frac{input+kernel-2}{stride}+1$



#### 代码实现

1. 滑动窗口版本

```python
#!/usr/bin/env python3    #加上这一句之后，在终端命令行模式下就可以直接输入这个文件的名字后运行文件中的代码
# -*- coding = utf-8 -*-
import numpy as np

# 为了简化运算，默认batch_size = 1
class my_conv(object):
    def __init__(self, input_data, weight_data, stride, padding = 'SAME'):
        self.input = np.asarray(input_data, np.float32)
        self.weights = np.asarray(weight_data, np.float32)
        self.stride = stride
        self.padding = padding
        
    def my_conv2d(self):
        """
        self.input: c * h * w  # 输入的数据格式
        self.weights: c * h * w
        """
        [c, h, w] = self.input.shape
        [kc, k, _] = self.weights.shape  # 这里默认卷积核的长宽相等
        assert c == kc  # 如果输入的channel与卷积核的channel不一致即报错
        output = []
        # 分通道卷积，最后再加起来
        for i in range(c):  
            f_map = self.input[i]
            kernel = self.weights[i]
            rs = self.compute_conv(f_map, kernel)
            if output == []:
                output = rs
            else:
                output += rs
        return output
      
    def compute_conv(self, fm, kernel):
        [h, w] = fm.shape
        [k, _] = kernel.shape

        if self.padding == 'SAME':
            pad_h = (self.stride * (h - 1) + k - h) // 2
            pad_w = (self.stride * (w - 1) + k - w) // 2
            rs_h = h
            rs_w = w
        elif self.padding == 'VALID':
            pad_h = 0
            pad_w = 0
            rs_h = (h - k) // self.stride + 1
            rs_w = (w - k) // self.stride + 1
        elif self.padding == 'FULL':
            pad_h = k - 1
            pad_w = k - 1
            rs_h = (h + 2 * pad_h - k) // self.stride + 1
            rs_w = (w + 2 * pad_w - k) // self.stride + 1
        padding_fm = np.zeros([h + 2 * pad_h, w + 2 * pad_w], np.float32)
        padding_fm[pad_h:pad_h+h, pad_w:pad_w+w] = fm  # 完成对fm的zeros padding
        rs = np.zeros([rs_h, rs_w], np.float32)

        for i in range(rs_h):
            for j in range(rs_w):
                roi = padding_fm[i*self.stride:(i*self.stride + k), j*self.stride:(j*self.stride + k)]
                rs[i, j] = np.sum(roi * kernel) # np.asarray格式下的 * 是对应元素相乘
        return rs

if __name__=='__main__':
    input_data = [
        [
            [1, 0, 1, 2, 1],
            [0, 2, 1, 0, 1],
            [1, 1, 0, 2, 0],
            [2, 2, 1, 1, 0],
            [2, 0, 1, 2, 0],
        ],
        [
            [2, 0, 2, 1, 1],
            [0, 1, 0, 0, 2],
            [1, 0, 0, 2, 1],
            [1, 1, 2, 1, 0],
            [1, 0, 1, 1, 1],

        ],
    ]
    weight_data = [
        [
            [1, 0, 1],
            [-1, 1, 0],
            [0, -1, 0],
        ],
        [
            [-1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    ]
    conv = my_conv(input_data, weight_data, 1, 'SAME')
    print(conv.my_conv2d())
```



2.矩阵乘法版本

```python
#!/usr/bin/env python3    #加上这一句之后，在终端命令行模式下就可以直接输入这个文件的名字后运行文件中的代码
# _*_ coding = utf-8 _*_
import numpy as np

# 为了简化运算，默认batch_size = 1
class my_conv(object):
    def __init__(self, input_data, weight_data, stride, padding = 'SAME'):
        self.input = np.asarray(input_data, np.float32)
        self.weights = np.asarray(weight_data, np.float32)
        self.stride = stride
        self.padding = padding
    def my_conv2d(self):
        """
        self.input: c * h * w  # 输入的数据格式
        self.weights: c * h * w
        """
        [c, h, w] = self.input.shape
        [kc, k, _] = self.weights.shape  # 这里默认卷积核的长宽相等
        assert c == kc  # 如果输入的channel与卷积核的channel不一致即报错
        # rs_h与rs_w为最后输出的feature map的高与宽
        if self.padding == 'SAME':
            pad_h = (self.stride * (h - 1) + k - h) // 2
            pad_w = (self.stride * (w - 1) + k - w) // 2
            rs_h = h
            rs_w = w
        elif self.padding == 'VALID':
            pad_h = 0
            pad_w = 0
            rs_h = (h - k) // self.stride + 1
            rs_w = (w - k) // self.stride + 1
        elif self.padding == 'FULL':
            pad_h = k - 1
            pad_w = k - 1
            rs_h = (h + 2 * pad_h - k) // self.stride + 1
            rs_w = (w + 2 * pad_w - k) // self.stride + 1
        # 对输入进行zeros padding，注意padding后依然是三维的
        pad_fm = np.zeros([c, h+2*pad_h, w+2*pad_w], np.float32)
        for i in range(c):
            pad_fm[i, pad_h:pad_h+h, pad_w:pad_w+w] = self.input[i]
        # 将输入和卷积核转化为矩阵相乘的规格
        mat_fm = np.zeros([rs_h*rs_w, kc*k*k], np.float32)
        mat_kernel = self.weights
        mat_kernel.shape = (kc*k*k, 1) # 转化为列向量
        row = 0   
        for i in range(rs_h):
            for j in range(rs_w):
                roi = pad_fm[:, i*self.stride:(i*self.stride+k), j*self.stride:(j*self.stride+k)]
                mat_fm[row] = roi.flatten()  # 将roi扁平化，即变为行向量
                row += 1
        # 卷积的矩阵乘法实现
        rs = np.dot(mat_fm, mat_kernel).reshape(rs_h, rs_w) 
        return rs

if __name__=='__main__':
    input_data = [
        [
            [1, 0, 1, 2, 1],
            [0, 2, 1, 0, 1],
            [1, 1, 0, 2, 0],
            [2, 2, 1, 1, 0],
            [2, 0, 1, 2, 0],
        ],
        [
            [2, 0, 2, 1, 1],
            [0, 1, 0, 0, 2],
            [1, 0, 0, 2, 1],
            [1, 1, 2, 1, 0],
            [1, 0, 1, 1, 1],

        ],
    ]
    weight_data = [
        [
            [1, 0, 1],
            [-1, 1, 0],
            [0, -1, 0],
        ],
        [
            [-1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    ]
    conv = my_conv(input_data, weight_data, 1, 'SAME')
    print(conv.my_conv2d())
```







### 4. 反向传播过程





### 5. 讲解一下卷积神经网络发展历史（LeNet——VGG——Inception——ResNet——DenseNet——SENet）



### 6. 对于 conv，pool层的操作本质

参考：

- [如何评价 MSRA 最新的 Deformable Convolutional Networks？](https://www.zhihu.com/question/57493889/answer/153369805)
- [【VALSE 前沿技术选介17-02期】可形变的神经网络](https://mp.weixin.qq.com/s/Ulu8Kw4FDty-dMOu7qNxxQ?)



来自来自地平线的ALAN Huang同学在知乎上给出了一个很精辟的回答（https://www.zhihu.com/question/57493889/answer/153369805）。 在这里跟大家分享，同时加入个人的一些comments。

> *conv，pooling这种操作，其实可以分成三阶段： indexing（im2col） ，reduce(sum), reindexing（col2im). 在每一阶段都可以做一些事情。 用data driven的方式去学每一阶段的参数，也是近些年的主流方向。*

来自：[【VALSE 前沿技术选介17-02期】可形变的神经网络](https://mp.weixin.qq.com/s/Ulu8Kw4FDty-dMOu7qNxxQ?)

个人认为，其实可以更细分为**四个阶段**，每个阶段其实都值得深入思考：

1. **Indexing (im2col)**：这也就是本篇文章关注的部分。
2. **Computation (gemm)**：在im2col之后，conv就被转化为了一个dense matrix multiplication的问题。本质上，conv还是一个线性模型就是因为在这一步还是一个线性变化。有若干工作试图增强计算步骤的表示能力。从最开始的Network In Network到后来的Neural Decision Forest，再到最近我们的Factorized Bilinear Layer，都是在这一步试图做出一些变化。
3. **Reduce (sum)**：最简单的reduce操作就是求和，但是这个步骤还是有大量变化的余地。例如，是否可以通过类似于attention一样的机制做加权求和？是否可以通过random projection引入随机性？
4. **Reindex (col2im)**：这步骤是第一步的逆操作。



### 7. 如何计算 FLOPs 和参数量

参考文章：

- [CNN 模型所需的计算力（flops）和参数（parameters）数量是怎么计算的？](https://www.zhihu.com/question/65305385)



#### FLOPs

首先是给出两个定义：

- **FLOPS**：全大写，是floating point operations per second的缩写，**意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标**。
- **FLOPs**：注意 s 小写，是floating point operations的缩写（s表复数），**意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。**

这里说的自然是第二种 FLOPs，计算量，也就是模型的复杂度。



##### 卷积层的 FLOPs

不考虑激活函数，对于单个输入特征图的计算公式为（没有考虑 batch ）:
$$
(2\times C_i \times K^2-1)\times H \times W\times C_o
$$
这里每个参数的含义：$C_i$ 是输入通道数量， K 表示卷积核的大小，H 和 W 是输出特征图(feature map)的大小，$C_o$ 是输出通道。

因为是乘法和加法，所以括号内是 2 ，表示两次运算操作。另外，不考虑 bias 的时候，有 -1，而考虑 bias 的时候是没有 -1。

对于括号内的理解是这样的：
$$
(2\times C_i \times K^2-1)=(C_i\times K^2)+(C_i\times K^2-1)
$$


第一项是乘法的运算数量，第二项是加法运算数量，因为 n 个数相加，是执行 n-1 次的加法次数，如果考虑 bias，就刚好是 n 次，也就是变成 $(2\times C_i \times K^2)$



对于整个公式来说就是分两步计算：

1. 括号内是计算得到输出特征图的一个像素的数值；
2. 括号外则是乘以整个输出特征图的大小，拓展到整个特征图。

举个例子，如下图所示是一个输出特征图的计算，其中输入特征图是 $5*5 $，卷积核是 $3*3$，输出的特征图大小也是 $3*3$，所以这里对应公式中的参数，就是 K=3, H=W=3, 假设输入和输出通道数量都是 1，那么下图得到右边的特征图的一个像素的数值的计算量就是 $(3*3)次乘法+（3  *3-1）次加法 = 17$，然后得到整个输出特征图的计算量就是 $17*  9 = 153$.

![image](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/convolution_schematic.gif)

##### 深度可分离卷积的FLOPs

深度可分离卷积分成两部分，一部分是分通道卷积，另一部分是1*1卷积。（如果不知道深度可分离卷积的朋友可以先看下[这个博客](https://baijiahao.baidu.com/s?id=1634399239921135758&wfr=spider&for=pc)，这是一种可大大减少计算量的卷积方法）

这里的讨论以考虑bias为准：
第一部分：$(2*k^2 )*H*W*C_{int}$
第二部分：$2*C_{int}*H*W*C_{out}$

最终的结果就是两部分相加。



##### 全连接层的 FLOPs

计算公式为：
$$
(2\times I-1)\times O
$$
每个参数的含义：I 是输入数量，O 是输出数量。

同样 2 也是表示乘法和加法，然后不考虑 bias 是 -1，考虑的时候没有 -1。

对于这个公式也是和卷积层的一样，括号内考虑一个输出神经元的计算量，然后扩展到所有的输出神经元.

##### 池化层的FLOPS

这里又分为全局池化和一般池化两种情况：

###### 全局池化

针对输入所有值进行一次池化操作，不论是max、sum还是avg，都可以简单地看做是只需要对每个值算一次。

所以结果为：$H_{int}*W_{int}*C_{int}$

###### 一般池化

答案是：$k^2*H_{out}*W_{out}*C_{out}$

注意池化层的：$C_{out} = C_{int}$

##### 激活层的FLOPs

###### ReLU

ReLU一般都是跟在卷积层的后面，这里假设卷积层的输出为$H*W*C$，因为ReLU函数的计算只涉及到一个判断，因此计算量就是$H*W*C$

###### sigmoid

根据sigmoid的公式可以知道，每个输入都需要经历4次运算，因此计算量是$H*W*C*4$（参数含义同ReLU）



##### 相关实现代码库

GitHub 上有几个实现计算模型的 FLOPs 的库：

- https://github.com/Lyken17/pytorch-OpCounter
- https://github.com/sagartesla/flops-cnn
- https://github.com/sovrasov/flops-counter.pytorch

非常简单的代码实现例子，来自 https://github.com/sagartesla/flops-cnn/blob/master/flops_calculation.py

```python
input_shape = (3 ,300 ,300) # Format:(channels, rows,cols)
conv_filter = (64 ,3 ,3 ,3)  # Format: (num_filters, channels, rows, cols)
stride = 1
padding = 1
activation = 'relu'

if conv_filter[1] == 0:
    n = conv_filter[2] * conv_filter[3] # vector_length
else:
    n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length

flops_per_instance = n + ( n -1)    # general defination for number of flops (n: multiplications and n-1: additions)

num_instances_per_filter = (( input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # for rows
num_instances_per_filter *= ((input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # multiplying with cols

flops_per_filter = num_instances_per_filter * flops_per_instance
total_flops_per_layer = flops_per_filter * conv_filter[0]  # multiply with number of filters

if activation == 'relu':
    # Here one can add number of flops required
    # Relu takes 1 comparison and 1 multiplication
    # Assuming for Relu: number of flops equal to length of input vector
    total_flops_per_layer += conv_filter[0] * input_shape[1] * input_shape[2]


if total_flops_per_layer / 1e9 > 1:   # for Giga Flops
    print(total_flops_per_layer/ 1e9 ,'{}'.format('GFlops'))
else:
    print(total_flops_per_layer / 1e6 ,'{}'.format('MFlops'))
```



#### 参数量

##### 卷积层的参数量

卷积层的参数量与输入特征图大小无关

考虑bias：$(k^2*C_{int}+1)*C_{out}$
不考虑bias：$(k^2*C_{int})*C_{out}$

##### 深度可分离卷积的参数量

不考虑bias：
第一部分：$k^2*C_{int}$
第二部分：$(1*1*C_{int})*C_{out}$
最终结果为两者相加。

##### 池化层的参数量

池化层没有需要学习的参数，所以参数量为0。

##### 全连接层的参数量

考虑bias：$I*O+1$





### 8. 网络结构的优化方式（scale up）有哪些，各自的优缺点？







### 9. 残差指的是什么？



![image](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/resnet_%E6%AE%8B%E5%B7%AE%E5%AD%A6%E4%B9%A0%E5%8D%95%E5%85%83.png)

其中ResNet提出了两种 `mapping`，分别对应公式 `y=F(x) + x` ：

- **identity mapping**，指的就是上图中”弯弯的曲线”，`identity mapping` 顾名思义，就是指本身，也就是公式中的x，
- 另一种 **residual mapping**，指的就是除了”弯弯的曲线“那部分，也就是 `y−x`，所以残差指的就是`F(x`)部分。



### 10. 为什么ResNet可以解决“随着网络加深，准确率不下降”的问题？

理论上，对于“随着网络加深，准确率下降”的问题，Resnet提供了两种选择方式，也就是`identity mapping`和`residual mapping`，**如果网络已经到达最优，继续加深网络，residual mapping将被push为0，只剩下identity mapping，这样理论上网络一直处于最优状态了，网络的性能也就不会随着深度增加而降低了。**



### 11. 1*1 卷积的作用

NIN(Network in Network)是第一篇探索$1\times 1$卷积核的论文，这篇论文通过在卷积层中使用MLP替代传统线性的卷积核，使单层卷积层内具有非线性映射的能力，也因其网络结构中嵌套MLP子网络而得名NIN。NIN对不同通道的特征整合到MLP自网络中，**让不同通道的特征能够交互整合，使通道之间的信息得以流通**，其中的MLP子网络恰恰可以用$1\times 1$的卷积进行代替。



GoogLeNet 则采用 $1\times 1$ 卷积核来减少模型的参数量。在原始版本的Inception模块中，由于每一层网络采用了更多的卷积核，大大增加了模型的参数量。此时在每一个较大卷积核的卷积层前引入 $1\times 1$卷积，可以通过分离通道与宽高卷积来减少模型参数量。

以下图为例，在不考虑参数偏置项的情况下，若输入和输出的通道数为 $C_1=16$ ，则左半边网络模块所需的参数为$(1\times 1+3\times 3+5\times 5+0)\times C_1 \times C_1=8960$；假定右半边网络模块采用的 $1\times 1$ 卷积通道数为 $C_2=8$ (满足$C_1 > C_2$)，则右半部分的网络结构所需参数量为 $(1\times 1\times(3C_1+C_2)+3\times 3\times C_2+5\times 5\times C_2)\times C_1=5248$，可以在不改变模型表达能力的前提下大大减少所使用的参数量。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/inception_modules.png" alt="image" style="zoom:75%;" />



综上所述，$1\times 1$卷积的作用主要为以下两点：

- 实现信息的跨通道交互和整合。
- 对卷积核通道数进行降维和升维，减小参数量。



### 12. 卷积层和池化层有什么区别？

卷积层和池化层在结构上具有一定的相似性，都是对感受域内的特征进行提取，并且根据步长设置获取到不同维度的输出，但是其内在操作是有本质区别的，如表所示。

|            | 卷积层                                 | 池化层                           |
| ---------- | -------------------------------------- | -------------------------------- |
| **结构**   | 零填充时输出维度不变，而通道数改变     | 通常特征维度会降低，通道数不变   |
| **稳定性** | 输入特征发生细微改变时，输出结果会改变 | 感受域内的细微变化不影响输出结果 |
| **作用**   | 感受域内提取局部关联特征               | 感受域内提取泛化特征，降低维度   |
| **参数量** | 与卷积核尺寸、卷积核个数相关           | 不引入额外参数                   |



### 13. 如何减少卷积层的参数量

减少卷积层参数量的方法可以简要地归为以下几点：

- 使用堆叠小卷积核代替大卷积核：VGG网络中2个$3\times 3$的卷积核可以代替1个的$5\times 5$卷积核
- 使用分离卷积操作：将原本 $K\times K\times C$的卷积操作分离为 $K\times K\times 1$和 $1\times 1\times C$的两部分操作
- 添加 $1\times 1$的卷积操作：与分离卷积类似，但是通道数可变，在 $K\times K\times C_1$卷积前添加 $1\times 1\times C_2$的卷积核（满足 $C_2 < C_1$）
- 在卷积层前使用池化操作：池化可以降低卷积层的输入特征维度



### 14. CNN 凸显共性的方法

#### 14.1 局部连接

感受野，即每个神经元仅与输入神经元相连接的一块区域。

在图像卷积操作中，神经元在空间维度上是局部连接，但在深度上是全连接。局部连接的思想，是受启发于生物学里的视觉系统结构，视觉皮层的神经元就是仅用局部接受信息。对于二维图像，**局部像素关联性较强**。**这种局部连接保证了训练后的滤波器能够对局部特征有最强的响应，使神经网络可以提取数据的局部特征**；

下图是一个很经典的图示，左边是全连接，右边是局部连接。

![image](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/%E5%B1%80%E9%83%A8%E8%BF%9E%E6%8E%A5.png)



对于一个1000 × 1000的输入图像而言，如果下一个隐藏层的神经元数目为10^6个，采用全连接则有1000 × 1000 × 10^6 = 10^12个权值参数，如此巨大的参数量几乎难以训练；

而采用局部连接，隐藏层的每个神经元仅与图像中10 × 10的局部图像相连接，那么此时的权值参数数量为10 × 10 × 10^6 = 10^8，将直接减少4个数量级。



#### 14.2 权值共享

**权值共享，即计算同一深度的神经元时采用的卷积核参数是共享的**。权值共享在一定程度上讲是有意义的，是由于在神经网络中，提取的底层边缘特征与其在图中的位置无关。但是在另一些场景中是无意的，如在人脸识别任务，我们期望在不同的位置学到不同的特征。

需要注意的是，**权重只是对于同一深度切片的神经元是共享的**。在卷积层中，通常采用多组卷积核提取不同的特征，即对应的是不同深度切片的特征，**而不同深度切片的神经元权重是不共享**。

相反，偏置这一权值对于同一深度切片的所有神经元都是共享的。

**权值共享带来的好处是大大降低了网络的训练难度**。如下图，假设在局部连接中隐藏层的每一个神经元连接的是一个10 × 10的局部图像，因此有10 × 10个权值参数，将这10 × 10个权值参数共享给剩下的神经元，也就是说隐藏层中10^6个神经元的权值参数相同，那么此时不管隐藏层神经元的数目是多少，需要训练的参数就是这 10 × 10个权值参数（也就是卷积核的大小）。

![image](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/%E6%9D%83%E5%80%BC%E5%85%B1%E4%BA%AB.png)



#### 14.3 池化操作

池化操作与多层次结构一起，实现了数据的降维，将低层次的局部特征组合成为较高层次的特征，从而对整个图片进行表示。如下图：

<img src="images/%E6%B1%A0%E5%8C%96%E6%93%8D%E4%BD%9C.png" alt="池化操作.png" style="zoom:75%;" />

### 15. 局部卷积的作用

并不是所有的卷积都会进行权重共享，在某些特定任务中，会使用不权重共享的卷积。下面通过人脸这一任务来进行讲解。在读人脸方向的一些paper时，会发现很多都会在最后加入一个 Local Connected Conv，也就是不进行权重共享的卷积层。总的来说，这一步的作用就是使用3D模型来将人脸对齐，从而使CNN发挥最大的效果。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/%E5%B1%80%E9%83%A8%E5%8D%B7%E7%A7%AF.png" alt="image" style="zoom:75%;" />



截取论文中的一部分图，经过3D对齐以后，形成的图像均是152×152，输入到上述的网络结构中。该结构的参数如下：

Conv：32个11×11×3的卷积核，

Max-pooling: 3×3，stride=2，

Conv: 16个9×9的卷积核，

Local-Conv: 16个9×9的卷积核，

Local-Conv: 16个7×7的卷积核，

Local-Conv: 16个5×5的卷积核，

Fully-connected: 4096维，

Softmax: 4030维。



前三层的目的在于提取低层次的特征，比如简单的边和纹理。其中Max-pooling层使得卷积的输出对微小的偏移情况更加鲁棒。但不能使用更多的Max-pooling层，因为太多的Max-pooling层会使得网络损失图像信息。全连接层将上一层的每个单元和本层的所有单元相连，用来捕捉人脸图像不同位置特征之间的相关性。最后使用softmax层用于人脸分类。

中间三层都是使用参数不共享的卷积核，之所以使用参数不共享，有如下原因：

1. 对齐的人脸图片中，**不同的区域会有不同的统计特征**，因此并不存在特征的局部稳定性，所以使用相同的卷积核会导致信息的丢失。
2. 不共享的卷积核并不增加inference时特征的计算量，**仅会增加训练时的计算量**。使用不共享的卷积核，由于需要训练的参数量大大增加，因此往往需要通过其他方法增加数据量。



### 16. 空洞卷积及其优点

- pooling 操作虽然能增大感受野，但是会丢失一些信息。空洞卷积在卷积核中插入权重为0的值，因此每次卷积中会skip掉一些像素点
- 空洞卷积**增大了卷积输出每个点的感受野，并且不像pooling会丢失信息**，在**图像需要全局信息或者需要较长sequence依赖的语音序列问题**上有着较广泛的应用



### 17. 卷积层和全连接层的区别（or CNN 在图像上表现很好的原因）

CNN 在图像上表现很好的原因，实际上就是问卷积层和全连接层的区别。

卷积层相比于全连接层，主要有两个特点：

1. **局部连接：**全连接层是一种稠密连接方式，而卷积层却只使用卷积核对局部进行处理，这种处理方式其实也刚好对应了图像的特点。在视觉识别中，关键性的图像特征、边缘、角点等只占据了整张图像的一小部分，相隔很远的像素之间存在联系和影响的可能性是很低的，**而局部像素具有很强的相关性**。
2. **共享参数：**如果借鉴全连接层的话，对于1000×1000大小的彩色图像，一层全连接层便对应于三百万数量级维的特征，即会导致庞大的参数量，不仅计算繁重，还会导致过拟合。而卷积层中，卷积核会与局部图像相互作用，**是一种稀疏连接，大大减少了网络的参数量**。另外从直观上理解，依靠卷积核的滑动去提取图像中不同位置的相同模式也刚好符合图像的特点，不同的卷积核提取不同的特征，组合起来后便可以提取到高级特征用于最后的识别检测了。



### 18. **CNN最成功的应用是在CV，那为什么NLP和Speech的很多问题也可以用CNN解出来？为什么AlphaGo里也用了CNN？这几个不相关的问题的相似性在哪里？CNN通过什么手段抓住了这个共性？**

参考：

- [深度学习岗位面试问题整理笔记](https://zhuanlan.zhihu.com/p/25005808)



1. 以上几个不相关问题的相关性在于，都存在**局部与整体的关系**，由低层次的特征经过组合，组成高层次的特征，并且得到不同特征之间的空间相关性。
2. **CNN抓住此共性的手段主要有四个：局部连接／权值共享／池化操作／多层次结构。**

- **局部连接使网络可以提取数据的局部特征**；
- 权值共享大大降低了网络的训练难度，一个Filter只提取一个特征，在整个图片（或者语音／文本） 中进行卷积；
- 池化操作与多层次结构一起，实现了数据的降维，将低层次的局部特征组合成为较高层次的特征，从而对整个图片进行表示。







------

## RNN&LSTM

### 1. **CNN**和**RNN**的区别 ？

| 类别   | 特点描述                                                     |
| ------ | ------------------------------------------------------------ |
| 相同点 | 1、传统神经网络的扩展。<br />2、前向计算产生结果，反向计算模型更新。<br />3、每层神经网络横向可以多个神经元共存,纵向可以有多层神经网络连接。 |
| 不同点 | 1、CNN空间扩展，神经元与特征卷积；RNN时间扩展，神经元与多个时间输出计算<br />2、RNN可以用于描述时间上连续状态的输出，有记忆功能，CNN用于静态输出 |



### 2. RNNs和FNNs有什么区别？

1. 不同于传统的前馈神经⽹络(FNNs)，RNNs 引⼊了定向循环，**能够处理输⼊之间前后**

**关联问题**。

2. RNNs 可以**记忆之前步骤的训练信息**。 定向循环结构如下图所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/RNN_3.png" style="zoom:80%;" />

### 3. RNNs 训练和传统 ANN 训练异同点？

相同点：

1. RNNs与传统ANN都使⽤BP（Back Propagation）误差反向传播算法。

不同点：

1. RNNs ⽹络参数W,U,V是共享的，⽽传统神经⽹络各层参数间没有直接联系。

2. 对于RNNs，在使⽤梯度下降算法中，每⼀步的输出不仅依赖当前步的⽹络，还依赖

于之前若⼲步的⽹络状态。





### 4. 为什么 RNN 训练的时候 Loss 波动很⼤

由于 RNN 特有的 memory 会影响后期其他的RNN的特点，梯度时⼤时⼩，learning rate 没法个性化的调整，导致RNN在train的过程中，Loss会震荡起伏。

为了解决RNN的这个问题，**在训练的时候，可以设置临界值，当梯度⼤于某个临界值，直接截断**，⽤这个临界值作为梯度的⼤⼩，防⽌⼤幅震荡。



### 5. RNN中为什么会出现梯度消失？

首先来看tanh函数的函数及导数图如下所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/RNN_tanh.png" style="zoom:80%;" />

sigmoid 函数及其导数图如下所示：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/RNN_sigmoid.png)

从上图观察可知，sigmoid函数的导数范围是(0,0.25]，tanh函数的导数范围是(0,1]，他们的导数最大都不大于1。

由前向输出的公式推导可知，RNN的激活函数是嵌套在里面的，如果选择激活函数为$tanh$或$sigmoid$，把激活函数放进去，拿出中间累乘的那部分可得：
$$
\prod_{j=k+1}^{t}{\frac{\partial{h^{j}}}{\partial{h^{j-1}}}} = \prod_{j=k+1}^{t}{tanh^{'}}\cdot W_{s}
$$

$$
\prod_{j=k+1}^{t}{\frac{\partial{h^{j}}}{\partial{h^{j-1}}}} = \prod_{j=k+1}^{t}{sigmoid^{'}}\cdot W_{s}
$$

**梯度消失现象**：基于上式，会发现累乘会导致激活函数导数的累乘，如果取 tanh 或 sigmoid 函数作为激活函数的话，那么必然是一堆小数在做乘法，**结果就是越乘越小**。随着时间序列的不断深入，小数的累乘就会导致梯度越来越小直到接近于0，这就是“梯度消失“现象。

实际使用中，会优先选择tanh函数，**原因是 tanh 函数相对于 sigmoid 函数来说梯度较大，收敛速度更快且引起梯度消失更慢**。



### 6. 如何解决**RNN**中的梯度消失问题？

梯度消失是在⽆限的利⽤历史数据⽽造成，但是 RNN 的特点本来就是能利⽤历史数据获取更多的可利⽤信息，解决RNN中的梯度消失⽅法主要有：

1. **选取更好的激活函数**，如 Relu激活函数。ReLU函数的左侧导数为0，右侧导数恒为1，这就避免了“梯度消失“的发⽣。**但恒为1的导数容易导致“梯度爆炸“，但设定合适的阈值可以解决这个问题**。
2. **加⼊BN层**，其优点包括可加速收敛、控制过拟合，可以少⽤或不⽤ Dropout 和正则、降低⽹络对初始化权重不敏感，且能允许使⽤较⼤的学习率等。
3. **改变传播结构**，LSTM 结构可以有效解决这个问题。



### 7. RNNs 在 NLP 中的典型应用

**（1）语言模型与文本生成(Language Modeling and Generating Text)**

给定一组单词序列，需要根据前面单词预测每个单词出现的可能性。语言模型能够评估某个语句正确的可能性，可能性越大，语句越正确。另一种应用便是使用生成模型预测下一个单词的出现概率，从而利用输出概率的采样生成新的文本。

**（2）机器翻译(Machine Translation)**

机器翻译是将一种源语言语句变成意思相同的另一种源语言语句，如将英语语句变成同样意思的中文语句。与语言模型关键的区别在于，需要将源语言语句序列输入后，才进行输出，即输出第一个单词时，便需要从完整的输入序列中进行获取。

**（3）语音识别(Speech Recognition)**

语音识别是指给定一段声波的声音信号，预测该声波对应的某种指定源语言语句以及计算该语句的概率值。 

**（4）图像描述生成 (Generating Image Descriptions)**

同卷积神经网络一样，RNNs已经在对无标图像描述自动生成中得到应用。CNNs与RNNs结合也被应用于图像描述自动生成。



### 8. 为什么 LSTM 在梯度消失上表示更好？

参考：

- [为什么相比于RNN，LSTM在梯度消失上表现更好？](https://www.zhihu.com/question/44895610)
- [为什么LSTM会减缓梯度消失？](https://zhuanlan.zhihu.com/p/109519044)



在 RNN 中，之所以会导致梯度消失的原因主要也是 RNN 会用同一套的权值矩阵，从反向传播来看：原始 RNN 中，隐层向量和输出计算如下：
$$
h_t = tanh(W_Ix_t+W_Rh_{t-1})\\
y_t = tanh(W_Oh_t)
$$
训练采用 BPTT 来计算梯度并更新参数，即 $E_t$ 对 $W_R$ 的梯度，整体的导数是时间 t 从 0 到 t-1 的梯度之和。对于时刻 t，利用链式求导法则有：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/rnn_equation1.png)

对于 $\frac{\partial h_t}{\partial h_i}$ ，再次利用链式求导法则，有：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/rnn_equation2.png)

接下来我们先看其中一项 $h_{k+1}$ 对  $h_k$ 的梯度计算：(diag为对角矩阵)

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/rnn_equation3.png)

那么如果要计算 $h_k$ 对于 $h_1$ 的梯度：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/rnn_equation4.png)

按照论文[https://arxiv.org/pdf/1211.5063.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1211.5063.pdf) 中所述:

- 如果 $W_R$ 的主本征值大于1，那么会发生梯度爆炸；
- 如果小于1，会出现梯度消失。

通过公式可以直观感受到梯度消失或者爆炸。要知道 $f^\prime$ 这个函数值是一直小于 1 的（p.s.其实取决于激活函数的选择，对于sigmoid来说确实是）,如果  $W_R$ 的值比较小，那么这个梯度不可避免的要趋于0。小于1的数连乘会压倒   $W_R$ 的连乘。相反的，如果  $W_R$  的值太大，梯度会不可避免的趋于无穷大，因为此时连乘的 $W_R$  会压倒小于1的数的连乘。

在应用中，梯度消失会更常见，所以我们需要更加关注它。（p.s.对于梯度爆炸，我们可以采用clip解决，一般问题不大）。而如果发现梯度消失，更早的隐层状态对于后面的隐层状态的影响几乎很小，也就是没有捕捉到长距离依赖。

根据上述推导，造成梯度消失的根因就是递归计算梯度 $\frac{\partial h_t}{\partial h_i}$，也就是只有这个梯度不趋于 0 也不趋于无穷大的时候，才可以学习到长距离依赖。

LSTM 的解决方法就是给这个递归的梯度增加了一个常量，即引入了一个 cell 状态 $C_t$。其计算方式为：
$$
C_t = f_t * C_{t-1} + i_t * \hat C_t
$$
同样是通过链式求导法则，可以得到：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/rnn_equation5.png)

简化一下：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/rnn_equation6.png)

所以如果要计算 t 时刻的，只需要将上式相乘 t 次即可。在递归梯度计算上，LSTM 和原始 RNN 最大的不同之处，RNN 的梯度 $\frac{\partial h_t}{\partial h_i}$ 是一直大于 1 或者在 [0, 1] 区间，这会导致梯度爆炸或者消失。而 LSTM 中则是可以大于 1 也可以在 [0, 1] 之间，也就是当考虑无穷大的时候，LSTM 不会趋于 0 或者无穷大。当要收敛到 0 时，可以增大遗忘门 $f_t$，使得 $\frac{\partial C_t}{\partial C_{t-1}}$ 的值拉向 1，这就可以减缓梯度的消失。

最重要的就是 $f_t, i_t, o_t, \hat C_t$ 都是神经网络自己学习的，即可以学到改变门控的值来决定什么时候遗忘梯度，什么时候保留梯度。

这看起来很神奇，但它确实是以下两点的结果：

1. cell 状态的加法更新策略使得梯度传递更恰当。

2. 门控单元可以决定遗忘多少梯度，他们可以在不同的时刻取不同的值。这些值都是通过隐层状态和输入的函数学习到的。





------

## 生成对抗网络（GAN）

### 1. GAN 为何会训练不稳定，如何解决？

参考：

- [GAN不稳定原因](https://blog.csdn.net/weixin_43698821/article/details/85003226)
- [提高GAN训练稳定性的9大tricks](https://zhuanlan.zhihu.com/p/68120231)




GAN 训练不稳定的原因如下：

1. **因为 G 和 D 互为对抗，此消彼长，很难让他们同时收敛**。大多深度模型的训练都使用优化算法寻找损失函数比较低的值。优化算法通常是个可靠的“下山”过程。生成对抗神经网络要求双方在博弈的过程中达到势均力敌（均衡）。每个模型在更新的过程中（比如生成器）成功的“下山”，但同样的更新可能会造成博弈的另一个模型（比如判别器）“上山”。甚至有时候博弈双方虽然最终达到了均衡，但双方在不断的抵消对方的进步并没有使双方同时达到一个有用的地方。对所有模型同时梯度下降使得某些模型收敛但不是所有模型都达到收敛最优。
2. **生成器G发生模式崩溃**：对于不同的输入生成相似的样本，最坏的情况仅生成一个单独的样本，判别器的学习会拒绝这些相似甚至相同的单一样本。在实际应用中，完全的模式崩溃很少，**局部的模式崩溃很常见**。局部模式崩溃是指生成器使不同的图片包含相同的颜色或者纹理主题，或者不同的图片包含同一只狗的不同部分。MinBatch GAN缓解了模式崩溃的问题但同时也引发了counting, perspective和全局结构等问题，这些问题通过设计更好的模型框架有可能解决。
3. **生成器梯度消失问题**：当判别器非常准确时，判别器的损失很快收敛到0，从而无法提供可靠的路径使生成器的梯度继续更新，造成生成器梯度消失。GAN的训练因为一开始随机噪声分布，与真实数据分布相差距离太远，两个分布之间几乎没有任何重叠的部分，这时候判别器能够很快的学习把真实数据和生成的假数据区分开来达到判别器的最优，造成生成器的梯度无法继续更新甚至梯度消失。



解决的办法有：

**1. 替代损失函数**

修复 GAN 缺陷的最流行的补丁是 Wasserstein GAN （[https://arxiv.org/pdf/1701.07875.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1701.07875.pdf)）。该 GAN 用 Earth Mover distance ( Wasserstein-1 distance 或 EM distance) 来替换传统 GAN 的 Jensen Shannon divergence ( J-S 散度) 。EM 距离的原始形式很难理解，因此使用了双重形式。这需要判别网络是 1-Lipschitz，通过修改判别网络的权重来维护。其优势和劣势分别如下：

- 使用 Earth Mover distance 的**优势在于即使真实的生成数据分布是不相交的，它也是连续的**。同时，在生成的图像质量和损失值之间存在一定关系。
- 使用 Earth Mover distance 的**劣势在于对于每个生成模型 G 都要执行许多判别网络 D 的更新**。而且，研究人员认为权重修改是确保 1-Lipschitz 限制的极端方式。

另一个解决方案是**使用均方损失（ mean squared loss ）替代对数损失（ log loss ）**。LSGAN （[https://arxiv.org/abs/1611.04076](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1611.04076)）的作者认为传统 GAN 损失函数并不会使收集的数据分布接近于真实数据分布。

原来 GAN 损失函数中的对数损失并不影响生成数据与决策边界（decision boundary）的距离。另一方面，LSGAN 也会对距离决策边界较远的样本进行惩罚，使生成的数据分布与真实数据分布更加靠近，这是通过将均方损失替换为对数损失来完成的。

**2. Two Timescale Update Rule (TTUR)**

在 TTUR 方法中，研究人员对判别网络 D 和生成网络 G 使用不同的学习速度。低速更新规则用于生成网络 G ，判别网络 D使用 高速更新规则。使用 TTUR 方法，研究人员可以让生成网络 G 和判别网络 D 以 1:1 的速度更新。 SAGAN （[https://arxiv.org/abs/1805.08318](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1805.08318)） 就使用了 TTUR 方法。



**3. Gradient Penalty （梯度惩罚）**

论文Improved Training of WGANs（[https://arxiv.org/abs/1704.00028](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1704.00028)）中，作者称权重修改会导致优化问题。权重修改会迫使神经网络学习学习更简单的相似（simpler approximations）达到最优数据分布，导致结果质量不高。同时如果 WGAN 超参数设置不合理，权重修改可能会出现梯度消失或梯度爆炸的问题，论文作者在损失函数中加入了一个简单的梯度惩罚机制以缓解该问题。

DRAGAN （[https://arxiv.org/abs/1705.07215](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1705.07215)）的作者称，当 GAN 的博弈达到一个局部平衡态（local equilibrium state），就会出现 mode collapse 的问题。而且判别网络 D 在这种状态下产生的梯度是非常陡（sharp）的。一般来说，使用梯度惩罚机制可以帮助避免这种状态的产生，极大增强 GAN 的稳定性，尽可能减少 mode collapse 问题的产生。



**4. Spectral Normalization（谱归一化）**

Spectral normalization 是用在判别网络 D 来增强训练过程的权重正态化技术 （weight normalization technique），可以确保判别网络 D 是 K-Lipschitz 连续的。 SAGAN ([https://arxiv.org/abs/1805.08318](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1805.08318))这样的实现也在判别网络 D 上使用了谱正则化。而且该方法在计算上要比梯度惩罚方法更加高效。



**5. Unrolling and Packing (展开和打包)**

文章 Mode collapse in GANs（[http://aiden.nibali.org/blog/2017-01-18-mode-collapse-gans/](https://link.zhihu.com/?target=http%3A//aiden.nibali.org/blog/2017-01-18-mode-collapse-gans/)）中提到一种预防 mode hopping 的方法就是在更新参数时进行预期对抗（anticipate counterplay）。展开的 GAN ( Unrolled GANs ）可以使用生成网络 G 欺骗判别网络 D，然后判别网络 D 就有机会进行响应。

另一种预防 mode collapse 的方式就是把多个属于同一类的样本进行打包，然后传递给判别网络 D 。PacGAN （[https://arxiv.org/abs/1712.04086](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1712.04086)）就融入了该方法，并证明可以减少 mode collapse 的发生。



**6. 多个 GAN**

一个 GAN 可能不足以有效地处理任务，因此研究人员提出使用多个连续的 GAN ，每个 GAN 解决任务中的一些简单问题。比如，FashionGAN（[https://www.cs.toronto.edu/~urtasun/publications/zhu_etal_iccv17.pdf](https://link.zhihu.com/?target=https%3A//www.cs.toronto.edu/~urtasun/publications/zhu_etal_iccv17.pdf)）就使用 2 个 GAN 来执行图像定位翻译。

因此，可以让 GAN 慢慢地解决更难的问题。比如 Progressive GANs (ProGANs，[https://arxiv.org/abs/1710.10196](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1710.10196)) 就可以生成分辨率极高的高质量图像。



**7. Relativistic GANs（相对生成对抗网络）**

传统的 GAN 会测量生成数据为真的可能性。Relativistic GANs 则会测量生成数据“逼真”的可能性。研究人员可以使用相对距离测量方法（appropriate distance measure）来测量相对真实性（relative realism），相关论文链接：[https://arxiv.org/abs/1807.00734](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1807.00734)。

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/relative_GAN.png)

图 A 表示 JS 散度的最优解，图 B 表示使用标准 GAN 损失时判别网络 D 的输出，图 C 表示输出曲线的实际图。

在论文中，作者提到判别网络 D 达到最优状态时，D 的输出应该聚集到 0.5。但传统的 GAN 训练算法会让判别网络 D 对图像输出“真实”（real，1）的可能性，这会限制判别网络 D 达到最优性能。不过这种方法可以很好地解决这个问题，并得到不错的结果。

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/relative_GAN2.png)经过 5000 次迭代后，标准 GAN (左)和相对 GAN (右)的输出。



**8. Self Attention Mechanism（自注意力机制）**

Self Attention GANs（[https://arxiv.org/abs/1805.08318](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1805.08318)）作者称用于生成图像的卷积会关注本地传播的信息。也就是说，由于限制性接收域这会错过广泛传播关系。

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/self_attention_gan.png)

将 attention map (在黄色框中计算)添加到标准卷积操作中。

Self-Attention Generative Adversarial Network 允许图像生成任务中使用注意力驱动的、长距依赖的模型。自注意力机制是对正常卷积操作的补充，全局信息（长距依赖）会用于生成更高质量的图像，而用来忽略注意力机制的神经网络会考虑注意力机制和正常的卷积。（相关论文链接：[https://arxiv.org/pdf/1805.08318.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1805.08318.pdf)）。



**9. 其他技术**

其他可以用来改善 GAN 训练过程的技术包括：

- 特征匹配
- Mini Batch Discrimination（小批量判别）
- 历史平均值
- One-sided Label Smoothing（单侧标签平滑）
- Virtual Batch Normalization（虚拟批量正态化）






------

## 计算机视觉

参考：

- [计算机视觉知识点总结](https://zhuanlan.zhihu.com/p/58776542)





### 图像分类

#### 1. 图像分类的损失函数，从网络输出到交叉熵是如何计算的



#### 2. 二分类是否好于多分类，以及如何进行二分类，二分类的损失函数





### 目标检测

#### 1. R-CNN 和 SSD 的区别



#### 2. NMS 原理及代码实现

参考资料：

- [NMS算法详解（附Pytorch实现代码）](https://zhuanlan.zhihu.com/p/54709759)
- [非极大值抑制（Non-Maximum Suppression，NMS）](https://www.bbsmax.com/A/A2dmV1YOze/)

##### 背景

NMS (Non-maximum suppression) 非极大值抑制，即抑制不是极大值的检测框，根据什么去抑制？在目标检测领域，当然是根据 IOU (Intersection over Union) 去抑制。下图是绿色检测框与红色检测框的 IOU 计算方法：

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/NMS_IOU.jpg)

##### NMS 原理及示例

**注意 NMS 是针对一个特定的类别进行操作的**。例如假设一张图中有要检测的目标有“人脸”和“猫”，没做NMS之前检测到10个目标框，每个目标框变量表示为: $[x_1,y_1,x_2,y_2,score_1,score_2]$ ，其中 $(x_1,y_1)$ 表示该框左上角坐标，$(x_2,y_2)$ 表示该框右下角坐标，$score_1$ 表示"人脸"类别的置信度，$score_2$ 表示"猫"类别的置信度。当 $score_1$ 比 $score_2$ 大时，将该框归为“人脸”类别，反之归为“猫”类别。最后我们假设10个目标框中有6个被归类为“人脸”类别。

接下来演示如何对“人脸”类别的目标框进行 NMS 。

首先对6个目标框按照 $score_1$ 即置信度降序排序：

| 目标框 | score_1 |
| :----: | :-----: |
|   A    |   0.9   |
|   B    |  0.85   |
|   C    |   0.7   |
|   D    |   0.6   |
|   E    |   0.4   |
|   F    |   0.1   |

(1) 取出最大置信度的那个目标框 A 保存下来
(2) 分别判断 B-F 这５个目标框与 A 的重叠度 IOU ，**如果 IOU 大于我们预设的阈值（一般为 0.5），则将该目标框丢弃**。假设此时丢弃的是 C和 F 两个目标框，这时候该序列中只剩下 B D E 这三个。
(3) 重复以上流程，直至排序序列为空。



##### 代码实现

```python
# bboxees维度为 [N, 4]，scores维度为 [N, 1]，均为np.array()
def single_nms(self, bboxes, scores, thresh = 0.5):
    # x1、y1、x2、y2以及scores赋值
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    
    # 计算每个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    
    # 按照 scores 置信度降序排序, order 为排序的索引
    order = scores.argsort() # argsort为python中的排序函数，默认升序排序
    order = order[::-1] # 将升序结果翻转为降序
    
    # 保留的结果框索引
    keep = []
    
    # torch.numel() 返回张量元素个数
    while order.size > 0:
        if order.size == 1:
            i = order[0]
            keep.append(i)
            break
        else:
            i = order[0]  # 在pytorch中使用item()来取出元素的实值，即若只是 i = order[0]，此时的 i 还是一个 tensor，因此不能赋值给 keep
            keep.append(i)
            
        # 计算相交区域的左上坐标及右下坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # 计算相交的面积，不重叠时为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # 计算 IOU = 重叠面积 / (面积1 + 面积2 - 重叠面积)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留 IOU 小于阈值的 bboxes
        inds = np.where(iou <= thresh)[0]
        if inds.size == 0:
            break
        order = order[inds + 1] # 因为我们上面求iou的时候得到的结果索引与order相比偏移了一位，因此这里要补回来
    return keep  # 这里返回的是bboxes中的索引，根据这个索引就可以从bboxes中得到最终的检测框结果
```











### 图像分割





### 迁移学习



## 图像处理基础

### 1. Canny 边缘检测流程



### 2. 边界检测算法有哪些



### 3. 边缘检测算法有哪些





### 4. 仿射变换矩阵，透视变换矩阵，双线性二插值





### 5. 锐化算法





### 6. 去噪算法







## 深度学习框架

### Tensorflow

#### 1. TF 中卷积的计算

#### 2. 转置卷积计算

#### 3. 多卡训练，一个模型在一个卡放不下，如何在多卡中运行





### PyTorch

#### 1. `DataParallel` 和 `DistributedDataParallel` 区别

参考文章：

- [pytorch分布式数据并行DistributedDataParallel（DDP）](https://zhuanlan.zhihu.com/p/107139605)

DistributedDataParallel（DDP）在module级别实现数据并行性。它使用[torch.distributed](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/intermediate/dist_tuto.html)包communication collectives来同步梯度，参数和缓冲区。并行性在单个进程内部和跨进程均有用。在一个进程中，DDP将input module 复制到 device_ids 指定的设备，相应地按 batch 维度分别扔进模型，并将输出收集到output_device，这与[DataParallel](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)相似。Across processes, DDP inserts necessary parameter synchronizations in forward passes and gradient synchronizations in backward passes. It is up to users to map processes to available resources, as long as processes do not share GPU devices.

**推荐（通常是最快的方法）为每个module 副本创建一个进程，即在一个进程中不进行任何module复制**。





 `DataParallel` 和 `DistributedDataParallel` 区别：

1. 如果模型太大而无法容纳在单个GPU上，则必须使用 **model parallel** 将其拆分到多个GPU中。 DistributedDataParallel与模型并行工作； DataParallel目前不提供。
2. **DataParallel是单进程，多线程，并且只能在单台计算机上运行**，而DistributedDataParallel是**多进程**，并且可以在**单机和分布式训练**中使用。因此，即使在单机训练中，您的数据足够小以适合单机，DistributedDataParallel仍要比DataParallel更快。 
3. DistributedDataParallel还可以**预先复制模型**，而不是在每次迭代时复制模型，并且可以避免PIL全局解释器锁定。
4. 如果数据和模型同时很大而无法用一个GPU训练，则可以将model parallel（与DistributedDataParallel结合使用。在这种情况下，每个DistributedDataParallel进程都可以model parallel，并且所有进程共同用数据并行；
5. DataParallel 在每个训练批次（batch）中，因为模型的权重都是在 一个进程上先算出来 然后再把他们分发到每个GPU上，所以网络通信就成为了一个瓶颈，而GPU使用率也通常很低；并且需要所有的GPU都在一个节点（一台机器）上，且并不支持 [Apex](https://link.zhihu.com/?target=https%3A//nvidia.github.io/apex/amp.html) 的 [混合精度训练](https://link.zhihu.com/?target=https%3A//devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/).





#### 2. 多 gpu 训练机制的原理

Pytorch 的多 GPU 处理接口是 `torch.nn.DataParallel(module, device_ids)`，其中 module 参数是所要执行的模型，而 device_ids 则是指定并行的 GPU id 列表。

并行处理机制是，首先将模型加载到主 GPU 上，然后再将模型复制到各个指定的从 GPU 中，然后将输入数据按 batch 维度进行划分，具体来说就是每个 GPU 分配到的数据 batch 数量是总输入数据的 batch 除以指定 GPU 个数。每个 GPU 将针对各自的输入数据独立进行 forward 计算，最后将各个 GPU 的 loss 进行求和，再用反向传播更新单个 GPU 上的模型参数，再将更新后的模型参数复制到剩余指定的 GPU 中，这样就完成了一次迭代计算。





------

## 优化算法



## 超参数调整



## 模型压缩、加速

### 1. 为什么 MobileNet 会这么快？

参考：[轻量级神经网络“巡礼”（二）—— MobileNet，从V1到V3](https://zhuanlan.zhihu.com/p/70703846)

不管是在GPU还是在CPU运行，**最重要的“耗时杀手”就是conv，卷积层**。也就是说，**想要提高网络的运行速度，就得到提高卷积层的计算效率**。

而**MobileNet的95%的计算都花费在了1×1的卷积上**，那1×1卷积有什么好处吗？

在实现标准卷积的时候，我们是采用 im2col 的操作，这是一个**通过牺牲空间的手段（约扩增 $K\times K 倍，K 是卷积核大小），将特真土转成庞大的矩阵进行卷积计算**的方法。

其实现思路就是：

> 把每一次循环所需要的数据都排列成列向量，然后逐一堆叠起来形成矩阵（按通道顺序在列方向上拼接矩阵）。
> 比如Ci×Wi×Hi大小的输入特征图，K×K大小的卷积核，输出大小为Co×Wo×Ho，
> 输入特征图将按需求被转换成(K∗K)×(Ci∗Wo∗Ho)的矩阵，卷积核将被转换成Co×(K∗K)的矩阵，

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/im2col.jpeg" style="zoom:80%;" />

然后调用**GEMM（矩阵乘矩阵）库**加速两矩阵相乘也就完成了卷积计算。由于**按照计算需求排布了数据顺序，每次计算过程中总是能够依次访问特征图数据，极大地提高了计算卷积的速度。** *（不光有GEMM，还有FFt（快速傅氏变换））*

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/im2col_2.jpeg" style="zoom:67%;" />

换一种表示方法能更好地理解，图片来自[High Performance Convolutional Neural Networks for Document Processing](https://link.zhihu.com/?target=https%3A//hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf)：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/im2col.jpg" alt="im2col" style="zoom:80%;" />

这样可以更清楚的看到卷积的定义进行卷积操作（上图上半部分），**内存访问会非常不规律**，以至于性能会非常糟糕。而Im2col()以一种**内存访问规则的方式排列数据**，**虽然Im2col操作增加了很多数据冗余，但使用Gemm的性能优势超过了这个数据冗余的劣势**。

所以标准卷积运算大概就是这样的一个过程：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/im2col_3.jpeg)

那我们现在回到1×1的卷积上来，有点特殊。按照我们之前所说的，1×1的卷积的原始储存结构和进行im2col的结构如下图所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/im2col_4.jpeg" style="zoom:67%;" />

可以看到矩阵是完全相同的。标准卷积运算和1×1卷积运算对比如下图：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/im2col_5.jpeg" style="zoom:50%;" />

也就是说，**1x1卷积不需要im2col的过程**，所以底层可以有更快的实现，拿起来就能直接算，大大节省了数据重排列的时间和空间。

当然，这些也不是那么绝对的，因为毕竟MobileNet速度快不快，与CONV1x1运算的优化程度密切相关。如果使用了定制化的硬件（比如用FPGA直接实现3x3的卷积运算单元），那么im2col就失去了意义而且反而增加了开销。

回到之前的MobileNet的资源分布，95%的1×1卷积和优化的网络结构就是MobileNet能如此快的原因了。







------

## 参考

- [深度学习500问](https://github.com/scutan90/DeepLearning-500-questions)
- [BAT面试1000题](https://zhuanlan.zhihu.com/c_140166199)
- [machine-learning-interview-questions](https://github.com/Sroy20/machine-learning-interview-questions)
- [深度学习面试中文版](https://github.com/elviswf/DeepLearningBookQA_cn)
- [图像处理100问](https://github.com/gzr2017/ImageProcessing100Wen)
- [计算机视觉及深度学习_面试问题（一）](https://mp.weixin.qq.com/s/y3bCUC8Mb3lhtsgwusXNWA)
- [深度学习和机器学习_面试问题（二）](https://mp.weixin.qq.com/s/kg8gIgHFGS3DyJg2OhxO-w)
- [阿里、百度、字节跳动、京东、地平线等计算机视觉实习生面试经历分析，已成功上岸！](https://mp.weixin.qq.com/s/BJAeIKULuoSnFAlO1IvVBg)
- [推荐收藏 | 决策树，逻辑回归，PCA-算法面经](https://mp.weixin.qq.com/s/ujjHw2Qym2R2B7tvO3LPSA)
- [自己整理的一点和深度学习相关的面试考点](https://zhuanlan.zhihu.com/p/48374690)
- [200 道机器学习面试题 | 吊打面试官系列](https://mp.weixin.qq.com/s/sN-Y3xds8_kbPerhmgc26g)
- [深度学习岗位面试问题整理笔记](https://zhuanlan.zhihu.com/p/25005808)
- [深度学习六十问！一位算法工程师经历30+场CV面试后总结的常见问题合集下篇（含答案）](https://mp.weixin.qq.com/s/GBaYuTrQkMVpH3mEZLFDLw)



