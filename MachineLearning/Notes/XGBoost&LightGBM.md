# XGBoost&LightGBM

参考：

- [XGBoost、GBDT超详细推导](https://zhuanlan.zhihu.com/p/92837676)
- [珍藏版 | 20道XGBoost面试题](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzI1MzY0MzE4Mg%3D%3D%26mid%3D2247485159%26idx%3D1%26sn%3Dd429aac8370ca5127e1e786995d4e8ec%26chksm%3De9d01626dea79f30043ab80652c4a859760c1ebc0d602e58e13490bf525ad7608a9610495b3d%26scene%3D21%23wechat_redirect)
- [阿泽：【机器学习】决策树（下）——XGBoost、LightGBM（非常详细）](https://zhuanlan.zhihu.com/p/87885678)



------

## XGBoost

首先需要说一说GBDT，它是一种基于boosting增强策略的加法模型，训练的时候采用前向分布算法进行贪婪的学习，每次迭代都学习一棵CART树来拟合之前 t-1 棵树的预测结果与训练样本真实值的残差。

XGBoost （eXtreme Gradient Boosting）是基于Boosting框架的一个算法工具包（包括工程实现），在并行计算效率、缺失值处理、预测性能上都非常强大，它对GBDT进行了一系列优化，比如损失函数进行了二阶泰勒展开、目标函数加入正则项、支持并行和默认缺失值处理等，在可扩展性和训练速度上有了巨大的提升，但其核心思想没有大的变化。



### 1. 模型介绍

#### 1.1 目标函数的二阶泰勒展开

和GBDT一样，XGBoost是一个加法模型，在每一步迭代中只优化当前步中的子模型。 在第 m 步中：
$$
F_m(x_i)=F_{m-1}(x_i)+f_m(x_i)
$$
$f_m(x_i)$ 为当前步的子模型， $F_{m-1}(x_i)$为训练完已经固定了的前  m-1 个子模型。

目标函数为经验风险+结构风险（正则项）：
$$
Obj = \sum^N_{i=1}L[F_m(x_i),y_i]+\sum_{j=1}^m\Omega (f_j)\\

 =\sum^N_{i=1}L[F_{m-1}(x_i)+f_m(x_i),y_i]+\sum^m_{j=1}\Omega(f_j) \tag{1}
$$
其中，正则项 $\Omega(f)$ 表示子模型 f 的复杂度，与二阶泰勒展开无关，将在下一节具体展开。

泰勒公式是将一个在 $x=x_0$ 处具有 n 阶导数的函数 $f(x)$ 利用关于 $\Delta x=x-x_0$ 的 n 次多项式来逼近 f(x) 的方法。XGBoos t运用二阶展开来近似表达损失函数。
$$
f(x_0+\Delta x) \approx f(x_0)+f^\prime (x_0)\Delta x + \frac{f^{\prime \prime}(x_0)}{2}(\Delta x)^2
$$
式（1）中，将 $F_{m-1}(x_i)$ 视作 $x_0$ ，$f_m(x_i)$ 视作 $\Delta x$ , $L(\hat y_i, y_i)$ 视作关于 $\hat y_i$ 的函数，可得：
$$
Obj = \sum_{i=1}^N \Big[ L[F_{m-1}(x_i),y_i] + \frac{\partial L}{\partial F_{m-1}(x_i)} f_m(x_i) + \frac{1}{2} \frac{\partial^2 L}{\partial^2 F_{m-1}(x_i)} f_m^2(x_i) \Big] +\sum_{j=1}^m \Omega (f_j) \\
$$
前 m-1 个子模型已经确定了，故上式中**除了关于 $f_m(x)$ 的部分都是常数**，不影响对  $f_m(x)$的优化求解。目标函数可转化为：
$$
\begin{align*} Obj = \sum_{i=1}^N \Big[g_i f_m(x_i)+\frac{1}{2} h_i f_m^2(x_i)\Big]+\Omega (f_m) \tag{2} \end{align*}
$$
其中
$$
g_i = \frac{\partial L}{\partial F_{m-1}(x_i)} ,\ \  h_i = \frac{\partial^2 L}{\partial^2 F_{m-1}(x_i)} \\
$$
这里的 L 是损失函数，度量一次预测的好坏。在 $F_{m-1}(x)$ 确定了的情况下，对每个样本点 i 都可以轻易计算出一个  $g_i$ 和 $h_i$

#### 1.2 基于树的正则化

XGBoost支持的基分类器包括决策树和线性模型，我们这里只讨论更常见的基于树的情况。为防止过拟合，XGBoost设置了基于树的复杂度作为正则项：
$$
\Omega(f)=\gamma T + \frac{1}{2} \lambda ||w||^2 \tag{3}
$$
T 为树 f 的叶节点个数，w 为所有叶节点输出回归值构成的向量，$||w||^2$ 为该向量L2范数（模长）的平方，$\gamma,\lambda$ 为超参数。作为回归树，叶子节点越多、输出的回归值

由（2）（3），目标函数如下：
$$
Obj = \sum_{i=1}^N \Big[g_i f_m(x_i)+\frac{1}{2} h_i f_m^2(x_i)\Big]+\gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 \\
$$
接下来通过一个数学处理，可以使得正则项和经验风险项合并到一起。经验风险项是在样本层面上求和，我们将其转换为叶节点层面上的求和。

定义节点 j 上的样本集为 $I(j)={x_i|q(x_i)=j}$，其中 $q(x_i)$ 为将样本映射到叶节点上的索引函数，叶节点 j 上的回归值为 $w_j=f_m(x_i),i\in I(j)$.
$$
Obj = \sum_{j=1}^{T} \Big[ (\sum_{i\in I(j)} g_i) w_j + \frac{1}{2}(\sum_{i\in I(j)} h_i + \lambda) w_j^2 \Big] + \gamma T \\
$$
进一步简化表达，令 $\sum_{i\in I(j)} g_i=G_j, \sum_{i\in I(j)} h_i=H_j$ , 注意这里 G 和 H 都是关于  j 的函数：
$$
Obj = \sum_{j=1}^{T} \Big[ G_j w_j + \frac{1}{2}(H_j + \lambda) w_j^2 \Big] + \gamma T \\
$$
此时，若一棵树的结构已经确定，则各个节点内的样本 $(x_i,y_i,g_i,h_i)$ 也是确定的，即 $G_j, H_j, T$被确定，每个叶节点输出的回归值应该使得上式最小，由二次函数极值点：
$$
w_j^*=-\frac{G_j}{H_j+\lambda} \\
$$
按此规则输出回归值后，目标函数值也就是树的评分如下，越小代表树的结构越好。观察下式，**树的评分也可以理解成所有叶节点的评分之和**：
$$
\begin{align*} Obj^* =  \sum_{j=1}^T \Big( -\frac{1}{2}\frac{G_j^2}{H_j + \lambda} + \gamma \Big) \tag{4} \end{align*} 
$$

#### 1.3 节点分裂准则

XGBoost的子模型树和决策树模型一样，要依赖节点递归分裂的贪心准则来实现树的生成。除此外，XGBoost还支持近似算法，解决数据量过大超过内存、或有并行计算需求的情况。

##### 1.3.1 贪心准则

基本思路和CART一样，**对特征值排序后遍历划分点，将其中最优的分裂收益作为该特征的分裂收益，选取具有最优分裂收益的特征作为当前节点的划分特征，按其最优划分点进行二叉划分，得到左右子树**。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/xgboost_fig.jpeg" style="zoom:47%;" />

上图是一次节点分裂过程，很自然地，分裂收益是树A的评分减去树B的评分。由（4），虚线框外的叶节点，即非分裂节点的评分均被抵消，只留下分裂后的LR节点和分裂前的S节点进行比较，因此分裂收益的表达式为：
$$
Gain = \frac{1}{2} \Big[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} -\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\lambda  \\
$$


##### 1.3.2 近似算法

XGBoost还提供了上述贪心准则的近似版本，简言之，**将特征分位数作为划分候选点**。 这样将划分候选点集合由全样本间的遍历缩减到了几个分位数之间的遍历。

具体而言，**特征分位数的选取有 global 和 local 两种可选策略：**

- global在全体样本上的特征值中选取，在根节点分裂之前进行一次即可；
- local则是**在待分裂节点包含的样本特征值上选取**，每个节点分裂前都要进行。

通常，**global由于只能划分一次，其划分粒度需要更细**。

在XGB原始论文中，作者在Higgs Boson数据集上比较了**精确贪心准则、global近似和local近似**三类配置的测试集AUC，用eps代表取分位点的粒度，如eps=0.25代表将数据集划分为1/0.25=4个buckets，发现global（eps=0.05）和local（eps=0.3）均能达到和精确贪心准则几乎相同的性能。

这三类配置在XGBoost包均有支持。

##### 1.3.3 加权分位数

查看（2）式表示的目标函数，令偏导为0, 易得 $f_m^*(x_i)=-\frac{g_i}{h_i}$，此目标函数可理解为以 $h_i$ 为权重，  为$-\frac{g_i}{h_i}$标签的二次损失函数：
$$
\begin{align*} Obj &= \sum_{i=1}^N \Big[g_i f_m(x_i)+\frac{1}{2} h_i f_m^2(x_i)\Big]+\Omega (f_m) \\  &= \sum_{i=1}^N \frac{1}{2} h_i\Big[ f_m(x_i)-(-\frac{g_i}{h_i}) \Big]^2+\Omega (f_m) + C \end{align*} \\
$$
因此，在近似算法取分位数时，实际上XGBoost会取以二阶导 $h_i$ 为权重的分位数（Weighted Quantile Sketch），如下图表示的三分位。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/xgboost_fig2.png" alt="img" style="zoom:67%;" />

#### 1.4 列采样和学习率

XGBoost还引入了两项特性：列采样和学习率。

**列采样**，即随机森林中的做法，**每次节点分裂的待选特征集合不是剩下的全部特征，而是剩下特征的一个子集。**是为了更好地对抗过拟合（我不是很清楚GBDT中列采样降低过拟合的理论依据。原文这里提到的动机是某GBDT的软件用户反馈列采样比行采样更能对抗过拟合），还能减少计算开销。

**学习率**，或者叫步长、shrinkage，**是在每个子模型前（即在每个叶节点的回归值上）乘上该系数，削弱每颗树的影响，使得迭代更稳定**。可以类比梯度下降中的学习率。XGBoost默认设定为0.3。



#### 1.5 稀疏感知

缺失值应对策略是算法需要考虑的。特征稀疏问题也同样需要考虑，如部分特征中出现大量的0或干脆是one-hot encoding这种情况。

XGBoost用稀疏感知策略来同时处理这两个问题：**概括地说，将缺失值和稀疏0值等同视作缺失值，再将这些缺失值“绑定”在一起，分裂节点的遍历会跳过缺失值的整体。这样大大提高了运算效率**。

> 0 值在XGB中被处理为数值意义上的0还是NA，不同平台上的默认设置不同，**[可参考本处](https://link.zhihu.com/?target=https%3A//blog.csdn.net/sinat_26811377/article/details/100064947)**。总的来说需要结合具体平台的设置，预处理区分开作为数值的0（不应该被处理为NA）和作为稀疏值的0（应该被处理为NA）。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/xgboost_fig3.png" alt="img" style="zoom:45%;" />

分裂节点依然通过遍历得到，**NA的方向有两种情况，在此基础上对非缺失值进行切分遍历。或者可以理解NA被分到一个固定方向**，非缺失值在升序和降序两种情况下进行切分遍历。

如上图所示，若某个特征值取值为1,2,5和大量的NA，XGBoost会遍历以上6种情况（3个非缺失值的切分点 × 缺失值的两个方向），最大的分裂收益就是本特征上的分裂收益，同时，NA将被分到右节点。



#### 1.6 优缺点

##### 优点

1. **精度更高：**GBDT 只用到一阶泰勒展开，而 XGBoost 对损失函数进行了二阶泰勒展开。XGBoost 引入二阶导一方面是为了增加精度，另一方面也是为了能够自定义损失函数，二阶泰勒展开可以近似大量损失函数；
2. **灵活性更强：**GBDT 以 CART 作为基分类器，XGBoost 不仅支持 CART 还支持线性分类器，（使用线性分类器的 XGBoost 相当于带 L1 和 L2 正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题））。此外，XGBoost 工具支持自定义损失函数，只需函数支持一阶和二阶求导；
3. **正则化：**XGBoost 在目标函数中加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、叶子节点权重的 L2 范式。正则项降低了模型的方差，使学习出来的模型更加简单，有助于防止过拟合；
4. **Shrinkage（缩减）：**相当于学习速率。XGBoost 在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间；
5. **列抽样：**XGBoost 借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算；
6. **缺失值处理：**XGBoost 采用的稀疏感知算法极大的加快了节点分裂的速度；
7. **可以并行化操作：**块结构可以很好的支持并行计算。



##### 缺点

1. 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，**但在节点分裂过程中仍需要遍历数据集**；
2. 预排序过程的**空间复杂度过高**，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，**相当于消耗了两倍的内存**。



#### 1.7 特点

**这部分内容参考了知乎上的一个问答—[机器学习算法中GBDT和XGBOOST的区别有哪些？](https://www.zhihu.com/question/41354392)，答主是wepon大神**

1. 传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。 **—可以通过booster [default=gbtree]设置参数:gbtree: tree-based models/gblinear: linear models**

2. 传统GBDT在优化时只用到一阶导数信息，xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶和二阶导数。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。 **—对损失函数做了改进（泰勒展开，一阶信息g和二阶信息h）**

3. xgboost在代价函数里加入了**正则项**，用于**控制模型的复杂度**。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和。从Bias-variance tradeoff角度来讲，正则项降低了模型variance，使学习出来的模型更加简单，防止过拟合，这也是xgboost优于传统GBDT的一个特性 
   **—正则化包括了两个部分，都是为了防止过拟合，剪枝是都有的，叶子结点输出L2平滑是新增的。**

4. shrinkage and column subsampling —**还是为了防止过拟合**

> （1）shrinkage缩减类似于学习速率，在每一步tree boosting之后增加了一个参数n（权重），通过这种方式来减小每棵树的影响力，给后面的树提供空间去优化模型。
>
> （2）column subsampling列(特征)抽样，说是从随机森林那边学习来的，防止过拟合的效果比传统的行抽样还好（行抽样功能也有），并且有利于后面提到的并行化处理算法。

5. split finding algorithms(划分点查找算法)：

   （1）exact greedy algorithm—**贪心算法获取最优切分点** 
   （2）approximate algorithm— **近似算法，提出了候选分割点概念，先通过直方图算法获得候选分割点的分布情况，然后根据候选分割点将连续的特征信息映射到不同的buckets中，并统计汇总信息。** 
   （3）Weighted Quantile Sketch—**分布式加权直方图算法** 
   **这里的算法（2）、（3）是为了解决数据无法一次载入内存或者在分布式情况下算法（1）效率低的问题，以下引用的还是wepon大神的总结：**

> 可并行的近似直方图算法。树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以xgboost还提出了一种可并行的近似直方图算法，用于高效地生成候选的分割点。

6.对缺失值的处理。对于特征的值有缺失的样本，xgboost可以自动学习出它的分裂方向。 **—稀疏感知算法**

7.**Built-in Cross-Validation（内置交叉验证)**

> XGBoost allows user to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run. 
> This is unlike GBM where we have to run a grid-search and only a limited values can be tested.

8. **continue on Existing Model（接着已有模型学习）**

> User can start training an XGBoost model from its last iteration of previous run. This can be of significant advantage in certain specific applications. 
> GBM implementation of sklearn also has this feature so they are even on this point.

9. **High Flexibility（高灵活性）**

> **XGBoost allow users to define custom optimization objectives and evaluation criteria. 
> This adds a whole new dimension to the model and there is no limit to what we can do.**

10. 并行化处理 **—系统设计模块,块结构设计等**

> xgboost工具支持并行。boosting不是一种串行的结构吗?怎么并行的？注意xgboost的并行不是tree粒度的并行，xgboost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。**xgboost的并行是在特征粒度上的**。我们知道，**决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点）**，xgboost在训练之前，**预先对数据进行了排序，然后保存为block结构**，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。



此外xgboost还设计了高速缓存压缩感知算法，这是系统设计模块的效率提升。 
当梯度统计不适合于处理器高速缓存和高速缓存丢失时，会大大减慢切分点查找算法的速度。 

- 针对 exact greedy algorithm采用缓存感知预取算法 
- 针对 approximate algorithms选择合适的块大小





#### 1.8 代码实现

下面给出简单使用**xgboost**这个框架的例子。

```python
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1729)
print(X_train.shape, X_test.shape)

#模型参数设置
xlf = xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.1, 
                        n_estimators=10, 
                        silent=True, 
                        objective='reg:linear', 
                        nthread=-1, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)

xlf.fit(X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, y_test)],early_stopping_rounds=100)

# 计算 auc 分数、预测
preds = xlf.predict(X_test)
```

一个运用到实际例子的代码，来自[xgboost入门与实战（实战调参篇）](http://blog.csdn.net/sb19931201/article/details/52577592)

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split

#from xgboost.sklearn import XGBClassifier
#from sklearn import cross_validation, metrics   #Additional scklearn functions
#from sklearn.grid_search import GridSearchCV   #Perforing grid search
#
#import matplotlib.pylab as plt
#from matplotlib.pylab import rcParams

#记录程序运行时间
import time 
start_time = time.time()

#读入数据
train = pd.read_csv("Digit_Recognizer/train.csv")
tests = pd.read_csv("Digit_Recognizer/test.csv") 

params={
'booster':'gbtree',
'objective': 'multi:softmax', #多分类的问题
'num_class':10, # 类别数，与 multisoftmax 并用
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':12, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, # 如同学习率
'seed':1000,
'nthread':7,# cpu 线程数
#'eval_metric': 'auc'
}

plst = list(params.items())
num_rounds = 5000 # 迭代次数

train_xy,val = train_test_split(train, test_size = 0.3,random_state=1)
#random_state is of big influence for val-auc
y = train_xy[:, 0]
X = train_xy[:, 1:]
val_y = val[:, 0]
val_X = val[:, 1:]

xgb_val = xgb.DMatrix(val_X,label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
xgb_test = xgb.DMatrix(tests)


watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

# training model 
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)

model.save_model('./model/xgb.model') # 用于存储训练出的模型
print "best best_ntree_limit",model.best_ntree_limit 

print "跑到这里了model.predict"
preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)

np.savetxt('xgb_submission.csv',np.c_[range(1,len(tests)+1),preds],delimiter=',',header='ImageId,Label',comments='',fmt='%d')

#输出运行时长
cost_time = time.time()-start_time
print "xgboost success!",'\n',"cost time:",cost_time,"(s)"
```

所使用的数据集是Kaggle上的[Classify handwritten digits using the famous MNIST data](https://www.kaggle.com/c/digit-recognizer/data)--手写数字识别数据集，即`Mnist`数据集。





------

### 2. 工程优化

#### 2.1 并行列块设计

XGBoost将**每一列特征提前进行排序，以块（Block）的形式储存在缓存中**，并以索引将特征值和梯度统计量 $g_i, h_i$ 对应起来，每次节点分裂时会重复调用排好序的块。而且不同特征会分布在独立的块中，**因此可以进行分布式或多线程的计算**。

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/xgboost_fig4.png)

#### **2.2 缓存访问**

特征值排序后通过索引来取梯度 $g_i, h_i$ 会导致访问的内存空间不一致，进而降低缓存的命中率，影响算法效率。为解决这个问题，**XGBoost为每个线程分配一个单独的连续缓存区，用来存放梯度信息**。



#### **2.3 核外块计算**

数据量过大时，不能同时全部载入内存。XGBoost将数据分为多个blocks并储存在硬盘中，使用一个独立的线程专门从磁盘中读取数据到内存中，实现计算和读取数据的同时进行。

为了进一步提高磁盘读取数据性能，XGBoost还使用了两种方法：

- **压缩block，用解压缩的开销换取磁盘读取的开销**；
- **将block分散储存在多个磁盘中，有助于提高磁盘吞吐量**。





------

### 3. 和其他模型的对比

#### 3.1 和 gbdt 的区别

- **基分类器**：XGBoost的基分类器**不仅支持CART决策树，还支持线性分类器**，此时XGBoost相当于带L1和L2正则化项的Logistic回归（分类问题）或者线性回归（回归问题）。
- **导数信息**：XGBoost对**损失函数做了二阶泰勒展开**，GBDT只用了一阶导数信息，并且XGBoost还支持自定义损失函数，只要损失函数一阶、二阶可导。
- **正则项**：XGBoost的目标函数**加了正则项**， 相当于预剪枝，使得学习出来的模型更加不容易过拟合。
- **列抽样**：XGBoost**支持列采样，与随机森林类似，用于防止过拟合**。
- **缺失值处理**：对树中的每个非叶子结点，XGBoost可以自动学习出它的默认分裂方向。如果某个样本该特征值缺失，会将其划入默认分支。
- **并行化**：**注意不是tree维度的并行，而是特征维度的并行**。XGBoost 预先将每个特征按特征值**排好序**，存储为块结构，分裂结点时可以**采用多线程并行查找每个特征的最佳分割点**，极大提升训练速度。



#### 3.2 和 RF 的区别

##### 相同点

都是由多棵树组成，最终的结果都是由多棵树一起决定。



##### 区别

- **集成学习**：RF属于bagging思想，而GBDT是boosting思想
- **偏差-方差权衡**：RF不断的降低模型的方差，而GBDT不断的降低模型的偏差
- **训练样本**：RF每次迭代的样本是从全部训练集中有放回抽样形成的，而GBDT每次使用全部样本
- **并行性**：RF的树可以并行生成，而GBDT只能顺序生成(需要等上一棵树完全生成)
- **最终结果**：RF最终是多棵树进行多数表决（回归问题是取平均），而GBDT是加权融合
- **数据敏感性**：RF对异常值不敏感，而GBDT对异常值比较敏感
- **泛化能力**：RF不易过拟合，而GBDT容易过拟合



#### 3.3  和LightGBM的区别

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/xgboost_fig5.png" alt="图片" style="zoom:67%;" />

（1）**树生长策略**：XGB采用`level-wise`的分裂策略，LGB采用`leaf-wise`的分裂策略。XGB对每一层所有节点做无差别分裂，但是可能有些节点增益非常小，对结果影响不大，带来不必要的开销。Leaf-wise是在所有叶子节点中选取分裂收益最大的节点进行的，但是很容易出现过拟合问题，所以需要对最大深度做限制 。

（2）**分割点查找算法**：XGB使用特征预排序算法，LGB使用基于直方图的切分点算法，其优势如下：

- **减少内存占用**，比如离散为256个bin时，只需要用8位整形就可以保存一个样本被映射为哪个bin(这个bin可以说就是转换后的特征)，对比预排序的exact greedy算法来说（用int_32来存储索引+ 用float_32保存特征值），可以节省7/8的空间。
- **计算效率提高**，预排序的Exact greedy对每个特征都需要遍历一遍数据，并计算增益，复杂度为𝑂(#𝑓𝑒𝑎𝑡𝑢𝑟𝑒×#𝑑𝑎𝑡𝑎)。而直方图算法在建立完直方图后，只需要对每个特征遍历直方图即可，复杂度为𝑂(#𝑓𝑒𝑎𝑡𝑢𝑟𝑒×#𝑏𝑖𝑛𝑠)。
- LGB还可以使用直方图做差加速，一个节点的直方图可以通过父节点的直方图减去兄弟节点的直方图得到，从而加速计算

> 但实际上xgboost的近似直方图算法也类似于lightgbm这里的直方图算法，为什么xgboost的近似算法比lightgbm还是慢很多呢？
>
> xgboost在每一层都动态构建直方图， 因为xgboost的直方图算法不是针对某个特定的feature，而是所有feature共享一个直方图(每个样本的权重是二阶导)，所以每一层都要重新构建直方图，而lightgbm中对每个特征都有一个直方图，所以构建一次直方图就够了。

（3）**支持离散变量**：无法直接输入类别型变量，因此需要事先对类别型变量进行编码（例如独热编码），而LightGBM可以直接处理类别型变量。

（4）**缓存命中率**：XGB使用Block结构的一个缺点是取梯度的时候，是通过索引来获取的，而这些梯度的获取顺序是按照特征的大小顺序的，这将导致非连续的内存访问，可能使得CPU cache缓存命中率低，从而影响算法效率。而LGB是基于直方图分裂特征的，梯度信息都存储在一个个bin中，所以访问梯度是连续的，缓存命中率高。

（5）**LightGBM 与 XGboost 的并行策略不同**：

- **特征并行** ：LGB特征并行的前提是每个worker留有一份完整的数据集，但是每个worker仅在特征子集上进行最佳切分点的寻找；worker之间需要相互通信，通过比对损失来确定最佳切分点；然后将这个最佳切分点的位置进行全局广播，每个worker进行切分即可。XGB的特征并行与LGB的最大不同在于XGB每个worker节点中仅有部分的列数据，也就是垂直切分，每个worker寻找局部最佳切分点，worker之间相互通信，然后在具有最佳切分点的worker上进行节点分裂，再由这个节点广播一下被切分到左右节点的样本索引号，其他worker才能开始分裂。二者的区别就导致了LGB中worker间通信成本明显降低，只需通信一个特征分裂点即可，而XGB中要广播样本索引。

- **数据并行** ：当数据量很大，特征相对较少时，可采用数据并行策略。LGB中先对数据水平切分，每个worker上的数据先建立起局部的直方图，然后合并成全局的直方图，采用直方图相减的方式，先计算样本量少的节点的样本索引，然后直接相减得到另一子节点的样本索引，这个直方图算法使得worker间的通信成本降低一倍，因为只用通信以此样本量少的节点。XGB中的数据并行也是水平切分，然后单个worker建立局部直方图，再合并为全局，不同在于根据全局直方图进行各个worker上的节点分裂时会单独计算子节点的样本索引，因此效率贼慢，每个worker间的通信量也就变得很大。

- **投票并行（LGB）**：当数据量和维度都很大时，选用投票并行，该方法是数据并行的一个改进。数据并行中的合并直方图的代价相对较大，尤其是当特征维度很大时。大致思想是：每个worker首先会找到本地的一些优秀的特征，然后进行全局投票，根据投票结果，选择top的特征进行直方图的合并，再寻求全局的最优分割点。

  



------

### 4. 模型细节

#### 4.1 XGBoost为什么使用泰勒二阶展开

- **精准性**：相对于GBDT的一阶泰勒展开，XGBoost采用二阶泰勒展开，可以更为精准的逼近真实的损失函数
- **可扩展性**：损失函数支持自定义，只需要新的损失函数二阶可导。



#### 4.2 XGBoost为什么可以并行训练

- XGBoost的并行，并不是说每棵树可以并行训练，XGB本质上仍然采用boosting思想，每棵树训练前需要等前面的树训练完成才能开始训练。
- XGBoost的并行，指的是特征维度的并行：在训练之前，每个特征按特征值对样本进行预排序，并存储为Block结构，在后面查找特征分割点时可以重复使用，而且特征已经被存储为一个个block结构，那么在寻找每个特征的最佳分割点时，可以利用多线程对每个block并行计算。



#### 4.3 XGBoost为什么快

- **分块并行**：训练前每个特征按特征值进行排序并存储为Block结构，后面查找特征分割点时重复使用，并且支持并行查找每个特征的分割点
- **候选分位点**：每个特征采用常数个分位点作为候选分割点
- **CPU cache 命中优化**： 使用缓存预取的方法，对每个线程分配一个连续的buffer，读取每个block中样本的梯度信息并存入连续的Buffer中。
- **Block 处理优化**：Block预先放入内存；Block按列进行解压缩；将Block划分到不同硬盘来提高吞吐



#### 4.4 XGBoost防止过拟合的方法

XGBoost在设计时，为了防止过拟合做了很多优化，具体如下：

- **目标函数添加正则项**：叶子节点个数+叶子节点权重的L2正则化
- **列抽样**：训练的时候只用一部分特征（不考虑剩余的block块即可）
- **子采样**：每轮计算可以不使用全部样本，使算法更加保守
- **shrinkage**: 可以叫学习率或步长，为了给后面的训练留出更多的学习空间



#### 4.5 XGBoost如何处理缺失值

XGBoost模型的一个优点就是允许特征存在缺失值。对缺失值的处理方式如下：

- 在特征k上寻找最佳 split point 时，不会对该列特征 missing 的样本进行遍历，而只对该列特征值为 non-missing 的样本上对应的特征值进行遍历，通过这个技巧来减少了为稀疏离散特征寻找 split point 的时间开销。
- 在逻辑实现上，为了保证完备性，会将该特征值missing的样本分别分配到左叶子结点和右叶子结点，两种情形都计算一遍后，选择分裂后增益最大的那个方向（左分支或是右分支），作为预测时特征值缺失样本的默认分支方向。
- 如果在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子结点。



#### 4.6 XGBoost中的一棵树的停止生长条件

- 当新引入的一次分裂所带来的增益Gain<0时，放弃当前的分裂。这是训练损失和模型结构复杂度的博弈过程。
- 当树达到最大深度时，停止建树，因为树的深度太深容易出现过拟合，这里需要设置一个超参数max_depth。
- 当引入一次分裂后，重新计算新生成的左、右两个叶子结点的样本权重和。如果任一个叶子结点的样本权重低于某一个阈值，也会放弃此次分裂。这涉及到一个超参数:最小样本权重和，是指如果一个叶子节点包含的样本数量太少也会放弃分裂，防止树分的太细。



#### 4.7 XGBoost如何处理不平衡数据

对于不平衡的数据集，例如用户的购买行为，肯定是极其不平衡的，这对XGBoost的训练有很大的影响，XGBoost有两种自带的方法来解决：

第一种，如果你在意AUC，采用AUC来评估模型的性能，那你可以通过设置scale_pos_weight来平衡正样本和负样本的权重。例如，当正负样本比例为1:10时，scale_pos_weight可以取10；

第二种，如果你在意概率(预测得分的合理性)，你不能重新平衡数据集(会破坏数据的真实分布)，应该设置max_delta_step为一个有限数字来帮助收敛（基模型为LR时有效）。



#### 4.8 XGBoost中如何对树进行剪枝

- 在目标函数中增加了正则项：使用叶子结点的数目和叶子结点权重的L2模的平方，控制树的复杂度。
- 在结点分裂时，定义了一个阈值，如果分裂后目标函数的增益小于该阈值，则不分裂。
- 当引入一次分裂后，重新计算新生成的左、右两个叶子结点的样本权重和。如果任一个叶子结点的样本权重低于某一个阈值（最小样本权重和），也会放弃此次分裂。
- XGBoost 先从顶到底建立树直到最大深度，再从底到顶反向检查是否有不满足分裂条件的结点，进行剪枝。



#### 4.9 XGBoost如何选择最佳分裂点？ 

XGBoost在训练前预先将特征按照特征值进行了排序，并存储为block结构，以后在结点分裂时可以重复使用该结构。

因此，可以采用特征并行的方法利用多个线程分别计算每个特征的最佳分割点，根据每次分裂后产生的增益，最终选择增益最大的那个特征的特征值作为最佳分裂点。

如果在计算每个特征的最佳分割点时，对每个样本都进行遍历，计算复杂度会很大，这种全局扫描的方法并不适用大数据的场景。XGBoost还提供了一种直方图近似算法，对特征排序后仅选择常数个候选分裂位置作为候选分裂点，极大提升了结点分裂时的计算效率。



#### 4.10 XGBoost的Scalable性如何体现

- **基分类器的scalability**：弱分类器可以支持CART决策树，也可以支持LR和Linear。
- **目标函数的scalability**：支持自定义loss function，只需要其一阶、二阶可导。有这个特性是因为泰勒二阶展开，得到通用的目标函数形式。
- **学习方法的scalability**：Block结构支持并行化，支持 Out-of-core计算。

#### 4.11 XGBoost如何评价特征的重要性

我们采用三种方法来评判XGBoost模型中特征的重要程度：

```
 官方文档：（1）weight - the number of times a feature is used to split the data across all trees. （2）gain - the average gain of the feature when it is used in trees. （3）cover - the average coverage of the feature when it is used in trees.
```

- **weight** ：该特征在所有树中被用作分割样本的特征的总次数。
- **gain** ：该特征在其出现过的所有树中产生的平均增益。
- **cover** ：该特征在其出现过的所有树中的平均覆盖范围。

> 注意：覆盖范围这里指的是一个特征用作分割点后，其影响的样本数量，即有多少样本经过该特征分割到两个子节点。



#### 4.11 XGBooost参数调优的一般步骤

首先需要初始化一些基本变量，例如：

- max_depth = 5
- min_child_weight = 1
- gamma = 0
- subsample, colsample_bytree = 0.8
- scale_pos_weight = 1

**(1) 确定learning rate和estimator的数量**

learning rate可以先用0.1，用cv来寻找最优的estimators

**(2) max_depth和 min_child_weight**

我们调整这两个参数是因为，这两个参数对输出结果的影响很大。我们首先将这两个参数设置为较大的数，然后通过迭代的方式不断修正，缩小范围。

max_depth，每棵子树的最大深度，check from range(3,10,2)。

min_child_weight，子节点的权重阈值，check from range(1,6,2)。

如果一个结点分裂后，它的所有子节点的权重之和都大于该阈值，该叶子节点才可以划分。

**(3) gamma**

也称作最小划分损失`min_split_loss`，check from 0.1 to 0.5，指的是，对于一个叶子节点，当对它采取划分之后，损失函数的降低值的阈值。

- 如果大于该阈值，则该叶子节点值得继续划分
- 如果小于该阈值，则该叶子节点不值得继续划分

**(4) subsample, colsample_bytree**

subsample是对训练的采样比例

colsample_bytree是对特征的采样比例

both check from 0.6 to 0.9

**(5) 正则化参数**

alpha 是L1正则化系数，try 1e-5, 1e-2, 0.1, 1, 100

lambda 是L2正则化系数

**(6) 降低学习率**

降低学习率的同时增加树的数量，通常最后设置学习率为0.01~0.1



#### 4.12 XGBoost模型如果过拟合了怎么解决

当出现过拟合时，有两类参数可以缓解：

第一类参数：用于直接控制模型的复杂度。包括`max_depth,min_child_weight,gamma` 等参数

第二类参数：用于增加随机性，从而使得模型在训练时对于噪音不敏感。包括`subsample,colsample_bytree`

还有就是直接减小`learning rate`，但需要同时增加`estimator` 参数。



#### 4.13 为什么XGBoost相比某些模型对缺失值不敏感

对存在缺失值的特征，一般的解决方法是：

- 离散型变量：用出现次数最多的特征值填充；
- 连续型变量：用中位数或均值填充；

一些模型如SVM和KNN，其模型原理中涉及到了对样本距离的度量，如果缺失值处理不当，最终会导致模型预测效果很差。

而树模型对缺失值的敏感度低，大部分时候可以在数据缺失时时使用。原因就是，一棵树中每个结点在分裂时，寻找的是某个特征的最佳分裂点（特征值），完全可以不考虑存在特征值缺失的样本，也就是说，如果某些样本缺失的特征值缺失，对寻找最佳分割点的影响不是很大。

XGBoost对缺失数据有特定的处理方法，[详情参考上篇文章第7题](http://mp.weixin.qq.com/s?__biz=Mzg2MjI5Mzk0MA==&mid=2247484181&idx=1&sn=8d0e51fb0cb974f042e66659e1daf447&chksm=ce0b59cef97cd0d8cf7f9ae1e91e41017ff6d4c4b43a4c19b476c0b6d37f15769f954c2965ef&scene=21#wechat_redirect)。

因此，对于有缺失值的数据在经过缺失处理后：

- 当数据量很小时，优先用朴素贝叶斯
- 数据量适中或者较大，用树模型，优先XGBoost
- 数据量较大，也可以用神经网络
- 避免使用距离度量相关的模型，如KNN和SVM





------

## LightGBM

参考：

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- [LightGBM 文档](https://link.zhihu.com/?target=https%3A//lightgbm.readthedocs.io/en/latest/)
- [论文阅读——LightGBM 原理](https://link.zhihu.com/?target=https%3A//blog.csdn.net/shine19930820/article/details/79123216)
- [机器学习算法之 LightGBM](https://link.zhihu.com/?target=https%3A//www.biaodianfu.com/lightgbm.html)
- [关于sklearn中的决策树是否应该用one-hot编码？ - 柯国霖的回答 - 知乎](https://www.zhihu.com/question/266195966/answer/306104444)
- [如何玩转LightGBM](https://link.zhihu.com/?target=https%3A//v.qq.com/x/page/k0362z6lqix.html)
- [A Communication-Efficient Parallel Algorithm for Decision Tree](https://link.zhihu.com/?target=http%3A//papers.nips.cc/paper/6381-a-communication-efficient-parallel-algorithm-for-decision-tree)
- [阿泽：【机器学习】决策树（下）——XGBoost、LightGBM（非常详细）](https://zhuanlan.zhihu.com/p/87885678)



LightGBM 由微软提出，主要用于解决 GDBT 在海量数据中遇到的问题，以便其可以更好更快地用于工业实践中。

从 LightGBM 名字我们可以看出其是轻量级（Light）的梯度提升机（GBM），**其相对 XGBoost 具有训练速度快、内存占用低的特点**。

那么 LightGBM 到底如何做到更快的训练速度和更低的内存使用的呢？

主要是从以下几点来解决：

1. 单边梯度抽样算法；
2. 直方图算法；
3. 互斥特征捆绑算法；
4. 基于最大深度的 Leaf-wise 的垂直生长算法；
5. 类别特征最优分割；
6. 特征并行和数据并行；
7. 缓存优化。



### 1. 数学原理

#### 1.1 单边梯度抽样算法

GBDT 算法的梯度大小可以反应样本的权重，**梯度越小说明模型拟合的越好**。

单边梯度抽样算法（Gradient-based One-Side Sampling, GOSS）利用这一信息对样本进行抽样，减少了大量梯度小的样本，在接下来的计算锅中**只需关注梯度高的样本，极大的减少了计算量**。

GOSS 算法**保留了梯度大的样本，并对梯度小的样本进行随机抽样**，为了不改变样本的数据分布，在计算增益时为梯度小的样本引入一个常数进行平衡。具体算法如下所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/LightGBM_fig1.png" alt="img" style="zoom:67%;" />

我们可以看到 GOSS 事先基于梯度的绝对值对样本进行排序（**无需保存排序后结果**），然后拿到前 a% 的梯度大的样本，和总体样本的 b%，在计算增益时，通过乘上 $\frac{1-a}{b}$ 来放大梯度小的样本的权重。

**一方面算法将更多的注意力放在训练不足的样本上，另一方面通过乘上权重来防止采样对原始数据分布造成太大的影响。**



#### 1.2 直方图算法

##### 直方图算法

直方图算法的基本思想是**将连续的特征离散化为 k 个离散特征，同时构造一个宽度为 k 的直方图用于统计信息（含有 k 个 bin）**。利用直方图算法我们无需遍历数据，只需要遍历 k 个 bin 即可找到最佳分裂点。

我们知道特征离散化的具有很多优点，如**存储方便、运算更快、鲁棒性强、模型更加稳定**等等。对于直方图算法来说最直接的有以下两个优点（以 k=256 为例）：

- **内存占用更小：**XGBoost 需要用 32 位的浮点数去存储特征值，并用 32 位的整形去存储索引，而 LightGBM 只需要用 8 位去存储直方图，相当于减少了 1/8；
- **计算代价更小：**计算特征分裂增益时，XGBoost 需要遍历一次数据找到最佳分裂点，而 LightGBM 只需要遍历特征 k 次，直接将时间复杂度从 $O(\#data * \#feature)$ 降低到  $O(k*\# feature)$，而我们知道  $\# data >> k$

虽然将特征离散化后无法找到精确的分割点，**可能会对模型的精度产生一定的影响**，但较粗的分割也起到了正则化的效果，**一定程度上降低了模型的方差**。



##### 直方图加速

在构建叶节点的直方图时，我们还可以**通过父节点的直方图与相邻叶节点的直方图相减的方式构建，从而减少了一半的计算量**。在实际操作过程中，我们还可以先计算直方图小的叶子节点，然后利用直方图作差来获得直方图大的叶子节点。

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/LightGBM_fig2.png)

##### 稀疏特征优化

XGBoost 在进行预排序时只考虑非零值进行加速，而 LightGBM 也采用类似策略：**只用非零特征构建直方图**。



#### 1.3 互斥特征捆绑算法

高维特征往往是稀疏的，**而且特征间可能是相互排斥的**（如两个特征不同时取非零值），如果两个特征并不完全互斥（如只有一部分情况下是不同时取非零值），可以用互斥率表示互斥程度。

**互斥特征捆绑算法（Exclusive Feature Bundling, EFB）指出如果将一些特征进行融合绑定，则可以降低特征数量**。

针对这种想法，我们会遇到两个问题：

1. 哪些特征可以一起绑定？
2. 特征绑定后，特征值如何确定？

**对于问题一：**EFB 算法利用特征和特征间的关系构造一个加权无向图，**并将其转换为图着色算法**。我们知道图着色是个 NP-Hard 问题，故采用**贪婪算法**得到近似解，具体步骤如下：

1. 构造一个加权无向图，顶点是特征，边是两个特征间互斥程度；
2. 根据节点的度进行降序排序，度越大，与其他特征的冲突越大；
3. 遍历每个特征，将它分配给现有特征包，或者新建一个特征包，是的总体冲突最小。

算法允许两两特征并不完全互斥来增加特征捆绑的数量，通过设置最大互斥率 $\gamma$ 来平衡算法的精度和效率。EFB 算法的伪代码如下所示：
<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/LightGBM_fig3.png" alt="img" style="zoom:50%;" />

我们看到时间复杂度为 $O(\# feature ^2)$ ，在特征不多的情况下可以应付，但如果特征维度达到百万级别，计算量则会非常大，为了改善效率，我们提出了一个更快的解决方案：**将 EFB 算法中通过构建图，根据节点度来排序的策略改成了根据非零值的技术排序，因为非零值越多，互斥的概率会越大**。

**对于问题二：**论文给出特征合并算法，其关键在于原始特征能从合并的特征中分离出来。假设 Bundle 中有两个特征值，A 取值为 [0, 10]、B 取值为 [0, 20]，为了保证特征 A、B 的互斥性，我们可以给特征 B 添加一个偏移量转换为 [10, 30]，Bundle 后的特征其取值为 [0, 30]，这样便实现了特征合并。具体算法如下所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/LightGBM_fig4.png" alt="img" style="zoom:67%;" />

#### 1.4 带深度限制的 Leaf-wise 算法

在建树的过程中有两种策略：

- Level-wise：基于层进行生长，直到达到停止条件；
- Leaf-wise：每次分裂增益最大的叶子节点，直到达到停止条件。

XGBoost 采用 Level-wise 的增长策略，方便并行计算每一层的分裂节点，**提高了训练速度**，但同时也因为节点增益过小增加了很多不必要的分裂，**降低了计算量**；

LightGBM 采用 Leaf-wise 的增长策略**减少了计算量**，配合最大深度的限制防止过拟合，由于每次都需要计算增益最大的节点，**所以无法并行分裂**。

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/LightGBM_fig5.png)

#### 1.5 类别特征最优分割

大部分的机器学习算法都不能直接支持类别特征，一般都会对类别特征进行编码，然后再输入到模型中。常见的处理类别特征的方法为 one-hot 编码，但我们知道对于决策树来说并不推荐使用 one-hot 编码：

1. **会产生样本切分不平衡问题，切分增益会非常小**。如，国籍切分后，会产生是否中国，是否美国等一系列特征，这一系列特征上只有少量样本为 1，大量样本为 0。这种划分的增益非常小：较小的那个拆分样本集，它占总样本的比例太小。无论增益多大，乘以该比例之后几乎可以忽略；较大的那个拆分样本集，它几乎就是原始的样本集，增益几乎为零；
2. 影响决策树学习：决策树依赖的是数据的统计信息，而独热码编码会把数据切分到零散的小空间上。在这些零散的小空间上统计信息不准确的，学习效果变差。**本质是因为独热码编码之后的特征的表达能力较差的**，特征的预测能力被人为的拆分成多份，每一份与其他特征竞争最优划分点都失败，最终该特征得到的重要性会比实际值低。

LightGBM 原生支持类别特征，采用 many-vs-many 的切分方式将类别特征分为两个子集，实现类别特征的最优切分。假设有某维特征有 k 个类别，则有 $2^{(k-1)}-1$ 中可能，时间复杂度为 $O(2^k)$ ，LightGBM 基于 Fisher 大佬的 《[On Grouping For Maximum Homogeneity](https://link.zhihu.com/?target=http%3A//www.csiss.org/SPACE/workshops/2004/SAC/files/fisher.pdf)》实现了 $O(klogk)$ 的时间复杂度。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/LightGBM_fig6.png" alt="img" style="zoom:67%;" />

上图为左边为基于 one-hot 编码进行分裂，后图为 LightGBM 基于 many-vs-many 进行分裂，在给定深度情况下，后者能学出更好的模型。

其基本思想在于**每次分组时都会根据训练目标对类别特征进行分类，根据其累积值 $\frac{\sum gradient}{\sum hessian}$ 对直方图进行排序，然后在排序的直方图上找到最佳分割**。

此外，LightGBM 还加了约束条件正则化，防止过拟合。

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/LightGBM_fig7.png)

我们可以看到这种处理类别特征的方式使得 AUC 提高了 1.5 个点，且时间仅仅多了 20%。



### 2 工程实现

#### 2.1 特征并行

传统的特征并行算法在于对数据进行垂直划分，然后使用不同机器找到不同特征的最优分裂点，基于通信整合得到最佳划分点，然后基于通信告知其他机器划分结果。

传统的特征并行方法有个很大的缺点：**需要告知每台机器最终划分结果，增加了额外的复杂度**（因为对数据进行垂直划分，每台机器所含数据不同，划分结果需要通过通信告知）。

**LightGBM 则不进行数据垂直划分**，每台机器都有训练集完整数据，在得到最佳划分方案后可在本地执行划分而减少了不必要的通信。

#### 2.2 数据并行

传统的数据并行策略主要为水平划分数据，然后本地构建直方图并整合成全局直方图，最后在全局直方图中找出最佳划分点。

这种数据划分有一个很大的缺点：**通讯开销过大**。如果使用点对点通信，一台机器的通讯开销大约为 $O(\# machine*\#feature * \# bin)$ ；如果使用集成的通信，则通讯开销为  $O(2*\#feature*\#bin)$，

**LightGBM 采用分散规约（Reduce scatter）的方式将直方图整合的任务分摊到不同机器上，从而降低通信代价，并通过直方图做差进一步降低不同机器间的通信**。

#### 2.3 投票并行

针对数据量特别大特征也特别多的情况下，可以采用投票并行。

**投票并行主要针对数据并行时数据合并的通信代价比较大的瓶颈进行优化，其通过投票的方式只合并部分特征的直方图从而达到降低通信量的目的**。

大致步骤为两步：

1. 本地找出 Top K 特征，并基于投票筛选出可能是最优分割点的特征；
2. 合并时只合并每个机器选出来的特征。

#### 2.4 缓存优化

上边说到 XGBoost 的预排序后的特征是通过索引给出的样本梯度的统计值，因其索引访问的结果并不连续，XGBoost 提出缓存访问优化算法进行改进。

而 LightGBM 所使用直方图算法对 Cache 天生友好：

1. 首先，所有的特征都采用相同的方法获得梯度（区别于不同特征通过不同的索引获得梯度），只需要对梯度进行排序并可实现连续访问，大大提高了缓存命中；
2. 其次，因为不需要存储特征到样本的索引，降低了存储消耗，而且也不存在 Cache Miss的问题。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/LightGBM_fig8.png" alt="img" style="zoom:67%;" />

### 3 与 XGBoost 的对比

本节主要总结下 LightGBM 相对于 XGBoost 的优点，从内存和速度两方面进行介绍。

#### 3.1 内存更小

1. XGBoost 使用预排序后需要记录特征值及其对应样本的统计值的索引，而 LightGBM 使用了直方图算法将特征值转变为 bin 值，**且不需要记录特征到样本的索引**，将空间复杂度从 $O(2*\#data)$ 降低为 $O(\# bin)$ ，极大的减少了内存消耗；
2. LightGBM 采用了直方图算法**将存储特征值转变为存储 bin 值**，降低了内存消耗；
3. LightGBM 在训练过程中**采用互斥特征捆绑算法减少了特征数量**，降低了内存消耗。

#### 3.2 速度更快

1. LightGBM 采用了**直方图算法将遍历样本转变为遍历直方图**，极大的降低了时间复杂度；
2. LightGBM 在训练过程中**采用单边梯度算法过滤掉梯度小的样本**，减少了大量的计算；
3. LightGBM 采用了**基于 Leaf-wise 算法的增长策略构建树**，减少了很多不必要的计算量；
4. LightGBM 采用**优化后的特征并行、数据并行方法加速计算**，当数据量非常大的时候还可以采用投票并行的策略；
5. LightGBM **对缓存也进行了优化，增加了 Cache hit 的命中率**。





