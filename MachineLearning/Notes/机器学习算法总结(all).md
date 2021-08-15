# 机器学习算法总结

------

### 1. 线性回归

#### 简述

在统计学中，**线性回归（Linear Regression）是利用称为线性回归方程的最小平方函数对一个或多个自变量和因变量之间关系进行建模的一种回归分析**。这种函数是一个或多个称为回归系数的模型参数的线性组合（自变量都是一次方）。**只有一个自变量的情况称为简单回归，大于一个自变量情况的叫做多元回归**。

优点：结果易于理解，计算上不复杂。
缺点：对非线性数据拟合不好。
适用数据类型：数值型和标称型数据。
算法类型：回归算法

线性回归的模型函数如下：
$$
h_\theta = \theta ^T x
$$

它的损失函数如下：
$$
J(\theta) = {1\over {2m}} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
$$
通过训练数据集寻找参数的最优解，即求解可以得到 $minJ(\theta)$ 的参数向量 $\theta$ ,其中这里的参数向量也可以分为参数 $w和b$ , 分别表示权重和偏置值。

求解最优解的方法有**最小二乘法和梯度下降法**。

#### 梯度下降法

梯度下降算法的思想如下(这里以一元线性回归为例)：

> 首先，我们有一个代价函数，假设是$J(\theta_0,\theta_1)$，我们的目标是$min_{\theta_0,\theta_1}J(\theta_0,\theta_1)$。
> 接下来的做法是：
>
> - 首先是随机选择一个参数的组合$(\theta_0,\theta_1)$,一般是设$\theta_0 = 0,\theta_1 = 0$;
> - 然后是不断改变$(\theta_0,\theta_1)$，并计算代价函数，直到一个**局部最小值**。之所以是**局部最小值**，是因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是**全局最小值**，选择不同的初始参数组合，可能会找到不同的局部最小值。
>   下面给出梯度下降算法的公式：

> repeat until convergence{
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0,\theta_1)\quad (for\quad j=0 \quad and\quad j=1)
$$
> } 

也就是在梯度下降中，不断重复上述公式直到收敛，也就是找到$\color{red}{局部最小值}$。其中符号`:=`是赋值符号的意思。

而应用梯度下降法到线性回归，则公式如下：
$$
\theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\ \\
 \theta_1 := \theta_1 - \alpha \frac{1}{m}\sum_{i=1}^m ((h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)})
$$

公式中的$\alpha$称为**学习率(learning rate)**，它决定了我们沿着能让代价函数下降程度最大的方向向下迈进的步子有多大。

在梯度下降中，还涉及到一个参数更新的问题，即更新$(\theta_0,\theta_1)$，一般我们的做法是**同步更新。**

最后，上述梯度下降算法公式实际上是一个叫**批量梯度下降(batch gradient descent)**，即它在每次梯度下降中都是使用整个训练集的数据，所以公式中是带有$\sum_{i=1}^m$。

#### 岭回归（ridge regression）:

岭回归是一种专用于共线性数据分析的有偏估计回归方法，实质上是一种改良的最小二乘估计法，通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价，获得回归系数更为符合实际、更可靠的回归方法，对病态数据的耐受性远远强于最小二乘法。

岭回归分析法是从根本上消除复共线性影响的统计方法。岭回归模型通过在相关矩阵中引入一个很小的岭参数K（1>K>0），并将它加到主对角线元素上，从而降低参数的最小二乘估计中复共线特征向量的影响，减小复共线变量系数最小二乘估计的方法，以保证参数估计更接近真实情况。岭回归分析将所有的变量引入模型中，比逐步回归分析提供更多的信息。

其他回归还可以参考 [各种回归全解：传统回归、逻辑回归、加权回归/核回归、岭回归、广义线性模型/指数族](http://blog.csdn.net/ownfed/article/details/41181665)。

#### 代码实现

Python实现的代码如下：

```python
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays

x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)

#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

#Predict Output
predicted= linear.predict(x_test)
```

上述是使用`sklearn`包中的线性回归算法的代码例子，下面是一个实现的具体例子。

```python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:36:06 2016

@author: cai
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn import linear_model

# 计算损失函数
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# 梯度下降算法
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            # 计算误差对权值的偏导数
            term = np.multiply(error, X[:, j])
            # 更新权值
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

dataPath = os.path.join('data', 'ex1data1.txt')
data = pd.read_csv(dataPath, header=None, names=['Population', 'Profit'])
# print(data.head())
# print(data.describe())
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
# 在数据起始位置添加1列数值为1的数据
data.insert(0, 'Ones', 1)
print(data.shape)

cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# 从数据帧转换成numpy的矩阵格式
X = np.matrix(X.values)
y = np.matrix(y.values)
# theta = np.matrix(np.array([0, 0]))
theta = np.matrix(np.zeros((1, cols-1)))
print(theta)
print(X.shape, theta.shape, y.shape)
cost = computeCost(X, y, theta)
print("cost = ", cost)

# 初始化学习率和迭代次数
alpha = 0.01
iters = 1000

# 执行梯度下降算法
g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g)

# 可视化结果
x = np.linspace(data.Population.min(),data.Population.max(),100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# 使用sklearn 包里面实现的线性回归算法
model = linear_model.LinearRegression()
model.fit(X, y)

x = np.array(X[:, 1].A1)
# 预测结果
f = model.predict(X).flatten()
# 可视化
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size(using sklearn)')
plt.show()
```

上述代码参考自 [Part 1 - Simple Linear Regression](http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/)。具体可以查看 [我的Github](https://github.com/ccc013/CodingPractise/blob/master/Python/MachineLearning/linearRegressionPractise.py)。

### 2. 逻辑回归

#### 简述

Logistic 回归算法基于 Sigmoid 函数，或者说 Sigmoid 就是逻辑回归函数。Sigmoid 函数定义如下：
$\frac{1}{1+e^{-z}}$。函数值域范围(0,1)。

因此逻辑回归函数的表达式如下：
$$
h_\theta(x) =g(\theta^T X) = \frac{1}{1+e^{-\theta^TX}} \\
其中，g(z) = \frac{1}{1+e^{-z}}
$$
其导数形式为：
$$
g\prime (z)  =  \frac{d}{dz} \frac{1}{1+e^{-z}} \\
		 = \frac{1}{(1+e^{-z})^2} (e^{-z}) \\
		 =  \frac{1}{1+e^{-z}} (1-  \frac{1}{1+e^{-z}}) \\
		 = g(z)(1-g(z))
$$

先验分布是伯努利分布



#### 代价函数

逻辑回归方法主要是用最大似然估计来学习的，所以单个样本的后验概率为：
$$
p(y | x; \theta) = (h_\theta(x))^y(1-h_\theta(x))^{1-y}
$$
到整个样本的后验概率就是:
$$
L(\theta) = p(y | X;\theta) \\
	      = \prod_{i=1}^{m} p(y^{(i)} | x^{(i)};\theta)\\
	      = \prod_{i=1}^{m} (h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
$$
其中，$P(y=1|x;\theta) = h_\theta(x), P(y=0|x;\theta)=1-h_\theta(x)$。

通过对数进一步简化有：$l(\theta) = logL(\theta) = \sum_{i=1}^{m}y^{(i)}logh(x^{(i)})+(1-y^{(i)})log(1-h(x^{(i)}))$.

而逻辑回归的代价函数就是$-l(\theta)$。也就是如下所示：
$$
J(\theta) = -\frac{1}{m} [\sum_{i=1}^my^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)}log(1-h_\theta(x^{(i)}))]
$$

同样可以使用梯度下降算法来求解使得代价函数最小的参数。其梯度下降法公式为：

![](https://img-blog.csdn.net/20170212181541232?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![](https://img-blog.csdn.net/20170212181600234?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 逻辑回归与 SVM

##### 相同点

1. 都是分类算法
2. 都是监督学习算法
3. 都是判别模型
4. 都能通过核函数方法针对非线性情况分类
5. 目标都是找一个分类超平面
6. 都能减少离群点的影响

##### 不同点

1. 损失函数不同，逻辑回归是cross entropy loss，svm是hinge loss
2. 逻辑回归在优化参数时所有样本点都参与了贡献，svm则只取离分离超平面最近的支持向量样本。这也是为什么逻辑回归不用核函数，它需要计算的样本太多。并且由于逻辑回归受所有样本的影响，当样本不均衡时需要平衡一下每一类的样本个数。
3. 逻辑回归对概率建模，svm对分类超平面建模
4. 逻辑回归是处理经验风险最小化，svm是结构风险最小化。这点体现在svm自带L2正则化项，逻辑回归并没有
5. 逻辑回归通过非线性变换减弱分离平面较远的点的影响，svm则只取支持向量从而消去较远点的影响
6. 逻辑回归是统计方法，svm是几何方法
7. 线性SVM依赖数据表达的距离测度，所以需要对数据先做normalization，LR不受其影响

#### 优缺点

##### 优点

1. 实现简单，广泛的应用于工业问题上；
2. 分类时计算量非常小，速度很快，存储资源低；
3. 便于观测样本概率分数
4. 对逻辑回归而言，多重共线性并不是问题，它可以结合L2正则化来解决该问题。

##### 缺点

1. 容易**欠拟合，一般准确度不太高**
2. 只能处理**两分类**问题（在此基础上衍生出来的softmax可以用于多分类），且必须**线性可分**；
3. **特征空间很大**时，逻辑回归的性能不是很好；
4. 不能很好地处理**大量多类特征或变量**
5. 对于非线性特征，需要进行转换。

适用数据类型：数值型和标称型数据。
类别：分类算法。
试用场景：解决二分类问题。

#### 代码实现

首先是采用`sklearn`包中的逻辑回归算法代码：

```python
#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# Create logistic regression object

model = LogisticRegression()

# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)

#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

#Predict Output
predicted= model.predict(x_test)
```

接下来则是应用例子，如下所示：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2016/10/19 21:35
@Author  : cai

实现多类的逻辑回归算法
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import minimize
from scipy.io import loadmat

# 定义Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义 cost函数
def costReg(theta, X, y, lambdas):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    h = X * theta.T
    first = np.multiply(-y, np.log(sigmoid(h)))
    second = np.multiply((1-y), np.log(1 - sigmoid(h)))
    reg = (lambdas / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

# 梯度下降算法的实现, 输出梯度对权值的偏导数
def gradient(theta, X, y, lambdas):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    # 计算误差
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((lambdas / len(X)) * theta)

    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()

# 实现一对多的分类方法
def one_vs_all(X, y, num_labels, lambdas):
    rows = X.shape[0]
    params = X.shape[1]

    # 每个分类器有一个 k * (n+1)大小的权值数组
    all_theta = np.zeros((num_labels, params + 1))

    # 增加一列，这是用于偏置值
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # 标签的索引从1开始
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # 最小化损失函数
        fmin = minimize(fun=costReg, x0=theta, args=(X, y_i, lambdas), method='TNC', jac=gradient)
        all_theta[i-1, :] = fmin.x

    return all_theta

def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # 增加一列，这是用于偏置值
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # 对每个训练样本计算其类的概率值
    h = sigmoid(X * all_theta.T)

    # 获取最大概率值的数组索引
    h_argmax = np.argmax(h, axis=1)
    # 数组是从0开始索引，而标签值是从1开始，所以需要加1
    h_argmax = h_argmax + 1

    return h_argmax

dataPath = os.path.join('data', 'ex3data1.mat')
# 载入数据
data = loadmat(dataPath)
print(data)
print(data['X'].shape, data['y'].shape)

# print(np.unique(data['y']))
# 测试
# rows = data['X'].shape[0]
# params = data['X'].shape[1]
#
# all_theta = np.zeros((10, params + 1))
#
# X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
#
# theta = np.zeros(params + 1)
#
# y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
# y_0 = np.reshape(y_0, (rows, 1))
# print(X.shape, y_0.shape, theta.shape, all_theta.shape)

all_theta = one_vs_all(data['X'], data['y'], 10, 1)
print(all_theta)

# 计算分类准确率
y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))
```

实现代码来自[Part 4 - Multivariate Logistic Regression](http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-4/)。具体可以查看[我的github](https://github.com/ccc013/CodingPractise/blob/master/Python/MachineLearning/mulLogisticRegressionPractise.py)。

### 3 决策树

#### 简介

**定义**：分类决策树模型是一种描述对实例进行分类的树形结构。决策树由结点和有向边组成。结点有两种类型：内部结点和叶结点。内部结点表示一个特征或属性，叶结点表示一个类。

决策树学习通常包括3个步骤：**特征选择、决策树的生成和决策树的修剪。**

决策树学习本质上是从训练数据集中归纳出一组分类规则，也可以说是**由训练数据集估计条件概率模型**。它使用的损失函数通常是**正则化的极大似然函数**，其策略是以损失函数为目标函数的最小化。

决策树学习的算法通常是一个递归地选择最优特征，并根据该特征对训练数据进行分割，使得对各个子数据集有一个最好的分类的过程。

**决策树的生成对应于模型的局部选择，决策树的剪枝对应于模型的全局选择。决策树的生成只考虑局部最优，相对地，决策树的剪枝则考虑全局最优。**

#### 特征选择

特征选择的准则通常是**信息增益或者信息增益比**。

首先是给出信息熵的计算公式 $H(p) = -\sum_{i=1}^{n} p_i log p_i$，**熵越大，随机变量的不确定性就越大**。

公式中 $p_i$ 表示随机变量 X 属于类别 $i$ 的概率，因此 $n$ 表示类别的总数。

条件熵的定义为：$H(Y|X) = \sum_{i=1}^n p_iH(Y|X=x_i)$

已经有了**熵作为衡量训练样例集合纯度**的标准，现在可以定义属性分类训练数据的效力的度量标准。这个标准被称为“**信息增益（information gain）**”。简单的说，一个属性的信息增益就是由于使用这个属性分割样例而导致的期望熵降低(或者说，**样本按照某属性划分时造成熵减少的期望,个人结合前面理解，总结为用来衡量给定的属性区分训练样例的能力**)。更精确地讲，**一个属性 A 相对样例集合 S 的信息增益 Gain(S,A) 被定义为**：

![](https://img-blog.csdn.net/20170213171939623?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其中 Values(A) 是属性 A 所有可能值的集合，Sv 是 S 中属性 A 的值为 v 的子集，注意上式第一项就是原集合S 的熵，第二项是用 A 分类 S 后的熵的期望值，第二项描述的期望熵就是每个子集的熵的加权和，权值为属性 Sv 的样例占原始样例 S 的比例 |Sv|/|S| ,所以 Gain(S,A) 是由于知道属性 A 的值而导致的期望熵减少，换句话来讲，Gain(S,A) 是由于给定属性 A 的值而得到的关于目标函数值的信息。

信息增益的缺点是**存在偏向于选择取值较多的特征的问题**。为了解决这个问题，可以使用**信息增益比**。

因此，特征 A 对训练数据集 D 的信息增益比 $g_R(D,A)$ 的定义如下：
$$
g_R(D, A) = \frac{g(D,A)}{H_A(D)}
$$
其中 $g(D,A)$ 是信息增益，而 $H_A(D)=-\sum_{i=1}^n \frac{|D_i|}{|D|} log_2 \frac{|D_i|}{|D|}$ ,其中 $n$ 是特征 A 取值的个数。

不过对于信息增益比，其也存在**对可取值数目较少的属性有所偏好的问题**。

#### 决策树的生成

接下来会介绍决策树生成的算法，包括**ID3, C4.5**算法。

##### ID3算法

ID3 算法的核心是在决策树各个结点上应用信息增益准则选择特征，递归地构建决策树。具体步骤如下所示：

![](https://img-blog.csdn.net/20170213204003966?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

ID3 算法思路总结如下：

1. 首先是针对当前的集合，计算每个特征的信息增益
2. 然后选择信息增益最大的特征作为当前节点的决策决策特征
3. 根据特征不同的类别划分到不同的子节点（比如年龄特征有青年，中年，老年，则划分到 3 颗子树）
4. 然后继续对子节点进行递归，直到所有特征都被划分

**ID3的缺点是**

1）容易造成过度拟合（over fitting）； 
2）只能处理标称型数据（离散型）； 
3）信息增益的计算依赖于特征数目较多的特征，而属性取值最多的属性并不一定最优； 
4）抗噪性差，训练例子中正例和反例的比例较难控制

##### C4.5算法

C4.5算法继承了 ID3 算法的优点，并在以下几方面对 ID3 算法进行了改进：

- 用信息增益率来选择属性，克服了用信息增益选择属性时偏向选择取值多的属性的不足；
- 在树构造过程中进行剪枝；
- 能够完成对连续属性的离散化处理；
- 能够对不完整数据进行处理。

C4.5算法有如下优点：**产生的分类规则易于理解，准确率较高**。

其缺点是：

1. **算法低效**，在构造树的过程中，需要对数据集进行多次的顺序扫描和排序，因而导致算法的低效 
2. **内存受限，**只适合于能够驻留于内存的数据集，当训练集大得无法在内存容纳时程序无法运行。 

算法的实现过程如下:

![](https://img-blog.csdn.net/20170213204817800?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**实际上由于信息增益比的缺点，C4.5 算法并没有直接选择信息增益比最大的候选划分属性，而是先从候选划分属性中找出信息增益高于平均水平的属性，再从中选择信息增益比最高的。**

**无论是 ID3 还是 C4.5 最好在小数据集上使用，决策树分类一般只适用于小数据。当属性取值很多时最好选择 C4.5 算法，ID3 得出的效果会非常差。**

#### 剪枝

在生成树的过程中，如果没有剪枝的操作的话，就会长成每一个叶都是单独的一类的样子。这样对我们的训练集是完全拟合的，但是对测试集则是非常不友好的，泛化能力不行。**因此，我们要减掉一些枝叶，使得模型泛化能力更强。** 
根据剪枝所出现的时间点不同，分为预剪枝和后剪枝。**预剪枝是在决策树的生成过程中进行的；后剪枝是在决策树生成之后进行的。**

决策树的剪枝往往是通过极小化决策树整体的损失函数或代价函数来实现的。简单来说，就是对比剪枝前后整体树的损失函数或者是准确率大小来判断是否需要进行剪枝。

决策树剪枝算法有多种，具体参考[决策树剪枝算法](http://blog.csdn.net/yujianmin1990/article/details/49864813)这篇文章。

#### CART

分类回归树(Classification And Regression Tree)是一个**决策二叉树**，在通过递归的方式建立，每个节点在分裂的时候都是希望通过最好的方式将剩余的样本划分成两类，这里的分类指标：

1. **分类树：基尼指数最小化(gini_index)**
2. **回归树：平方误差最小化**

分类树的生成步骤如下所示：

![](https://img-blog.csdn.net/20170213212619889?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

简单总结如下：

1. 首先是根据当前特征计算他们的基尼增益
2. 选择**基尼增益最小**的特征作为划分特征
3. 从该特征中查找基尼指数最小的分类类别作为最优划分点
4. 将当前样本划分成两类，一类是划分特征的类别等于最优划分点，另一类就是不等于
5. 针对这两类递归进行上述的划分工作，直达所有叶子指向同一样本目标或者叶子个数小于一定的阈值

基尼指数的计算公式为$Gini(p) = 1 - \sum_{k=1}^K p_k^2$。K是类别的数目，$p_k$表示样本属于第k类的概率值。它可以用来度量分布不均匀性（或者说不纯），总体的类别越杂乱，GINI指数就越大（跟熵的概念很相似）。

给定一个数据集D，在特征A的条件下，其基尼指数定义为$Gini(D,A) = \sum_{i=1}^n \frac{|D_i|}{|D|} Gini(D_i)$。

回归树：

> 回归树是以平方误差最小化的准则划分为两块区域

1. 遍历特征计算最优的划分点$s$，
   使其最小化的平方误差是：$min \{min(∑_i^{R1}((y_i−c_1)^2))+min(∑_i^{R2}((y_i−c_2)^2))\}$
   计算根据s划分到左侧和右侧子树的目标值与预测值之差的平方和最小，这里的预测值是两个子树上输入$x_i$样本对应$y_i$的均值

2. 找到最小的划分特征j以及其最优的划分点$s$,根据特征$j$以及划分点$s$将现有的样本划分为两个区域，一个是在特征$j$上小于等于$s$，另一个在特征$j$上大于$s$
   $$
   R1(j)= \{x|x(j)≤s\} \\
   R2(j)=\{x|x(j)>s\}  \\
   c_m = \frac{1}{N_m} \sum_{x_i \in R_m(j, s)} y_i, m = 1,2, \quad x\in R_m
   $$

3. 进入两个子区域按上述方法继续划分，直到到达停止条件

回归树的缺点：

- **不如线性回归普遍；**
- **要求大量训练数据；**
- **难以确定某个特征的整体影响；**
- **比线性回归模型难解释**

关于CART剪枝的方法可以参考[决策树系列（五）——CART](http://www.cnblogs.com/yonghao/p/5135386.html)。

#### 停止条件

1. 直到每个叶子节点都只有一种类型的记录时停止，（这种方式很容易过拟合）
2. 另一种是当叶子节点的样本数目小于一定的阈值或者节点的信息增益小于一定的阈值时停止

#### 关于特征与目标值

1. 特征离散 目标值离散：可以使用ID3，cart
2. 特征连续 目标值离散：将连续的特征离散化 可以使用ID3，cart

#### 连续值属性的处理

​	**C4.5既可以处理离散型属性，也可以处理连续性属性。**在选择某节点上的分枝属性时，对于离散型描述属性，C4.5的处理方法与ID3相同。对于连续分布的特征，其处理方法是：

　　**先把连续属性转换为离散属性再进行处理。**虽然本质上属性的取值是连续的，但对于有限的采样数据它是离散的，如果有N条样本，那么我们有N-1种离散化的方法：$<=v_j$的分到左子树，$>v_j$的分到右子树。计算这N-1种情况下最大的信息增益率。另外，对于连续属性先进行排序（升序），只有在决策属性（即分类发生了变化）发生改变的地方才需要切开，这可以显著减少运算量。**经证明，在决定连续特征的分界点时采用增益这个指标**（因为若采用增益率，splittedinfo影响分裂点信息度量准确性，若某分界点恰好将连续特征分成数目相等的两部分时其抑制作用最大），**而选择属性的时候才使用增益率这个指标能选择出最佳分类特征。**

在C4.5中，对连续属性的处理如下：

1、对特征的取值进行升序排序

2、两个特征取值之间的中点作为可能的分裂点，将数据集分成两部分，计算**每个可能的分裂点的信息增益**（InforGain）。**优化算法就是只计算分类属性发生改变的那些特征取值。**

3、选择修正后**信息增益(InforGain)最大的分裂点**作为该特征的最佳分裂点

4、计算**最佳分裂点的信息增益率（Gain Ratio）作为特征的Gain Ratio**。注意，此处需对最佳分裂点的信息增益进行修正：减去log2(N-1)/|D|（N是连续特征的取值个数，D是训练数据数目，此修正的原因在于：**当离散属性和连续属性并存时，C4.5算法倾向于选择连续特征做最佳树分裂点**）

#### 决策树的分类与回归

- 分类树
  输出叶子节点中所属类别最多的那一类
- 回归树
  输出叶子节点中各个样本值的平均值

#### 理想的决策树

1. 叶子节点数尽量少
2. 叶子节点的深度尽量小(太深可能会过拟合)

#### 过拟合原因

采用上面算法生成的决策树在事件中往往会导致过滤拟合。也就是该决策树对训练数据可以得到很低的错误率，但是运用到测试数据上却得到非常高的错误率。过渡拟合的原因有以下几点：

- **噪音数据**：训练数据中存在噪音数据，决策树的某些节点有噪音数据作为分割标准，导致决策树无法代表真实数据。
- **缺少代表性数据**：训练数据没有包含所有具有代表性的数据，导致某一类数据无法很好的匹配，这一点可以通过观察混淆矩阵（Confusion Matrix）分析得出。
- **多重比较（Mulitple Comparition）**：举个列子，股票分析师预测股票涨或跌。假设分析师都是靠随机猜测，也就是他们正确的概率是0.5。每一个人预测10次，那么预测正确的次数在8次或8次以上的概率为 [![image](http://images.cnitblog.com/blog/349490/201303/15154352-dd92afccc91e4e2e9a08578d8ba9ab04.png)](http://images.cnitblog.com/blog/349490/201303/15154352-831a596c9a3c4fcca3d6e8f863b2f91f.png)，只有5%左右，比较低。但是如果50个分析师，每个人预测10次，选择至少一个人得到8次或以上的人作为代表，那么概率为 [![image](http://images.cnitblog.com/blog/349490/201303/15154353-c827fc2a20e74c31a3f6a1f1d64a436c.png)](http://images.cnitblog.com/blog/349490/201303/15154352-be8974a2a79a4662a4579210187d31fa.png)，概率十分大，随着分析师人数的增加，概率无限接近1。但是，选出来的分析师其实是打酱油的，他对未来的预测不能做任何保证。上面这个例子就是**多重比较**。这一情况和决策树选取分割点类似，需要在每个变量的每一个值中选取一个作为分割的代表，所以选出一个噪音分割标准的概率是很大的。

#### 解决决策树的过拟合

1. 剪枝
   1. 前置剪枝：在分裂节点的时候设计比较苛刻的条件，如不满足则直接停止分裂（这样干决策树无法到最优，也无法得到比较好的效果）
   2. 后置剪枝：在树建立完之后，用单个节点代替子树，节点的分类采用子树中主要的分类（这种方法比较浪费前面的建立过程）
2. 交叉验证
3. 随机森林

#### 优缺点

##### 优点

1. 计算量简单，可解释性强，比较适合处理有缺失属性值的样本，能够处理不相关的特征；

2. 效率高，决策树只需要一次构建，反复使用。

3. 训练时间复杂度较低，预测的过程比较快速，每一次预测的最大计算次数不超过决策树的深度。对于N个样本，每个样本都包含M个属性，在不考虑连续属性离散化以及子树增长的代价情况下，决策树算法的平均时间复杂度仅为$O(M*N*logN)$。构建一个决策树，最坏情况下的复杂度是$O(tree  depth)$，其中树的深度一般呈对数增长。


##### 缺点

1. 单颗决策树分类能力弱，并且对连续值变量难以处理；
2. 容易过拟合（后续出现了随机森林，减小了过拟合现象）；
3. 可能或陷于局部最小值中
4. 没有在线学习

#### 代码实现

使用 sklearn 中决策树函数的简单代码例子如下所示：

```python
#Import Library
#Import other necessary libraries like pandas, numpy...

from sklearn import tree
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  

# model = tree.DecisionTreeRegressor() for regression

# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)

#Predict Output
predicted= model.predict(x_test)
```

决策树的代码在开源库 OpenCV 中有实现，具体的源码分析可以参考[Opencv2.4.9源码分析——Decision Trees](http://blog.csdn.net/zhaocj/article/details/50503450)，这篇文章也比较详细总结了决策树的知识点以及对OpenCV中决策树部分的源码进行了分析。

### 4 随机森林

#### 简介

> **随机森林**指的是利用多棵树对样本进行训练并预测的一种分类器。它是由多棵CART(Classification And Regression Tree)构成的。对于每棵树，其**使用的训练集是从总的训练集中有放回采样出来的**，这意味着总训练集中有些样本可能多次出现在一棵树的训练集中，也可能从未出现在一棵树的训练集中。在训练每棵树的节点时，**使用的特征是从所有特征中按照一定比例随机地无放回的抽取的**，假设总的特征数是`M`,则这个比例可以是$\sqrt(M), \frac{1}{2} \sqrt(M), 2\sqrt(M)$。

#### 训练过程

随机森林的训练过程可以总结如下：

(1)给定训练集`S`，测试集`T`，特征维数`F`。确定参数：使用到的CART的数量`t`，每棵树的深度`d`，每个节点使用到的特征数量`f`，终止条件：节点上最少样本数`s`，节点上最少的信息增益`m`

对于第1-t棵树，`i=1-t`：

(2)从S中有放回的抽取大小和S一样的训练集S(i)，作为根节点的样本，从根节点开始训练

(3)如果当前节点上达到终止条件，则设置当前节点为叶子节点，如果是分类问题，该叶子节点的预测输出为当前节点样本集合中数量最多的那一类`c(j)`，概率`p`为`c(j)`占当前样本集的比例；如果是回归问题，预测输出为当前节点样本集各个样本值的平均值。然后继续训练其他节点。**如果当前节点没有达到终止条件，则从F维特征中无放回的随机选取f维特征。利用这f维特征，寻找分类效果最好的一维特征`k`及其阈值`th`，当前节点上样本第k维特征小于`th`的样本被划分到左节点，其余的被划分到右节点。**继续训练其他节点。有关分类效果的评判标准在后面会讲。

(4)重复(2)(3)直到所有节点都训练过了或者被标记为叶子节点。

(5)重复(2),(3),(4)直到所有CART都被训练过。

#### 预测过程

预测过程如下：

对于第1-t棵树，i=1-t：

(1)从当前树的根节点开始，根据当前节点的阈值th，判断是进入左节点(`<th`)还是进入右节点(`>=th`)，直到到达，某个叶子节点，并输出预测值。

(2)重复执行(1)直到所有t棵树都输出了预测值。**如果是分类问题，则输出为所有树中预测概率总和最大的那一个类，即对每个c(j)的p进行累计；如果是回归问题，则输出为所有树的输出的平均值。**

**有关分类效果的评判标准，因为使用的是CART，因此使用的也是CART的评判标准，和C3.0,C4.5都不相同。**

**对于分类问题（将某个样本划分到某一类），也就是离散变量问题，CART使用Gini值作为评判标准。定义为$Gini(p) = 1 - \sum_{k=1}^K p_k^2$，  $p_k$为当前节点上数据集中第k类样本的比例。**

**例如：分为2类，当前节点上有100个样本，属于第一类的样本有70个，属于第二类的样本有30个，则$Gini=1-0.7×07-0.3×03=0.42$，可以看出，类别分布越平均，Gini值越大，类分布越不均匀，Gini值越小。在寻找最佳的分类特征和阈值时，评判标准为：$argmax（Gini-GiniLeft-GiniRight）$，即寻找最佳的特征f和阈值th，使得当前节点的Gini值减去左子节点的Gini和右子节点的Gini值最大。**

**对于回归问题，相对更加简单，直接使用$argmax(Var-VarLeft-VarRight)$作为评判标准，即当前节点训练集的方差Var减去减去左子节点的方差VarLeft和右子节点的方差VarRight值，求其最大值**。

#### 特征重要性度量

计算某个特征 X 的重要性时，具体步骤如下：

1. 对每一颗决策树，选择相应的袋外数据（out of bag，OOB）计算袋外数据误差，记为errOOB1.

   所谓袋外数据是指，每次建立决策树时，通过重复抽样得到一个数据用于训练决策树，这时还有**大约1/3的数据没有被利用**，没有参与决策树的建立。这部分数据可以用于对决策树的性能进行评估，计算模型的预测错误率，称为袋外数据误差。

   **这已经经过证明是无偏估计的,所以在随机森林算法中不需要再进行交叉验证或者单独的测试集来获取测试集误差的无偏估计。**

2. 随机对袋外数据OOB所有样本的特征X加入**噪声干扰**（可以随机改变样本在特征X处的值），再次计算袋外数据误差，记为errOOB2。

3. 假设森林中有N棵树，则特征X的重要性=$∑\frac{errOOB2-errOOB1}{N}$。这个数值之所以能够说明特征的重要性是因为，**如果加入随机噪声后，袋外数据准确率大幅度下降（即errOOB2上升），说明这个特征对于样本的预测结果有很大影响，进而说明重要程度比较高。**

#### 特征选择

在特征重要性的基础上，特征选择的步骤如下：

1. 计算每个特征的重要性，并按降序排序
2. 确定要剔除的比例，依据特征重要性剔除相应比例的特征，得到一个新的特征集
3. 用新的特征集重复上述过程，直到剩下m个特征（m为提前设定的值）。
4. 根据上述过程中得到的各个特征集和特征集对应的袋外误差率，选择袋外误差率最低的特征集。

#### 优缺点

##### 优点

- 在数据集上表现良好，在当前的很多数据集上，相对其他算法有着很大的优势
- 它能够处理很高维度（特征很多）的数据，并且不用做特征选择
- 可以评估特征的重要性
- 在创建随机森林的时候，对 generlization error 使用的是无偏估计
- 训练速度快，容易做成并行化方法
- 在训练过程中，能够检测到特征间的互相影响
- 实现比较简单
- 对于不平衡的数据集来说，它可以平衡误差
- 可以应用在特征缺失的数据集上，并仍然有不错的性能

##### 缺点

1. 随机森林已经被证明在某些**噪音较大**的分类或回归问题上会过拟合
2. 对于有不同取值的属性的数据，**取值划分较多的属性会对随机森林产生更大的影响**，所以随机森林在这种数据上产出的属性权值是不可信的。

#### 与 bagging 的区别

1. Random forest是选与**输入样本的数目相同多**的样本（可能一个样本会被选取多次，同时也会造成一些样本不会被选取到），而bagging一般选取比**输入样本的数目少**的样本；

2. bagging是用**全部特征**来得到分类器，而Random forest是需要从全部特征中**选取其中的一部分**来训练得到分类器； **一般Random forest效果比bagging效果好！**

   

#### 代码实现

简单使用 sklearn 中随机森林算法的例子：

```python
#Import Library
from sklearn.ensemble import RandomForestClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# Create Random Forest object
model= RandomForestClassifier()

# Train the model using the training sets and check score
model.fit(X, y)

#Predict Output
predicted= model.predict(x_test)
```

此外，OpenCV中也实现了随机森林算法。具体使用例子可以查看[RandomForest随机森林总结](http://www.cnblogs.com/hrlnw/p/3850459.html)。

### 5 SVM

#### 简介

> SVM是一种二类分类模型，其基本模型定义为特征空间上的间隔最大的线性分类器，即支持向量机的学习策略便是**间隔最大化**，最终可转化为一个凸二次规划问题的求解。或者简单的可以理解为就是在高维空间中寻找一个合理的超平面将数据点分隔开来，其中涉及到非线性数据到高维的映射以达到数据线性可分的目的。

训练数据线性可分时，通过**硬间隔最大化**，学习一个线性分类器，即**线性可分支持向量机**，又称为硬间隔支持向量机；训练数据近似线性可分时，通过**软间隔最大化**，也学习一个线性分类器，即**线性支持向量机**，也称为软间隔支持向量机；训练数据线性不可分时，通过**使用核技巧和软间隔最大化**，学习**非线性支持向量机**。

#### 线性可分支持向量机和硬间隔最大化

接下来主要是手写的笔记，主要参考

* 《统计学习方法》
* [SVM详解(包含它的参数C为什么影响着分类器行为)-scikit-learn拟合线性和非线性的SVM](http://blog.csdn.net/xlinsist/article/details/51311755)
* [机器学习常见算法个人总结（面试用）](http://kubicode.me/2015/08/16/Machine%20Learning/Algorithm-Summary-for-Interview/)
* [SVM-支持向量机算法概述](http://blog.csdn.net/passball/article/details/7661887)
* [机器学习算法与Python实践之（二）支持向量机（SVM）初级](http://blog.csdn.net/zouxy09/article/details/17291543)
* [机器学习算法与Python实践之（三）支持向量机（SVM）进阶](http://blog.csdn.net/zouxy09/article/details/17291805)
* [机器学习算法与Python实践之（四）支持向量机（SVM）实现](http://blog.csdn.net/zouxy09/article/details/17292011)

![](https://img-blog.csdn.net/20170216123438940?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

关于凸函数，**凸集是指有这么一个点的集合，其中任取两个点连一条直线，这条线上的点仍然在这个集合内部，因此说“凸”是很形象的**。例如下图，对于凸函数（在数学表示上，满足约束条件是仿射函数，也就是线性的Ax+b的形式）来说，局部最优就是全局最优，但对非凸函数来说就不是了。

![](https://img-blog.csdn.net/20170216133841807?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



![](https://img-blog.csdn.net/20170216123848270?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

> **支持向量**是训练数据集的样本点中与分离超平面距离最近的样本点的实例。

如下图所示：

![](https://img-blog.csdn.net/20170216125100136?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

图中对类1，即Class 1的支持向量就在超平面$H_2: wx+b=-1$上，而对于类2，即Class 2正类的支持向量就在超平面$H_1: wx+b=1$上。而在这两个超平面中间的距离，即图中标着`m`的距离称为**间隔**，它依赖于分离超平面的法向量$w$，等于$\frac{2}{|| w ||}$，而两个超平面$H_1 和H_2$称为**间隔平面**。

在决定分离超平面时只有支持向量其作用，其他实例点并不起作用。如果移动支持向量将改变所求的解。**正是因为支持向量在确定分离超平面中起着决定性作用 ，所以将这种分类模型称为支持向量机。**支持向量的个数一般很少，所以支持向量机由很少的“重要的”训练样本确定。

#### 线性支持向量机和软间隔最大化

![](https://img-blog.csdn.net/20170216125731017?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![](https://img-blog.csdn.net/20170216125743970?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 核函数

![](https://img-blog.csdn.net/20170216140507376?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![](https://img-blog.csdn.net/20170216140517591?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

采用不同的核函数就相当于采用不同的相似度的衡量方法。从计算的角度，不管$Φ(x)$变换的空间维度有多高，甚至是无限维（函数就是无限维的），**这个空间的线性支持向量机的求解都可以在原空间通过核函数进行**，这样就可以避免了高维空间里的计算，而**计算核函数的复杂度和计算原始样本内积的复杂度没有实质性的增加**。

> 一般情况下RBF效果是不会差于Linear
> 但是时间上RBF会耗费更多
> 下面是吴恩达的见解：
>
> 1. 如果Feature的数量很大，跟样本数量差不多，这时候选用LR或者是Linear Kernel的SVM
> 2. 如果Feature的数量比较小，样本数量一般，不算大也不算小，选用SVM+Gaussian Kernel
> 3. 如果Feature的数量比较小，而样本数量很多，需要手工添加一些feature变成第一种情况

更多有关核函数可以参考[【模式识别】SVM核函数](http://blog.csdn.net/xiaowei_cqu/article/details/35993729)和[SVM的核函数如何选取?--知乎](https://www.zhihu.com/question/21883548)。

这里总结一下：支持向量机的基本思想可以概括为，**首先通过非线性变换将输入空间变换到一个高维的空间，然后在这个新的空间求最优分类面即最大间隔分类面，而这种非线性变换是通过定义适当的内积核函数来实现的。**SVM实际上是根据统计学习理论依照**结构风险最小化**的原则提出的，要求实现两个目的：**1）两类问题能够分开（经验风险最小）2）margin最大化（风险上界最小），既是在保证风险最小的子集中选择经验风险最小的函数。**

#### 优缺点

##### 优点

1. 使用核函数可以向高维空间进行映射
2. 使用核函数可以解决非线性的分类
3. 分类思想很简单，就是将样本与决策面的间隔最大化
4. 分类效果较好

##### 缺点

1. 对大规模数据训练比较困难
2. 无法直接支持多分类，但是可以使用间接的方法来做
3. 噪声也会影响SVM的性能，因为SVM主要是由少量的支持向量决定的。

#### 多类分类

##### 直接法

直接在目标函数上进行修改，将多个分类面的参数求解合并到一个最优化问题中，通过求解该优化就可以实现多分类。但是计算复杂度很高，实现起来较为困难。

##### 间接法

> **1 一对多方法**就是每次训练的时候设置其中某个类为一类，其余所有类为另一个类。比如有A,B,C,D四个类，第一次A是一个类，{B,C,D}是一个类，训练一个分类器，第二次B是一个类，然后A,C,D是一个类，训练一个分类器，依次类推。因此，如果总共有$n$个类，最终将训练$n$个分类器。测试的时候，将测试样本都分别送入所有分类器中，取得到最大值的类别作为其分类结果。这是因为到分类面距离越大，分类越可信。

这种方法的优点是**每个优化问题的规模比较小，而且分类速度很快**，因为分类器数目和类别数目相同；但是，有时会出现这样两种情况：对一个测试样本，每个分类器都得到它属于分类器所在类别；或者都不属于任意一个分类器的类别。**前者称为分类重叠现象，后者叫不可分类现象**。前者可以任意选择一个结果或者就按照其到每个超平面的距离来分，哪个远选哪个类别；而后者只能分给新的第$n+1$个类别了。最大的缺点还是由于将$n-1$个类别作为一个类别，其数目会数倍于只有1个类的类别，这样会人为造成**数据集偏斜**的问题。

> **2 一对一方法**是任意两个类都训练一个分类器，那么$n$个类就需要$\frac{n(n-1)}{2}$个分类器。预测的时候通过投票选择最终结果。

这个方法同样会有分类重叠的现象，但不会有不可分类现象，因为不可能所有类别的票数都是0。**这种方法会比较高效，每次训练使用的样本其实就只有两类数据，而且预测会比较稳定，但是缺点是预测时间会很久。**

> **3 层次支持向量机（H-SVMs）。**层次分类法首先将所有类别分成两个子类，再将子类进一步划分成两个次级子类，如此循环，直到得到一个单独的类别为止。

> **4 DAG-SVMS**是由Platt提出的决策导向的循环图DDAG导出的,是针对“一对一”SVMS存在误分、拒分现象提出的。

#### 序列最小最优化算法(SMO)

SMO是用于快速求解SVM的，是一种启发式算法。
基本思路如下：如果所有变量的解都满足此最优化问题的**KKT**条件，那么这个最优化问题的解就得到了。**因为KKT条件是该最优化问题的充分必要条件**。否则它选择**凸二次规划的两个变量**，其他的变量保持不变，然后根据这两个变量构建一个二次规划问题，这个二次规划关于这两个变量解会更加的接近原始二次规划的解，通过这样的子问题划分可以大大增加整个算法的计算速度，关于这两个变量：

1. 其中一个是**严重违反KKT条件**的一个变量
2. 另一个变量由约束条件自动确定。

整个SMO算法分为两部分：求解两个变量二次规划的解析方法和选择变量的启发式方法。

![](https://img-blog.csdn.net/20170216155938379?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![](https://img-blog.csdn.net/20170216155947160?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

> **SMO称选择第一个变量的过程为外层循环**。外层循环在训练样本中选取违反KKT条件最严重的样本点，并将其对应的变量作为第一个变量。具体的，检验训练样本($x_i, y_i$)是否满足KKT条件，也就是：

![这里写图片描述](https://img-blog.csdn.net/20170216160228381?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

 该检验是在$ε$范围内进行的。在检验过程中，外层循环**首先遍历所有满足条件$0<α_j<C$的样本点，即在间隔边界上的支持向量点，**检验他们是否满足KKT条件，然后选择违反KKT条件最严重的$α_i$。如果这些样本点都满足KKT条件，那么遍历整个训练集，检验他们是否满足KKT条件，然后选择违反KKT条件最严重的$α_i$。

       **优先选择遍历非边界数据样本，因为非边界数据样本更有可能需要调整，边界数据样本常常不能得到进一步调整而留在边界上**。由于大部分数据样本都很明显不可能是支持向量，因此对应的$α$乘子一旦取得零值就无需再调整。遍历非边界数据样本并选出他们当中违反KKT 条件为止。**当某一次遍历发现没有非边界数据样本得到调整时，遍历所有数据样本，以检验是否整个集合都满足KKT条件**。如果整个集合的检验中又有数据样本被进一步进化，则有必要再遍历非边界数据样本。这样，**不停地在遍历所有数据样本和遍历非边界数据样本之间切换，直到整个样本集合都满足KKT条件为止**。**以上用KKT条件对数据样本所做的检验都以达到一定精度ε就可以停止为条件。如果要求十分精确的输出算法，则往往不能很快收敛。**

       对整个数据集的遍历扫描相当容易，而实现对非边界$α_i$的扫描时，首先需要将所有非边界样本的$α_i$值（也就是满足$0<α_i<C$）保存到新的一个列表中，然后再对其进行遍历。同时，该步骤跳过那些已知的不会改变的$α_i$值。

> SMO称选择第2个变量的过程为内层循环。第2个变量选择的标准是希望能使$\alpha_2$有足够大的变化。

记$g(x) = \sum_{i=1}^N \alpha_i y_iK(x_i, x) + b$, 令$E_i = g(x_i) - y_i= (\sum_{i=1}^N \alpha_i y_iK(x_i, x) + b) - y_i,\qquad i = 1, 2$。

当$i = 1, 2$时，$E_i$是函数$g(x_i)$对输入$x_i$的预测值与真实输出$y_i$之差。

对于第2个变量$\alpha_2$的选择，一个简单的做法是选择让$|E_1 - E_2|$最大的变化。为了节省计算时间，将所有$E_i$值保存在一个列表中。

如果上述方法选择的变量不能使目标函数有足够的下降，那么采用以下启发式规则继续选择第2个变量。遍历在**间隔边界上**的支持向量点，依次将其对应的向量作为$\alpha_2$试用，直到目标函数有足够的下降。若找不到合适的，则遍历训练数据集；若仍找不到合适的$\alpha_2$，则放弃第一个变量$\alpha_1$，再通过外层循环寻求另外的$\alpha_1$。

选择这两个拉格朗日乘子后，我们需要先计算这些参数的约束值。然后再求解这个约束最大化问题,下面用$\alpha_i, \alpha_j$表示选择的第1个和第2个变量。

 首先，我们需要给$α_j$找到边界$L\le α_j \le H$，以保证$α_j$满足$0\le α_j\le C$的约束。这意味着$α_j$必须落入这个盒子中。由于只有两个变量($α_i, α_j$)，约束可以用二维空间中的图形来表示，如下图：

![](https://img-blog.csdn.net/20170216163029434?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

不等式约束使得($α_i, α_j$)在盒子[0, C]x[0, C]内，等式约束使得($α_i, α_j$)在平行于盒子[0, C]x[0, C]的对角线的直线上。**因此要求的是目标函数在一条平行于对角线的线段上的最优值**。这使得两个变量的最优化问题成为实质的单变量的最优化问题。由图可以得到，$α_j$的上下界可以通过下面的方法得到：

![](https://img-blog.csdn.net/20170216163047736?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

我们优化的时候，$α_j $必须要满足上面这个约束。也就是说上面是$α_j$的可行域。然后我们开始寻找$α_j$，使得目标函数最大化。通过推导得到 $α_j $ 的更新公式如下：

![](https://img-blog.csdn.net/20170216163407907?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

这里 $E_k$ 可以看做对第 $k$ 个样本，SVM 的输出与期待输出，也就是样本标签的误差。

![](https://img-blog.csdn.net/20170216163623191?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

而$η$实际上是度量两个样本$i和j$的相似性的。在计算$η$的时候，我们需要使用核函数，那么就可以用核函数来取代上面的内积。

得到新的$α_j$后，我们需要保证它处于边界内。换句话说，如果这个优化后的值跑出了边界L和H，我们就需要简单的裁剪，将$α_j$收回这个范围：

![](https://img-blog.csdn.net/20170216163645193?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

最后，得到优化的$α_j$后，我们需要用它来计算$α_i$：

![](https://img-blog.csdn.net/20170216163655723?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

 到这里，$α_i和α_j$的优化就完成了。

最后就是更新阈值$b$了，使得两个样本$i和j$都满足KKT条件。如果优化后$α_i$不在边界上（也就是满足$0<α_i<C$，这时候根据KKT条件，可以得到$y_ig_i(x_i)=1$，这样我们才可以计算$b$），那下面的阈值$b_1$是有效的，因为当输入$x_i$时它迫使SVM输出$y_i$。

![](https://img-blog.csdn.net/20170216164702252?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

同样，如果$0 < \alpha_j < C$,那么下面的$b_2$也是有效的：

![](https://img-blog.csdn.net/20170216165055215?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

 如果$0 < \alpha_i < C$和$0 < \alpha_j < C$都满足，那么$b_1和b_2$都有效，而且他们是相等的。如果他们两个都处于边界上（也就是$α_i=0$或者$α_i=C$，同时$α_j=0$或者$α_j=C$），那么在$b_1$和$b_2$之间的阈值都满足KKT条件，一般我们取他们的平均值$b=\frac{b1+b2}{2}$。所以，总的来说对$b$的更新如下：

![](https://img-blog.csdn.net/20170216165345667?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

每次完成两个变量的优化后，还必须更新对应的$E_i$值，并将它们保存在列表中。

#### KKT条件分析

KKT条件具体可以查看[深入理解拉格朗日乘子法（Lagrange Multiplier) 和KKT条件](http://blog.csdn.net/xianlingmao/article/details/7919597)。

SVM的KKT条件应该是

![](http://images.cnblogs.com/cnblogs_com/zgw21cn/031309_1004_SVM12.png)

即满足:

* L对各个x求导为零； 
* $h(x)=0$; 
* $∑α_ig_i(x)=0，α_i≥0$

拉格朗日乘子法(Lagrange Multiplier)和KKT(Karush-Kuhn-Tucker)条件是**求解约束优化问题**的重要方法，**在有等式约束时使用拉格朗日乘子法，在有不等约束时使用KKT条件**。前提是：只有当**目标函数为凸函数**时，使用这两种方法才保证求得的是**最优解**。

假设我们优化得到的最优解是：$α_i,β_i, ξ_i, w和b$。我们的最优解需要满足KKT条件：

![](https://img-blog.csdn.net/20170216170048177?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

同时$β_i, ξ_i$都需要大于等于0，而$α_i$需要在0和C之间。那可以分三种情况讨论：

![](https://img-blog.csdn.net/20170216170217710?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

因此，KKT条件变成了：

![](https://img-blog.csdn.net/20170216170256658?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

第一个式子表明**如果$α_i=0$，那么该样本落在两条间隔线外**。第二个式子表明**如果$α_i=C$，那么该样本有可能落在两条间隔线内部，也有可能落在两条间隔线上面，主要看对应的松弛变量的取值是等于0还是大于0**，第三个式子表明**如果$0<α_i<C$，那么该样本一定落在分隔线上**（这点很重要，$b$就是拿这些落在分隔线上的点来求的，因为在分割线上$w^Tx+b=1$或者$w^Tx+b=-1$嘛，才是等式，在其他地方，都是不等式，求解不了$b$）。具体形象化的表示如下：

![](https://img-blog.csdn.net/20170216170440432?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

 通过KKT条件可知，$α_i$不等于0的都是支持向量，它有可能落在分隔线上，也有可能落在两条分隔线内部。KKT条件是非常重要的，在SMO也就是SVM的其中一个实现算法中，我们可以看到它的重要应用。

#### 代码实现

下面是线性SVM的代码实现：

```python
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris() # 由于Iris是很有名的数据集，scikit-learn已经原生自带了。
X = iris.data[:, [2, 3]]
y = iris.target # 标签已经转换成0，1，2了
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 为了看模型在没有见过数据集上的表现，随机拿出数据集中30%的部分做测试

# 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) # 估算每个特征的平均值和标准差
sc.mean_ # 查看特征的平均值，由于Iris我们只用了两个特征，所以结果是array([ 3.82857143,  1.22666667])
sc.scale_ # 查看特征的标准差，这个结果是array([ 1.79595918,  0.77769705])
X_train_std = sc.transform(X_train)
# 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# 导入SVC
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0) # 用线性核，你也可以通过kernel参数指定其它的核。
svm.fit(X_train_std, y_train)
# 打印决策边界，这个函数是我自己写的，如果你想要的话，我发给你
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
```

接下来是使用非线性SVM的代码：

```python
svm = SVC(kernel='rbf', random_state=0, gamma=x, C=1.0) # 令gamma参数中的x分别等于0.2和100.0
svm.fit(X_train_std, y_train) # 这两个参数和上面代码中的训练集一样
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
```

SVM 的知识点就总结到这里，还是参考了不少文章和看书才完成，但是需要继续通过实践才能加深对 SVM 的了解。

### 6. 朴素贝叶斯

参考文章：

- 《统计学习方法》
- [机器学习常见算法个人总结（面试用）](http://kubicode.me/2015/08/16/Machine%20Learning/Algorithm-Summary-for-Interview/)
- [朴素贝叶斯理论推导与三种常见模型](http://blog.csdn.net/u012162613/article/details/48323777)
- [朴素贝叶斯的三个常用模型：高斯、多项式、伯努利](http://www.letiantian.me/2014-10-12-three-models-of-naive-nayes/)

#### 简介

> **朴素贝叶斯**是基于贝叶斯定理与特征条件独立假设的分类方法。

**贝叶斯定理**是基于条件概率来计算的，条件概率是在已知事件B发生的前提下，求解事件A发生的概率，即$P(A|B)=\frac{P(AB)}{P(B)}$，而贝叶斯定理则可以通过$P(A|B)$来求解$P(B|A)$：
$$
P(B|A) = \frac{P(A|B)P(B)}{P(A)}
$$
其中分母$P(A)$可以根据全概率公式分解为：$P(A)=\sum_{i=1}^n P(B_i)P(A|B_i)$

而**特征条件独立假设**是指假设各个维度的特征$x_1,x_2,...,x_n$互相独立，则条件概率可以转化为：
$$
P(x|y_{k})=P(x_{1},x_{2},...,x_{n}|y_{k})=\prod_{i=1}^{n}P(x_{i}|y_{k})
$$
朴素贝叶斯分类器可表示为：
$$
f(x)=argmax_{y_{k}} P(y_{k}|x)=argmax_{y_{k}} \frac{P(y_{k})\prod_{i=1}^{n}P(x_{i}|y_{k})}{\sum_{k}P(y_{k})\prod_{i=1}^{n}P(x_{i}|y_{k})}
$$
而由于对上述公式中分母的值都是一样的，所以可以忽略分母部分，即可以表示为：
$$
f(x)=argmax P(y_{k})\prod_{i=1}^{n}P(x_{i}|y_{k})
$$
这里$P(y_k)$是先验概率，而$P(y_k|x)$则是后验概率，朴素贝叶斯的目标就是最大化后验概率，这等价于期望风险最小化。

#### 参数估计

##### 极大似然估计

朴素贝叶斯的学习意味着估计$P(y_k)$和$P(x_i|y_k)$,可以通过**极大似然估计**来估计相应的概率。

![](http://img.blog.csdn.net/20170218105353279?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

如上图所示，分别是$P(y_k)$和$P(x_i|y_k)$的极大似然估计。

当求解完上述两个概率，就可以对测试样本使用朴素贝叶斯分类算法来预测其所属于的类别，简单总结的算法流程如下所示：

![](http://img.blog.csdn.net/20170218105758081?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

##### 贝叶斯估计/多项式模型

用**极大似然估计可能会出现所要估计的概率值为0**的情况，这会影响到后验概率的计算，使分类产生偏差。解决这个问题的办法是使用**贝叶斯估计**，也被称为多项式模型。

当**特征是离散的时候，使用多项式模型**。多项式模型在计算先验概率$P(y_k)$和条件概率$P(x_i|y_k)$时，会做一些**平滑处理**，具体公式为：
$$
P(y_k)=\frac{N_{y_k}+α}{N+kα}
$$

> $N$是总的样本个数，$k$是总的类别个数，$N_{y_k}$是类别为$y_k$的样本个数，$α$是平滑值。

$$
P(x_i|y_k) = \frac{N_{y_k,x_i} + \alpha}{N_{y_k}+n\alpha}
$$

> $N_{y_k}$是类别为$y_k$的样本个数，$n$是特征的维数，$N_{y_k,x_i}$是类别为$y_k$的样本中，第$i$维特征的值是$x_i$的样本个数，$α$是平滑值。

当$α=1$时，称作**Laplace平滑**，当$0<α<1$时，称作**Lidstone**平滑，$α=0$时不做平滑。

如果不做平滑，当某一维特征的值$x_i$没在训练样本中出现过时，会导致$P(x_i|y_k)=0$，从而导致后验概率为0。加上平滑就可以克服这个问题。

##### 高斯模型

当特征是连续变量的时候，运用多项式模型会导致很多$P(x_i|y_k) = 0$（不做平滑的情况下），即使做平滑，所得到的条件概率也难以描述真实情况，所以处理连续变量，应该采用高斯模型。

高斯模型是假设每一维特征都服从高斯分布（正态分布）：
$$
P(x_{i}|y_{k}) = \frac{1}{\sqrt{2\pi\sigma_{y_{k}}^{2}}}exp( -\frac{(x_{i}-\mu_{y_{k}})^2}  {2\sigma_{y_{k}}^{2}}   )
$$
$\mu_{y_{k},i}$表示类别为$y_k$的样本中，第$i$维特征的均值；
$\sigma_{y_{k},i}^{2}$表示类别为$y_k$的样本中，第$i$维特征的方差。

##### 伯努利模型

与多项式模型一样，伯努利模型适用于**离散特征**的情况，所不同的是，**伯努利模型中每个特征的取值只能是1和0**(以文本分类为例，某个单词在文档中出现过，则其特征值为1，否则为0).

伯努利模型中，条件概率$P(x_i|y_k)$的计算方式是：

当特征值$x_i$为1时，$P(x_i|y_k)=P(x_i=1|y_k)$；

当特征值$x_i$为0时，$P(x_i|y_k)=1−P(x_i=1|y_k)$；

#### 工作流程

1. 准备阶段
   确定特征属性，并对每个特征属性进行适当划分，然后由人工对一部分待分类项进行分类，形成训练样本。
2. 训练阶段
   计算每个类别在训练样本中的出现频率及每个特征属性划分对每个类别的条件概率估计
3. 应用阶段
   使用分类器进行分类，输入是分类器和待分类样本，输出是样本属于的分类类别

#### 属性特征

1. 特征为离散值时直接统计即可（表示统计概率）
2. 特征为连续值的时候假定特征符合高斯分布，则有

$$
P(x_{i}|y_{k}) = \frac{1}{\sqrt{2\pi\sigma_{y_{k}}^{2}}}exp( -\frac{(x_{i}-\mu_{y_{k}})^2}  {2\sigma_{y_{k}}^{2}}   )
$$

#### 与逻辑回归的不同

1. **Naive Bayes是一个生成模型**，在计算P(y|x)之前，先要从训练数据中计算P(x|y)和P(y)的概率，从而利用贝叶斯公式计算P(y|x)。

   **Logistic Regression是一个判别模型**，它通过在训练数据集上最大化判别函数P(y|x)学习得到，不需要知道P(x|y)和P(y)。

2. Naive Bayes是建立在**条件独立假设**基础之上的，设特征X含有n个特征属性（X1，X2，...Xn），那么在给定Y的情况下，X1，X2，...Xn是条件独立的。

   Logistic Regression的限制则要**宽松很多**，如果数据满足条件独立假设，Logistic Regression能够取得非常好的效果；当数据不满足条件独立假设时，Logistic Regression仍然能够通过调整参数让模型最大化的符合数据的分布，从而训练得到在现有数据集下的一个最优模型。

3. **当数据集比较小的时候，应该选用Naive Bayes**，为了能够取得很好的效果，数据的需求量为O(log n)

   **当数据集比较大的时候，应该选用Logistic Regression，**为了能够取得很好的效果，数据的需求量为O( n)

#### 与逻辑回归的相同

1. 两者都是对特征的线性表达
2. 两者建模的都是条件概率，对最终求得的分类结果有很好的解释性。

#### 优缺点

优点

- 对小规模的数据表现很好，适合多分类任务，适合增量式训练。

缺点

- 对输入数据的表达形式很敏感（离散、连续，值极大极小之类的）。

#### 代码实现

下面是使用`sklearn`的代码例子，分别实现上述三种模型,例子来自 [朴素贝叶斯的三个常用模型：高斯、多项式、伯努利](http://www.letiantian.me/2014-10-12-three-models-of-naive-nayes/)。
下面是高斯模型的实现

```python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> iris.feature_names  # 四个特征的名字
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
>>> iris.data
array([[ 5.1,  3.5,  1.4,  0.2],
       [ 4.9,  3. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [ 4.6,  3.1,  1.5,  0.2],
       [ 5. ,  3.6,  1.4,  0.2],
       [ 5.4,  3.9,  1.7,  0.4],
       [ 4.6,  3.4,  1.4,  0.3],
       [ 5. ,  3.4,  1.5,  0.2],
       ......
       [ 6.5,  3. ,  5.2,  2. ],
       [ 6.2,  3.4,  5.4,  2.3],
       [ 5.9,  3. ,  5.1,  1.8]]) #类型是numpy.array
>>> iris.data.size  
600  #共600/4=150个样本
>>> iris.target_names
array(['setosa', 'versicolor', 'virginica'], 
      dtype='|S10')
>>> iris.target
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,....., 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ......, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
>>> iris.target.size
150
>>> from sklearn.naive_bayes import GaussianNB
>>> clf = GaussianNB()
>>> clf.fit(iris.data, iris.target)
>>> clf.predict(iris.data[0])
array([0])   # 预测正确
>>> clf.predict(iris.data[149])
array([2])   # 预测正确
>>> data = numpy.array([6,4,6,2])
>>> clf.predict(data)
array([2])  # 预测结果很合理
```

多项式模型如下：

```python
>>> import numpy as np
>>> X = np.random.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import MultinomialNB
>>> clf = MultinomialNB()
>>> clf.fit(X, y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
>>> print(clf.predict(X[2]))
[3]
```

值得注意的是，多项式模型在训练一个数据集结束后可以继续训练其他数据集而无需将两个数据集放在一起进行训练。在 sklearn 中，MultinomialNB() 类的partial_fit() 方法可以进行这种训练。这种方式特别适合于训练集大到内存无法一次性放入的情况。

在第一次调用 `partial_fit()`  时需要给出所有的分类标号。

```python
>>> import numpy
>>> from sklearn.naive_bayes import MultinomialNB
>>> clf = MultinomialNB() 
>>> clf.partial_fit(numpy.array([1,1]), numpy.array(['aa']), ['aa','bb'])
GaussianNB()
>>> clf.partial_fit(numpy.array([6,1]), numpy.array(['bb']))
GaussianNB()
>>> clf.predict(numpy.array([9,1]))
array(['bb'], 
      dtype='|S2')
```

伯努利模型如下：

```python
>>> import numpy as np
>>> X = np.random.randint(2, size=(6, 100))
>>> Y = np.array([1, 2, 3, 4, 4, 5])
>>> from sklearn.naive_bayes import BernoulliNB
>>> clf = BernoulliNB()
>>> clf.fit(X, Y)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
>>> print(clf.predict(X[2]))
[3]
```



朴素贝叶斯的总结就到这里为止。

### 7. K-近邻算法(KNN)

#### 简介

> k 近邻（KNN)是一种基本分类与回归方法。

其思路如下：给一个训练数据集和一个新的实例，在训练数据集中找出与这个新实例最近的$k$个训练实例，然后统计最近的$k$个训练实例中**所属类别计数最多的那个类**，就是新实例的类。其流程如下所示：

1. 计算训练样本和测试样本中每个样本点的距离（常见的距离度量有欧式距离，马氏距离等）；
2. 对上面所有的距离值进行排序；
3. 选前 k 个最小距离的样本；
4. 根据这 k 个样本的标签进行投票，得到最后的分类类别；

KNN 的特殊情况是 k=1 的情况，称为**最近邻算法**。对输入的实例点（特征向量）x，最近邻法将训练数据集中与 x 最近邻点的类作为其类别。

#### 三要素

1. k 值的选择
2. 距离的度量（常见的距离度量有欧式距离，马氏距离）
3. 分类决策规则（多数表决规则）

#### k 值的选择

1. **k 值越小表明模型越复杂，更加容易过拟合**，其**偏差小，而方差大**
2. 但是 **k 值越大，模型越简单**，如果 k=N 的时候就表明无论什么点都是**训练集中类别最多**的那个类，这种情况，则是**偏差大，方差小**。

> 所以一般 k 会取一个**较小的值，然后用过交叉验证来确定**
> 这里所谓的交叉验证就是将样本划分一部分出来为预测样本，比如 95% 训练，5% 预测，然后 k 分别取1，2，3，4，5 之类的，进行预测，计算最后的分类误差，选择误差最小的 k

#### 距离的度量

KNN 算法使用的距离一般是欧式距离，也可以是更一般的 $L_p$ 距离或者马氏距离，其中 $L_p$ 距离定义如下：
$$
L_p(x_i, x_j) = (\sum_{l=1}^n |x_i^{(l)} - x_j^{(l)} |^p)^{\frac{1}{p}}
$$
这里 $x_i = (x_i^{(1)}, x_i^{(2)},...x_i^{(n)})^T, x_j = (x_j^{(1)}, x_j^{(2)}, ... , x_j^{(n)})^T$，然后 $p \ge 1$。

当 $p=2$，称为欧式距离，即
$$
L_2(x_i, x_j) = (\sum_{l=1}^n |x_i^{(l)} - x_j^{(l)} |^2)^{\frac{1}{2}}
$$
当 $p=1$，称为曼哈顿距离，即
$$
L_1(x_i, x_j) = \sum_{l=1}^n |x_i^{(l)} - x_j^{(l)} |
$$
当 $p = \infty$，它是各个坐标距离的最大值，即
$$
L_\infty(x_i, x_j) =max_l |x_i^{(l)} - x_j^{(l)} |
$$
马氏距离如下定义：

![这里写图片描述](http://img.blog.csdn.net/20170219212021208?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### KNN的回归

在找到最近的 k 个实例之后，可以计算这 k 个实例的平均值作为预测值。或者还可以给这 k 个实例添加一个权重再求平均值，这个权重与度量距离成反比（越近权重越大）。

#### 优缺点

##### 优点

1. **思想简单，理论成熟，既可以用来做分类也可以用来做回归**；
2. 可用于**非线性分类**；
3. 训练时间复杂度为$O(n)$；
4. 准确度高，对数据没有假设，对**异常值**不敏感；

##### 缺点

1. **计算量大**；
2. **样本不平衡问题**（即有些类别的样本数量很多，而其它样本的数量很少）；
3. 需要**大量的内存**；

#### KD树

> KD树是一个二叉树，表示对K维空间的一个划分，可以进行快速检索（那KNN计算的时候不需要对全样本进行距离的计算了）

##### 构造KD树

在k维的空间上循环找子区域的中位数进行划分的过程。
假设现在有K维空间的数据集$T={x_1,x_2,x_3,…x_n},x_i={a_1,a_2,a_3..a_k}$

1. 首先构造根节点，以坐标$a_1$的中位数$b$为切分点，将根结点对应的矩形局域划分为两个区域，区域1中$a_1<b$,区域2中$a_1>b$
2. 构造叶子节点，分别以上面两个区域中$a_2$的中位数作为切分点，再次将他们两两划分，作为深度1的叶子节点，（如果$中位数a_2=中位数$，则$a_2$的实例落在切分面）
3. 不断重复2的操作，深度为$j$的叶子节点划分的时候，选择维度为$l=j(mod k) + 1$，索取的$a_i$ 的$i=j%k+1$，直到两个子区域没有实例时停止

##### KD树的搜索

1. 首先从根节点开始递归往下找到包含$x$的叶子节点，每一层都是找对应的$x_i$
2. 将这个叶子节点认为是当前的“**近似最近点**”
3. 递归向上回退，如果以$x$圆心，以“近似最近点”为半径的球与根节点的另一半子区域边界**相交**，则说明另一半子区域中存在与$x$**更近的点**，则进入另一个子区域中查找该点并且更新”近似最近点“
4. 重复3的步骤，直到**另一子区域与球体不相交或者退回根节点**
5. 最后更新的”近似最近点“与$x$真正的最近点

##### KD树进行KNN查找

通过 KD 树的搜索找到与搜索目标最近的点，这样KNN的搜索就可以被限制在空间的局部区域上了，可以大大增加效率。

##### KD树搜索的复杂度

当实例随机分布的时候，搜索的复杂度为$log(N)$，$N$为实例的个数，KD树更加适用于**实例数量远大于空间维度**的KNN搜索，如果实例的**空间维度与实例个数差不多**时，它的效率基于等于**线性扫描**。

#### 代码实现

使用`sklearn`的简单代码例子：

```python
#Import Library
from sklearn.neighbors import KNeighborsClassifier

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create KNeighbors classifier object model 

KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5

# Train the model using the training sets and check score
model.fit(X, y)

#Predict Output
predicted= model.predict(x_test)
```

最后，**在用KNN前你需要考虑到：**

- KNN的计算成本很高
- 所有特征应该**标准化数量级**，否则数量级大的特征在计算距离上会有偏移。
- 在进行KNN前**预处理数据**，例如去除异常值，噪音等。



### 9 提升方法

参考自：

- 《统计学习方法》
- [浅谈机器学习基础（上）](http://www.jianshu.com/p/ed9ae5385b89?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io#)
- [Ensemble learning:Bagging,Random Forest,Boosting](http://blog.csdn.net/taoyanqi8932/article/details/54098100)

#### 简介

> 提升方法(boosting)是一种常用的统计学习方法，在分类问题中，它**通过改变训练样本的权重**，学习多个分类器，并将这些分类器进行线性组合，提供分类的性能。

#### boosting 和 bagging

**boosting 和 bagging **都是**集成学习（ensemble learning）**领域的基本算法，**两者使用的多个分类器的类型是一致的。**

##### Bagging

**bagging **也叫**自助汇聚法（bootstrap aggregating）**，比如原数据集中有 N 个样本，我们每次从原数据集中**有放回**的抽取，抽取 N 次，就得到了一个新的有 N 个样本的数据集，然后我们抽取 S 个 N 次，就得到了 S 个有 N 个样本的新数据集，然后拿这 S 个数据集去训练 S 个分类器，之后应用这 S 个分类器进行分类，选择分类器**投票最多的类别**作为最后的分类结果。一般来说**自助样本的包含有 63% 的原始训练数据**，因为：

假设共抽取 N 个样本，则 N 次都没有抽到的概率是 $p=(1-\frac{1}{N})^N$

则一个样本被抽到的概率有 $p = 1- (1- \frac{1}{N})^N$

所以，当 N 很大时有：$p = 1- \frac{1}{e} = 0.632$。

这样，在一次 bootstrap 的过程中，会有 36% 的样本没有被采样到，它们被称为 **out-off-bag(oob)**，这是自助采样带给 `bagging` 的里一个优点，因为我们可以用 `oob` 进行**“包外估计”(out-of-bag estimate)**。

**bagging 通过降低基分类器的方差改善了泛化误差，bagging 的性能依赖于基分类器的稳定性**。如果基分类器是**不稳定**的，bagging **有助于减少训练数据的随机波动导致的误差**，如果基分类器是稳定的，即对训练数据集中的微小变化是鲁棒的，则组合分类器的误差主要由**基分类器偏移**所引起的，这种情况下，**bagging 可能不会对基分类器有明显的改进效果，甚至可能降低分类器的性能。**

##### boosting 与 bagging 的区别

- bagging 通过**有放回的抽取**得到了 S 个数据集，而 boosting 用的始终是**原数据集**，但是**样本的权重会发生改变。**
- **boosting 对分类器的训练是串行的**，每个新分类器的训练都会受到上一个分类器分类结果的影响。
- **bagging 里面各个分类器的权重是相等的，但是 boosting 不是**，每个分类器的权重代表的是其对应分类器在上一轮分类中的成功度。

**AdaBoost 是 boosting 方法中最流行的版本**

#### AdaBoost 算法

> **AdaBoost（adaptive boosting）是元算法，通过组合多个弱分类器来构建一个强分类器。**我们为训练数据中的每一个样本都赋予其一个权重，这些权重构成了向量$D$，一开始，这些权重都初始化成相等值，然后每次添加一个弱分类器对样本进行分类，从第二次分类开始，**将上一次分错的样本的权重提高，分对的样本权重降低，持续迭代**。此外，对于每个弱分类器而言，**每个分类器也有自己的权重，取决于它分类的加权错误率，加权错误率越低，则这个分类器的权重值$α$越高**，最后综合多个弱分类器的分类结果和其对应的权重$α$得到预测结果，AdaBoost是最好的监督学习分类方法之一。

其算法过程如下所示：

![这里写图片描述](http://img.blog.csdn.net/20170222110646742?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其中，注意：

![这里写图片描述](http://img.blog.csdn.net/20170222110716909?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

##### 训练误差分析

**AdaBoost算法的最基本性质是在学习过程中不断减小训练误差**，对训练误差的上界有如下定理：

> 定理1：AdaBoost最终分类器的训练误差界为：
> $$
> \frac{1}{N} \sum_{i=1}^N I(G(x_i) \neq y_i) \le \frac{1}{N} \sum_i exp(-y_i f(x_i)) = \prod_m Z_m
> $$
> 定理2：二类分类问题
> $$
> \prod_{m=1}^M Z_m = \prod_{m=1}^M [2\sqrt{e_m(1-e_m)}] = \prod_{m=1}^M [\sqrt{1-4\gamma_m^2} \le exp(-2\sum_{m=1}^M \gamma_m^2)
> $$

##### 算法解释

**AdaBoost**算法还可以解释为模型是加法模型，损失函数是指数函数，学习算法是前向分步算法的二类分类学习方法。

加法模型是形如$f(x) = \sum_{i=1}^M \beta_m b(x; \gamma_m)$的函数形式，其中$b(x;\gamma_m)$是基函数，而$\beta_m$是基函数的系数，$\gamma_m$是基函数的参数。对于**AdaBoost**算法，其基本分类器的线性组合为$f(x) = \sum_{m=1}^M \alpha_m G_m(x)$正是一个加法模型。

**AdaBoost**算法的损失函数是指数函数，公式为$E = \sum_{i=1}^N exp(-y_i G_m(x_i))$。

此外，经过$m$轮迭代可以得到$f_m(x) = f_{m-1}(x) + \alpha_m G_m(x)$。而前向分步算法的过程如下所示：

![这里写图片描述](http://img.blog.csdn.net/20170222162813661?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

通过上述步骤，前向分步算法将同时求解从$m=1$到$M$所有参数$\beta_m, \gamma_m$的优化问题简化为逐步求解各个$\beta_m, \gamma_m$的优化问题。

#### 优缺点

##### 优点

1. 泛化误差低
2. 容易实现，分类准确率较高，没有太多参数可以调

##### 缺点

- 对异常值比较敏感
- 训练时间过长
- 执行效果依赖于弱分类器的选择

### 10 GBDT

参考如下

- [机器学习（四）--- 从gbdt到xgboost](http://www.cnblogs.com/mfryf/p/5946815.html)
- [机器学习常见算法个人总结（面试用）](http://kubicode.me/2015/08/16/Machine%20Learning/Algorithm-Summary-for-Interview/)
- [xgboost入门与实战（原理篇）](http://blog.csdn.net/sb19931201/article/details/52557382)

#### 简介

> GBDT是一个基于迭代累加的决策树算法，它通过构造一组弱的学习器（树），并把多颗决策树的结果累加起来作为最终的预测输出。

#### 算法介绍

**GBDT**是希望组合一组弱的学习器的线性组合，即有：
$$
F^* = argminE_{x,y}[L(y,F(x))] \\
F(x; p_m, a_m) = \sum_{m=0}^M p_m h(x;a_m)
$$
上述公式中$p_m$表示步长，我们可以在函数空间形式上使用梯度下降法求解，首先固定$x$，然后对$F(x)$求其最优解。下面先给出框架流程：

![](http://img.blog.csdn.net/20170223110849276?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

我们需要做的是估计$g_m(x)$,它是梯度方向；通过使用决策树实现来逼近$g_m(x)$，使得两者之间的距离尽可能的近，而距离的衡量方式有多种，包括均方误差和`LogLoss`误差。下面给出使用`LogLoss`损失函数的具体推导：
$$
L(y, F) = log(1+exp(-2yF)) \qquad y\in [-1,1]
$$
**Step1** 首先求解初始值$F_0$，令其偏导为0。（实现时是第1棵树需要拟合的残差）：

![](http://img.blog.csdn.net/20170223111557862?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**Step 2** 估计$g_m(x)$，并用决策树对其进行拟合。$g_m(x)$是梯度，实现时是第$m$棵树需要拟合的残差：

![](http://img.blog.csdn.net/20170223111723163?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**Step 3** 使用牛顿法求解下降方向步长。$r_jm$是拟合的步长，实现时是每棵树的预测值。（通常实现中这一步是被省略的，改为使用**Shrinkage**的策略通过参数设置步长，避免过拟合。

![](http://img.blog.csdn.net/20170223111916894?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**Step 4** 预测时只需要把每棵树的预测值乘以缩放因子然后相加即可得到最终的预测值：
$$
p = predict(0) + \sum_{m=1}^M shrinkage * predict(d_m)
$$
若需要预测值输出区间在$[0,1]$，可作如下转换：
$$
probability = \frac{1}{1+e^{-2 * predict}}
$$
**GBDT中的树是回归树，不是分类树。**

#### RF 与 GBDT 对比

（1）RF中树的棵树是**并行**生成的；GBDT中树是**顺序**生成的；两者中过多的树都会过拟合，但是**GBDT更容易过拟合；**

（2）RF中每棵树分裂的特征**比较随机**；GBDT中前面的树优先分裂对大部分样本区分的特征，后面的树分裂对小部分样本区分特征；

（3）RF中主要参数是**树的棵数**；GBDT中主要参数是**树的深度，一般为1**；

#### Shrinkage

**Shrinkage**认为，每次走一小步逐步逼近的结果要比每次迈一大步逼近结果更加容易避免过拟合。
$$
y(1\sim i) = y(1 \sim i-1) + step * y_i
$$

#### 优缺点

##### 优点

1. 精度高
2. 能处理非线性数据
3. 能处理多特征类型
4. 适合低维稠密数据
5. 模型可解释性好
6. 不需要做特征的归一化，可以自动选择特征
7. 能适应多种损失函数，包括均方误差和`LogLoss`等

##### 缺点

1. **boosting**是个串行的过程，所以并行麻烦，需要考虑上下树之间的联系
2. 计算复杂度大
3. 不使用高维稀疏特征

#### 调参

1. 树的个数 100~10000
2. 叶子的深度 3~8
3. 学习速率 0.01~1
4. 叶子上最大节点树 20
5. 训练采样比例 0.5~1
6. 训练特征采样比例 $\sqrt(n)$

#### xgboost

**xgboost是boosting Tree的一个很牛的实现**，它在最近Kaggle比赛中大放异彩。它 有以下几个优良的特性：

1. 显示的把树模型复杂度作为正则项加到优化目标中。
2. 公式推导中用到了二阶导数，用了二阶泰勒展开。
3. 实现了分裂点寻找近似算法。
4. 利用了特征的稀疏性。
5. 数据事先排序并且以block形式存储，有利于并行计算。
6. 基于分布式通信框架rabit，可以运行在MPI和yarn上。（最新已经不基于rabit了）
7. 实现做了面向体系结构的优化，针对cache和内存做了性能优化。

在项目实测中使用发现，Xgboost的训练速度要远远快于传统的GBDT实现，10倍量级。

##### 特点

**这部分内容参考了知乎上的一个问答—[机器学习算法中GBDT和XGBOOST的区别有哪些？](https://www.zhihu.com/question/41354392)，答主是wepon大神**

1.传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。 **—可以通过booster [default=gbtree]设置参数:gbtree: tree-based models/gblinear: linear models**

2.传统GBDT在优化时只用到一阶导数信息，xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶和二阶导数。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。 **—对损失函数做了改进（泰勒展开，一阶信息g和二阶信息h）**

3.xgboost在代价函数里加入了**正则项**，用于**控制模型的复杂度**。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和。从Bias-variance tradeoff角度来讲，正则项降低了模型variance，使学习出来的模型更加简单，防止过拟合，这也是xgboost优于传统GBDT的一个特性 
**—正则化包括了两个部分，都是为了防止过拟合，剪枝是都有的，叶子结点输出L2平滑是新增的。**

4.shrinkage and column subsampling —**还是为了防止过拟合**

> （1）shrinkage缩减类似于学习速率，在每一步tree boosting之后增加了一个参数n（权重），通过这种方式来减小每棵树的影响力，给后面的树提供空间去优化模型。
>
> （2）column subsampling列(特征)抽样，说是从随机森林那边学习来的，防止过拟合的效果比传统的行抽样还好（行抽样功能也有），并且有利于后面提到的并行化处理算法。

5.split finding algorithms(划分点查找算法)：
（1）exact greedy algorithm—**贪心算法获取最优切分点** 
（2）approximate algorithm— **近似算法，提出了候选分割点概念，先通过直方图算法获得候选分割点的分布情况，然后根据候选分割点将连续的特征信息映射到不同的buckets中，并统计汇总信息。** 
（3）Weighted Quantile Sketch—**分布式加权直方图算法** 
**这里的算法（2）、（3）是为了解决数据无法一次载入内存或者在分布式情况下算法（1）效率低的问题，以下引用的还是wepon大神的总结：**

> 可并行的近似直方图算法。树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以xgboost还提出了一种可并行的近似直方图算法，用于高效地生成候选的分割点。

6.对缺失值的处理。对于特征的值有缺失的样本，xgboost可以自动学习出它的分裂方向。 **—稀疏感知算法**

7.**Built-in Cross-Validation（内置交叉验证)**

> XGBoost allows user to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run. 
> This is unlike GBM where we have to run a grid-search and only a limited values can be tested.

8.**continue on Existing Model（接着已有模型学习）**

> User can start training an XGBoost model from its last iteration of previous run. This can be of significant advantage in certain specific applications. 
> GBM implementation of sklearn also has this feature so they are even on this point.

9.**High Flexibility（高灵活性）**

> **XGBoost allow users to define custom optimization objectives and evaluation criteria. 
> This adds a whole new dimension to the model and there is no limit to what we can do.**

10.并行化处理 **—系统设计模块,块结构设计等**

> xgboost工具支持并行。boosting不是一种串行的结构吗?怎么并行的？注意xgboost的并行不是tree粒度的并行，xgboost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。**xgboost的并行是在特征粒度上的**。我们知道，**决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点）**，xgboost在训练之前，**预先对数据进行了排序，然后保存为block结构**，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。

此外xgboost还设计了高速缓存压缩感知算法，这是系统设计模块的效率提升。 
当梯度统计不适合于处理器高速缓存和高速缓存丢失时，会大大减慢切分点查找算法的速度。 
（1）针对 exact greedy algorithm采用缓存感知预取算法 
（2）针对 approximate algorithms选择合适的块大小

##### 代码使用

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

### 11 EM算法

参考自

- 《统计学习方法》
- [机器学习常见算法个人总结（面试用）](http://kubicode.me/2015/08/16/Machine%20Learning/Algorithm-Summary-for-Interview/)
- [从最大似然到EM算法浅解](http://blog.csdn.net/zouxy09/article/details/8537620)
- [（EM算法）The EM Algorithm](http://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html)

#### 简介

> EM算法，即期望极大算法，用于**含有隐变量的概率模型的极大似然估计或极大后验概率估计**，它一般分为两步：**第一步求期望(E),第二步求极大(M)。**

如果概率模型的变量都是观测变量，那么给定数据之后就可以直接使用极大似然法或者贝叶斯估计模型参数。
但是当模型含有隐含变量的时候就不能简单的用这些方法来估计，EM就是一种含有隐含变量的概率模型参数的极大似然估计法。

应用到的地方：混合高斯模型、混合朴素贝叶斯模型、因子分析模型

#### 算法推导

![这里写图片描述](http://img.blog.csdn.net/20170224165311795?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20170224165327407?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

上述公式相当于决定了$L(\theta)$的下界，而**EM**算法实际上就是通过不断求解下界的极大化来逼近对数似然函数极大化的算法。

![这里写图片描述](http://img.blog.csdn.net/20170224165456956?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 算法流程

算法流程如下所示：

![这里写图片描述](http://img.blog.csdn.net/20170224165544390?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 收敛性

收敛性部分可以主要看[（EM算法）The EM Algorithm](http://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html)的推导，最终可以推导得到如下公式：
$$
L(\theta^{(t+1)}) \ge \sum_i \sum_{z^{i}} Q_i^{(t)}(z^{(i)}) log \frac{p(x^{(i)}, z^{(i)} ; \theta^{(t+1)})}{Q_i^{(t)}(z^{(i)})} \\
			\ge  \sum_i \sum_{z^{i}} Q_i^{(t)}(z^{(i)}) log \frac{p(x^{(i)}, z^{(i)} ; \theta^{(t)})}{Q_i^{(t)}(z^{(i)})} \\
			= L(\theta^{(t)})
$$

#### 特点

1. 最大优点是简单性和普适性
2. **EM**算法不能保证找到全局最优点，在应用中，通常选取几个不同的初值进行迭代，然后对得到的几个估计值进行比较，从中选择最好的
3. **EM**算法对初值是敏感的，不同初值会得到不同的参数估计值

#### 使用例子

**EM算法**一个常见的例子就是**GMM模型**，即高斯混合模型。而高斯混合模型的定义如下：

> 高斯混合模型是指具有如下形式的概率分布模型：
> $$
> P(y| \theta) = \sum_{k=1}^K \alpha_k \phi(y | \theta_k) \\
> 其中， \alpha_k 是系数，\alpha_k \ge 0, \sum_{k=1}^K \alpha_k = 1; \phi(y|\theta_k)是高斯分布密度，\theta_k = (\mu_k, \sigma_k^2), \\
> \phi(y|\theta_k) = \frac{1}{\sqrt{2 \pi} \sigma_k} exp(-\frac{(y-\mu_k)^2}{2\sigma_k^2})
> $$

$\phi(y|\theta_k)$称为第$k$个分模型。

每个样本都有可能由$k$个高斯产生，只不过由每个高斯产生的概率不同而已，因此每个样本都有对应的高斯分布（$k$个中的某一个），此时的隐含变量就是每个样本对应的某个高斯分布。

GMM的E步公式如下（计算每个样本对应每个高斯的概率）：

 ![这里写图片描述](http://img.blog.csdn.net/20170224182036789?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

更具体的计算公式为：

![这里写图片描述](http://img.blog.csdn.net/20170224182218590?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

M步公式如下（计算每个高斯的**比重，均值，方差**这3个参数）：

 ![这里写图片描述](http://img.blog.csdn.net/20170224182243247?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

　　

### 12 优化算法

参考自

- [常见的几种最优化方法](http://www.cnblogs.com/maybe2030/p/4751804.html)

常见的最优化方法有梯度下降法、牛顿法和拟牛顿法、共轭梯度法等等

#### 梯度下降法

梯度下降法是最早最简单，也是最为常用的最优化方法。梯度下降法实现简单，**当目标函数是凸函数时，梯度下降法的解是全局解**。一般情况下，其解**不保证是全局最优解**，梯度下降法的速度也未必是最快的。**梯度下降法的优化思想是用当前位置负梯度方向作为搜索方向，因为该方向为当前位置的最快下降方向，所以也被称为是”最速下降法“。最速下降法越接近目标值，步长越小，前进越慢。**梯度下降法的搜索迭代示意图如下图所示：

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Gradient_descent.png/350px-Gradient_descent.png?_=4751804)

其缺点是：

**（1）靠近极小值时收敛速度减慢，如下图所示；**

**（2）直线搜索时可能会产生一些问题；**

**（3）可能会“之字形”地下降。**

![](https://upload.wikimedia.org/wikipedia/commons/6/60/Banana-SteepDesc.gif?_=4751804)

从上图可以看出，梯度下降法在接近最优解的区域收敛速度明显变慢，利用梯度下降法求解需要很多次的迭代。

在机器学习中，基于基本的梯度下降法发展了两种梯度下降方法，分别为**随机梯度下降法和批量梯度下降法**。

比如对一个线性回归模型，假设 $h(x)$ 是要拟合的函数，$J(\theta)$ 是损失函数，而 $\theta$ 是参数，要迭代求解的值，然后 $m$ 是训练集样本个数，$n$ 是特征个数，则有：
$$
h(\theta) = \sum_{j=0}^n \theta_j x_j \\
J(\theta) = \frac{1}{2m} \sum_{i=1}^m (y^i - h_\theta(x^i))^2
$$

#### 批量梯度下降法(Batch Gradient Descent, BGD)

首先是令损失函数对参数求偏导，即 $\frac{\partial J(\theta)}{\partial \theta_j} = -\frac{1}{m}\sum_{i=1}^m (y^i - h_\theta(x^i)) x_j^i$。然后需要最小化损失函数，所以每个参数会按照其梯度负方向更新，即有：
$$
\theta_j^` = \theta_j + \frac{1}{m} \sum_{i=1}^{m}(y^i - h_\theta(x^i))x_j^i
$$
从上述公式可知，其求解的是一个全局最优解，因为每次更新参数的时候，都需要使用整个训练集的样本，所以如果 $m$ 很大，那么每次迭代的速度就会很慢，这也是为何有随机梯度下降方法出现的原因。

对于这种方法，每次迭代的计算量是 $O(m*n^2)$，$m,n$ 分别是样本总数量和每个样本的特征维度。

#### 随机梯度下降(Stochastic Gradient Descent, SGD)

上述使用过的损失函数可以写成如下形式：
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^m \frac{1}{2}(y^i - h_\theta(x^i))^2 = \frac{1}{m} \sum_{i=1}^m cos t(\theta, (x^i, y^i)) \\
cos t(\theta, (x^i, y^i))  = \frac{1}{2}(y^i - h_\theta(x^i))^2
$$
而函数更新公式如下：
$$
\theta_j^` = \theta_j + (y^i - h_\theta(x^i))x_j^i
$$
所以 SGD 是通过每个样本来迭代更新一次，当样本数量很大的时候，那么可能只需要使用其中一部分即可找到最优解。相比 BGD 方法，迭代速度加快了很多。但是 SGD 的一个问题是噪音会更多，这使得它每次迭代并不是朝着最优解的方向。

**随机梯度下降每次迭代只使用一个样本**，迭代一次计算量为 $O(n^2)$，当样本个数 $m$ 很大的时候，随机梯度下降迭代一次的速度要远高于批量梯度下降方法。**两者的关系可以这样理解：随机梯度下降方法以损失很小的一部分精确度和增加一定数量的迭代次数为代价，换取了总体的优化效率的提升。增加的迭代次数远远小于样本的数量。**

**对批量梯度下降法和随机梯度下降法的总结：**

- **批量梯度下降---最小化所有训练样本的损失函数，使得最终求解的是全局的最优解，即求解的参数是使得风险函数最小，但是对于大规模样本问题效率低下。**
- **随机梯度下降---最小化每条样本的损失函数，虽然不是每次迭代得到的损失函数都向着全局最优方向， 但是大的整体的方向是向全局最优解的，最终的结果往往是在全局最优解附近，适用于大规模训练样本情况。**

#### 小批量随机梯度下降法

相比前两者方法，小批量随机梯度下降是两者的一个折中方法，也是现在常用的梯度下降方法，它随机使用一个训练集的一个子集，即不是整个训练集，也不是一个训练样本，而是 n 个样本的训练子集。

**小批量梯度下降在参数空间上的表现比随机梯度下降要好的多**，尤其在有大量的小型实例集
时。作为结果，小批量梯度下降会比随机梯度更靠近最小值。但是，另一方面，****它有可能陷**
在局部最小值中** 。

#### 牛顿法

牛顿法是一种在实数域和复数域上近似求解方程的方法。方法使用函数$f (x)$的泰勒级数的前面几项来寻找方程$f(x) = 0$的根。它是二阶算法，它使用了 Hessian 矩阵求权重的二阶偏导数，目标是采用损失函数的二阶偏导数寻找更好的训练方向。**牛顿法最大的特点就在于它的收敛速度很快。**

具体步骤如下：

首先，选择一个接近函数$f(x)$零点的$x_0$，计算相应的$f(x_0)$和切线斜率$f\prime(x_0)$($表示函数的导数f\prime 表示函数f的导数$)。然后我们计算穿过点$(x_0, f(x_0))$并且斜率是$f\prime (x_0)$的直线和$x$轴的交点的$x$坐标，也就是求下列方程的解：
$$
x*f\prime(x_0) + f(x_0) - x_0 * f\prime(x_0) = 0
$$
我们将求得的新点的$x$坐标命名为$x_1$，通常$x_1$会比$x_0$更接近方程$f(x)=0$的解。因此，我们可以利用$x_1$开始下一轮迭代。迭代公式可化简为如下所示：
$$
x_{n+1} = x_n - \frac{f(x_n)}{f\prime(x_n)}
$$
已经证明，如果$f\prime$ 是连续的，并且待求的零点$x$是孤立的，那么在零点$x$周围存在一个区域，只要初始值$x_0$位于这个邻近区域内，那么牛顿法必定收敛。 并且，如果$f\prime(x)$不为0, 那么牛顿法将具有平方收敛的性能. 粗略的说，这意味着每迭代一次，牛顿法结果的有效数字将增加一倍。

由于牛顿法是基于当前位置的切线来确定下一次的位置，所以牛顿法又被很形象地称为是"切线法"。牛顿法的搜索路径（二维情况）如下图所示：

牛顿法搜索动态示例图：

![](https://upload.wikimedia.org/wikipedia/commons/e/e0/NewtonIteration_Ani.gif?_=4751804)

**关于牛顿法和梯度下降法的效率对比：**

- 从本质上去看，**牛顿法是二阶收敛，梯度下降是一阶收敛，所以牛顿法就更快**。如果更通俗地说的话，比如你想找一条最短的路径走到一个盆地的最底部，梯度下降法每次只从你当前所处位置选一个坡度最大的方向走一步，牛顿法在选择方向时，**不仅会考虑坡度是否够大，还会考虑你走了一步之后，坡度是否会变得更大**。所以，可以说牛顿法比梯度下降法看得更远一点，能更快地走到最底部。（牛顿法目光更加长远，所以少走弯路；相对而言，梯度下降法只考虑了局部的最优，没有全局思想。）
- 根据 wiki 上的解释，**从几何上说，牛顿法就是用一个二次曲面去拟合当前所处位置的局部曲面，而梯度下降法是用一个平面去拟合当前的局部曲面**，通常情况下，**二次曲面的拟合会比平面更好**，所以牛顿法选择的下降路径会更符合真实的最优下降路径。

![](http://images0.cnblogs.com/blog2015/764050/201508/222309373784741.png)

​				注：红色的牛顿法的迭代路径，绿色的是梯度下降法的迭代路径。

##### 优缺点

###### 优点

**二阶收敛，收敛速度快；**

###### 缺点

1. **Hessian矩阵（海森矩阵的逆）计算量较大，当问题规模较大时，不仅计算量大而且需要的存储空间也多，因此牛顿法在面对海量数据时由于每一步迭代的开销巨大而变得不适用；**
2. **牛顿法在每次迭代时不能总是保证海森矩阵是正定的，一旦海森矩阵不是正定的，优化方向就会“跑偏”，从而使得牛顿法失效，也说明了牛顿法的鲁棒性较差。**

#### 拟牛顿法

**拟牛顿法的本质思想是改善牛顿法每次需要求解复杂的Hessian矩阵的逆矩阵的缺陷，它使用正定矩阵来近似Hessian矩阵的逆，从而简化了运算的复杂度。**拟牛顿法和最速下降法一样只要求每一步迭代时知道目标函数的梯度。通过测量梯度的变化，构造一个目标函数的模型使之足以产生超线性收敛性。这类方法大大优于最速下降法，尤其对于困难的问题。另外，**因为拟牛顿法不需要二阶导数的信息，而是在每次迭代的时候计算一个矩阵，其逼近海塞矩阵的逆。最重要的是，该逼近值只是使用损失函数的一阶偏导来计算**，所以有时比牛顿法更为有效。如今，优化软件中包含了大量的拟牛顿算法用来解决无约束，约束，和大规模的优化问题。

具体步骤如下：

首先，构造目标函数在当前迭代$x_k$的二次模型：
$$
m_k(p) = f(x_k) + \nabla f(x_k)^T p + \frac{p^TB_k p}{2} \\
p_k = -B_k^{-1} \nabla f(x_k)
$$
这里$B_k$是一个对称正定矩阵，我们取这个二次模型的最优解作为搜索方向，并且得到新的迭代点：$x_{k+1} = x_k + a_kp_k$

其中要求步长$a_k$满足Wolfe条件。**这样的迭代与牛顿法类似，区别就在于用近似的Hessian矩阵$B_k$代替真正的Hessian矩阵。**所以**拟牛顿法最关键的地方就是每次迭代中矩阵$B_k$的更新**。现在假设得到一个新的迭代$x_{k+1}$，并得到一个新的二次模型：
$$
m_{k+1}(p) = f(x_{k+1}) + \nabla f(x_{k+1})^T p + \frac{p^TB_{k+1} p}{2} 
$$
我们尽可能利用上一步的信息来选取$B_k$，具体地，我们要求$\nabla f(x_{k+1}) - \nabla f(x_k) = a_kB_{k+1}p_k$

从而得到$B_{k+1}(x_{k+1} - x_k) = \nabla f(x_{k+1}) - \nabla f(x_k)$

这个公式被称为割线方程。常用的拟牛顿法有DFP算法和BFGS算法。

#### 共轭梯度法(Conjugate Gradient)

共轭梯度法是介于最速下降法与牛顿法之间的一个方法，**它仅需利用一阶导数信息，但克服了最速下降法收敛慢的缺点，又避免了牛顿法需要存储和计算Hesse矩阵并求逆的缺点，**共轭梯度法不仅是解决大型线性方程组最有用的方法之一，也是解大型非线性最优化最有效的算法之一。在各种优化算法中，共轭梯度法是非常重要的一种。**其优点是所需存储量小，具有收敛快，稳定性高，而且不需要任何外来参数。**

在共轭梯度训练算法中，因为是沿着**共轭方向（conjugate directions）执行搜索的**，所以通常该算法要比沿着梯度下降方向优化收敛得更迅速。共轭梯度法的训练方向是与海塞矩阵共轭的。

共轭梯度法已经证实其在神经网络中要比梯度下降法有效得多。并且由于共轭梯度法并没有要求使用海塞矩阵，所以在大规模神经网络中其还是可以做到很好的性能。

具体的实现步骤请参加[wiki百科共轭梯度法](https://en.wikipedia.org/wiki/Conjugate_gradient_method#Example_code_in_MATLAB)。

#### 启发式优化方法

启发式方法指人在解决问题时所采取的一种根据经验规则进行发现的方法。其特点是在解决问题时,利用过去的经验,选择已经行之有效的方法，而不是系统地、以确定的步骤去寻求答案。启发式优化方法种类繁多，包括经典的模拟退火方法、遗传算法、蚁群算法以及粒子群算法等等。

还有一种特殊的优化算法被称之多目标优化算法，它主要针对同时优化多个目标（两个及两个以上）的优化问题，这方面比较经典的算法有NSGAII算法、MOEA/D算法以及人工免疫算法等。

具体可以参考文章[[Evolutionary Algorithm\] 进化算法简介](http://www.cnblogs.com/maybe2030/p/4665837.html)

#### 解决约束优化问题--拉格朗日乘数法

这个方法可以参考文章[拉格朗日乘数法](http://www.cnblogs.com/maybe2030/p/4946256.html)

**Levenberg-Marquardt 算法**

Levenberg-Marquardt 算法，也称之为衰减最小二乘法（damped least-squares method），该算法的损失函数采用平方误差和的形式。**该算法的执行也不需要计算具体的海塞矩阵，它仅仅只是使用梯度向量和雅可比矩阵（Jacobian matrix）。**

该算法的损失函数使用如下所示的方程，即平方误差和的形式：
$$
f = \sum_{i=0}^m e_i^2
$$
其中参数$m$是数据集样本的数量。

我们可以定义损失函数的雅克比矩阵是以误差对参数的偏导数为元素的，即$J_{i,j}f(w) = \frac{de_i}{dw_j}\quad i=1,2...,m\quad j=1,2,...,n$

$n$是神经网络的参数数量，所以雅克比矩阵就是$m*n$的矩阵。

损失函数的梯度向量就是$\nabla f = 2J^Te$，$e$在这里表示所有误差项的向量。

所以，我们最后可以使用下列表达式来逼近Hessian矩阵：
$$
H(f) \approx 2J^TJ + \lambda I
$$
其中$\lambda$是衰减因子，它确保了Hessian矩阵的正定性，$I$是单位矩阵。

所以，参数的更新和优化如下公式所示：
$$
w_{i+1} = w_i - (J_i^TJ_i + \lambda I)^{-1}(2J_i^Te_i),\quad i = 0,1,...
$$
当$\lambda = 0$，该算法就是使用Hessian矩阵逼近值的牛顿法，而当$\lambda$很大时，该算法就近似于采用很小学习率的梯度下降法。如果进行迭代导致了损失函数上升，衰减因子$λ$就会增加。如果损失函数下降，那么$λ$就会下降，**从而 Levenberg-Marquardt 算法更接近于牛顿法。该过程经常用于加速收敛到极小值点。**

Levenberg-Marquardt 算法是**为平方误差和函数所定制的**。这就让使用这种误差度量的神经网络训练地十分迅速。然而 Levenberg-Marquardt 算法还有一些缺点，**第一就是其不能用于平方根误差或交叉熵误差（cross entropy error）等函数，此外该算法还和正则项不兼容。最后，对于大型数据集或神经网络，雅可比矩阵会变得十分巨大，因此也需要大量的内存。**所以我们在大型数据集或神经网络中并不推荐采用 Levenberg-Marquardt 算法。

#### 内存与收敛速度的比较

下图展示了所有上文所讨论的算法，及其收敛速度和内存需求。其中收敛速度最慢的是梯度下降算法，但该算法同时也只要求最少的内存。相反，Levenberg-Marquardt 算法可能是收敛速度最快的，但其同时也要求最多的内存。比较折衷方法是拟牛顿法。

![](http://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW8FZ7Lp7l5hLOSOZOjypSoENNfb5uPgLVibx4t889M9rg1WPHDIE4iaen2rzQXYdiaLSXibddq1honibLQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

总而言之，如果我们的神经网络有数万参数，为了节约内存，我们可以使用梯度下降或共轭梯度法。如果我们需要训练多个神经网络，并且每个神经网络都只有数百参数、数千样本，那么我们可以考虑 Levenberg-Marquardt 算法。而其余的情况，拟牛顿法都能很好地应对。



------

### 参考

- 《统计学习方法》
- [各种回归全解：传统回归、逻辑回归、加权回归/核回归、岭回归、广义线性模型/指数族](http://blog.csdn.net/ownfed/article/details/41181665)
- [Part 1 - Simple Linear Regression](http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/)
- [决策树剪枝算法](http://blog.csdn.net/yujianmin1990/article/details/49864813)
- [决策树系列（五）——CART](http://www.cnblogs.com/yonghao/p/5135386.html)
- [RandomForest随机森林总结](http://www.cnblogs.com/hrlnw/p/3850459.html)
- [SVM详解(包含它的参数C为什么影响着分类器行为)-scikit-learn拟合线性和非线性的SVM](http://blog.csdn.net/xlinsist/article/details/51311755)
- [机器学习常见算法个人总结（面试用）](http://kubicode.me/2015/08/16/Machine%20Learning/Algorithm-Summary-for-Interview/)
- [SVM-支持向量机算法概述](http://blog.csdn.net/passball/article/details/7661887)
- [机器学习算法与Python实践之（二）支持向量机（SVM）初级](http://blog.csdn.net/zouxy09/article/details/17291543)
- [机器学习算法与Python实践之（三）支持向量机（SVM）进阶](http://blog.csdn.net/zouxy09/article/details/17291805)
- [机器学习算法与Python实践之（四）支持向量机（SVM）实现](http://blog.csdn.net/zouxy09/article/details/17292011)
- [【模式识别】SVM核函数](http://blog.csdn.net/xiaowei_cqu/article/details/35993729)
- [SVM的核函数如何选取?--知乎](https://www.zhihu.com/question/21883548)
- [朴素贝叶斯理论推导与三种常见模型](http://blog.csdn.net/u012162613/article/details/48323777)
- [朴素贝叶斯的三个常用模型：高斯、多项式、伯努利](http://www.letiantian.me/2014-10-12-three-models-of-naive-nayes/)
- [机器学习&数据挖掘笔记_16（常见面试之机器学习算法思想简单梳理）](http://www.cnblogs.com/tornadomeet/p/3395593.html)
- [K-Means Clustering](http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-7/)
- [斯坦福大学公开课 ：机器学习课程](http://v.163.com/special/opencourse/machinelearning.html)
- [浅谈机器学习基础（上）](http://www.jianshu.com/p/ed9ae5385b89?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io#)
- [Ensemble learning:Bagging,Random Forest,Boosting](http://blog.csdn.net/taoyanqi8932/article/details/54098100)
- [机器学习（四）--- 从gbdt到xgboost](http://www.cnblogs.com/mfryf/p/5946815.html)
- [xgboost入门与实战（原理篇）](http://blog.csdn.net/sb19931201/article/details/52557382)
- [机器学习算法中GBDT和XGBOOST的区别有哪些？](https://www.zhihu.com/question/41354392)
- [常见的几种最优化方法](http://www.cnblogs.com/maybe2030/p/4751804.html)



