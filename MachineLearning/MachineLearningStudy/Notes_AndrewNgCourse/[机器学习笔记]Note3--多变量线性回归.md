﻿# [机器学习笔记]Note3--多变量线性回归

标签（空格分隔）： 机器学习

---

继续是[机器学习课程](https://www.coursera.org/learn/machine-learning)的笔记，这节课介绍的是多变量的线性回归。

### 多变量线性回归
#### 多维特征
  上节课介绍的是单变量的线性回归，这节课则是进一步介绍多变量的线性回归方法。
  现在假设在房屋问题中增加更多的特征，例如房间数，楼层等，构成一个含有多个变量的模型，模型中的特征为$(x_1,x_2,\ldots,x_n)$.
  如下图所示：
  ![此处输入图片的描述][1]
在增加这么多特征后，需要引入一系列新的注释：

- n 代表特征的数量
- $x^{(i)}$代表第i个训练实例，是特征矩阵中的第i行，是一个**向量**
- $x_j^{(i)}$代表特征矩阵中第i行的第j个特征，也是第i个训练实例的第j个特征

所以在如上图中，特征数量`n=4`，然后$x^{(2)}=\left[\begin{matrix} 1416 \\ 3 \\ 2 \\ 40 \end{matrix} \right] $,这表示的就是图中第二行的数据，也是第二个训练实例的特征，而$x^{(2)}_3 = 2$,表示的就是第二行第3列的数据。

现在支持多变量的假设`h`表示为：
$$
h_\theta (x)= \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
$$
这个公式中有n+1个参数和n个变量，为了让公式可以简化一些，引入$x_0$ =1,则公式变为：
$$
h_\theta (x)= \theta_0 x_0+ \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
$$
此时模型中的参数是一个n+1维的向量，任何一个训练实例也是n+1维的向量，特征矩阵X的维度是`m*(n+1)`。
此时特征矩阵$x=\left[\begin{matrix} x_0 \\ x_1 \\ x_2 \\ \vdots \\ x_n \end{matrix} \right]$,参数$\theta = \left[\begin{matrix} \theta_0 \\ \theta_1 \\ \theta_2 \\ \vdots \\ \theta_n \end{matrix} \right]$,所以假设`h`就可以如下表示：
$$
 h_\theta (x) = \theta^T x
$$
上述公式中的`T`表示矩阵转置。

#### 多变量梯度下降
  与单变量一样，我们也需要为多变量构建一个代价函数，同样也是一个所有建模误差的平方和，即:
$$
J(\theta_0,\theta_1,\ldots,\theta_n) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2
$$
目标也是找到让代价函数最小的一系列参数，使用的也是梯度下降法，多变量线性回归的批量梯度下降算法如下所示：
> Repeat{
$$
 \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0,\theta_1,\ldots,\theta_n)
$$
}

也就是
> Repeat{
$$
 \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2
$$
}

通过求导数后，可以得到
> Repeat{
$$
 \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})\cdot x_j^{(i)} \\
 (同时更新参数\theta_j, for \; j = 0,1,\ldots,n)
$$
}

其更新方法如下所示:
$$
 \theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})\cdot x_0^{(i)} \\
  \theta_1 := \theta_1 - \alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})\cdot x_1^{(i)} \\
   \theta_2 := \theta_2 - \alpha \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})\cdot x_2^{(i)} \\
 \ldots
$$

#### 特征缩放
 在面对多特征问题的时候，我们要保证这些特征都具有相近的尺度，这将帮助梯度下降算法更快地收敛。
 
   以房价问题为例，假设我们使用两个特征，房屋的尺寸和房间的数量，前者的值是0-2000平方英尺，而后者的值是0-5，以两个参数分别为横纵坐标，绘制代价函数的等高线图，如下图所示，能看出图像会显得很扁，梯度下降算法需要非常多次的迭代才能收敛。
   ![此处输入图片的描述][2]
   
**解决的方法就是尝试将所有特质的尺度都尽量缩放到$-1\le x_i \le 1$之间。**最简单的方法如下所示：
$$
 x_n = \frac{x_n-\mu_n}{S_n} \\
 其中\mu_n是平均值，S_n可以是标准差或者是最大值减去最小值
$$

#### 学习率
  对于梯度下降，我们还需要解决的问题有：
  
- 如何判断当前的梯度下降是正常工作，即最终可以收敛；
- 如何选择一个学习率

对于第一个问题，由于迭代次数会随着模型不同而不同，我们也不能提前预知，但可以通过绘制迭代次数和代价函数的图表来观察算法在何时收敛。如下图所示：
![此处输入图片的描述][3]
 由上图所示，当曲线在每次迭代后都是呈下降趋势，那么可以表明梯度下降是正常工作的，然后图中可以看出在迭代次数达到400后，曲线基本是平坦的，可以说明梯度下降法在迭代到这个次数后已经收敛了。
 
 当然也有一些自动测试是否收敛的，例如将代价函数的变化值与某个阈值(如0.001)进行比较。但选择这个阈值是比较困难的，所以通常看上面的图表会更好。
 
 对于第二个问题，如何选择一个学习率。由于梯度下降算法的每次迭代都会受到学习率的影响，如果学习率过小，那么达到收敛需要的迭代次数会非常大；但如果学习率过大，每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛。
 
 通常可以考虑尝试这些学习率$\alpha = 0.001,0.003,0.01,0.03,0.1,0.3,1,\ldots$。
 
#### 特征和多项式回归
  线性回归并不适用于所有数据，有时需要曲线来适应数据，比如一个二次方模型$h_\theta(x) = \theta_0 + \theta_1 x_1+\theta_2 x_2^2$，或者三次方模型$h_\theta(x) = \theta_0 + \theta_1 x_1+\theta_2 x_2^2+\theta_3 x_3^3$
而这就是多项式回归，比如在房屋问题中，我们可以选择3个特征，一个房屋的价格，房屋的面积，房屋的体积，这样就会用到三次方模型，其曲线如下图所示：
![此处输入图片的描述][4]
  
当然，如果我们希望继续使用线性回归模型，可以令：
$$
x_2 = x_2^2 \\
x_3 = x_3^3
$$

这样就可以将模型转换为线性回归模型$h_\theta(x) = \theta_0 + \theta_1 x_1+\theta_2 x_2+\theta_3 x_3$。**但是如果使用多项式回归模型，在运行梯度下降算法前，有必要使用特征缩放。**
 
### 正规方程(Normal Equation)
#### 正规方程简介
  到目前为止，我们都是使用梯度下降算法来解决线性回归问题，但是对于某些线性回归问题，**正规方程方法是更好的解决方案。**
>  正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：
$$
\frac{\partial}{\partial \theta_j} J(\theta_j) = 0
$$

假设我们的数据如下所示：
![此处输入图片的描述][5]
然后我们在每行数据都添加一个$x_0=1$，可以得到下列表格数据：
![此处输入图片的描述][6]
那么可以得到我们的训练集特征矩阵X以及训练结果向量y：
$$
  X = \left[\begin{matrix}
  1 & 2104 & 5 & 1 & 45 \\
  1 & 1416 & 3 & 2 & 40 \\
  1 & 1534 & 3 & 2 & 30 \\
  1 & 852 & 2 & 1 & 36 \end{matrix}\right]
  \quad y = \left[\begin{matrix} 460 \\ 232 \\ 315 \\ 178 \end{matrix}\right]
$$
则利用正规方法可以得到向量$\color{red}{\theta = (X^TX)^{-1}X^Ty}$,其中T代表矩阵转置，上标-1表示矩阵的逆。

> 注意：对于那些不可逆的矩阵(通常是因为特征之间不独立，如同时包含英尺为单位的尺寸和米为单位的尺寸两个特征，也有可能是因为特征数据量大于训练集的数量)，正规方程方法是不能用的。

#### 梯度下降法与正规方程的比较
 下面给出梯度下降方法和正规方法的比较：
 | 梯度下降 | 正规方程 |
 | :-------:| :-------:|
 |需要选择学习率$\alpha$| 不需要|
 |需要多次迭代|一次运算得到|
 |当特征量n大时也能较好使用|如果特征数量n比较大则运算代价大，因为矩阵逆的运算时间复杂度为$O(n^3)$,通常来说n小于10000还是可以接受的|
 |适用于各种类型的模型|只适用于线性模型，不适合逻辑回归模型等其他模型|
 
### 小结
  本节课内容主要是介绍了多变量的线性回归方法，跟单变量线性回归方法还是比较类型的，只是需要增加多几个变量，同样是使用误差平方和函数作为代价函数，然后也是使用梯度下降算法。但需要注意的是由于是多个变量，每个变量的取值范围可能相差很大，这就需要使用特征缩放，通常是将每个特征缩放到$[-1,1]$，然后就是介绍了如何选择学习率以及判断梯度下降是否收敛的问题。
  
  接着就是介绍了多项式回归方法，这是由于线性回归可能对某些数据并不适用，所以需要使用如二次方模型，三次方模型等训练数据，但可以通过变量转换来重新使用线性回归模型，但是需要使用特征缩放方法。
  
  最后就是介绍了一种跟梯度下降方法有同样效果的正规方程方法，主要是通过求解$\frac{\partial}{\partial \theta_j} J(\theta_j) = 0$来得到参数值，并给出两种方法的对比。


  [1]: http://img.blog.csdn.net/20160608152421331
  [2]: http://img.blog.csdn.net/20160608163102186
  [3]: http://img.blog.csdn.net/20160608164733557
  [4]: http://img.blog.csdn.net/20160608201046727
  [5]: http://img.blog.csdn.net/20160608152421331
  [6]: http://img.blog.csdn.net/20160608203443897