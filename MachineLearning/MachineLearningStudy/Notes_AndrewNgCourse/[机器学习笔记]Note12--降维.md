﻿# [机器学习笔记]Note12--降维

标签（空格分隔）： 机器学习

---
[TOC]

继续是[机器学习课程](https://www.coursera.org/learn/machine-learning)的笔记，本节介绍的是降维方法，这也是非监督学习中的一个问题，主要介绍主要成分分析（Principal Component Analysis，PCA）算法。

### 降维
#### 动机一：数据压缩
  使用降维的一个原因是数据压缩。下面通过几个例子来介绍降维。
  
  第一个例子是将数据从二维降至一维。假设我们要采用两种不同的仪器来测量一些东西的尺寸，其中一个仪器测量结果的单位是英尺，另一个仪器测量的结果单位是厘米，我们希望将测量的结果作为机器学习的特征。现在的问题是，两种仪器对同一个东西测量的结果不完全相等（由于误差、精度等），而将两者都作为特征有些重复，因而，我们希望将这个二维的数据降至一维。如下图所示：
  
  ![此处输入图片的描述][1]
  
  具体做法就是，找出一条合适的直线，如上图下方那条直线，然后将所有的数据点都投射到该直线上，然后用$z^{(i)}$标识，这样便完成了从二维数据$x^{(i)}$向一维数据$z^{(i)}$的映射。这样得到的新特征只是原有特征的近似，但好处是**将需要的存储、内存占用量减半，而已可以让使用这些数据的算法运行得更快**。
  
  第二个例子是将数据从三维降至二维。这个例子的过程与上面类似，是将三维向量投射到一个二维平面上，强迫使得所有的数据都在同一个平面上，降至二维的特征向量。

![此处输入图片的描述][2]

这样的处理过程可以被用于把任何维度的数据都降到任何想要的维度，如将1000维的特征降至100维。

#### 动机二：数据可视化
  在许多机器学习问题中，如果我们能将数据可视化，这有助于我们寻找到一个更好的解决方案，而降维可以帮助做到数据可视化。
  
  一个例子是假设现在有关于许多不同国家的数据，每一个特征向量都有50个特征（如，GDP,人均GDP,平均寿命等），如下图所示。
  
  ![此处输入图片的描述][3]
  
  如果要将这个50维的数据可视化是不可能的，但是使用降维的方法将其降至2维，那就可以将其可视化。如下图所示，用新的特征$z_1和z_2$来表现。
  
  ![此处输入图片的描述][4]
  
  这样的问题就是，**降维的算法只负责减少维度，而新特征的意义就必须由我们自己去发现了。**对于上述例子，我们根据新的二维特征画出一个二维图，如下图所示，用点$z^{(i)}$表示每个国家，那么可能会发现水平轴可能对应的是一个国家的面积或者是GDP，而纵轴计算对应人均GDP或者幸福感等。
  
  ![此处输入图片的描述][5]
  
### 主要成分分析（Principal Component Analysis，PCA）
  主要成分分析时最常见的降维算法。
  
  在PCA中，如果是将二维数据降至一维，我们要做的就是找到一个**方向向量（Vector direction)**，当我们将所有的数据都投射到该向量上时，我们希望投射平均均方误差可以尽可能地小。**方向向量时一个经过原点的向量，而投射误差是从特征向量向该方向向量作垂线的长度。**如下图所示

  ![此处输入图片的描述][6]
  
  下面给出PCA问题的一般描述：

* 问题是将n维数据降至k维
* 目标是找到向量$u^{(1)},u^{(2)},\ldots,u^{(k)}$使得总的投射误差最小

  然后是比较PCA和线性回归的，这两种算法是不同的算法。**PCA最小化的是投射误差，而线性回归尝试的是最小化预测误差。线性回归的目的是预测结果，而PCA不作任何预测。**如下图所示
  
  ![此处输入图片的描述][7]
  
  左图是线性回归的误差，而右图是PCA的误差。
  
### PCA算法
  接下来是介绍PCA的具体实现过程。
  
  首先是预处理过程，做的是**均值归一化**。需要计算出所有特征的均值$\mu_j=\frac{1}{m}\sum_{i=1}^m x_j^{(i)}$,然后令$x_j = x_j-\mu_j$。如果特征是不同数量级的，还需要将其除以标准差$\sigma^2$。
  
  接下来就是正式的PCA算法过程了。也就是要**计算协方差矩阵(covariance matrix)$\sum$**。而协方差矩阵$\sum=\frac{1}{m}\sum_{i=1}^m (x^{(i)})(x^{(i)})^T$。
  
  然后就是计算协方差矩阵的**特征向量（eigenvectors)**。在Octave语言中可以利用**奇异值分解（singular value decomposition,SVD)来求解，`[U,S,V] = svd(sigma)`。
  
  对于一个$n \times n$维度的矩阵，上式中的U是一个具有与数据之间最小投射误差的方向向量构成的矩阵。如果我们希望将数据从n维降至k维，我们只需要从U中选取前K个向量，获得一个$n\times k$维度的矩阵，这里用$U_{reduce}$表示，然后通过如下计算获得要求的新特征向量$z^{(i)}$:
$$
z^{(i)}=U_{reduce}^T \times x^{(i)}
$$

其中$x是n \times 1$维的，因此结果是$k \times 1$维。

注意，这里我们部队偏倚特征进行处理。

在压缩过数据后，我们可以采用如下方法来近似地获得原有的特征：$x_{approx}^{(i)}=U_{reduce}z^{(i)}$

### 选择主要成分的数量
  PCA需要将n维数据降至k维数据，这里的k也就是PCA需要确定的参数K，也就是主要成分的数量。
  
  主要成分分析是要**减少投射的平均均方误差**：
$$
\frac{1}{m}\sum_{i=1}^m ||x^{(i)}-x_{approx}^{(i)}||^2
$$

而训练集的方差是$\frac{1}{m}\sum_{i=1}^m ||x^{(i)}||^2$。

我们希望的是**在平均均方误差与训练集方差的比例尽可能小的情况下选择尽可能小的K值**。

一般来说，我们希望这个比例，如下所示，是小于1%，即意味着原本数据的偏差有99%都保留下来了。
$$
\frac{\frac{1}{m}\sum_{i=1}^m ||x^{(i)}-x_{approx}^{(i)} ||^2}{\frac{1}{m}\sum_{i=1}^m ||x^{(i)}||^2} \le 0.01
$$

而如果选择保留95%的偏差，便能显著地降低模型中特征的维度了。

所以做法可以是，先令K=1，然后进行PCA，获得$U_{reduce}和z$，然后计算比例是否小于1%。如果不是，再令K=2，如此类推，直到找到可以使得比例小于1%的最小K值（原因是各个特征之间通常情况存储某种相关性）。

还有一些更好的方式来选择K，在Octave语言中调用**svd**函数的时候，我们获得三个参数：`[U,S,V]=svd(sigma)`。其中S是一个$n\times n$的矩阵，只有对角线上有值，其他单元都是0，我们可以使用这个矩阵来计算平均均方误差与训练集方差的比例：
$$
\frac{\frac{1}{m}\sum_{i=1}^m ||x^{(i)}-x_{approx}^{(i)} ||^2}{\frac{1}{m}\sum_{i=1}^m ||x^{(i)}||^2} = 1-\frac{\sum_{i=1}^k S_{ii}}{\sum_{i=1}^m S_{ii}} \le 1\%
$$
也就是：
$$
\frac{\sum_{i=1}^k S_{ii}}{\sum_{i=1}^m S_{ii}} \ge 0.99
$$

### 应用PCA
  假设我们正在针对一张$100\times 100$像素的图片进行某个计算机视觉的机器学习，即总共有10000个特征。这里可以使用PCA来降维来提高算法的速度。做法如下：

1. 第一步是使用PCA将数据压缩至1000个特征；
2. 对训练集运行学习算法；
3. 在预测的时候，使用第一步学来的$U_{reduce}$将测试集中的特征x转换成新的特征向量z，然后再进行预测。

**注意：只有在训练集才运行PCA算法，而将训练集中学到的$U_{reduce}$应用到交叉验证集和测试集中。**

错误使用PCA的情况有：

1. **将PCA用于减少过拟合（减少特征的数量）。**这样做并不好，不如尝试归一化处理。**原因是PCA只是近似地丢弃掉一些特征，它并不考虑与结果变量有关的信息，因此可能丢掉非常重要的特征。而当进行归一化处理时，会考虑到结果变量，不会丢掉重要的数据。**
2. **默认地将PCA作为学习过程中的一部分**。这虽然很多时候有效果，但最好还是从原始特征开始，只在有必要的时候（算法运行太慢或者占用太多内存）才考虑使用PCA。

### 小结
  本节内容介绍了使用降维的两大原因，一个是进行数据压缩，减少内存的使用，提高算法速度，第二个是为了数据可视化，从而找到一个更好的解决问题的方法。
  
  降维方法中最常用的就是PCA算法，所以本节内容主要是介绍PCA算法的基本做法，具体实现过程，以及使用的方法和注意事项。

  
  
  
  


  [1]: http://img.blog.csdn.net/20160723200106464
  [2]: http://img.blog.csdn.net/20160723200625388
  [3]: http://img.blog.csdn.net/20160723201507909
  [4]: http://img.blog.csdn.net/20160723201900822
  [5]: http://img.blog.csdn.net/20160723202307240
  [6]: http://img.blog.csdn.net/20160723203144433
  [7]: http://img.blog.csdn.net/20160723203514963