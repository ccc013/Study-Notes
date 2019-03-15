# Note2 矩阵消元

标签（空格分隔）： 线性代数

---
[TOC]

第二节介绍[矩阵消元](http://open.163.com/movie/2010/11/P/P/M6V0BQC4M_M6V29EGPP.html).

### 消元法

首先是给出一个例子来说明消元法的使用，例子如下所示：

$$
\begin{cases}
x+2y+z=2 \\
3x+8y+z=12 \\
 4y+z=2
\end{cases}
$$

用矩阵表示就是
$$
A = \left[ \begin{matrix} 1 & 2 & 1 \\ 3 & 8 & 1 \\ 0 & 4 & 1 \end{matrix} \right]  b = \left[ \begin{matrix} 2 \\ 12 \\ 2 \end{matrix} \right]
$$

消元法的步骤首先是方程1乘以某个系数，然后方程2减去它，使得让方程2的x的系数变为0，然后同理让方程3的y的系数变为0。做法如下所示：

$$\left[ \begin{matrix} 1 & 2 & 1 \\ 3 & 8 & 1 \\ 0 & 4 & 1 \end{matrix} \right]  \left[ \begin{matrix} 2 \\ 12 \\ 2 \end{matrix} \right] =>(方程1乘以3 ) \left[ \begin{matrix} 1 & 2 & 1 \\ 0 & 2 & -2 \\ 0  & 4 & 1 \end{matrix} \right]  \left[ \begin{matrix} 2 \\ 6 \\ 2 \end{matrix}\right]$ =>(方程2乘以2)  \left[ \begin{matrix} 1 & 2 & 1 \\ 0 & 2 & -2 \\ 0  & 0 & 5\end{matrix} \right]  \left[ \begin{matrix} 2 \\ 6 \\ -10 \end{matrix} \right]$$

这里就得到了
$$
u = \left[ \begin{matrix} 1 & 2 & 1 \\ 0 & 2 & -2 \\ 0  & 0 & 5 \end{matrix} \right]  c = \left[ \begin{matrix} 2 \\ 6 \\ -10 \end{matrix} \right]
$$

当得到矩阵u和c后，就可以进行会代，即如下方程组
$$
\begin{cases}
x+2y+z=2 \\
2y-2z=6 \\
 5z=-10
\end{cases}
$$

自然就得到答案
$$
\begin{cases}
x=2 \\
y=1 \\
z=-2
\end{cases}
$$

这里的矩阵是可逆的，所以可以使用消元法，但是还是存在一些矩阵是不适用于消元法的，比如如果该例子中方程组1的`x`系数是0，这个时候需要使用如行交换的方法来得到适合使用消元法的矩阵。

### 矩阵消元

  这里将介绍使用矩阵变换来使用消元法。

###### 第一步
$$
\left[ \begin{matrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ 0 & 0 & 1 \end{matrix} \right] \left[ \begin{matrix} 1 & 2 & 1 \\ 3 & 8 & 1 \\ 0 & 4 & 1 \end{matrix} \right] = \left[ \begin{matrix} 1 & 2 & 1 \\ 0 & 2 & -2 \\ 0  & 4 & 1 \end{matrix} \right] 
$$

第一个矩阵称之为初等矩阵，记为$E_{21}$,表示修改的是第二行第一列的位置，而保持第一行和第三行不变，实际上是在单位矩阵$\left[ \begin{matrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{matrix} \right]$的基础上进行修改，如果是直接跟单位矩阵相乘，那么就是得到相同的结果，而现在是需要将第二行减去第一行乘以3的结果，而第二行第一列的值乘以的就是第二个方程的第一行的值，然后再相加，实现的效果是一样的。

###### 第二步
$$
\left[ \begin{matrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -2 & 1 \end{matrix} \right] \left[ \begin{matrix} 1 & 2 & 1 \\ 0 & 2 & -2 \\ 0  & 4 & 1 \end{matrix} \right] = \left[ \begin{matrix} 1 & 2 & 1 \\ 0 & 2 & -2 \\ 0  & 0 & 5 \end{matrix} \right] 
$$

第一个矩阵也是初等矩阵，记为$E_{32}$,表示修改的是第三行第二列的位置，而保持第一行和第二行不变。

上述两步可以表示为
$E_{32}$($E_{21}$A) = u

这里可以使用乘法的结合律，也就是($E_{32}$$E_{21}$) A= u。

但是注意这里是不适用交换律的。

这里可以求解$E_{32}$$E_{21}$的值，但是老师说可以有更好的方法，就是求逆矩阵。即求**让矩阵U变回矩阵A**的矩阵。如下所示

$$
\left[ \begin{matrix} 1 & 0 & 0 \\ 3 & 1 & 0 \\ 0 & 0 & 1 \end{matrix} \right]  \left[ \begin{matrix} 1 & 0 & 0 \\ -3 & 1 & 0 \\ 0 & 0 & 1 \end{matrix} \right]  = \left[ \begin{matrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{matrix} \right]
$$

三个矩阵分别记为$E^{-1}$,**E**,**I**。

（这个求解逆矩阵的方法，暂时还没想明白为什么更好）

### 置换矩阵

最后老师讲解了一个置换矩阵的知识点，这个和本节课的内容并不太相关。

首先是一个行交换的例子。
$$
\left[ \begin{matrix} 0 & 1 \\ 1 & 0 \end{matrix} \right] \left[ \begin{matrix} a & b \\ c & d \end{matrix} \right] = \left[ \begin{matrix} c & d \\ a & b \end{matrix} \right]
$$

第一个矩阵就是置换矩阵P，实现对第二个矩阵的行交换。

而如果是列交换，则如下所示
$$
 \left[ \begin{matrix} a & b \\ c & d \end{matrix} \right] \left[ \begin{matrix} 0 & 1 \\ 1 & 0 \end{matrix} \right] = \left[ \begin{matrix} b & a \\ d & c \end{matrix} \right]
$$

这里也说明了矩阵的交换律，即如`BA = AB`是不成立的。

### 总结

这节课讲的是矩阵消元法，还算是比较基础的内容。





