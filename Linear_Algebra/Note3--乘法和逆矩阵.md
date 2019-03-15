# Note3--乘法和逆矩阵

标签（空格分隔）： 线性代数

---
继续是线性代数的学习笔记，第三节课[乘法和逆矩阵](http://open.163.com/movie/2010/11/H/O/M6V0BQC4M_M6V29FCHO.html)

### 矩阵乘法
 首先是对于矩阵相乘，如矩阵A和B相乘得到C，即`A*B=C`;那么如果要得到矩阵C的一个元素，如$c_{34}$,其求解如下所示：
$$
 c_{34} = a_{31}*b_{14} + a_{32}*b_{24}+\ldots = \sum_{k=1}^n a_{3k}b_{k4} 
$$
这里的n是矩阵A的列数，也是矩阵B的行数，所以两者要能相乘得到矩阵C，其要满足矩阵A的大小是`m*n`，而B的大小是`n*p`，这样得到的矩阵C的大小就是`m*p`。

所以矩阵相乘的条件是：

- **如果不是方阵，第一个矩阵的列数要等于第二个矩阵的行数。**
- **如果是方阵，则两者必须大小相同。**

  除了上述方法来进行矩阵相乘外，还是有其他方法的。**第二种方法就是整列考虑，也就是变成矩阵乘以向量的方法，**就是第一节中提到的方法。

  比如对于上述例子，矩阵A乘以矩阵B的第一列会得到矩阵C的第一列，以此类推，就可以得到一个完整的矩阵C，**那么矩阵C的每一列都是矩阵A中列的线性组合。**

  **第三种方法就是根据行向量了，即A的每一行乘以矩阵B得到C对应的行，那么C中每一行就是B的行的线性组合。**

  **第四种方法就是使用A的一列乘以B的一行，同样可以得到矩阵C。**一个例子如下所示：

$$
\begin{bmatrix}2 & 7 \\ 3 & 8 \\ 4 & 9 \end{bmatrix} \begin{bmatrix}1 & 6 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix}2\\ 3  \\ 4 \end{bmatrix}\begin{bmatrix}1 & 6\end{bmatrix}+ \begin{bmatrix}7  \\ 8 \\ 9 \end{bmatrix} \begin{bmatrix}0 & 0 \end{bmatrix} = \begin{bmatrix}2 & 12 \\ 3 & 18 \\ 4 & 24 \end{bmatrix}
$$

**第五种方法就是分块方法**。将矩阵A和矩阵B分别分成若干块，每块大小不需要一定相同，但是必须满足矩阵相乘的条件。其分块乘法规则就是相当于构建一个新的矩阵相乘。例子如下所示：将矩阵A、B都分成4块，如下所示，
$$
A=\begin{bmatrix}A1 & A2 \\ A3 & A4 \end{bmatrix}，B=\begin{bmatrix}B1 & B2 \\ B3 & B4 \end{bmatrix}
$$
则矩阵A、B相乘得到的C为$\begin{bmatrix}A1B1+A2B3 & A1B2+A2B4 \\ A3B1+A4B2 & A3B2+A4B4 \end{bmatrix}$

### 逆
  接下来介绍逆矩阵的内容。令I表示单位矩阵，则**若方阵A可逆，即有逆矩阵$A^{-1}$，则有$AA^{-1}=A^{-1}A=I$成立，同时矩阵A被称为可逆的，或者非奇异的矩阵**。

  这里要注意公式一定成立的前提是A必须是一个方阵。

#### 奇异矩阵
  首先介绍如何判断奇异矩阵，也就是不可逆的矩阵。

> 对于一个矩阵$A$,如果可以找到一个非零向量$X$,使得$AX=0$成立，则矩阵$A$是不可逆的。

所以，假设有一个不可逆的矩阵$A=\begin{bmatrix}1 & 3 \\ 2 & 6 \end{bmatrix}$,可以找到一个非零向量$X=\begin{bmatrix} 3 \\ -1 \end{bmatrix}$，使得$AX=0$。

#### 可逆矩阵
  对于一个可逆矩阵A，我们应该如何找到其逆矩阵$A^{-1}$。这里将用到**“Gaussian-Jordan”消元法**。

  假设有一个可逆矩阵$A=\begin{bmatrix}1 & 3 \\ 2 & 7 \end{bmatrix}$，令其逆矩阵$A^{-1}=\begin{bmatrix}a & c \\ b & d \end{bmatrix}$,因为$AA^{-1}=I=\begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix}$,也就是有：
$$
\begin{bmatrix}1 & 3 \\ 2 & 7 \end{bmatrix} \begin{bmatrix}a \\ b \end{bmatrix}=\begin{bmatrix}1  \\ 0 \end{bmatrix} \\
\begin{bmatrix}1 & 3 \\ 2 & 7 \end{bmatrix} \begin{bmatrix}c \\ d \end{bmatrix}=\begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

而利用“Gaussian-Jordan”消元法来解，有如下过程，其中使用到增广矩阵的知识：
$$
\left[
    \begin{array}{cc|cc}
      1 & 3 & 1 & 0 \\
      2 & 7 & 0 & 1
    \end{array}
\right]
(row2 = row2-2*row1)-->
\left[
    \begin{array}{cc|cc}
      1 & 3 & 1 & 0 \\
      0 & 1 & -2 & 1
    \end{array}
\right]
(row1 = row1-3*row2)-->
\left[
    \begin{array}{cc|cc}
      1 & 0 & 7 & -3 \\
      0 & 1 & -2 & 1
    \end{array}
\right]
$$

对于上述消元过程，首先第一步给出的矩阵
$$
\left[
\begin{array}{cc|cc}
      1 & 3 & 1 & 0 \\
      2 & 7 & 0 & 1
\end{array}
\right]
$$
就是一个增广矩阵，它用中间的一条竖线分为左右两部分，左边就是矩阵$A$，右边部分就是单位矩阵$I$,然后首先是令第二行减去第一行乘以2后的结果，也就是中间部分的矩阵，此时矩阵A部分的第二行变成$[0\ 1]$了，然后就是让第一行减去第二行乘以3的结果，得到最后一个矩阵，此时可以发现左边就是一个单位矩阵$I$，而右边就是我们需要的结果$A^{-1}$，这里可以令$AA^{-1}$来验证是否等于单位矩阵，从而判断得到的是否就是矩阵A的逆矩阵。






