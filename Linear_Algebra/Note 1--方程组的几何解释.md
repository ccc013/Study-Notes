# Note 1--方程组的几何解释

标签（空格分隔）： 线性代数 机器学习

---
[TOC]

这是记录麻省理工学院公开课：线性代数的笔记，网址是[麻省理工公开课：线性代数](http://open.163.com/special/opencourse/daishu.html)
第一节课说的是有关方程组的几何解释。网址是[方程组的几何解释](http://open.163.com/movie/2010/11/7/3/M6V0BQC4M_M6V29E773.html)

----------
 
 首先是介绍方程组的几何解释，提出可以用矩阵表示，然后矩阵表示有两种表达方式，分别是行图像和列图像。行图像比较常见，比如两条直线相交，而列图像则比较少见。
 
### 两个未知数两个方程 ###

 然后老师举例说明，首先是两个方程组两个未知数的例子，例子如下所示：
 $$
 \begin{cases}  
 \ 2x-y = 0 \\
 \ -x+3y= 3 
 \end{cases}  
 $$
 
用行图像表示如下所示：
$$
\left[\begin{matrix}2 & -1  \\-1 & 3  \\\end{matrix} \right] \left[\begin{matrix} x\\ y \end{matrix}\right]  = \left[\begin{matrix} 0\\ 3 \end{matrix}\right] 
$$

这里用**A**=$\left[\begin{matrix}2 & -1  \\-1 & 3  \\\end{matrix} \right]$,**x** = $\left[\begin{matrix} x\\ y \end{matrix}\right]$,**b**=$\left[\begin{matrix} 0\\ 3 \end{matrix}\right] $,可以得到`Ax =b`

这里表示的就是两条直线，并且它们相交于点`(1,2)`。

如果是用列向量，则如下所示：
$$
x\left[\begin{matrix}2 \\-1 \end{matrix} \right] + y\left[\begin{matrix}-1\\3\end{matrix}\right] = \left[\begin{matrix} 0\\ 3 \end{matrix}\right] 
$$

对于这种写法，老师称之为列向量的线性组合，然后在二维坐标平面上表示了这两个向量，而这个列向量的线性组合的解，其实在用行图像表示的时候已经得到了，就是`x=1, y=2`。

### 三个未知数三个方程组 ###
  接着老师给出了三个未知数的情况，举例如下所示
  $$
 \begin{cases}  
 \ 2x-y = 0 \\
 \ -x+2y-z = -1 \\
 \    -3y+4z = 4
 \end{cases}  
 $$ 
 使用行图像表示，**A** = $\left[\begin{matrix}2 & -1 & 0\\-1 & 2 & -1 \\ 0 & -3 & 4\end{matrix}\right]$,**b**=$\left[\begin{matrix}0\\-1\\4\end{matrix}\right]$,
 
 使用列图像表示是如下所示：
 $$
 x\left[ \begin{matrix}2\\-1\\0\end{matrix} \right]+y\left[ \begin{matrix}-1\\2\\-3\end{matrix} \right]+z\left[ \begin{matrix}0\\-1\\4\end{matrix} \right]=\left[ \begin{matrix}0\\-1\\4\end{matrix} \right]
 $$
 
 如果通过行图像来求解，需要通过在三维坐标轴上画出3个平面求平面的交点，这是非常困难的。(这里老师也说了下一节课会介绍消元法来求解)。
 
 而如果看列图像，则可以轻松得到答案：`x=0,y=0,z=1`，当然这是老师特意设计的题目，所以才这么容易得到这个答案。
 
 然后老师就问了一个问题：
> 对任意的**b**，都能令`Ax = b`有解吗？
这个问题对于这个三个未知数的例子来说，等价于这个例子中的列向量的线性组合是否能覆盖整个三维空间？

这里的答案当然是不能确定的，如果三个列向量都是在同一个平面上，那么得到的解也就只是在同一个平面的。

### 矩阵向量相乘的解法 ###
最后老师介绍了矩阵与向量相乘的两种解法，首先是一个例子
$$
\left[ \begin{matrix}2 & 5\\1 & 3\end{matrix}\right] \left[\begin{matrix}1\\2\end{matrix}\right]
$$
两种解法分别是按照行向量还是列向量来解答的。

第一种，如果是按照列向量解答，则可以写成如下所示：
$$
\left[ \begin{matrix}2 & 5\\1 & 3\end{matrix}\right] \left[\begin{matrix}1\\2\end{matrix}\right] = 1\left[ \begin{matrix}2 \\1\end{matrix}\right]+ 2 \left[ \begin{matrix}5\\3\end{matrix}\right] = \left[ \begin{matrix}12\\7\end{matrix}\right]
$$

第二种，就是按行来求解，如下所示：
$$
\left[ \begin{matrix}2 & 5\\1 & 3\end{matrix}\right] \left[\begin{matrix}1\\2\end{matrix}\right] =\left[ \begin{matrix}2*1+5*2\\1*1+3*2\end{matrix}\right]= \left[ \begin{matrix}12\\7\end{matrix}\right]
$$
也就是第一个矩阵的第一行乘以第二个向量的对应列，然后第二行乘以第二个向量的对应列。

这种解法也是当初刚开始学习线性代数所学习的方法。

### 总结 ###
这节课的收获主要是了解到列向量这种求法，之前对于矩阵的求解，还是通过按行来相乘求解的。不过在这节课中的例子都是矩阵乘以向量得到一个向量，如果是矩阵之间的相乘，不知道是否还是可以如此解决。

最后是手写笔记如下所示
 ![image](http://7xrluf.com1.z0.glb.clouddn.com/%E6%89%AB%E6%8F%8F_20160529215117.jpg)



