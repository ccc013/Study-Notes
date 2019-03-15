# 论文阅读(5)--Mining Discriminative Triplets of Patches for Fine-Grained Classification

------
来自马里兰大学的Yaming Wang等人提出了挖掘**mid-level patch** 的一种方法，文章中提出了基于**mid-level representation**的特征构建方法并结合`SVM`分类器进行分类，提出了一种**基于顺序与形状条件限制的Triplet Mining方法。**

下面展示了整个方法的流程示意图：

![](http://img.blog.csdn.net/20161014194019172)

论文中的方法首先是只需要物体的`bounding box`的标注以及类的标签，它是一种基于`patch`的方法。在仅使用`bounding box`的标注下，论文为了更精确的定位这些有区分性的`patches`，使用带有几何限制的`triplets of patches`，这里的`triplets`是包含了三个外观描述符以及两个简单但有效的几何限制。

此外，如何在所有可能的`triplets of patches`中发现非常有区分性的`triplets`也是一个必须解决的问题，论文与以往工作不同，是采用所有训练集或者一大部分训练集中所有相似的一组图片来寻找并通过测量它们的区分度来选择更好的`triplets`。

下面会介绍下关键的`triplets of patches`。

#### 1. Triplets of Patches with Geometric Constraints 

首先这里的`triplets`与两种几何限制如下图所示：

![](http://img.blog.csdn.net/20161014195845758)

上图中，`A,B,C`可以被看做是`patch`的外观模型(`Appearance model`)，然后左半图表示**顺序限制**，右半图表示**形状限制**。

这里的顺序限制分为顺时针，还是逆时针，并且可以用一个公式表示--$G_{ABC}=1$表示顺时针排列，也就是上述最左侧的三个点就是顺时针，左边第二个就是逆时针，即$G_{ABC}=-1$。

而在形状限制中主要使用到三个点，两两之间形成的夹角$\theta_A,\theta_B,\theta_C$ 。












