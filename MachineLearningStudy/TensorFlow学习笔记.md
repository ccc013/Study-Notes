这里是学习使用`tensorflow`的笔记。



---
#### 知识点总结

1. [**关于padding操作中的两种方式`SAME`和`VALID`的区别**](http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t)。

---
##### 05-15

今天是成功安装了`tensorflow`，在Ubuntu上，并且是在`anaconda`中。

具体安装教程是参考了(anaconda + tensorflow +ubuntu)[http://www.th7.cn/Program/Python/201604/840421.shtml]。

安装TF的过程如下：

1. 命令行输入`conda create -n tensorflow python=2.7`,创建一个新环境。
2. 然后激活环境，`source activate tensorflow`.
3. 安装TF，**CPU版本**的是`pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl`.
而**GPU版本**的是`pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl`，这里的TF版本都是0.8，而且`cp27`是指明了用的是`python 2.7`版本。
4. 然后将anaconda文件夹里面的envs文件夹里的tensorflow文件夹里面找到site-packages里的文件都复制到属于python2.7的site-packages文件夹里面。我的路径分别是`/home/cai/anaconda3/envs/tensorflow/lib/python2.7/site-packages`和`/home/cai/anaconda3/envs/python2/lib/python2.7/site-packages`。

完成上述步骤后，使用jupyter notebook 或者spyder都能正常使用TF，而且由于该版本的TF可以只需要CUDA7.5版本，所以能跑GPU版本。


---
##### 05-16

首先是根据官网的[Mnist教程](https://www.tensorflow.org/get_started/mnist/pros)，简单熟悉了TF是如何实现CNN的，看起来非常类似opencv，需要自己定义好整个流程，运用它提供的函数，比如实现卷积运算，pooling，relu函数，矩阵相乘，还有就是梯度下降等优化方法，然后搭建起来。

接着就是找下如何实现alexNet，并且进行微调。首先是找到一个代码可以将`Caffe`跑好的模型转换为`Tensorflow`的模型--[caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)。

接着找到一个教程--[finetuning-alexnet-with-tensorflow](https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html)，这是一步步介绍如何实现代码，然后还有一篇文章，介绍了几种实现代码--[TensorFlow之深入理解AlexNet](http://hacker.duanshishi.com/?p=1661),以及一个博客--[小石头的码疯窝-ML DL CV](http://hacker.duanshishi.com/),写了不少有关TF的，有实现不同网络结构，包括alexnet，vgg等，还有跟GAN相关的文章也有。



