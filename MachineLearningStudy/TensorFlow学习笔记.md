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


---
##### 05-20

这几天一直想跑下微调的alexNet，但一开始出现变量已经存在或者不存在问题，这个应该是跟其定义有关系，如果是第一次定义，那么在用`variable_scope`的时候，就不用定义参数`reuse`，如果是第二次，就需要设置这个参数`reuse=True`，一开始我是将权值参数定义放在辅助函数`conv`里面，然后出错，就将其放到定义整个计算图，即网络结构前面，定义好所有层的权值和偏置值，但是还有错，因为会载入预训练好的模型，这里也说需要修改。

接着就是因为版本问题，一些函数名字改变了，比如有关记录训练内容的，`tf.summary`，至少0.8版本是没有`summary`这个模块的，应该是后面增加了，所以也在官网找到这些函数的原来名字，参考了(Tensorflow: 'module' object has no attribute 'scalar_summary')[http://stackoverflow.com/questions/41066244/tensorflow-module-object-has-no-attribute-scalar-summary],以及[官网](https://www.tensorflow.org/install/migration)，不过如果是新增加的模块或者函数，那就没办法了，但现在不打算更换CUDA版本，所以只能先这么使用着。

接下来就是输入问题，一开始是用`opencv`来读取图片，对图片进行尺寸调整，减去均值，还有就是做下水平翻转，但是可以`import cv2`，但是读取图片，想打印图片尺寸，会出现`'NoneType' has no attribute 'shape'`，发现是根本没有读取图片成功，这个问题真是弄了很久没搞好，然后就换个库，用`PIL`，这个倒可以成功，但是运行了一段时间，可能是训练了20来次，训练时间还是很久的，感觉比Caffe慢很多，然后就会出现减均值出错，说是尺寸不匹配，因为均值是自定义一个`np.array`，一个3维的，然后发现PIL读取的图片是二维的，所以当我将减均值去掉后，还是有新的错误，一开始是会定义一个`np.ndarray`的数组，是`[batch,227,227,3]`，即每个图片应该是[227,227,3],可是这种情况都会出现在训练了好几次才出错，还不知道如何修改这种问题，或许需要更换成`matplotlib`?或者看看怎么让PIL读取图片出来显示3维吧。

