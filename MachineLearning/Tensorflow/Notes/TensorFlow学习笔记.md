这里是学习使用`tensorflow`的笔记。

---
#### 知识点总结

1. [**关于padding操作中的两种方式`SAME`和`VALID`的区别**](http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t)。
2. [详解 TensorBoard－如何调参](https://www.jianshu.com/p/d059ffea9ec0)

---
#### 2018

##### 09-21

windows下采用 tensorboard的时候，出现`ModuleNotFoundError: No module named 'tensorboard.main'`。

解决方法是安装`tensorflow-tensorboard`。

tensorboard的使用命令：

```
tensorboard --logdir==/path/to/log

例如：tensorboard --logdir==/home/lc/logs/
```

##### 09-25

关于多GPU的问题：

1. 一直打印变量信息的原因是定义tf.Session()的时候，设置`log_device_placement=True`导致的问题
2. 计算梯度均值遇到空的情况--在调用`compute_gradients()`方法是制定计算的参数
3. loss值比采用单GPU时候的大很多---没有对loss进行平均，输出的是4个GPU的loss求和,采用`tf.reduce_mean()`
4. G和D的loss都变成Nan---D的输出没有采用`sigmoid`函数，将其限制在(0,1)的范围
5. 训练速度并没有提升太多，主要是多GPU的调用还是通过一个for循环，如果循环内的处理太耗时间，总体时间还是需要比较久。

参考实现多gpu文章：

- [TensorFlow多GPU并行](http://hongbomin.com/2017/06/27/tensorflow-mnist-multi-gpu/)
- [Tensorflow GAN discriminator loss NaN since negativ discriminator output](https://stackoverflow.com/questions/45506455/tensorflow-gan-discriminator-loss-nan-since-negativ-discriminator-output)
- [cifar10_multi_gpu_train](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py)
- [inception_train](https://github.com/tensorflow/models/blob/master/research/inception/inception/inception_train.py)

##### 09-26

问题:每次训练到一定次数，在保存模型代码时候会出现下列错误：
```
libprotobuf FATAL google/protobuf/wire_format.cc:830] CHECK failed: (output->ByteCount()) == (expected_endpoint): : Protocol message serialized to a size different from what was originally expected.  Perhaps it was modified by another thread during serialization?
terminate called after throwing an instance of 'google::protobuf::FatalException'
  what():  CHECK failed: (output->ByteCount()) == (expected_endpoint): : Protocol message serialized to a size different from what was originally expected.  Perhaps it was modified by another thread during serialization?
Command terminated by signal 6
```
根据[Libprotobuf error causes crash in the middle of training](https://discuss.pytorch.org/t/libprotobuf-error-causes-crash-in-the-middle-of-training/7691)这篇文章的说法，可能是由于tensorboard的问题，刚好在我的代码中保存模型，接着就是将图片写入summary中，而且在固定次数出问题的时候，保存的模型大小已经达到2G，也可能是protobuf对于处理这种大于2G大小的模型时候会出现这个问题。

[Saver producing large meta files after long run #2654](https://github.com/tensorflow/tensorflow/issues/2654) 关于meta文件越来越大的问题，meta文件主要是保存训练的图Graph，即网络结构。

最终发现问题所在是采用`tf.summary.image()`保存图片的操作，这里设置了每相隔k次迭代就会保存一次图片，每次保存图片的命名也是按照当前epoch和iterations来命名，相当于每次都新加入一个summary，这个操作让网络结构不断增大，也就是meta文件变大。

因此，问题来了，`tf.summary.image()`操作如何才能保存不同训练阶段的图片呢，如果是在训练前定义好，就每次都会覆盖最新图片。

参考文章：

- [进一步理解 TensorFlow 的 Graph 机制](https://zhuanlan.zhihu.com/p/32315232)
- [tensorflow sess.run()越来越慢的原因分析及其解决方法](https://zhuanlan.zhihu.com/p/31619020)
- [tensorflow: assigning weights after finalizing graph](https://stackoverflow.com/questions/50895128/tensorflow-assigning-weights-after-finalizing-graph)


---
#### 2017

##### 05-15

今天是成功安装了`tensorflow`，在Ubuntu上，并且是在`anaconda`中。

具体安装教程是参考了[anaconda + tensorflow +ubuntu](http://www.th7.cn/Program/Python/201604/840421.shtml)。

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

接着就是因为版本问题，一些函数名字改变了，比如有关记录训练内容的，`tf.summary`，至少0.8版本是没有`summary`这个模块的，应该是后面增加了，所以也在官网找到这些函数的原来名字，参考了[Tensorflow: 'module' object has no attribute 'scalar_summary'](http://stackoverflow.com/questions/41066244/tensorflow-module-object-has-no-attribute-scalar-summary),以及[官网](https://www.tensorflow.org/install/migration)，不过如果是新增加的模块或者函数，那就没办法了，但现在不打算更换CUDA版本，所以只能先这么使用着。

接下来就是输入问题，一开始是用`opencv`来读取图片，对图片进行尺寸调整，减去均值，还有就是做下水平翻转，但是可以`import cv2`，但是读取图片，想打印图片尺寸，会出现`'NoneType' has no attribute 'shape'`，发现是根本没有读取图片成功，这个问题真是弄了很久没搞好，然后就换个库，用`PIL`，这个倒可以成功，但是运行了一段时间，可能是训练了20来次，训练时间还是很久的，感觉比Caffe慢很多，然后就会出现减均值出错，说是尺寸不匹配，因为均值是自定义一个`np.array`，一个3维的，然后发现PIL读取的图片是二维的，所以当我将减均值去掉后，还是有新的错误，一开始是会定义一个`np.ndarray`的数组，是`[batch,227,227,3]`，即每个图片应该是[227,227,3],可是这种情况都会出现在训练了好几次才出错，还不知道如何修改这种问题，或许需要更换成`matplotlib`?或者看看怎么让PIL读取图片出来显示3维吧。


---
##### 06-27

现在是实现了将输入图片转换成TFRecord格式，并且成功读取和解码成图片，通过TF的队列来读取，并用`tf.train.batch`生成指定`batch_size`大小的`image_batch`和`label_batch`，这样方便训练时候调用。

不过现在遇到一个问题，训练了一定次数后，会出现权值`NAN`的情况，因为用`tf.summary`记录的是不微调的网络层的参数，一开始学习率是`0.001`，降低到`0.0001`，情况还是一样，现在尝试提高到`0.01`。

提高学习率或者降低还是会有参数爆炸的问题，即使关掉记录参数的操作，后面也会出现梯度爆炸的情况，学习率改变还是出现问题，根据搜索的结果，可能还是输入数据的问题，出现有一个batch为空的情况，或者数据处理出现问题了。

现在换一种方式输入数据，不转换成TFRecords格式，用tf来读取数据，然后输入，读取数据使用CPU来处理。但发现代码中要添加的库是在1.2.0版本中的。

---
##### 07-01

今天是完成了在windows上安装的过程。主要参考这篇文章--[在Windows上安装GPU版Tensorflow](http://www.jianshu.com/p/1fad663dabc3)，不过我下载的是1.2.0版本，算是最新版本，当然最新的是周五发布的1.2.1版本。

首先是下载了cuda8.0和cudnn6.0，但是出现了问题，问题如[ImportError: No module named '_pywrap_tensorflow' (MSVCP140.dll is present) #7705](https://github.com/tensorflow/tensorflow/issues/7705),然后我按照其中一种做法，就是换成cudnn5.1版本，就可以成功使用了。


---
##### 07-05

昨天是修改了下数据保存文件的路径，修改了代码，改下函数接口，因为版本提升了，然后开始跑，早上看下日志，发现出错了，在第一次测试完成后，出现`iterator`相关的错误，搜索下错误，居然没有搜索结果，将测试时间从1000变成100，还是出错，有个猜测，因为测试是可以完成的，是测试完后，不能进行训练，即数据出现问题，可能因为训练集和验证集都是用同个`iterator`来进行取一个batch数据的操作，所以感觉测试完成后需要重新初始化训练集的`iterator`。

不过笔记本跑是比较慢，训练20次需要1分钟到1分钟15秒，测试一次需要6分钟左右。

看了下`Tensorboard`的内容，这次就正常了，可以看到记录的各种数据，准确率的变化，loss，还有可以记录参数及其梯度值变化，然后其实还可以记录图片数据，还可以看整个网络结构的设置，功能还是很强大。


---
##### 07-20

发现在`finetuing`时候，如果载入是`ckpt`类型的模型，可以选择不微调的层，但是训练过程保存的时候是需要保存所有层的，所以这个时候应该定义两个`tf.train.Saver`变量，保存模型的变量是：

```
# Create a saver.
saver = tf.train.Saver(tf.global_variables())
```

而载入模型的变量写法是:


```
train
restorer = tf.train.Saver([v for v in tf.trainable_variables() if v.name.split('/')[0] not in train_layers])
```

---
##### 07-26

今天，增加了滑动平均模型后，发现loss下降更多，训练集准确率提高了，增加的代码如下：


```
global_step = tf.Variable(0, trainable=False)
# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

variable_averages = tf.train.ExponentialMovingAverage(
            0.999, global_step)
# Another possibility is to use tf.slim.get_variables().
variables_to_average = (tf.trainable_variables() +
                        tf.moving_average_variables())
variables_averages_op = variable_averages.apply(variables_to_average)

with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')
```

根据《Tensorflow 实战Google深度学习框架》中介绍，这个滑动平均模型可以增强模型的鲁棒性，在测试集上性能会更好。




