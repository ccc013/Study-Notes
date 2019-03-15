
TF 对实现卷积层有两个方法，即两种接口，分别是`tf.nn.conv2d ` 和` tf.layers.conv2d`。

查阅了官方文档和多篇文章，可以说得到的结论是：
- 两者实现的效果应该是一样的，只不过`tf.layers.conv2d`是一个更高层的封装，使用`tf.nn.convolution`作为后端
- `tf.layers.conv2d`参数更加丰富，可以用于**从头训练一个模型**， 其参数 filters 是一个整数，表示卷积核的数量
- `tf.nn.conv2d`一般是**加载预训练模型**使用，并且需要先定义 filter 和 bias 两个变量，且 filter 的 shape 是`[filter_height, filter_width, in_channels, out_channels]`


参考文章：

- [TensorFlow中的两种conv2d方法和kernel_initializer](https://www.cnblogs.com/wxshi/p/8734715.html)
- [tf-nn-conv2d-vs-tf-layers-conv2d](https://stackoverflow.com/questions/42785026/tf-nn-conv2d-vs-tf-layers-conv2d)
- [tensorflow学习：tf.nn.conv2d 和 tf.layers.conv2d](https://blog.csdn.net/wanglitao588/article/details/77162351)
- [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)
- [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
- [【Tensorflow】tf.nn.atrous_conv2d如何实现空洞卷积？](https://blog.csdn.net/mao_xiao_feng/article/details/77924003)

#### tf.nn.conv2d

`tf.nn.conv2d`函数的定义如下所示：
```
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```
使用的例子如下：

```
kernels = (3, 3)
channels = 64
filters = tf.get_variable("kernel", shape=[kernel[0], kernel[1], channels, x.get_shape()[-1]])
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME", name='conv')
```


作用就是对给定的 4 维输入`input`和卷积核`filter`实现二维卷积操作，函数中每个参数的定义如下：
- input: 输入数据。`Tensor`类型，维度是 4 维，数据类型必须是`half`、`bfloat16`、`float32`和`float64`中的一种，根据默认格式，四个维度分别表示`[batch, height, width, channels]`
- filter：卷积核，也是四维的`Tensor`类型，分别是`[filter_height, filter_width, in_channels, out_channels]`
- strides：步长值，是一个包含`int`类型的列表`list`，如上述例子所示，即`[1,2,2,1]`，分别表示每个维度需要做的步长，一般就是输入数据的`height`和`width`两个维度，也就是`[1,strides, strides, 1]`，这里其实步长值是`strides=2`
- padding：字符串类型的变量，表示填充的方式，只能是`SAME`、`VALID`两种之一，它们的计算方式如下图所示
![tensorflow_padding.png](WEBRESOURCE3218efd96eac82f683cbb00ce08fb626)

也就是当使用`VALID`的时候，如果卷积计算过程中，剩下的不够一步，则剩下的像素会被抛弃，`SAME`则会补0

- use_cudnn_on_gpu：可选的`bool`类型，默认是`True`，这是选择是否在 GPU 上使用 cudnn，cudnn 的作用就是计算加速。
- data_format：可选字符串类型，分别是`NHWC`和`NCHW`，默认是`NHWC`，指定输入和输出数据的格式，对于默认格式，输入和输出数据格式就是`[batch, height, width, channels]`，否则，对于`NCHW`格式，就是`[batch, channels, height, width]`
- dilations：可选的包含`int`数据的列表，默认是`[1,1,1,1]`，一个长度为 4 的一维`Tensor`，对应`input`的每个维度，这个参数主要是实现空洞卷积，也就是如果设置数值`k>1`，那么每个卷积核的元素之间将添加`k-1`个 0 值，其实就是扩大了卷积核的大小，比如卷积核大小是`3*3`，设置`dilations=(1,2,2,1)`，那么卷积核会扩大到`5*5`的大小。具体可以参考[【Tensorflow】tf.nn.atrous_conv2d如何实现空洞卷积？](https://blog.csdn.net/mao_xiao_feng/article/details/77924003)
- name：操作的名字，虽然是可选，但一般都需设置，主要是 Tensorflow 会有对变量是否复用，即`reuse`，如果不设置名字可能会出错。



#### tf.layers.conv2d

定义如下：

```
tf.layers.conv2d(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
```
基本每个参数用法和`tf.nn.conv2d`是一样的，不过区别也是有的：

1. 卷积核的输入不需要自定义一个变量输入，而是通过两个参数输入，`filters`表示卷积核的数量，即`out_channels`，而`kernel_size`是一个整数或者是一个`tuple/list`类型的包含两个整数的，比如`[3,3]`或者`(3,3)`或者`3`，都表示创建大小是`3*3`的卷积核
2. `activation`可以选择想采用的激活函数名称，也就是不仅仅是卷积，还完成了非线性转换的操作
3. `use_bias`是一个布尔类型，默认是`True`，表示是否用偏置值；
4. 可以定义权重`weight`和偏置`bias`的初始化方式--`kernel_initializer`和`bias_initializer`
5. 还有权重和偏执的正则化方式，改层参数是否可训练等

简单说，这个方法的参数更加丰富，两者的调用方式如下：


```
tf.layers.conv2d-> tf.nn.convolution

tf.layers.conv2d->Conv2D->Conv2D.apply()->_Conv->_Conv.apply()->_Layer.apply()->_Layer.\__call__()->_Conv.call()->nn.convolution()...
```


