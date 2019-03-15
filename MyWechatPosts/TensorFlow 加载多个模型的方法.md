
采用 TensorFlow 的时候，有时候我们需要加载的不止是一个模型，那么如何加载多个模型呢？

原文：https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/

---

> 关于 TensorFlow 可以有很多东西可以说。但这次我只介绍如何导入训练好的模型（图），因为我做不到导入第二个模型并将它和第一个模型一起使用。并且，这种导入非常慢，我也不想重复做第二次。另一方面，将一切东西都放到一个模型也不实际。

在这个教程中，我会介绍如何保存和载入模型，更进一步，如何加载多个模型。

### 加载 TensorFlow 模型

在介绍加载多个模型之前，我们先介绍下如何加载单个模型，官方文档：https://www.tensorflow.org/programmers_guide/meta_graph。

首先，我们需要创建一个模型，训练并保存它。这部分我不想过多介绍细节，只需要关注如何保存模型以及不要忘记给每个操作命名。

创建一个模型，训练并保存的代码如下：

```
import tensorflow as tf
### Linear Regression 线性回归###
# Input placeholders
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
# Model parameters 定义模型的权值参数
W1 = tf.Variable([0.1], tf.float32)
W2 = tf.Variable([0.1], tf.float32)
W3 = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.1], tf.float32)

# Output 模型的输出
linear_model = tf.identity(W1 * x + W2 * x**2 + W3 * x**3 + b,
                           name='activation_opt')

# Loss 定义损失函数
loss = tf.reduce_sum(tf.square(linear_model - y), name='loss')
# Optimizer and training step 定义优化器运算
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss, name='train_step')

# Remember output operation for later aplication
# Adding it to a collections for easy acces
# This is not required if you NAME your output operation
# 记得将输出操作添加到一个集合中，但如何你命名了输出操作，这一步可以省略
tf.add_to_collection("activation", linear_model)

## Start the session ##
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#  CREATE SAVER
saver = tf.train.Saver()

# Training loop 训练
for i in range(10000):
    sess.run(train, {x: data, y: expected})
    if i % 1000 == 0:
        # You can also save checkpoints using global_step variable
        saver.save(sess, "models/model_name", global_step=i)

# SAVE TensorFlow graph into path models/model_name
# 保存模型到指定路径并命名模型文件名字
saver.save(sess, "models/model_name")
```

注意，这里是第一个重点--**对变量和运算命名**。这是为了在加载模型后可以使用指定的一些权值参数，如果不命名的话，这些变量会自动命名为类似“Placeholder_1”的名字。在复杂点的模型中，使用领域(scopes)是一个很好的做法，但这里不做展开。

总之，==重点就是为了在加载模型的时候能够调用权值参数或者某些运算操作，你必须给他们命名或者是放到一个集合中。==

当保存模型后，在指定保存模型的文件夹中就应该包含这些文件：`model_name.index`、`model_name.meta`以及其他文件。如果是采用`checkpoints`后缀命名模型名字，还会有名字包含`model_name-1000`的文件，其中的数字是对应变量`global_step`，也就是当前训练迭代次数。

现在我们就可以开始加载模型了。加载模型其实很简单，我们需要的只是两个函数即可：`tf.train.import_meta_graph`和`saver.restore()`。此外，就是提供正确的模型保存路径位置。另外，如果我们希望在不同机器使用模型，那么还需要设置参数：`clear_device=True`。

接着，我们就可以通过之前命名的名字或者是保存到的集合名字来调用保存的运算或者是权值参数了。如果使用了领域，那么还需要包含领域的名字才行。而在实际调用这些运算的时候，还必须采用类似`{'PlaceholderName:0': data}`的输入占位符，否则会出现错误。

加载模型的代码如下：


```
sess = tf.Session()

# Import graph from the path and recover session
# 加载模型并恢复到会话中
saver = tf.train.import_meta_graph('models/model_name.meta', clear_devices=True)
saver.restore(sess, 'models/model_name')

# There are TWO options how to access the operation (choose one)
# 两种方法来调用指定的运算操作，选择其中一个都可以
  # FROM SAVED COLLECTION: 从保存的集合中调用
activation = tf.get_collection('activation')[0]
  # BY NAME: 采用命名的方式
activation = tf.get_default_graph.get_operation_by_name('activation_opt').outputs[0]

# Use imported graph for data
# You have to feed data as {'x:0': data}
# Don't forget on ':0' part!
# 采用加载的模型进行操作，不要忘记输入占位符
data = 50
result = sess.run(activation, {'x:0': data})
print(result)
```

### 多个模型

上述介绍了如何加载单个模型的操作，但如何加载多个模型呢？

如果使用加载单个模型的方式去加载多个模型，那么就会出现变量冲突的错误，也无法工作。这个问题的原因是因为一个默认图的缘故。冲突的发生是因为我们将所有变量都加载到当前会话采用的默认图中。当我们采用会话的时候，我们可以通过`tf.Session(graph=MyGraph)`来指定采用不同的已经创建好的图。因此，如果我们希望加载多个模型，那么我们需要做的就是把他们加载在不同的图，然后在不同会话中使用它们。

这里，自定义一个类来完成加载指定路径的模型到一个局部图的操作。这个类还提供`run`函数来对输入数据使用加载的模型进行操作。这个类对于我是有用的，因为我总是将模型输出放到一个集合或者对它命名为`activation_opt`，并且将输入占位符命名为`x`。你可以根据自己实际应用需求对这个类进行修改和拓展。

代码如下：

```
import tensorflow as tf

class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            # 从指定路径加载模型到局部图中
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
            # 两种方式来调用运算或者参数
              # FROM SAVED COLLECTION:            
            self.activation = tf.get_collection('activation')[0]
              # BY NAME:
            self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def run(self, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={"x:0": data})
      
      
### Using the class ###
# 测试样例
data = 50         # random data
model = ImportGraph('models/model_name')
result = model.run(data)
print(result)
```

### 总结

如果你理解了 TensorFlow 的机制的话，加载多个模型并不是一件困难的事情。上述的解决方法可能不是完美的，但是它简单且快速。最后给出总结整个过程的样例代码，这是在 Jupyter notebook 上的，代码地址如下：

https://gist.github.com/Breta01/f205a9d27090c18d394fbaab98de7c65#file-importmodulesnotebook-ipynb

---

最后，给出文章中几个代码例子的 github 地址：

1. Code for creating, training and saving TensorFlow model.：https://gist.github.com/Breta01/d61e526a6d1969085faa17eea8f02bb4#file-savingmodeltf-py
2. Importing and using TensorFlow graph (model)：https://gist.github.com/Breta01/c8b2df2b05b3ec54f1aa95e7a03a2907#file-importing-tf-model-py
3. Class for importing multiple TensorFlow graphs.：https://gist.github.com/Breta01/cabbb5c7d9bbd3d9b4ec404828ac24bb#file-multiple-tf-graph-class-py
4. Example of importing multiple TensorFlow modules：https://gist.github.com/Breta01/f205a9d27090c18d394fbaab98de7c65#file-importmodulesnotebook-ipynb



---
欢迎关注我的微信公众号--机器学习与计算机视觉或者扫描下方的二维码，在后台留言，和我分享你的建议和看法，指正文章中可能存在的错误，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)

**推荐阅读**

1.[机器学习入门系列(1)--机器学习概览(上)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483667&idx=1&sn=c6b6feb241897ede16bd745d595cef92&chksm=fe3b0f66c94c86701e9b071e62750d189c254fd3ebe9bb6251505162139efefdf866093b38c3&token=2134085567&lang=zh_CN#rd)

2.[机器学习入门系列(2)--机器学习概览(下)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483672&idx=1&sn=34b6687030db92fd3e04dcdebd09fffc&chksm=fe3b0f6dc94c867b2a72c427ebb90e2a683e6ad97ea2c5fbdc3a3bb86a8b159b8e5f107d2dcc&token=2134085567&lang=zh_CN#rd)

3.[[GAN学习系列] 初识GAN](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483711&idx=1&sn=ead88d5b21e08d9df853b72f31d4b5f4&chksm=fe3b0f4ac94c865cfc243123eb4815539ef2d5babdc8346f79a29b681e55eee5f964bdc61d71&token=1493836032&lang=zh_CN#rd)

4.[[GAN学习系列2] GAN的起源](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483732&idx=1&sn=99cb91edf6fb6da3c7d62132c40b0f62&chksm=fe3b0f21c94c8637a8335998c3fc9d0adf1ac7dea332c2bd45e63707eac6acad8d84c1b3d16d&token=985117826&lang=zh_CN#rd)

5.[谷歌开源的 GAN 库--TFGAN](https://mp.weixin.qq.com/s/Kd_nsit-JMaEjT5o8rEkKQ)

如果你觉得我写得还不错，可以给我点个赞！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/0.jpg)

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/02.gif)
