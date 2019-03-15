
> 本文大约 8000 字，阅读大约需要 12 分钟

> 第一次翻译，限于英语水平，可能不少地方翻译不准确，请见谅！

最近谷歌开源了一个基于 TensorFlow 的库--TFGAN，方便开发者快速上手 GAN 的训练，其 Github 地址如下：

https://github.com/tensorflow/models/tree/master/research/gan

原文网址：[Generative Adversarial Networks: Google open sources TensorFlow-GAN (TFGAN)](https://hub.packtpub.com/google-opensources-tensorflow-gan-tfgan-library-for-generative-adversarial-networks-neural-network-model/#)

---

如果你玩过波斯王子，那你应该知道你需要保护自己不被”影子“所杀掉，但这也是一个矛盾：如果你杀死“影子”，那游戏就结束了；但你不做任何事情，那么游戏也会输掉。

尽管生成对抗网络（GAN）有不少优点，但它也面临着相似的区分问题。大部分支持 GAN 的深度学习专业也是非常谨慎的支持它，并指出它确实存在稳定性的问题。

GAN 的这个问题也可以称做**整体收敛性问题**。尽管判别器 D 和 生成器 D 相互竞争博弈，但同时也相互依赖对方来达到有效的训练。如果其中一方训练得很差，那整个系统也会很差(这也是之前提到的梯度消失或者模式奔溃问题)。并且你也需要确保他们不会训练太过度，造成另一方无法训练了。因此，波斯王子是一个很有趣的概念。

首先，神经网络的提出就是为了模仿人类的大脑（尽管是人为的）。它们也已经在物体识别和自然语言处理方面取得成功。但是，想要在思考和行为上与人类一致，这还有非常大的差距。

那么是什么让 GANs 成为机器学习领域一个热门话题呢？因为它不仅只是一个相对新的结构，它更加是一个比之前其他模型都能更加准确的对真实数据建模，可以说是深度学习的一个革命性的变化。

最后，它是一个同时训练两个独立的网络的新模型，这两个网络分别是判别器和生成器。这样一个非监督神经网络却能比其他传统网络得到更好性能的结果。

但目前事实是我们对 GANs 的研究还只是非常浅层，仍然有着很多挑战需要解决。GANs 目前也存在不少问题，比如无法区分在某个位置应该有多少特定的物体，不能应用到 3D 物体，以及也不能理解真实世界的整体结构。当然现在有大量研究正在研究如何解决上述问题，新的模型也取得更好的性能。

而最近谷歌为了让 GANs 更容易实现，设计开发并开源了一个基于 TensorFlow 的轻量级库--**TFGAN**。

根据谷歌的介绍，TFGAN 提供了一个基础结构来减少训练一个 GAN 模型的难度，同时提供非常好测试的损失函数和评估标准，以及给出容易上手的例子[1]，这些例子强调了 TFGAN 的灵活性和易于表现的优点。

此外，还提供了一个教程[2]，包含一个高级的 API 可以快速使用自己的数据集训练一个模型。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/image_compress.png)

上图是展示了对抗损失在图像压缩[3]方面的效果。最上方第一行图片是来自 ImageNet[4] 数据集的图片，也是原始输入图片，中间第二行展示了采用传统损失函数训练得到的图像压缩神经网络的压缩和解压缩效果，最底下一行则是结合传统损失函数和对抗损失函数训练的网络的结果，可以看到尽管基于对抗损失的图片并不像原始图片，但是它比第二行的网络得到更加清晰和细节更好的图片。

TFGAN 既提供了几行代码就可以实现的简答函数来调用大部分 GAN 的使用例子，也是建立在包含复杂 GAN 设计的模式化方式。这就是说，我们可以采用自己需要的模块，比如损失函数、评估策略、特征以及训练等等，这些都是独立的模块。TFGAN 这样的设计方式其实就满足了不同使用者的需求，对于入门新手可以快速训练一个模型来看看效果，对于需要修改其中任何一个模块的使用者也能修改对应模块，而不会牵一发而动全身。

最重要的是，谷歌也保证了这个代码是经过测试的，不需要担心一般的 GAN 库造成的数字或者统计失误。

#### 开始使用

首先添加以下代码来导入 tensorflow 和 声明一个 TFGAN 的实例：

```
import tensorflow as tf
tfgan = tf.contrib.gan
```

#### 为何使用 TFGAN

- 采用良好测试并且很灵活的调用接口[5]实现快速训练生成器和判别器网络，此外，还可以混合 TFGAN、原生 TensorFlow以及其他自定义框架代码；
- 使用实现好的 GAN 的损失函数和惩罚策略[6] (比如 Wasserstein loss、梯度惩罚等)
- 训练阶段对 GAN 进行监控和可视化操作[7]，以及评估生成结果[8]
- 使用实现好的技巧来稳定和提高性能[9]
- 基于常规的 GAN 训练例子来开发
- 采用 GANEstimator[10] 接口里快速训练一个 GAN 模型
- TFGAN 的结构改进也会自动提升你的 TFGAN 项目的性能
- TFGAN 会不断添加最新研究的算法成果

#### TFGAN 的部件有哪些呢？

TFGAN 是由多个设计为独立的部件组成的，分别是：

- core[5]：提供了一个主要的训练 GAN 模型的结构。训练过程分为四个阶段，每个阶段都可以采用自定义代码或者 调用 TFGAN 库接口来完成；
- features[9]：包含许多常见的 GAN 运算和正则化技术，比如实例正则化(instance normalization)
- losses[11]：包含常见的 GAN 的损失函数和惩罚机制，比如 Wasserstein loss、梯度惩罚、相互信息惩罚等
- evaulation[12]：使用一个预训练好的 Inception 网络来利用`Inception Score`或者`Frechet Distance`评估标准来评估非条件生成模型。当然也支持利用自己训练的分类器或者其他方法对有条件生成模型的评估
- examples[1] and tutorial[2]：使用 TFGAN 训练 GAN 模型的例子和教程。包含了使用非条件和条件式的 GANs 模型，比如 InfoGANs 等。

#### 训练一个 GAN 模型

典型的 GAN 模型训练步骤如下：

1. 为你的网络指定输入，比如随机噪声，或者是输入图片（一般是应用在图片转换的应用，比如 pix2pixGAN 模型）
2. 采用`GANModel`接口定义生成器和判别器网络
3. 采用`GANLoss`指定使用的损失函数
4. 采用`GANTrainOps`设置训练运算操作，即优化器
5. 开始训练

当然，GAN 的设置有多种形式。比如，你可以在非条件下训练生成器生成图片，或者可以给定一些条件，比如类别标签等输入到生成器中来训练。无论是哪种设置，TFGAN 都有相应的实现。下面将结合代码例子来进一步介绍。

#### 实例

##### 非条件 MNIST 图片生成

第一个例子是训练一个生成器来生成手写数字图片，即 MNIST 数据集。生成器的输入是从多变量均匀分布采样得到的随机噪声，目标输出是 MNIST 的数字图片。具体查看论文“Generative Adversarial Networks”[17]。代码如下：


```
# 配置输入
# 真实数据来自 MNIST 数据集
images = mnist_data_provider.provide_data(FLAGS.batch_size)
# 生成器的输入，从多变量均匀分布采样得到的随机噪声
noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

# 调用 tfgan.gan_model() 函数定义生成器和判别器网络模型
gan_model = tfgan.gan_model(
    generator_fn=mnist.unconditional_generator,  
    discriminator_fn=mnist.unconditional_discriminator,  
    real_data=images,
    generator_inputs=noise)

# 调用 tfgan.gan_loss() 定义损失函数
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan_losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan_losses.wasserstein_discriminator_loss)

# 调用 tfgan.gan_train_ops() 指定生成器和判别器的优化器
train_ops = tfgan.gan_train_ops(
    gan_model,
    gan_loss,
    generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5))

# tfgan.gan_train() 开始训练，并指定训练迭代次数 num_steps
tfgan.gan_train(
    train_ops,
    hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps)],
    logdir=FLAGS.train_log_dir)
```

##### 条件式 MNIST 图片生成

第二个例子同样还是生成 MNIST 图片，但是这次输入到生成器的不仅仅是随机噪声，还会给类别标签，这种 GAN 模型也被称作条件 GAN，其目的也是为了让 GAN 训练不会太过自由。具体可以看论文“Conditional Generative Adversarial Nets”[13]。

代码方面，仅仅需要修改输入和建立生成器与判别器模型部分，如下所示：

```
# 配置输入
# 真实数据来自 MNIST 数据集，这里增加了类别标签--one_hot_labels
images, one_hot_labels = mnist_data_provider.provide_data(FLAGS.batch_size)
# 生成器的输入，从多变量均匀分布采样得到的随机噪声
noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

# 调用 tfgan.gan_model() 函数定义生成器和判别器网络模型
gan_model = tfgan.gan_model(
    generator_fn=mnist.conditional_generator,  
    discriminator_fn=mnist.conditional_discriminator,  
    real_data=images,
    generator_inputs=(noise, one_hot_labels)) # 生成器的输入增加了类别标签
    
# 剩余的代码保持一致
...
```

##### 对抗损失

第三个例子结合了 L1 pixel loss 和对抗损失来学习自动编码图片。瓶颈层可以用来传输图片的压缩表示。如果仅仅使用 pixel-wise loss，网络只回倾向于生成模糊的图片，但 GAN 可以用来让这个图片重建过程更加逼真。具体可以看论文“Full Resolution Image Compression with Recurrent Neural Networks”[3]来了解如何用 GAN 来实现图像压缩，以及论文“Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network”[14]了解如何用 GANs 来增强生成的图片质量。

代码如下：

```
# 配置输入
images = image_provider.provide_data(FLAGS.batch_size)

# 配置生成器和判别器网络
gan_model = tfgan.gan_model(
    generator_fn=nets.autoencoder,  # 自定义的 autoencoder
    discriminator_fn=nets.discriminator,  # 自定义的 discriminator
    real_data=images,
    generator_inputs=images)

# 建立 GAN loss 和 pixel loss
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan_losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan_losses.wasserstein_discriminator_loss,
    gradient_penalty=1.0)
l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data, ord=1)

# 结合两个 loss
gan_loss = tfgan.losses.combine_adversarial_loss(
    gan_loss, gan_model, l1_pixel_loss, weight_factor=FLAGS.weight_factor)

# 剩下代码保持一致
...
```

##### 图像转换

第四个例子是图像转换，它是将一个领域的图片转变成另一个领域的同样大小的图片。比如将语义分割图变成街景图，或者是灰度图变成彩色图。具体细节看论文“Image-to-Image Translation with Conditional Adversarial Networks”[15]。

代码如下：

```
# 配置输入，注意增加了 target_image
input_image, target_image = data_provider.provide_data(FLAGS.batch_size)

# 配置生成器和判别器网络
gan_model = tfgan.gan_model(
    generator_fn=nets.generator,  
    discriminator_fn=nets.discriminator,  
    real_data=target_image,
    generator_inputs=input_image)

#  建立 GAN loss 和 pixel loss
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan_losses.least_squares_generator_loss,
    discriminator_loss_fn=tfgan_losses.least_squares_discriminator_loss)
l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data, ord=1)

# 结合两个 loss
gan_loss = tfgan.losses.combine_adversarial_loss(
    gan_loss, gan_model, l1_pixel_loss, weight_factor=FLAGS.weight_factor)

# 剩下代码保持一致
...
```

##### InfoGAN

最后一个例子是采用 InfoGAN 模型来生成 MNIST 图片，但是可以不需要任何标签来控制生成的数字类型。具体细节可以看论文“InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets”[16]。

代码如下：

```
# 配置输入
images = mnist_data_provider.provide_data(FLAGS.batch_size)

# 配置生成器和判别器网络
gan_model = tfgan.infogan_model(
    generator_fn=mnist.infogan_generator,  
    discriminator_fn=mnist.infogran_discriminator,  
    real_data=images,
    unstructured_generator_inputs=unstructured_inputs,  # 自定义输入
    structured_generator_inputs=structured_inputs)  # 自定义

# 配置 GAN loss 以及相互信息惩罚
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan_losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan_losses.wasserstein_discriminator_loss,
    gradient_penalty=1.0,
    mutual_information_penalty_weight=1.0)

# 剩下代码保持一致
...
```

##### 自定义模型的创建

最后同样是非条件 GAN 生成 MNIST 图片，但利用`GANModel`函数来配置更多参数从而更加精确控制模型的创建。

代码如下：

```
# 配置输入
images = mnist_data_provider.provide_data(FLAGS.batch_size)
noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

# 手动定义生成器和判别器模型
with tf.variable_scope('Generator') as gen_scope:
  generated_images = generator_fn(noise)
with tf.variable_scope('Discriminator') as dis_scope:
  discriminator_gen_outputs = discriminator_fn(generated_images)
with variable_scope.variable_scope(dis_scope, reuse=True):
  discriminator_real_outputs = discriminator_fn(images)
generator_variables = variables_lib.get_trainable_variables(gen_scope)
discriminator_variables = variables_lib.get_trainable_variables(dis_scope)

# 依赖于你需要使用的 TFGAN 特征，你并不需要指定 `GANModel`函数的每个参数，不过
# 最少也需要指定判别器的输出和变量
gan_model = tfgan.GANModel(
    generator_inputs,
    generated_data,
    generator_variables,
    gen_scope,
    generator_fn,
    real_data,
    discriminator_real_outputs,
    discriminator_gen_outputs,
    discriminator_variables,
    dis_scope,
    discriminator_fn)

# 剩下代码和第一个例子一样
...
```

最后，再次给出 TFGAN 的 Github 地址如下：

https://github.com/tensorflow/models/tree/master/research/gan

---
文章链接：

1. TFGAN 例子：https://github.com/tensorflow/models/tree/master/research/gan
2. TFGAN 教程：https://github.com/tensorflow/models/tree/master/research/gan/tutorial.ipynb
3. Full Resolution Image Compression with Recurrent Neural Networks: https://arxiv.org/abs/1608.05148
4. ImageNet：http://www.image-net.org/
5. TFGAN 训练接口代码：https://www.tensorflow.org/code/tensorflow/contrib/gan/python/train.py
6. TFGAN loss接口代码：https://www.tensorflow.org/code/tensorflow/contrib/gan/python/losses/python/losses_impl.py
7. https://www.tensorflow.org/code/tensorflow/contrib/gan/python/eval/python/summaries_impl.py
8. https://www.tensorflow.org/code/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
9. Tricks：https://www.tensorflow.org/code/tensorflow/contrib/gan/python/features/python/
10. GANEstimator：https://www.tensorflow.org/code/tensorflow/contrib/gan/python/estimator/python/gan_estimator_impl.py
11. losses： https://www.tensorflow.org/code/tensorflow/contrib/gan/python/losses/python/
12. evaluation：https://www.tensorflow.org/code/tensorflow/contrib/gan/python/eval/python/
13. Conditional Generative Adversarial Nets：https://arxiv.org/abs/1411.1784
14. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network：https://arxiv.org/abs/1609.04802
15. Image-to-Image Translation with Conditional Adversarial Networks：https://arxiv.org/abs/1611.07004
16. InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets：https://arxiv.org/abs/1606.03657
17. Generative Adversarial Networks：https://arxiv.org/abs/1406.2661

---
如果有翻译不当的地方或者有任何建议和看法，可以点击左下角查看原文留言给出你对本文的建议和看法。

同时也欢迎关注我的微信公众号--机器学习与计算机视觉或者扫描下方的二维码，在后台留言，和我分享你的建议和看法，指正文章中可能存在的错误，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)

**推荐阅读**

1.[机器学习入门系列(1)--机器学习概览(上)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483667&idx=1&sn=c6b6feb241897ede16bd745d595cef92&chksm=fe3b0f66c94c86701e9b071e62750d189c254fd3ebe9bb6251505162139efefdf866093b38c3&token=2134085567&lang=zh_CN#rd)

2.[机器学习入门系列(2)--机器学习概览(下)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483672&idx=1&sn=34b6687030db92fd3e04dcdebd09fffc&chksm=fe3b0f6dc94c867b2a72c427ebb90e2a683e6ad97ea2c5fbdc3a3bb86a8b159b8e5f107d2dcc&token=2134085567&lang=zh_CN#rd)

3.[[GAN学习系列] 初识GAN](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483711&idx=1&sn=ead88d5b21e08d9df853b72f31d4b5f4&chksm=fe3b0f4ac94c865cfc243123eb4815539ef2d5babdc8346f79a29b681e55eee5f964bdc61d71&token=1493836032&lang=zh_CN#rd)

4.[[GAN学习系列2] GAN的起源](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483732&idx=1&sn=99cb91edf6fb6da3c7d62132c40b0f62&chksm=fe3b0f21c94c8637a8335998c3fc9d0adf1ac7dea332c2bd45e63707eac6acad8d84c1b3d16d&token=985117826&lang=zh_CN#rd)

如果你觉得我写得还不错，可以给我点个赞！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/0.jpg)

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/02.gif)


