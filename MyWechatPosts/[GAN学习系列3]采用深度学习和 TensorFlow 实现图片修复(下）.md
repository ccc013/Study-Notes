
这是本文的最后一部分内容了，前两部分内容的文章：

1. [[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(上）](https://mp.weixin.qq.com/s/S_uiSe74Ti6N_u4Y5Fd6Fw)
2. [[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(中）](https://mp.weixin.qq.com/s/nYDZA75JcfsADYyNdXjmJQ)

以及原文的地址：

http://bamos.github.io/2016/08/09/deep-completion/


最后一部分的目录如下：

- 第三步：为图像修复寻找最佳的假图片
    - 利用 DCGANs 实现图像修复
    - [ML-Heavy] 损失函数
    - [ML-Heavy] TensorFlow 实现 DCGANs 模型来实现图像修复
    - 修复你的图片

---
### 第三步：为图像修复寻找最佳的假图片

#### 利用 DCGANs 实现图像修复

在第二步中，我们定义并训练了判别器`D(x)`和生成器`G(z)`，那接下来就是如何利用`DCGAN`网络模型来完成图片的修复工作了。

在这部分，作者会参考论文"Semantic Image Inpainting with Perceptual and Contextual Losses"[1] 提出的方法。

对于部分图片`y`，对于缺失的像素部分采用最大化`D(y)`这种看起来合理的做法并不成功，它会导致生成一些既不属于真实数据分布，也属于生成数据分布的像素值。如下图所示，我们需要一种合理的将`y`映射到生成数据分布上。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/inpainting-projection.png)

#### [ML-Heavy] 损失函数

首先我们先定义几个符号来用于图像修复。用`M`表示一个二值的掩码(Mask)，即只有 0 或者是 1 的数值。其中 1 数值表示图片中要保留的部分，而 0 表示图片中需要修复的区域。定义好这个 Mask 后，接下来就是定义如何通过给定一个 Mask 来修复一张图片`y`，具体的方法就是让`y`和`M`的像素对应相乘，这种两个矩阵对应像素的方法叫做**哈大马乘积**[2]，并且表示为 `M ⊙ y ` ，它们的乘积结果会得到图片中原始部分，如下图所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/mask-example.png)

接下来，假设我们从生成器`G`的生成结果找到一张图片，如下图公式所示，第二项表示的是`DCGAN`生成的修复部分：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/math_1.png)

根据上述公式，我们知道最重要的就是第二项生成部分，也就是需要实现很好修复图片缺失区域的做法。为了实现这个目的，这就需要回顾在第一步提出的两个重要的信息，上下文和感知信息。而这两个信息的获取主要是通过损失函数来实现。损失函数越小，表示生成的`G(z)`越适合待修复的区域。

##### Contextual Loss

为了保证输入图片相同的上下文信息，需要让输入图片`y`（可以理解为标签）中已知的像素和对应在`G(z)`中的像素尽可能相似，因此需要对产生不相似像素的`G(z)`做出惩罚。该损失函数如下所示，采用的是 L1 正则化方法：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/math_2.png)

这里还可以选择采用 L2 正则化方法，但论文中通过实验证明了 L1 正则化的效果更好。

理想的情况是`y`和`G(z)`的所有像素值都是相同的，也就是说它们是完全相同的图片，这也就让上述损失函数值为0

##### Perceptual Loss

为了让修复后的图片看起来非常逼真，我们需要让判别器`D`具备正确分辨出真实图片的能力。对应的损失函数如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/math_3.png)

因此，最终的损失函数如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/math_4.png)

这里 λ 是一个超参数，用于控制两个函数的各自重要性。

另外，论文还采用泊松混合(poisson blending)[3] 方法来平滑重构后的图片。

#### [ML-Heavy] TensorFlow 实现 DCGANs 模型来实现图像修复

代码实现的项目地址如下：

https://github.com/bamos/dcgan-completion.tensorflow

首先需要新添加的变量是表示用于修复的 mask，如下所示，其大小和输入图片一样

```
self.mask = tf.placeholder(tf.float32, [None] + self.image_shape, name='mask')
```
对于最小化损失函数的方法是采用常用的梯度下降方法，而在 TensorFlow 中已经实现了自动微分[4]的方法，因此只需要添加待实现的损失函数代码即可。添加的代码如下所示：

```
self.contextual_loss = tf.reduce_sum(
    tf.contrib.layers.flatten(
        tf.abs(tf.mul(self.mask, self.G) - tf.mul(self.mask, self.images))), 1)
self.perceptual_loss = self.g_loss
self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)
```
接着，就是定义一个 mask。这里作者实现的是位置在图片中心部分的 mask，可以根据需求来添加需要的任意随机位置的 mask，实际上代码中实现了多种 mask

```
if config.maskType == 'center':
    scale = 0.25
    assert(scale <= 0.5)
    mask = np.ones(self.image_shape)
    l = int(self.image_size*scale)
    u = int(self.image_size*(1.0-scale))
    mask[l:u, l:u, :] = 0.0
```
因为采用梯度下降，所以采用一个 mini-batch 的带有动量的映射梯度下降方法，将`z`映射到`[-1,1]`的范围。代码如下：

```
for idx in xrange(0, batch_idxs):
    batch_images = ...
    batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)
    zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

    v = 0
    for i in xrange(config.nIter):
        fd = {
            self.z: zhats,
            self.mask: batch_mask,
            self.images: batch_images,
        }
        run = [self.complete_loss, self.grad_complete_loss, self.G]
        loss, g, G_imgs = self.sess.run(run, feed_dict=fd)
        # 映射梯度下降方法
        v_prev = np.copy(v)
        v = config.momentum*v - config.lr*g[0]
        zhats += -config.momentum * v_prev + (1+config.momentum)*v
        zhats = np.clip(zhats, -1, 1)
```

#### 修复你的图片

选择需要进行修复的图片，并放在文件夹`dcgan-completion.tensorflow/your-test-data/raw`下面，然后根据之前第二步的做法来对人脸图片进行对齐操作，然后将操作后的图片放到文件夹`dcgan-completion.tensorflow/your-test-data/aligned`。作者随机从数据集`LFW`中挑选图片进行测试，并且保证其`DCGAN`模型的训练集没有包含`LFW`中的人脸图片。

接着可以运行下列命令来进行修复工作了：

```
./complete.py ./data/your-test-data/aligned/* --outDir outputImages
```

上面的代码会将修复图片结果保存在`--outDir`参数设置的输出文件夹下，接着可以采用`ImageMagick`工具来生成动图。这里因为动图太大，就只展示修复后的结果图片：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/completion.png)

而原始的输入待修复图片如下：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/missing_faces.png)

---
### 小结

最后，再给出前两步的文章链接：

1. [[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(上）](https://mp.weixin.qq.com/s/S_uiSe74Ti6N_u4Y5Fd6Fw)
2. [[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(中）](https://mp.weixin.qq.com/s/nYDZA75JcfsADYyNdXjmJQ)

当然这个图片修复方法由于也是2016年提出的方法了，所以效果不算特别好，这两年其实已经新出了好多篇新的图片修复方法的论文，比如：

1. 2016CVPR Context encoders: Feature learning by inpainting

https://arxiv.org/abs/1604.07379
 
2. Deepfill 2018--Generative Image Inpainting with Contextual Attention

https://arxiv.org/abs/1801.07892

3. Deepfillv2--Free-Form Image Inpainting with Gated Convolution

https://arxiv.org/abs/1806.03589

4.2017CVPR--High-resolution image inpainting using multi-scale neural patch synthesis

https://arxiv.org/abs/1611.09969

5. 2018年的 NIPrus收录论文--Image Inpainting via Generative Multi-column Convolutional Neural Networks

https://arxiv.org/abs/1810.08771



---
文中的链接：
1. https://arxiv.org/abs/1607.07539
2. https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
3. http://dl.acm.org/citation.cfm?id=882269
4. https://en.wikipedia.org/wiki/Automatic_differentiation


---
欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，在后台留言，和我分享你的建议和看法，指正文章中可能存在的错误，大家一起交流，学习和进步！

如果你觉得我写得不错，欢迎给我点个**好看**！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)

由于暂时没有留言功能，也可以到我的个人博客和 CSDN 博客进行留言：

http://ccc013.github.io/

https://blog.csdn.net/lc013/article/details/84845439

---
**往期精彩推荐**

1.[机器学习入门系列(1)--机器学习概览(上)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483667&idx=1&sn=c6b6feb241897ede16bd745d595cef92&chksm=fe3b0f66c94c86701e9b071e62750d189c254fd3ebe9bb6251505162139efefdf866093b38c3&token=2134085567&lang=zh_CN#rd)

2.[机器学习入门系列(2)--机器学习概览(下)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483672&idx=1&sn=34b6687030db92fd3e04dcdebd09fffc&chksm=fe3b0f6dc94c867b2a72c427ebb90e2a683e6ad97ea2c5fbdc3a3bb86a8b159b8e5f107d2dcc&token=2134085567&lang=zh_CN#rd)

3.[[GAN学习系列] 初识GAN](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483711&idx=1&sn=ead88d5b21e08d9df853b72f31d4b5f4&chksm=fe3b0f4ac94c865cfc243123eb4815539ef2d5babdc8346f79a29b681e55eee5f964bdc61d71&token=1493836032&lang=zh_CN#rd)

4.[[GAN学习系列2] GAN的起源](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483732&idx=1&sn=99cb91edf6fb6da3c7d62132c40b0f62&chksm=fe3b0f21c94c8637a8335998c3fc9d0adf1ac7dea332c2bd45e63707eac6acad8d84c1b3d16d&token=985117826&lang=zh_CN#rd)

5.[[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(上）](https://mp.weixin.qq.com/s/S_uiSe74Ti6N_u4Y5Fd6Fw)
