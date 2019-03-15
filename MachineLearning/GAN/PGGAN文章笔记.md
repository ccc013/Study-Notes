主要参考[NVIDIA新作解读：用GAN生成前所未有的高清图像（附PyTorch复现）](http://www.paperweekly.site/papers/notes/175)。

这篇论文的贡献：

1. 训练方式

对于 G 和 D 的训练方式如下：

> 作者采用 progressive growing 的训练方式，先训一个小分辨率的图像生成，训好了之后再逐步过渡到更高分辨率的图像。然后稳定训练当前分辨率，再逐步过渡到下一个更高的分辨率。

2. 增加生成多样性

**增加生成样本的多样性有两种可行的方法：通过 loss 让网络自己调整、通过设计判别多样性的特征人为引导。**

WGAN 属于前者，它采用更好的分布距离的估计（Wasserstein distance）。模型收敛意味着生成的分布和真实分布一致，能够有多样性的保证。PG-GAN 则属于后者。

作者沿用 improved GAN 的思路，通过**人为地给 Discriminator 构造判别多样性的特征来引导 Generator 生成更多样的样本**。Discriminator 能探测到 mode collapse 是否产生了，一旦产生，Generator 的 loss 就会增大，通过优化 Generator 就会往远离 mode collapse 的方向走，而不是一头栽进坑里。

Improved GAN 引入了 minibatch discrimination 层，构造一个 minibatch 内的多样性衡量指标。它引入了新的参数。

而 PG-GAN 不引入新的参数，利用特征的标准差作为衡量标准。具体做法就是：

假设我们有 N 个样本的 feature maps，我们对**每个空间位置求标准差**，++用 numpy 的 std 函数来说就是沿着样本的维度求 std++。这样就得到一张新的 feature map（如果样本的 feature map 不止一个，那么这样构造得到的 feature map 数量应该是一致的），接着 feature map 求平均得到一个数。

这个过程简单来说就是求 mean std，**作者把这个数复制成一张 feature map 的大小，跟原来的 feature map 拼在一起送给 Discriminator。**

3. Normalization

从 DCGAN[3]开始，GAN 的网络使用 batch (or instance) normalization 几乎成为惯例。使用 batch norm 可以增加训练的稳定性，大大减少了中途崩掉的情况。作者采用了两种新的 normalization 方法，不引入新的参数。

第一种 normalization 方法叫 pixel norm，它是 local response normalization 的变种。Pixel norm 沿着 channel 维度做归一化，这样归一化的一个好处在于，feature map 的每个位置都具有单位长度。这个归一化策略与作者设计的 Generator 输出有较大关系，注意到 Generator 的输出层并没有 Tanh 或者 Sigmoid 激活函数

第二种 normalization 方法跟凯明大神的初始化方法（Delving deep into rectifiers: Surpassing human-level performance on imagenet classification）挂钩（即Xavier初始化方法）。He 的初始化方法能够确保网络初始化的时候，随机初始化的参数不会大幅度地改变输入信号的强度。

![image](http://cdn.wxsell.com/web/article/VfHUSDEN.jpg)

作者走得比这个要远一点，他不只是初始化的时候对参数做了调整，而是动态调整。**初始化采用标准高斯分布，但是每次迭代都会对 weights 按照上面的式子做归一化**。作者 argue 这样的归一化的好处在于它不用再担心参数的 scale 问题，起到均衡学习率的作用（euqalized learning rate）

4. 有针对性地给样本加噪声

**通过给真实样本加噪声能够起到均衡 Generator 和 Discriminator 的作用，起到缓解 mode collapse 的作用，这一点在 WGAN 的前传中就已经提到**（Towards principled methods for training generative adversarial networks）。尽管使用 LSGAN 会比原始的 GAN 更容易训练，然而它在 Discriminator 的输出接近 1 的适合，梯度就消失，不能给 Generator 起到引导作用。

针对 D 趋近 1 的这种特性，作者提出了下面这种添加噪声的方式：

![image](http://cdn.wxsell.com/web/article/aD5Os9wZ.jpg)

其中，![image](http://cdn.wxsell.com/web/article/djwR1yfg.jpg) 分别为第 t 次迭代判别器输出的修正值、第 t-1 次迭代真样本的判别器输出。


从式子可以看出，当真样本的判别器输出越接近 1 的时候，噪声强度就越大，而输出太小（<=0.5）的时候，不引入噪声，这是因为 0.5 是 LSGAN 收敛时，D 的合理输出（无法判断真假样本），而小于 0.5 意味着 D 的能力太弱。