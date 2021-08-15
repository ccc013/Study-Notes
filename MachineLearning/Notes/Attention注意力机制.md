# Attention注意力机制

参考文章：

- [浅谈Attention机制的理解](https://zhuanlan.zhihu.com/p/35571412)
- [一文看懂 Attention（本质原理+3大优点+5大类型）](https://easyaitech.medium.com/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82-attention-%E6%9C%AC%E8%B4%A8%E5%8E%9F%E7%90%86-3%E5%A4%A7%E4%BC%98%E7%82%B9-5%E5%A4%A7%E7%B1%BB%E5%9E%8B-e4fbe4b6d030)
- [深度学习中的注意力机制](https://blog.csdn.net/tg229dvt5i93mxaq5a6u/article/details/78422216)





------

## 1. 起源

Attention 机制最早是在计算机视觉里应用的，随后在 [NLP](https://easyai.tech/ai-definition/nlp/) 领域也开始应用了，真正发扬光大是在 NLP 领域，因为 2018 年 [BERT](https://easyai.tech/ai-definition/bert/) 和 GPT 的效果出奇的好，进而走红。而 [Transformer](https://easyai.tech/ai-definition/transformer/) 和 Attention 这些核心开始被大家重点关注。

如果用图来表达 Attention 的位置大致是下面的样子：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/attention_1.png" style="zoom:50%;" />





## 2. 本质

attention机制的本质是从人类视觉注意力机制中获得灵感(可以说很‘以人为本’了)。大致是我们视觉在感知东西的时候，**一般不会是一个场景从到头看到尾每次全部都看，而往往是根据需求观察注意特定的一部分**。而且当我们发现一个场景经常在某部分出现自己想观察的东西时，我们就会进行学习在将来再出现类似场景时把注意力放到该部分上。这可以说就是注意力机制的本质内容了。

也就是说注意力机制的核心逻辑就是「**从关注全部到关注重点**」。

Attention机制的实质其实是一个寻址（addressing）的过程，它可以缓解神经网络模型复杂度：不需要将所有的 N 个输入信息都输入到神经网络进行计算，只需要从 X 中选择一些和任务相关的信息输入给神经网络。





## 3. 原理

在 Encoder-Decoder 框架中常用到 attention，所以介绍 attention 原理的时候也会涉及到 Encoder-Decoder。



不过 attention 是可以脱离 Encoder-Decoder 框架的，下面的图片则是脱离 Encoder-Decoder 框架后的原理图解。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/attention_2.png" alt="img" style="zoom:50%;" />

上面的图看起来比较抽象，下面用一个例子来解释 attention 的原理：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/attention_3.png" alt="img" style="zoom:50%;" />

图书管（source）里有很多书（value），为了方便查找，我们给书做了编号（key）。当我们想要了解漫威（query）的时候，我们就可以看看那些动漫、电影、甚至二战（美国队长）相关的书籍。

为了提高效率，并不是所有的书都会仔细看，针对漫威来说，动漫，电影相关的会看的仔细一些（权重高），但是二战的就只需要简单扫一下即可（权重低）。

当我们全部看完后就对漫威有一个全面的了解了。



**Attention 原理的3步分解：**

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/attention_4.png" alt="img" style="zoom:50%;" />

第一步-**信息输入**： query 和 key 进行相似度计算，得到权值

第二步-**注意力分布计算**：将权值进行归一化，得到直接可用的权重：
$$
\alpha_i = softmax(s(key_i,q))=softmax(s(X_i, q))
$$
这里称 $\alpha_i$ 为注意力分布（概率分布），其中 $S(X_i, q)$ 是注意力打分机制，有几种计算方法：

- 加性模型：$s(x_i,q) = v^Ttanh(Wx_i+Uq)$
- 点积模型：$s(x_i, q)=x_i^Tq$
- 缩放点积模型：$s(x_i,q)=\frac{x_i^Tq}{d}$
- 双线性模型：$s(x_i,q)=x_i^TWq$



第三步-**信息加权平均**：将权重和 value 进行加权求和，注意力分布 $\alpha_i$ 可以解释为在上下文查询 q 的时候，第 i 个信息受关注的程度，采用一种“软性”的信息选择机制对输入信息 X 进行编码为：
$$
att(q, X)=\sum_{i=1}^N\alpha_iX_i
$$
这种编码方式为软性注意力机制（soft Attention），一般有两种：普通模式(Key=Value=X)，和键值对模型(Key!=Value)

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/soft_attention_fig.jpeg" style="zoom:67%;" />

> 从上面的建模，我们可以大致感受到 Attention 的思路简单，**四个字“带权求和”就可以高度概括**，大道至简。做个不太恰当的类比，人类学习一门新语言基本经历四个阶段：死记硬背（通过阅读背诵学习语法练习语感）->提纲挈领（简单对话靠听懂句子中的关键词汇准确理解核心意思）->融会贯通（复杂对话懂得上下文指代、语言背后的联系，具备了举一反三的学习能力）->登峰造极（沉浸地大量练习）。
>
> 这也如同attention的发展脉络，RNN 时代是死记硬背的时期，attention 的模型学会了提纲挈领，进化到 [*transformer*](https://easyai.tech/ai-definition/transformer/)，融汇贯通，具备优秀的表达学习能力，再到 GPT、BERT，通过多任务大规模学习积累实战经验，战斗力爆棚。
>
> 要回答为什么 attention 这么优秀？是因为它让模型开窍了，懂得了提纲挈领，学会了融会贯通。
>
> — — 阿里技术

想要了解更多技术细节，可以看看下面的文章或者视频：

「文章」[深度学习中的注意力机制](https://blog.csdn.net/tg229dvt5i93mxaq5a6u/article/details/78422216)

「文章」[遍地开花的 Attention，你真的懂吗？](https://zhuanlan.zhihu.com/p/77307258)

「文章」[探索 NLP 中的 Attention 注意力机制及 Transformer 详解](https://www.infoq.cn/article/lteUOi30R4uEyy740Ht2)

「视频」[李宏毅 — transformer](https://www.bilibili.com/video/av56239558?from=search&seid=14406218127146760248)

「视频」[李宏毅 — ELMO、BERT、GPT 讲解](https://www.bilibili.com/video/av56235038?from=search&seid=9558641265797595207)




## 4. 优缺点

### 优点

- **效果好，一步到位的全局联系捕捉**

  在 Attention 机制引入之前，有一个问题很苦恼，**就是长距离的信息会被弱化**。attention机制可以灵活的捕捉全局和局部的联系，而且是一步到位的。

  另一方面从attention函数就可以看出来，它先是进行序列的每一个元素与其他元素的对比，在这个过程中每一个元素间的距离都是一，因此它比时间序列RNNs的一步步递推得到长期依赖关系好的多，越长的序列RNNs捕捉长期依赖关系就越弱。

- **并行计算减少模型训练时间，速度快**

  Attention机制每一步计算不依赖于上一步的计算结果，因此可以和CNN一样并行处理。但是CNN也只是每次捕捉局部信息，通过层叠来获取全局的联系增强视野。

- **模型复杂度小，参数少**
  模型复杂度是与CNN和RNN同条件下相比较的。



### 缺点

缺点很明显，**attention机制不是一个"distance-aware"的，它不能捕捉语序顺序**(这里是语序哦，就是元素的顺序)。

这在NLP中是比较糟糕的，自然语言的语序是包含太多的信息。如果确实了这方面的信息，结果往往会是打折扣的。说到底，attention机制就是一个精致的"词袋"模型。

来自文章[浅谈Attention机制的理解](https://zhuanlan.zhihu.com/p/35571412)：

> 所以有时候我就在想，在NLP任务中，我把分词向量化后计算一波TF-IDF是不是会和这种attention机制取得一样的效果呢? 当然这个缺点也好搞定，我在添加位置信息就好了。所以就有了 position-embedding(位置向量)的概念了，这里就不细说了。



## 5. Attention 的 N 种类型

Attention 有很多种不同的类型：Soft Attention、Hard Attention、静态Attention、动态Attention、Self Attention 等等。下面就跟大家解释一下这些不同的 Attention 都有哪些差别。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/attention_5.png" alt="img" style="zoom:50%;" />

由于这篇文章《[Attention用于NLP的一些小结](https://zhuanlan.zhihu.com/p/35739040)》已经总结的很好的，下面就直接引用了：

本节从计算区域、所用信息、结构层次和模型等方面对Attention的形式进行归类。

### 5.1 计算区域

根据Attention的计算区域，可以分成以下几种：

- **Soft Attention**，这是比较常见的Attention方式，**对所有key求权重概率，每个key都有一个对应的权重，是一种全局的计算方式**（也可以叫Global Attention）。这种方式比较理性，参考了所有key的内容，再进行加权。但是计算量可能会比较大一些。
- **Hard Attention**，这种方式是直接精准定位到某个key，其余key就都不管了，相当于这个key的概率是1，其余key的概率全部是0。**因此这种对齐方式要求很高**，要求一步到位，如果没有正确对齐，会带来很大的影响。另一方面，**因为不可导，一般需要用强化学习的方法进行训练**。（或者使用gumbel softmax之类的）
- **Local Attention**，这种方式其实是以上两种方式的一个折中，对一个窗口区域进行计算。先用Hard方式定位到某个地方，以这个点为中心可以得到一个窗口区域，**在这个小区域内用Soft方式来算Attention**。

### 5.2 所用信息

假设我们要对一段原文计算Attention，这里原文指的是我们要做 attention 的文本，那么所用信息包括内部信息和外部信息，**内部信息指的是原文本身的信息，而外部信息指的是除原文以外的额外信息**。

- **General Attention**** ，这种方式利用到了外部信息，常用于需要构建两段文本关系的任务，query一般包含了额外信息，根据外部query对原文进行对齐。

比如在阅读理解任务中，需要构建问题和文章的关联，假设现在baseline是，对问题计算出一个问题向量 q，把这个 q 和所有的文章词向量拼接起来，输入到 LSTM 中进行建模。那么在这个模型中，文章所有词向量共享同一个问题向量，现在我们想让文章每一步的词向量都有一个不同的问题向量，也就是，在每一步使用文章在该步下的词向量对问题来算 attention，这里问题属于原文，文章词向量就属于外部信息。

- **Local Attention**，这种方式只使用内部信息，key和value以及query只和输入原文有关，**在 self attention 中，key=value=query**。既然没有外部信息，那么在原文中的每个词可以跟该句子中的所有词进行Attention计算，相当于寻找原文内部的关系。

还是举阅读理解任务的例子，上面的baseline中提到，对问题计算出一个向量q，那么这里也可以用上 attention，只用问题自身的信息去做attention，而不引入文章信息。

### 5.3 结构层次

结构方面根据是否划分层次关系，分为单层attention，多层attention和多头attention：

1. 单层Attention，这是比较普遍的做法，用一个query对一段原文进行一次attention。
2. 多层Attention，**一般用于文本具有层次关系的模型**，假设我们把一个document划分成多个句子，在第一层，我们分别对每个句子使用attention计算出一个句向量（也就是单层attention）；在第二层，我们对所有句向量再做attention计算出一个文档向量（也是一个单层attention），最后再用这个文档向量去做任务。
3. 多头Attention，这是 Attention is All You Need中提到的multi-head attention，用到了多个query对一段原文进行了多次attention，每个query都关注到原文的不同部分，相当于**重复做多次单层attention**：

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/attention_6.png)

最后再把这些结果拼接起来：

![img](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/attention_7.png)

### 5.4 模型方面

从模型上看，Attention 一般用在CNN和LSTM上，也可以直接进行纯 Attention 计算。

#### 1）CNN+Attention

CNN的卷积操作可以提取重要特征，我觉得这也算是Attention的思想，**但是CNN的卷积感受视野是局部的**，需要通过叠加多层卷积区去扩大视野。另外，**Max Pooling直接提取数值最大的特征，也像是hard attention的思想，直接选中某个特征**。

CNN上加Attention可以加在这几方面：

- 在卷积操作前做attention，比如Attention-Based BCNN-1，这个任务是文本蕴含任务需要处理两段文本，同时对两段输入的序列向量进行attention，计算出特征向量，再拼接到原始向量中，作为卷积层的输入。
- 在卷积操作后做attention，比如Attention-Based BCNN-2，对两段文本的卷积层的输出做attention，作为pooling层的输入。
- 在pooling层做attention，代替max pooling。比如Attention pooling，首先我们用LSTM学到一个比较好的句向量，作为query，然后用CNN先学习到一个特征矩阵作为key，再用query对key产生权重，进行attention，得到最后的句向量。

#### 2）LSTM+Attention

LSTM内部有Gate机制，其中input gate选择哪些当前信息进行输入，forget gate选择遗忘哪些过去信息，我觉得这算是一定程度的Attention了，而且号称可以解决长期依赖问题，实际上LSTM需要一步一步去捕捉序列信息，在长文本上的表现是会随着step增加而慢慢衰减，难以保留全部的有用信息。

LSTM通常需要得到一个向量，再去做任务，常用方式有：

- 直接使用最后的hidden state（可能会损失一定的前文信息，难以表达全文）
- 对所有step下的hidden state进行等权平均（对所有step一视同仁）。
- Attention机制，对所有step的hidden state进行加权，把注意力集中到整段文本中比较重要的hidden state信息。**性能比前面两种要好一点，而方便可视化观察哪些step是重要的，但是要小心过拟合，而且也增加了计算量**。



#### 3）纯Attention

Attention is all you need，没有用到CNN/RNN，乍一听也是一股清流了，但是仔细一看，本质上还是一堆向量去计算attention。



### 5.5 相似度计算方式

在做attention的时候，我们需要计算query和某个key的分数（相似度），常用方法有：

1）点乘：最简单的方法
$$
s(q, k)=q^Tk
$$


2）矩阵相乘：

$$
s(q,k)=q^Tk
$$



3）cos相似度：

$$
s(q,k)=\frac{q^Tk}{||q||\cdot ||k||}
$$


4）串联方式：把q和k拼接起来，

$$
s(q,k)=W[q;k]
$$


5）用多层感知机也可以：

$$
s(q,k)=v_a^Ttanh(W_q+U_k)
$$


------

## 6. self-attention

### 简介

在一般任务的 Encoder-Decoder 框架中，输入 Source 和输出 Target 内容是不一样的，比如对于英-中机器翻译来说，Source是英文句子，Target 是对应的翻译出的中文句子，**Attention 机制发生在Target的元素Query和Source中的所有元素之间**。

而 Self Attention 顾名思义，指的不是 Target 和 Source 之间的 Attention 机制，而是 Source 内部元素之间或者 Target 内部元素之间发生的 Attention 机制，**也可以理解为 Target=Source 这种特殊情况下的注意力计算机制**。其具体计算过程是一样的，只是计算对象发生了变化而已。

引入 Self Attention 后会更容易捕获句子中长距离的相互依赖的特征，因为如果是RNN或者LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。

但是Self Attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，**所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征**。

除此外，**Self Attention对于增加计算的并行性也有直接帮助作用**。这是为何Self Attention逐渐被广泛使用的主要原因。

**为什么自注意力模型（self-Attention model）如此强大：利用注意力机制来“动态”地生成不同连接的权重，从而处理变长的信息序列。**



### 计算

给出信息输入，用 $X=[x_1,x_2,\cdots,x_N]$ 表示 N 个输入信息，通过线性变换得到查询向量序列、键向量序列和值向量序列如下所示：
$$
Q=W_QX\\
K=W_KX\\
V=W_VX
$$
上面的公式可以看出，**self-Attention 中的 Q 是对自身（self）输入的变换，而在传统的 Attention 中，Q 来自于外部**。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/self_attention_fig.jpeg" style="zoom:80%;" />

上述的 self-Attention 计算过程剖解（来自《细讲 | Attention Is All You Need 》）

注意力公式为：
$$
h_i = att((K,V),q_i)=\sum^N_{j=1}\alpha_{ij}v_j = \sum^N_{j=1}softmax(s(k_j, q_i))v_j
$$
自注意力模型中，通常使用缩放点积来作为注意力打分函数，输出向量序列如下所示：
$$
H=Vsoftmax(\frac{K^TQ}{\sqrt{d_3}})
$$


























