
> 2019年第 15 篇文章，总第 39 篇文章
本文大约 1600  字，阅读大约需要 8分钟

练习题 3 的网址：

http://www.runoob.com/python/python-exercise-example3.html

---

### Example-3 完全平方数

> **题目**：一个整数，它加上100后是一个完全平方数，再加上168又是一个完全平方数，请问该数是多少？

#### 思路

首先我们可以假设这个整数是`x`，那么根据题目有：

```
x+100 = m**2 (1)
x+100+168 = n**2 (2)
```
`m, n`都是正整数，接着就是先根据求解一元二次方程组的做法，可以得到

``` 
n**2 - n**2 = 168 (3)
```
利用平方差分解上式，有`(n-m)(n+m)=168`，这个时候，我们再做一个变换：

```
m + n = i (4)
n - m = j (5)
i * j = 168 (6)
```
这个变换，其实只是再设置两个变量`i,j`，并且根据它们两者相乘是 168，这是一个偶数，由于两个数相乘是偶数，有两种情况，两者都是偶数，或者一个偶数和一个奇数，然后再求解(4)和(5)，有：

```
n = (i + j) / 2 (7)
m = (i - j) / 2 (8)
```
根据(7)式子，`i+j`必须是偶数，这样才可以被 2 整除，得到正整数`n`，这种情况下，结合(4)和(5)，可以推导得到`i,j`都是大于等于 2 的偶数，又根据(6)，可以推导到`i,j`的范围是：

```
1 < j < i < 85
```
这里是假设了`i > j`的情况，因为不存在一个偶数的平方就是`168`，所以假设`i>j`。

#### 代码实现

第一种实现：

```
def perfect_square():
    for i in range(1, 85):
        if 168 % i == 0:
            j = 168 / i;
            if i > j and (i + j) % 2 == 0 and (i - j) % 2 == 0:
                m = (i + j) / 2
                n = (i - j) / 2
                x = n * n - 100
                print(x)
```

第二种实现是网上大神的解法，参考文章：

- https://www.cnblogs.com/iderek/p/5954778.html
- http://www.cnblogs.com/CheeseZH/archive/2012/11/05/2755107.html

这种实现其实就是在分析过程中，只推导到`m,n`部分，即(3)式的部分，然后直接根据这个公式和范围来求解，这个时候`m,n`的范围就是`(1,169)`。

这是一个应用列表推导式的解法：

```
def perfect_square2():
    '''
    列表推导式
    :return:
    '''
    [print(m**2-100, end=',') for m in range(1, 169) for n in range(1, 169) if (n**2 - m**2) == 168]

def perfect_square2_loop():
    '''
    for 循环形式
    :return:
    '''
    for m in range(1, 169):
        for n in range(1, 169):
            if (n ** 2 - m ** 2) == 168:
                print(m ** 2 - 100, end=',')

```
输出结果都是：

```
-99,21,261,1581,
```

源代码在：

https://github.com/ccc013/CodesNotes/blob/master/Python_100_examples/example3.py

或者点击原文，也可以查看源代码。

---


欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)


#### 往期精彩推荐

##### Python-100 练习系列

- [Python-100 | 练习题 01 & 列表推导式](https://mp.weixin.qq.com/s/qSUJKYjGLkGcswdBpA8KLg)
- [Python-100 练习题 02](https://mp.weixin.qq.com/s/w2pmPqp_dmPNFfoaZP95JQ)


##### 学习笔记

- [机器学习入门系列（1）--机器学习概览](https://mp.weixin.qq.com/s/r_UkF_Eys4dTKMH7DNJyTA)
- [[GAN学习系列] 初识GAN](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483711&idx=1&sn=ead88d5b21e08d9df853b72f31d4b5f4&chksm=fe3b0f4ac94c865cfc243123eb4815539ef2d5babdc8346f79a29b681e55eee5f964bdc61d71&token=1493836032&lang=zh_CN#rd)
- [[GAN学习系列2] GAN的起源](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483732&idx=1&sn=99cb91edf6fb6da3c7d62132c40b0f62&chksm=fe3b0f21c94c8637a8335998c3fc9d0adf1ac7dea332c2bd45e63707eac6acad8d84c1b3d16d&token=985117826&lang=zh_CN#rd)
- [[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(上）](https://mp.weixin.qq.com/s/S_uiSe74Ti6N_u4Y5Fd6Fw)

##### 数学学习笔记

- [程序员的数学笔记1--进制转换](https://mp.weixin.qq.com/s/Sn7V27O77moGCLOpFzEKqg)
- [程序员的数学笔记2--余数](https://mp.weixin.qq.com/s/hv4cWzuca49VHLc92DicZQ)
- [程序员的数学笔记3--迭代法](https://mp.weixin.qq.com/s/uUtK2tTZa_b5jeiTyXYRYg)

##### Github项目 & 资源教程推荐

- [[Github 项目推荐] 一个更好阅读和查找论文的网站](https://mp.weixin.qq.com/s/ImQcGt8guLKZawNLS-_HzA)
- [[资源分享] TensorFlow 官方中文版教程来了](https://mp.weixin.qq.com/s/Si1YaYLfhL1upbjQkvireQ)
- [必读的AI和深度学习博客](https://mp.weixin.qq.com/s/0J2raJqiYsYPqwAV1MALaw)
- [[教程]一份简单易懂的 TensorFlow 教程](https://mp.weixin.qq.com/s/vXIM6Ttw37yzhVB_CvXmCA)
- [[资源]推荐一些Python书籍和教程，入门和进阶的都有！](https://mp.weixin.qq.com/s/jkIQTjM9C3fDvM1c6HwcQg)



