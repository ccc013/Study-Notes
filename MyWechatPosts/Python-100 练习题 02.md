
> 2019年第 10 篇文章，总第 34 篇文章

练习题2 的网址：

http://www.runoob.com/python/python-exercise-example2.html

---

### Example-2 企业发放奖金

> **题目**：企业发放的奖金根据利润提成。利润(I)低于或等于10万元时，奖金可提10%；利润高于10万元，低于20万元时，低于10万元的部分按10%提成，高于10万元的部分，可提成7.5%；20万到40万之间时，高于20万元的部分，可提成5%；40万到60万之间时高于40万元的部分，可提成3%；60万到100万之间时，高于60万元的部分，可提成1.5%，高于100万元时，超过100万元的部分按1%提成，从键盘输入当月利润I，求应发放奖金总数？

#### 思路

这道题目可以根据每个奖金发放区间来分界，先分别定义两个数组，一个数组是存放每个区间奖金的提成比例，记为`rat`；另一个数组是记录每个发放区间的上边界，表示当超过该边界时候，直接利用上边界乘以该区间的提成比例，例如对于在 10 万元以下的这个区间，就是上边界为 10 万，然后超过后，该区间发放奖金就是`100000*0.1`。

然后我们先考虑利润超过 100 万的情况，依次降低利润，对应每种情况。

#### 代码实现

```
def pay_award():
    profit = int(input('净利润:'))
    arr = [1000000, 600000, 400000, 200000, 100000, 0]
    rat = [0.01, 0.015, 0.03, 0.05, 0.075, 0.1]
    r = 0
    for idx in range(0, 6):
        if profit > arr[idx]:
            # 当前区间的利润
            r += (profit - arr[idx]) * rat[idx]
            print('current award=', (profit - arr[idx]) * rat[idx])
            # 重置下一个区间起始奖金数量
            profit = arr[idx]
    return r
```
简单的测试例子：

```
# 利润是 11000
净利润:11000
current award= 1100.0
award= 1100.0

# 利润是 1100000 （110万）
净利润:1100000
current award= 1000.0
current award= 6000.0
current award= 6000.0
current award= 10000.0
current award= 7500.0
current award= 10000.0
award= 40500.0
```

源代码在：

https://github.com/ccc013/CodesNotes/blob/master/Python_100_examples/example2.py

或者点击原文，也可以查看源代码。

---


欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)


#### 往期精彩推荐

##### Python-100 练习系列

- [Python-100 | 练习题 01 & 列表推导式](https://mp.weixin.qq.com/s/qSUJKYjGLkGcswdBpA8KLg)


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

