
最近打算好好练习下 python，因此找到一个练习题网站，打算每周练习 3-5 题吧。

http://www.runoob.com/python/python-100-examples.html

另外，这个网站其实也还有 Python 的教程，从基础到高级的知识都有。

---
### Example-1 三位数组合

> **题目**：有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？

#### 思路

最简单的方法，就是穷举法了，分别求出在百位、十位、个位上的数字，然后剔除出现重复数字的组合，剩余的就是答案了。

#### 代码实现

直接代码实现如下：

```
def create_three_digits(number_start=1, number_end=4):
    '''
    给定指定数字范围（比如1到4），求可以组成多少个无重复的三位数
    :param number_start: 起始数字
    :param number_end: 结束数字
    :return: 返回数量，以及可能的三位数的列表
    '''
    count = 0
    result_list = list()
    for i in range(number_start, number_end + 1):
        for j in range(number_start, number_end + 1):
            for k in range(number_start, number_end + 1):
                if (i != j) and (i != k) and (j != k):
                    count += 1
                    result_list.append(str(i) + str(j) + str(k))
    return count, result_list
```

写得更加简便点，可以采用列表推导式：

```
def create_three_digits2(number_start=1, number_end=4):
    '''
    采用列表推导式实现
    :param number_start:
    :param number_end:
    :return:
    '''
    return [str(i) + str(j) + str(k) for i in range(number_start, number_end + 1) for j in
            range(number_start, number_end + 1) for k in
            range(number_start, number_end + 1) if (i != j) and (i != k) and (j != k)]
```

输出结果如下，总共有 24 种不同的排列组合。

```
valid count=24, and they are:
123
124
132
134
142
143
213
214
231
234
241
243
312
314
321
324
341
342
412
413
421
423
431
432
```

当然，目前这种代码实现的时间复杂度是很高的，毕竟是三个`for`循环。如果有更好的解法，可以在后台留言，告诉我！

#### 知识点复习--列表推导式

列表推导式（又称列表解析式）提供了一种简明扼要的方法来创建列表。

**它的结构是在一个中括号里包含一个表达式，然后是一个 for 语句，然后是 0 个或多个 for 或者 if 语句**。那个表达式可以是任意的，意思是你可以在列表中放入任意类型的对象。返回结果将是一个新的列表，在这个以 if 和 for 语句为上下文的表达式运行完成之后产生。

用代码表示列表推导式如下：

```
variable = [out_exp for out_exp in input_list if out_exp == 2]
```

一个简明的例子如下：

```
multiples = [i for i in range(30) if i % 3 is 0]
print(multiples)
# Output: [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
```

那么，什么时候最适合用列表推导式呢？

其实是**当你需要使用 for 循环来生成一个新列表**。举个例子，你通常会这样做：

```
squared = []
for x in range(10):
    squared.append(x**2)
```
这时候，采用列表推导式最合适：

```
squared = [x**2 for x in range(10)]
```

源代码在：

https://github.com/ccc013/CodesNotes/blob/master/Python_100_examples/example1.py

或者点击原文，也可以查看源代码。

---

参考文章：

- [列表推导式（list comprehensions）](https://eastlakeside.gitbooks.io/interpy-zh/content/Comprehensions/list-comprehensions.html)


欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)


#### 往期精彩推荐

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



