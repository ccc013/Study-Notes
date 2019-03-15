
> 2019年第 18 篇文章，总第 42 篇文章

练习题 4 的网址：

http://www.runoob.com/python/python-exercise-example4.html

---
### Example-4 判断天数

> **题目**：输入某年某月某日，判断这一天是这一年的第几天？

#### 思路

判断输入的日期是一年中的第几天，因为一年有12个月，我们可以先考虑计算逐月累计的天数，假设输入的月份是 `m`，那么前 `m-1`个月份的天数是可以计算出来的，比如输入的是 2018 年 3 月 5 日，那么前两个月的天数就是`31+28=59`天，然后再加上输入的天，即 `59+5=64`天。

当然，涉及到日期，年份，都需要考虑闰年，闰年的定义如下，来自百度百科

> **普通闰年**: 能被4整除但不能被100整除的年份为普通闰年。（如2004年就是闰年，1999年不是闰年）；
>
> **世纪闰年**: 能被400整除的为世纪闰年。（如2000年是世纪闰年，1900年不是世纪闰年）；



#### 代码实现

实现的代码如下：

```python
def calculate_days():
    year = int(input('year:\n'))
    month = int(input('month:\n'))
    day = int(input('day:\n'))

    # 统计前 m-1 个月的天数
    months = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    sums = 0
    if 0 < month <= 12:
        sums = months[month - 1]
    else:
        print('Invalid month:', month)

    sums += day

    # 判断闰年
    is_leap = False
    if (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0)):
        is_leap = True
    if is_leap and month > 2:
        sums += 1
    return sums
```

测试例子如下，给出两个同样的日期，但年份不同，闰年的 2016 年和非闰年的 2018年。

```python
# 非闰年
year:
2018
month:
3
day:
5
it is the 64th day

# 闰年
year:
2016
month:
3
day:
5
it is the 65th day
```

源代码在：

https://github.com/ccc013/CodesNotes/blob/master/Python_100_examples/example4.py

或者点击原文，也可以查看源代码。

---


欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)


#### 往期精彩推荐

##### Python-100 练习系列

- [Python-100 | 练习题 01 & 列表推导式](https://mp.weixin.qq.com/s/qSUJKYjGLkGcswdBpA8KLg)
- [Python-100 练习题 02](https://mp.weixin.qq.com/s/w2pmPqp_dmPNFfoaZP95JQ)
- [Python-100 练习题 03 完全平方数](https://mp.weixin.qq.com/s/iHGr6vCJHgALPoHj8koy-A)

##### 机器学习系列

- [机器学习入门系列（1）--机器学习概览](https://mp.weixin.qq.com/s/r_UkF_Eys4dTKMH7DNJyTA)
- [机器学习入门系列(2)--如何构建一个完整的机器学习项目(一)](https://mp.weixin.qq.com/s/nMG5Z3CPdwhg4XQuMbNqbw)
- [机器学习数据集的获取和测试集的构建方法](https://mp.weixin.qq.com/s/HxGO7mhxeuXrloN61sDGmg)
- [特征工程之数据预处理（上）](https://mp.weixin.qq.com/s/BnTXjzHSb5-4s0O0WuZYlg)
- [特征工程之数据预处理（下）](https://mp.weixin.qq.com/s/Npy1-zrRmqETN8GydnIb8Q)
- [特征工程之特征缩放&特征编码](https://mp.weixin.qq.com/s/WYPUJbcT6UHvEFMJe8vteg)

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



