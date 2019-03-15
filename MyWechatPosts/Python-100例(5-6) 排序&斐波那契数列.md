
> 2019 年第 23 篇，总 47 篇文章

前面分享的四道题目如下：

- [Python-100 | 练习题 01 & 列表推导式](https://mp.weixin.qq.com/s/qSUJKYjGLkGcswdBpA8KLg)
- [Python-100 练习题 02](https://mp.weixin.qq.com/s/w2pmPqp_dmPNFfoaZP95JQ)
- [Python-100 练习题 03 完全平方数](https://mp.weixin.qq.com/s/iHGr6vCJHgALPoHj8koy-A)
- [Python-100 练习题 04 判断天数](https://mp.weixin.qq.com/s/2hXJq1k-BTCcHAR1tG_o3w)

这次是分享 Python-100 例的第五和第六题，分别是排序和斐波那契数列问题，这两道题目其实都是非常常见的问题，特别是后者，一般会在数据结构的教程中，讲述到递归这个知识点的时候作为例题进行介绍的。


---
### Example-5 排序

> **题目**：输入三个整数 x,y,z，请把这三个数由小到大输出。

#### 思路

考虑令 x 保存最小的数值，即先令 x 分别和 y，z 作比较，通过比较后，x变成最小值，接着 y 和 z 比较，即可完成排序

#### 代码实现

代码实现上有两种，一种就是手动实现排序过程，另一种就是采用内置函数。

```python
def sort_numbers_1():
    x = int(input('integer:\n'))
    y = int(input('integer:\n'))
    z = int(input('integer:\n'))
    print('input numbers: x=%d, y=%d, z=%d' % (x, y, z))
    if x > y:
        x, y = y, x
    if x > z:
        x, z = z, x
    if y > z:
        y, z = z, y
    print('sorted: x=%d, y=%d, z=%d' % (x, y, z))

# 利用列表的内置函数 sort()
def sort_numbers_2():
    l = []
    for i in range(3):
        x = int(input('integer:\n'))
        l.append(x)
    print('original list:', l)
    l.sort()
    print('sorted:', l)
```

测试样例如下：

```
# sort_numbers_1()运行结果
integer:
1
integer:
0
integer:
5
input numbers: x=1, y=0, z=5
sorted: x=0, y=1, z=5

# sort_numbers_2() 运行结果
integer:
1
integer:
0
integer:
5
original list: [1, 0, 5]
sorted: [0, 1, 5]
```

### Example-6 斐波那契数列

> **题目**：斐波那契数列

#### 思路

斐波那契数列（Fibonacci sequence），又称黄金分割数列，指的是这样一个数列：0、1、1、2、3、5、8、13、21、34、....

数学上的定义如下：

```
n=0: F(0)=0
n=1: F(1)=1
n>=2: F(n)=F(n-1)+F(n-2)
```

#### 代码实现

需要输出斐波那契数列的第 n 个数，实现方法如下，既可以通过迭代实现，也可以利用递归实现：

```python
# 采用迭代循环实现
def fib1(n):
    a, b = 1, 1
    # n 必须大于等于 2
    for i in range(n - 1):
        a, b = b, a + b
    return a


# 递归实现
def fib2(n):
    if 0 < n <= 2:
        return 1
    else:
        return fib2(n - 1) + fib2(n - 2)
```

如果是需要输出给定个数的所有斐波那契数列，代码如下：

```python
# 输出指定个数的斐波那契数列
def fib_array(n):
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    fibs = [1, 1]
    for i in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs
```

测试结果如下：

```python
a1 = fib1(10)
a2 = fib2(10)
fibs = fib_array(10)
print('fib1 result=', a1)
print('fib2 result=', a2)
print('fib array=', fibs)

# 输出结果
# fib1 result= 55
# fib2 result= 55
# fib array= [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
```

另外，这里更推荐采用迭代实现斐波那契数列，而不是递归做法，主要是递归实现一方面是调用函数自身，而函数调用是有时间和空间的消耗，这会影响效率问题，另一方面是递归中很多计算都是重复的，它本质上是将一个问题分解成多个小问题，这些多个小问题存在相互重叠的部分，也就会出现重复计算的问题。

这里选择 `n=30`，计算两种方法使用的时间，结果如下：

```python
start = time.time()
a1 = fib1(30)
print('fib1 cost time: ', time.time() - start)
print('fib1 result=', a1)
start2 = time.time()
a2 = fib2(30)
print('fib2 cost time: ', time.time() - start2)
print('fib2 result=', a2)
```

输出结果如下：

```
fib1 cost time:  0.0
fib1 result= 832040
fib2 cost time:  0.39077210426330566
fib2 result= 832040
```

可以看到递归实现所需要的时间明显大于迭代实现的方法。

因此，尽管递归的代码看上去更加简洁，但从实际应用考虑，需要选择效率更高的迭代实现方法。


---
### 小结

今天分享的两道题目就到这里，如果你有更好的解决方法，也可以后台留言，谢谢！

---


欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)


#### 往期精彩推荐

##### Python-100 练习系列

- [Python-100 | 练习题 01 & 列表推导式](https://mp.weixin.qq.com/s/qSUJKYjGLkGcswdBpA8KLg)
- [Python-100 练习题 02](https://mp.weixin.qq.com/s/w2pmPqp_dmPNFfoaZP95JQ)
- [Python-100 练习题 03 完全平方数](https://mp.weixin.qq.com/s/iHGr6vCJHgALPoHj8koy-A)
- [Python-100 练习题 04 判断天数](https://mp.weixin.qq.com/s/2hXJq1k-BTCcHAR1tG_o3w)


##### 机器学习系列

- [机器学习入门系列（1）--机器学习概览](https://mp.weixin.qq.com/s/r_UkF_Eys4dTKMH7DNJyTA)
- [机器学习入门系列(2)--如何构建一个完整的机器学习项目(一)](https://mp.weixin.qq.com/s/nMG5Z3CPdwhg4XQuMbNqbw)
- [机器学习数据集的获取和测试集的构建方法](https://mp.weixin.qq.com/s/HxGO7mhxeuXrloN61sDGmg)
- [特征工程之数据预处理（上）](https://mp.weixin.qq.com/s/BnTXjzHSb5-4s0O0WuZYlg)
- [特征工程之数据预处理（下）](https://mp.weixin.qq.com/s/Npy1-zrRmqETN8GydnIb8Q)
- [特征工程之特征缩放&特征编码](https://mp.weixin.qq.com/s/WYPUJbcT6UHvEFMJe8vteg)
- [特征工程(完)](https://mp.weixin.qq.com/s/0QkAOXg9nw8UwpnKuYdC-g)
- [常用机器学习算法汇总比较(上）](https://mp.weixin.qq.com/s/4Ban_TiMKYUBXTq4WcMr5g)
- [常用机器学习算法汇总比较(中）](https://mp.weixin.qq.com/s/ELQbsyxQtZYdtHVrfOFBFw)


##### Github项目 & 资源教程推荐

- [[Github 项目推荐] 一个更好阅读和查找论文的网站](https://mp.weixin.qq.com/s/ImQcGt8guLKZawNLS-_HzA)
- [[资源分享] TensorFlow 官方中文版教程来了](https://mp.weixin.qq.com/s/Si1YaYLfhL1upbjQkvireQ)
- [必读的AI和深度学习博客](https://mp.weixin.qq.com/s/0J2raJqiYsYPqwAV1MALaw)
- [[教程]一份简单易懂的 TensorFlow 教程](https://mp.weixin.qq.com/s/vXIM6Ttw37yzhVB_CvXmCA)
- [[资源]推荐一些Python书籍和教程，入门和进阶的都有！](https://mp.weixin.qq.com/s/jkIQTjM9C3fDvM1c6HwcQg)
- [[Github项目推荐] 机器学习& Python 知识点速查表](https://mp.weixin.qq.com/s/kn2DUJHL48UyuoUEhcfuxw)
