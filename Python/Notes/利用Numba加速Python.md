

在 [24式加速你的Python](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247484643&idx=1&sn=55ba87185102042bf8c4641e14573161&chksm=fe3b0a96c94c83802ef2a8c4f35fb2364bf732f7931739a3d09386fa244634720b271892fe5b&token=1679241518&lang=zh_CN#rd)中介绍对循环的加速方法中，一个办法就是采用 `Numba` 加速，刚好最近看到一篇文章介绍了利用 `Numba` 加速 Python ，文章主要介绍了两个例子，也是 `Numba` 的两大作用，分别是加速循环，以及对 `Numpy` 的计算加速。

原文：https://towardsdatascience.com/heres-how-you-can-get-some-free-speed-on-your-python-code-with-numba-89fdc8249ef3

------

相比其他语言，Python 确实在运行速度上是比较慢的。

一种常用解决方法，就是用如 C++ 改写代码，然后用 Python 进行封装，这样既可以实现 C++ 的运行速度又可以保持在主要应用中采用 Python 的方便。

这种办法的唯一难点就是改写为 C++ 部分的代码需要耗费不少时间，特别是如果你对 C++ 并不熟悉的情况。

`Numba`  可以实现提升速度但又不需要改写部分代码为其他编程语言。

#### Numba 简介

`Numba` 是一个可以将 Python 代码转换为优化过的机器代码的编译库。通过这种转换，对于数值算法的运行速度可以提升到接近 `C` 语言代码的速度。

采用 `Numba` 并不需要添加非常复杂的代码，只需要在想优化的函数前 添加一行代码，剩余的交给 `Numba` 即可。

`Numba` 可以通过 `pip` 安装：

```shell
$ pip install numba
```

`Numba` 对于有许多数值运算的，`Numpy` 操作或者大量循环操作的情况，都可以大大提升运行速度。

#### 加速 Python 循环

`Numba`  的最基础应用就是加速 Python 中的循环操作。

首先，如果你想使用循环操作，你先考虑是否可以采用 `Numpy` 中的函数替代，有些情况，可能没有可以替代的函数。这时候就可以考虑采用 `Numba` 了。

第一个例子是通过插入排序算法来进行说明。我们会实现一个函数，输入一个无序的列表，然后返回排序好的列表。

我们先生成一个包含 100,000  个随机整数的列表，然后执行 50 次插入排序算法，然后计算平均速度。

代码如下所示：

```python
import time
import random

num_loops = 50
len_of_list = 100000

def insertion_sort(arr):
    for i in range(len(arr)):
        cursor = arr[i]
        pos = i
        
        while pos > 0 and arr[pos-1] > cursor:
            # 从后往前对比，从小到大排序
            arr[pos] = arr[pos-1]
            pos = pos-1
        # 找到当前元素的位置
        arr[pos] = cursor
    return arr
start = time.time()
list_of_numbers = list()
for i in range(len_of_list):
    num = random.randint(0, len_of_list)
    list_of_numbers.append(num)

for i in range(num_loops):
    result = insertion_sort(list_of_numbers)

end = time.time()

run_time = end-start
print('Average time={}'.format(run_time/num_loops))
```

输出结果：

```python
Average time=22.84399790763855
```

从代码可以知道插入排序算法的时间复杂度是 $O(n^2)$，因为这里包含了两个循环，`for` 循环里面带有 `while` 循环，这是最差的情况。然后输入数量是 10 万个整数，再加上重复 50 次，这是非常耗时的操作了。

原作者采用的是电脑配置是 i7-8700k，所以其平均耗时是 `3.0104s`。但这里我的电脑配置就差多了，i5-4210M 的笔记本电脑，并且已经使用了接近 4 年，所以我跑的结果是，平均耗时为 `22.84s`。

那么，如何采用 `Numba` 加速循环操作呢，代码如下所示：

```python
import time
import random
from numba import jit

num_loops = 50
len_of_list = 100000

@jit(nopython=True)
def insertion_sort(arr):
    for i in range(len(arr)):
        cursor = arr[i]
        pos = i
        
        while pos > 0 and arr[pos-1] > cursor:
            # 从后往前对比，从小到大排序
            arr[pos] = arr[pos-1]
            pos = pos-1
        # 找到当前元素的位置
        arr[pos] = cursor
    return arr
start = time.time()
list_of_numbers = list()
for i in range(len_of_list):
    num = random.randint(0, len_of_list)
    list_of_numbers.append(num)

for i in range(num_loops):
    result = insertion_sort(list_of_numbers)

end = time.time()

run_time = end-start
print('Average time={}'.format(run_time/num_loops))
```

输出结果：

```
Average time=0.09438572406768798
```

可以看到，其实只增加了两行代码，第一行就是导入 `jit` 装饰器

```python
from numba import jit
```

 接着在函数前面增加一行代码，采用装饰器

```python
@jit(nopython=True)
def insertion_sort(arr):
```

使用 `jit` 装饰器表明我们希望将该函数转换为机器代码，然后参数 `nopython` 指定我们希望 `Numba` 采用纯机器代码，或者有必要的情况加入部分 `Python` 代码，这个参数必须设置为 `True` 来得到更好的性能，除非出现错误。

原作者得到的平均耗时是 `0,1424s` ，而我的电脑上则是提升到仅需 `0.094s` ，速度都得到非常大的提升。

#### 加速 Numpy 操作

`Numba` 的另一个常用地方，就是加速 `Numpy` 的运算。

这次将初始化 3 个非常大的 `Numpy` 数组，相当于一个图片的尺寸大小，然后采用 `numpy.square()` 函数对它们的和求平方。

代码如下所示：

```python
import time
import numpy as np

num_loops = 50
img1 = np.ones((1000, 1000), np.int64) * 5
img2 = np.ones((1000, 1000), np.int64) * 10
img3 = np.ones((1000, 1000), np.int64) * 15

def add_arrays(img1, img2, img3):
    return np.square(img1+img2+img3)

start1 = time.time()
for i in range(num_loops):
    result = add_arrays(img1, img2, img3)
end1 = time.time()
run_time1 = end1 - start1
print('Average time for normal numpy operation={}'.format(run_time1/num_loops))
```

输出结果：

```
Average time for normal numpy operation=0.040156774520874024
```

当我们对 `Numpy` 数组进行基本的数组计算，比如加法、乘法和平方，`Numpy` 都会自动在内部向量化，这也是它可以比原生 `Python` 代码有更好性能的原因。

上述代码在原作者的电脑运行的速度是 `0.002288s` ，而我的电脑需要 `0.04s` 左右。

但即便是 `Numpy` 代码也不会和优化过的机器代码速度一样快，因此这里依然可以采用 `Numba` 进行加速，代码如下所示：

```python
# numba 加速
from numba import vectorize, int64

@vectorize([int64(int64,int64,int64)], target='parallel')
def add_arrays_numba(img1, img2, img3):
    return np.square(img1+img2+img3)
    
start2 = time.time()
for i in range(num_loops):
    result = add_arrays_numba(img1, img2, img3)
end2 = time.time()
run_time2 = end2 - start2
print('Average time using numba accelerating={}'.format(run_time2/num_loops))
```

输出结果：

```
Average time using numba accelerating=0.007735490798950195
```

这里采用的是 `vectorize` 装饰器，它有两个数参数，第一个参数是指定需要进行操作的 `numpy` 数组的数据类型，这是必须添加的，因为 `numba` 需要将代码转换为最佳版本的机器代码，以便提升速度；

第二个参数是 `target` ，它有以下三个可选数值，表示如何运行函数：

- **cpu**：运行在单线程的 CPU 上
- **parallel**：运行在多核、多线程的 CPU
- **cuda**：运行在 GPU 上

`parallel` 选项在大部分情况是快过 `cpu` ，而 `cuda` 一般用于有非常大数组的情况。

上述代码在原作者的电脑运行时间是 `0.001196s` ，提升了 2 倍左右，而我的电脑是 `0.0077s`，提升了 5 倍左右速度。

#### 小结

`numba` 在以下情况下可以更好发挥它提升速度的作用：

-  `Python` 代码运行速度慢于 `C`代码的地方，典型的就是循环操作
- 在同个地方重复使用同个操作的情况，比如对许多元素进行同个操作，即 `numpy` 数组的操作

而在其他情况下，`Numba` 并不会带来如此明显的速度提升，当然，一般情况下尝试采用 `numba` 提升速度也是一个不错的尝试。

最后，练习代码：

https://github.com/ccc013/Python_Notes/blob/master/Python_tips/numba_example.ipynb







