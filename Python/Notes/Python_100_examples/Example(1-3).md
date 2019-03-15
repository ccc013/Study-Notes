
来自 [Python 100例](http://www.runoob.com/python/python-100-examples.html) 的练习题目

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

当然，目前这种代码实现的时间复杂度是很高的，毕竟是三个`for`循环。

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

- [Python练习题 003：完全平方数](https://www.cnblogs.com/iderek/p/5954778.html)
- [ZH奶酪：编程语言入门经典100例【Python版】](http://www.cnblogs.com/CheeseZH/archive/2012/11/05/2755107.html)

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

### Example-7 复制列表

> **题目**：将一个列表的数据复制到另一个列表

#### 思路

直接采用切片操作，即 `[:]`

#### 代码实现

这道题目比较简单，代码如下：

```python
print('original list: {}'.format(input_list))
copyed_list = input_list[:]
print('copyed_list: {}'.format(copyed_list))
```

输出结果如下：

```python
original list: [3, 2, '1', [1, 2]]
copyed_list: [3, 2, '1', [1, 2]]
```

这道题目只要知道列表的切片操作，就非常简单，当然如果不知道这个操作，也可以通过 for 循环来遍历实现复制的操作，就是没有这么简洁，一行代码搞定。

### Example-8 乘法口诀

> **题目**：输出 9*9 乘法口诀

#### 思路

最简单就是通过两层的 for 循环，两个参数，一个控制行，一个控制列，然后注意每行输出个数，即每层循环的起始和结束条件。

#### 代码实现

两种实现方法如下：

```python
# 第一种，for 循环实现
def multiplication_table1():
    for i in range(1, 10):
        for j in range(1, i + 1):
            print('%d*%d=%-2d ' % (i, j, i * j), end='')
        print('')


# 第二种，一行代码实现
def multiplication_table2():
    print('\n'.join([' '.join(['%s*%s=%-2s' % (y, x, x * y) for y in range(1, x + 1)]) for x in range(1, 10)]))
```

结果如下：

```
1*1=1 
1*2=2  2*2=4 
1*3=3  2*3=6  3*3=9 
1*4=4  2*4=8  3*4=12 4*4=16
1*5=5  2*5=10 3*5=15 4*5=20 5*5=25
1*6=6  2*6=12 3*6=18 4*6=24 5*6=30 6*6=36
1*7=7  2*7=14 3*7=21 4*7=28 5*7=35 6*7=42 7*7=49
1*8=8  2*8=16 3*8=24 4*8=32 5*8=40 6*8=48 7*8=56 8*8=64
1*9=9  2*9=18 3*9=27 4*9=36 5*9=45 6*9=54 7*9=63 8*9=72 9*9=81
```

### Example-9 暂停一秒输出

> **题目**：暂停一秒输出

#### 思路

使用 time 模块的 sleep() 函数。

#### 代码实现

```python
print("Start : %s" % time.ctime())
time.sleep(1)
print("End : %s" % time.ctime())
```

输出结果如下：

```
Start : Wed Mar  6 21:48:18 2019
End : Wed Mar  6 21:48:19 2019
```

这道题目主要使用`time.sleep()`实现暂停功能，它接收一个参数表示停顿的时间，单位是秒。

另外一个函数 `time.ctime()` 主要是把一个时间戳（按秒计算的浮点数）转化为 `time.asctime()` 的形式，而 `time asctime()` 函数返回一个可读的形式为"Wed Mar  6 21:48:18 2019"（2019年3月6日 周三 21 时 48 分 18 秒）的24个字符的字符串。

### Example-10

> **题目**: 暂停一秒输出，并格式化当前时间。

#### 思路

继续使用 time 模块的函数实现

#### 代码实现

```python
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
time.sleep(1)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
```

输出结果如下：

```
2019-03-06 21:43:36
2019-03-06 21:43:37
```

除了 `sleep` 函数外，还用到 `time` 模块的另外三个函数：

- **strftime**--接收时间元组，并返回以可读字符串表示的当地时间

- **localtime**--作用是格式化时间戳为本地的时间
- **time**--返回当前时间的时间戳（1970纪元后经过的浮点秒数）



### Example-11 



