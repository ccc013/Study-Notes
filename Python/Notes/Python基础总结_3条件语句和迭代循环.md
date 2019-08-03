### 3. 条件语句和迭代循环

#### 1. 条件语句

Python 的条件语句就是通过一条或者多条语句的执行结果(判断 True 或者 False)来决定执行的代码块。

整体上可以分为四种语句：

- if 语句
- if-else 语句
- if-elif-else 语句
- 嵌套语句(多个 if 语句)

##### if 语句

给定一个二元条件，满足条件执行语句 A，不满足就跳过，代码例子如下：

```python
a = 3
# if 语句
if a > 0:
    print('a =', a)
```

##### if-else 语句

同样是给定二元条件，满足条件执行语句 A，不满足执行语句 B，代码例子如下：

```python
a = 3
# if-else
if a > 2:
    print('a is ', a)
else:
    print('a is less 2')
```

##### if-elif-else 语句

给定多元条件，满足条件1，执行语句1，满足条件2，执行语句2，依次类推，简单的代码例子如下：

```python
a = 3
# if-elif-else
if a > 5:
    print('a>5')
elif a > 3:
    print('a>3')
else:
    print('a<=3')
```

##### 嵌套语句

嵌套语句中可以包含更多的 if 语句，或者是 if-else 、if-elif-else 的语句，简单的代码例子如下所示：

```python
a = 3
# 嵌套语句
if a < 0:
    print('a<0')
else:
    if a > 3:
        print('a>3')
    else:
        print('0<a<=3')
        
```

#### 2. 迭代循环

Python 中的循环语句主要是两种，`while` 循环和 `for` 循环，然后并没有 `do-while` 循环。

##### while 循环

一个简单的 while 循环如下，while 循环的终止条件就是 `while` 后面的语句不满足，即为 `False` 的时候，下面的代码例子中就是当 `n=0` 的时候，会退出循环。

```python
n = 3
while n > 0:
    print(n)
    n -= 1
```

另一个例子，用于输入的时候让用户不断输入内容，直到满足某个条件后，退出。

```python
promt = "\ninput something, and repeat it."
promt += "\nEnter 'q' to end the program.\n"
message = ""
while message != 'q':
    message = input(promt)
    print(message)
```



##### for 循环

`for` 循环可以显式定义循环的次数，并且通常经常用于列表、字典等的遍历。一个简单的例子如下：

```python
# for
l1 = [i for i in range(3)]
for v in l1:
    print(v)
```

上述例子其实用了两次 `for` 循环，第一次是用于列表推导式生成列表 `l1` ，并且就是采用 `range` 函数，指定循环次数是 3 次，第二次就是用于遍历列表。

对于 `range` 函数，还有以下几种用法：

```python
l2 = ['a', 'b', 'c', 'dd', 'nm']
# 指定区间
for i in range(2, 5):
    print(i)
# 指定区间，并加入步长为 10
for j in range(10, 30, 10):
    print(j)
# 结合 len 来遍历列表
for i in range(len(l2)):
    print('{}: {}'.format(i, l2[i]))
```



另外，对于列表的循环，有时候希望同时打印当前元素的数值和索引值，可以采用 `enumerate` 函数，一个坚定例子如下：

```python
l2 = ['a', 'b', 'c', 'dd', 'nm']
for i, v in enumerate(l2):
    print('{}: {}'.format(i, v))
```



##### break 和 continue 以及循环语句中的 else 语句

`break` 语句用于终止循环语句，例子如下：

```python
# break
for a in range(5):
    if a == 3:
        break
    print(a)
```

这里就是如果 `a = 3` ，就会终止 `for` 循环语句。

`continue` 用于跳过当前一次的循环，进入下一次的循环，例子如下：

```python
# continue
for a in range(5):
    if a == 3:
        continue
    print(a)
```

循环语句可以有 `else` 子句，它在穷尽列表(以 `for` 循环)或条件变为 `fals`e (以 `while` 循环)导致循环终止时被执行,但**循环被 `break` 终止时不执行**。例子如下：

```python
# else
for a in range(5):
    print(a)
else:
    print('finish!')
```







------

#### 参考

- 《Python 编程从入门到实践》
- [Python 基础教程](http://www.runoob.com/python/python-tutorial.html)
- [一天快速入门python](https://mp.weixin.qq.com/s/odBnvjO6dgc8HzV9N-aTzg)
- [超易懂的Python入门级教程，赶紧收藏！](https://mp.weixin.qq.com/s/ja8lZvEzZEuzC0C9kkXpag)

