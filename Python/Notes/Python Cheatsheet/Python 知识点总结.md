
参考自：

- [Comprehensive Python Cheatsheet](https://github.com/gto76/python-cheatsheet)
- [Python 列表(List)](http://www.runoob.com/python/python-lists.html)
- 《Python从入门到实践》
- [廖雪峰的 Python 教程](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000)

---
### 基本的脚本模板

Python 脚本或者程序的基本模板形式如下：
```
#!/usr/bin/env python3
#
# Usage: .py
# 
# 导入脚本运行的包
from collections import namedtuple
from enum import Enum
import re
import sys

# main 函数
def main():
    pass


###
##  UTIL
#

def read_file(filename):
    with open(filename, encoding='utf-8') as file:
        return file.readlines()

# 程序运行的起始位置
if __name__ == '__main__':
    main()
```
第一行`!/usr/bin/env python3`用于告诉操作系统执行该脚本的时候，调用位于`/usr/bin/env`下的 Python 解释器。

常用有以下几种写法：

- `#!/usr/bin/python` 是告诉操作系统执行这个脚本的时候，调用 /usr/bin 下的 python 解释器；
- `#!/usr/bin/env python` 这种用法是为了防止操作系统用户没有将 python 装在默认的 /usr/bin 路径里。当系统看到这一行的时候，首先会到 env 设置里查找 python 的安装路径，再调用对应路径下的解释器程序完成操作。
- `#!/usr/bin/python` 相当于写死了python路径;
- `#!/usr/bin/env python` 会去环境设置寻找python目录, 推荐这种写法

接着程序的入口是开始于

```
if __name__ == '__main__':
    main()
```

### 列表（List）

> 列表是最常用的 Python 数据类型，它是一个有序的集合，可以随时增加或者删除其中的元素。

#### 特点

1. 可以同时包含不同的数据类型元素，如数字、字符串、列表等；
2. 索引从 0 开始；

#### Python 列表函数&方法

##### 函数

```python
# 比较两个列表的元素
cmp(list1, list2)
# 返回列表元素个数
len(list)
# 返回列表元素最大值
max(list)
# 返回列表元素最小值
min(list)
# 将元组转换为列表
list(seq)
```

##### 方法

```python
# 在列表末尾添加新的对象
list.append(obj)
# 统计某个元素在列表中出现的次数
list.count(obj)
# 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
list.extend(seq)
# 从列表中找出某个值第一个匹配项的索引位置
list.index(obj)
# 将对象插入列表
list.insert(index, obj)
# 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
list.pop([index=-1])
# 移除列表中某个值的第一个匹配项
list.remove(obj)
# 反向列表中元素
list.reverse()
# 对原列表进行排序
list.sort(cmp=None, key=None, reverse=False)
```

接下来会举例介绍上述的某些函数和方法，比如增删查改、排序、反转的几个函数和方法。

#### 初始化

**定义**

```python
<list> = list()
<list> = [<el>]
```

其中 `<list>` 表示列表，`<el>` 表示一个变量。

**例子**

```python
# 创建空列表，两种方法
list1 = list()
list2 = []
# 初始化带有数据
list3 = [1, 2, 3]
list4 = ['a', 2, 'nb', [1,3,4]]]
```

#### 增删查改

##### 增加元素

**定义**

```python
<list>.append(<el>)
<list>.extend(<collection>)
<list> += [<el>]
<list> += [<collection>]
<list>.insert(index, <el>)
```

`<collection>` 表示一个集合元素，如列表 `[1, 2, 3]`。

**例子**

```python
# 末尾添加元素
list1.append('abc')
print('list1:', list1)
# 末尾添加另一个列表，并合并为一个列表
list1.extend(list3)
print('list1.extend(list3), list1:', list1)
list1.extend((1, 3))
print('list1.extend((1,3)), list1:', list1)
# 通过 += 添加元素
list2 += [1, 2, 3]
print('list2:', list2)
list2 += list4
print('list2:', list2)
# 在指定位置添加元素,原始位置元素右移一位
list3.insert(0, 'a')
print('list3:', list3)
# 末尾位置添加，原来末尾元素依然保持在末尾
list3.insert(-1, 'b')
print('list3:', list3)
```

这里定义的 `list1,list2,list3,list4` 列表都是沿用初始化的四个列表。输出结果如下：

```python
list1.append("abc"), list1: ['abc']
list1.extend(list3), list1: ['abc', 1, 2, 3]
list1.extend((1,3)), list1: ['abc', 1, 2, 3, 1, 3]
list2 += [1, 2, 3], ist2: [1, 2, 3]
list2 += list4, list2: [1, 2, 3, 'a', 2, 'nb', [1, 3, 4]]
list3.insert(0, "a"), list3: ['a', 1, 2, 3]
list3.insert(-1, "b"), list3: ['a', 1, 2, 'b', 3]
```

##### 删除元素

**定义**

```python
<el> = <list>.pop([index])  # Removes and returns item at index or from the end.
<list>.remove(<el>)         # Removes first occurrence of item or raises ValueError.
<list>.clear()              # Removes all items.   
del <list>[index]           # remove item at index 
```

**例子**

```python
# del 删除指定位置元素
del list3[-1]
print('del list3[-1], list3:', list3)
# pop 删除元素
pop_el = list3.pop()
print('list3:', list3)
print('pop element:', pop_el)
# pop 删除指定位置元素
pop_el2 = list3.pop(0)
print('list3:', list3)
print('pop element:', pop_el2)
# remove 根据值删除元素
list3.remove(1)
print('list3:', list3)
# clear 清空列表
list3.clear()
print('clear list3:', list3)
```

输出结果如下：

```python
del list3[-1], list3: ['a', 1, 2, 'b']
list3: ['a', 1, 2]
pop element: b
list3: [1, 2]
pop element: a
list3: [2]
clear list3: []
```



##### 查找元素

**定义**

```python
index = <list>.index(<el>)  # Returns first index of item.
```

**例子**

```python
# index 根据数值查询索引
print('list1:', list1)
ind = list1.index(3)
print('list1.index(3)，index=', ind)
```

输出结果

```python
list1: ['abc', 1, 2, 3, 1, 3]
list1.index(3)，index= 3
```

##### 修改&访问元素

**定义**

```python
<el> = <list>[index]   # get item at index of list
<list> = <list>[from_inclusive : to_exclusive : step_size]  # slice
<list> = <list>[:]  # copy list
```

**例子**

```python
# 访问列表第一个元素
print('list1[0]: ', list1[0])
# 访问列表最后一个元素
print('list1[-1]: ', list1[-1])
# 访问第一个到第三个元素
print('list1[:3]: ', list1[:3])
# 访问第一个到第三个元素,步长为2
print('list1[:3:2]: ', list1[:3:2])
# 复制列表
new_list = list1[:]
print('copy list1, new_list:', new_list)
```

输出结果：

```python
list1[0]:  abc
list1[-1]:  3
list1[:3]:  ['abc', 1, 2]
list1[:3:2]:  ['abc', 2]
copy list1, new_list: ['abc', 1, 2, 3, 1, 3]
```

#### 排序

**定义**

```python
<list>.sort()
<list> = sorted(<collection>)
sorted_by_second = sorted(<collection>, key=lambda el: el[1])
sorted_by_both   = sorted(<collection>, key=lambda el: (el[1], el[0]))
```

**例子**

```python
list5 = [3, 1, 4, 2, 5]
print('list5:', list5)
# use sorted
list6 = sorted(list5)
print('list6=sorted(list5), list5={}, list6={}'.format(list5, list6))
# use list.sort()
list5.sort()
print('list5.sort(), list5: ', list5)
```

输出结果如下：

```python
list5: [3, 1, 4, 2, 5]
list6=sorted(list5), list5=[3, 1, 4, 2, 5], list6=[1, 2, 3, 4, 5]
list5.sort(), list5:  [1, 2, 3, 4, 5]
```

**`sorted()` 都不会改变列表本身的顺序**，只是对列表临时排序，并返回一个新的列表对象；

相反，**列表本身的 `sort()` 会永久性改变列表本身的顺序**。

另外，如果列表元素不是单纯的数值类型，如整数或者浮点数，而是字符串、列表、字典或者元组，那么还可以自定义排序规则，这也就是定义中最后两行，例子如下：

```python
# 列表元素也是列表
list8 = [[4, 3], [5, 2], [1, 1]]
list9 = sorted(list8)
print('list9 = sorted(list8), list9=', list9)
# sorted by the second element
list10 = sorted(list8, key=lambda x: x[1])
print('list10 = sorted(list8, key=lambda x:x[1]), list10=', list10)
list11 = sorted(list8, key=lambda x: (x[1], x[0]))
print('list11 = sorted(list8, key=lambda x:(x[1],x[0])), list11=', list11)
# 列表元素是字符串
list_str = ['abc', 'pat', 'cda', 'nba']
list_str_1 = sorted(list_str)
print('list_str_1 = sorted(list_str), list_str_1=', list_str_1)
# 根据第二个元素排列
list_str_2 = sorted(list_str, key=lambda x: x[1])
print('list_str_2 = sorted(list_str, key=lambda x: x[1]), list_str_2=', list_str_2)
# 先根据第三个元素，再根据第一个元素排列
list_str_3 = sorted(list_str, key=lambda x: (x[2], x[0]))
print('list_str_3 = sorted(list_str, key=lambda x: (x[2], x[0])), list_str_3=', list_str_3)
```

输出结果如下：

```python
list9 = sorted(list8), list9= [[1, 1], [4, 3], [5, 2]]
list10 = sorted(list8, key=lambda x:x[1]), list10= [[1, 1], [5, 2], [4, 3]]
list11 = sorted(list8, key=lambda x:(x[1],x[0])), list11= [[1, 1], [5, 2], [4, 3]]

list_str_1 = sorted(list_str), list_str_1= ['abc', 'cda', 'nba', 'pat']
list_str_2 = sorted(list_str, key=lambda x: x[1]), list_str_2= ['pat', 'abc', 'nba', 'cda']
list_str_3 = sorted(list_str, key=lambda x: (x[2], x[0])), list_str_3= ['cda', 'nba', 'abc', 'pat']
```

#### 反转

**定义**

```python
<list>.reverse()
<iter> = reversed(<list>)
<list> = <list>[::-1]
```

**例子**

```python
# 反转列表
list5.reverse()
print('list5.reverse(), list5: ', list5)
list7 = reversed(list5)
print('list7=reversed(list5), list5={}, list7={}'.format(list5, list7))
#for val in list7:
#    print(val)
# 注意不能同时两次
list7_val = [val for val in list7]
print('采用列表推导式, list7_val=', list7_val)
list8 = list5[::-1]
print('list5 = {}\nlist_reversed = list5[::-1], list_reversed = {}'.format(list5, list_reversed))
```

输出结果

```python
list5.reverse(), list5:  [5, 4, 3, 2, 1]
list7=reversed(list5), list5=[5, 4, 3, 2, 1], list7=<list_reverseiterator object at 0x000001D0A17C5550>
采用列表推导式, list7_val= [1, 2, 3, 4, 5]
list5 = [5, 4, 3, 2, 1]
list_reversed = list5[::-1], list_reversed = [1, 2, 3, 4, 5]
```

`reverse()` 方法会永久改变列表本身，而 `reversed()` 不会改变列表对象，它返回的是一个迭代对象，如例子输出的  `<list_reverseiterator object at 0x000001D0A17C5550>` , 要获取其排序后的结果，需要通过 `for` 循环，或者列表推导式，但需要注意，**它仅仅在第一次遍历时候返回数值**。

以及，一个小小的技巧，利用切片实现反转，即 `<list> = <list>[::-1]`



#### 其他

**定义**

```python
# 元素求和
sum_of_elements  = sum(<collection>)
elementwise_sum  = [sum(pair) for pair in zip(list_a, list_b)]
# 字符串和列表的相互转换
list_of_chars    = list(<str>)
string = '(format)'.join(<list>)
```

**例子**

列表元素求和的例子

```python
# 列表元素求和
list5 = [3, 1, 4, 2, 5]
print('list5:', list5)
sum_list5 = sum(list5)
print('sum_list5=sum(list5), sum_list5=', sum_list5)
# 实现 list_a + list_b 的对应元素相加操作
list_a = [1, 3, 5]
list_b = [2, 4, 6]
sum_pair = [sum(pair) for pair in zip(list_a, list_b)]
print('sum_pair = [sum(pair) for pair in zip(list_a, list_b)], sum_pair =', sum_pair)

```

输出结果：

```python
list5: [3, 1, 4, 2, 5]
sum_list5=sum(list5), sum_list5= 15
sum_pair = [sum(pair) for pair in zip(list_a, list_b)], sum_pair = [3, 7, 11]
```

接着是字符串和列表互换的例子：

```python
strs = 'python'
list_str = list(strs)
print('strs = {}\nlist_str = list(strs), list_str={}'.format(strs, list_str))
new_strs = ' '.join(list_str)
print('new_strs= ', new_strs)
```

输出结果如下：

```python
strs = python
list_str = list(strs), list_str=['p', 'y', 't', 'h', 'o', 'n']
new_strs=  p-y-t-h-o-n
```

其中 `join` 方法用于连接列表中的元素，并且可以定义连接的方式，例子中是采用 `-`，也可以默认直接使用,如下所示：

```python
>>> ''.join(list_str)
python
```



### 元组(tuple)

> 元组也是 Python 的一种有序列表，但它不能**修改**。

元组和列表非常相似，唯一不同的是**元组初始化后就不能进行修改**，它用小括号 `()` 表示。



#### 内置函数

```python
# 比较两个元组的元素
cmp(tuple1, tuple2)
# 返回元组元素个数
len(tuple)
# 返回元组元素最大值
max(tuple)
# 返回元组元素最小值
min(tuple)
# 将列表转换为元组
tuple(seq)
```



