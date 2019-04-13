### 2. 基础语法和变量类型

注意：主要是基于 **Python 3** 的语法来介绍，并且代码例子也是在 Python3 环境下运行的。

#### 2.1 基础语法

##### 标识符

标识符由**字母、数字和下划线(_)组成，其中不能以数字开头，并且区分大小写**。

以下划线开头的标识符是有特殊意义的：

- 单下划线开头的如 `_foo`，表示不能直接访问的类属性，需要通过类提供的接口进行访问，不能通过 `from xxx import *` 导入；
- 双下划线开头的如 `__foo` ，表示类的私有成员；
- 双下划线开头和结尾的如 `__foo__` 代表 Python 中的特殊方法，比如 `__init()__` 代表类的构建函数

##### 保留字

**保留字是不能用作常数或变数，或者其他任何标识符名称**。 `keyword` 模块可以输出当前版本的所有关键字：

```python
import keyword
print(keyword.kwlist)
```

所有的保留字如下所示：

|          |         |        |
| :------: | :-----: | :----: |
|   and    |  exec   |  not   |
|  assert  | finally |   or   |
|  break   |   for   |  pass  |
|  class   |  from   | print  |
| continue | global  | raise  |
|   def    |   if    | return |
|   del    | import  |  try   |
|   elif   |   in    | while  |
|   else   |   is    |  with  |
|  except  | lambda  | yield  |

##### 行和缩进

和其他编程语言的最大区别就是，Python 的代码块不采用大括号 `{}` 来控制类、函数以及其他逻辑判断，反倒是采用**缩进**来写模块。

**缩进的空白数量是可变的**，但是所有代码块语句必须包含**相同的缩进空白数量**，这个必须严格执行，如下所示：

```python
# 正确示例
i = 2
if i == 3:
    print('true!')
else:
  print('False')

# 错误示例
if i == 3:
    print('i:')
    print(i)
else:
    print('wrong answer!')
    # 没有严格缩进，执行时会报错
  print('please check again')
```

这里将会报错 **`IndentationError: unindent does not match any outer indentation level`**，这个错误表示采用的缩进方式不一致，有的是 `tab` 键缩进，有的是空格缩进，改为一致即可。

而如果错误是 **`IndentationError: unexpected indent`**，表示格式不正确，可能是 `tab` 和空格没对齐的问题。

因此，按照约定俗成的管理，**应该始终坚持使用4个空格的缩进**，并且**注意不能混合使用 `tab` 键和四格空格**，这会报错！

##### 注释

注释分为两种，单行和多行的。

```python
# 单行注释
print('Hello, world!')

'''
这是多行注释，使用单引号。
这是多行注释，使用单引号。
这是多行注释，使用单引号。
'''

"""
这是多行注释，使用双引号。
这是多行注释，使用双引号。
这是多行注释，使用双引号。
"""
```



##### 输入输出

通常是一条语句一行，如果语句很长，我们可以使用**反斜杠(`\`)**来实现多行语句。在 `[], {}, 或 () `中的多行语句，则不需要反斜杠。

```python
sentence1 = "I love " + \
"python"

sentence2 = ["I", "love",
          "python"]
```

另外，我们也可以同一行显示多条语句，语句之间用分号(;)分割，示例如下：

```python
print('Hello');print('world')
```

对于用户输入，Python2 采用的是 `raw_input()`，而 3 版本则是 `input()` 函数：

```python
# 等待用户输入
# python 2 
user_input = raw_input('请输入一个数字:\n')
# python 3
user_input = input('请输入一个数字:\n')
print('user_input=', user_input)
```

其中 `\n` 实现换行。用户按下回车键(enter)退出，其他键显示。

对于 `print` 输出，默认输出是换行的，如果需要实现不换行，可以指定参数 `end`，如下所示：

```python
a = 3
b = 2
c = 4
d = 5
# 默认换行
print(a)
print(b)
# 不换行，并设置逗号分隔
print(c, end=',')
print(d)
```



#### 2.2 基本变量类型

计算机程序要处理不同的数据，需要定义不同的数据类型。Python 定义了六种标准的数据类型，分布如下所示：

- Numbers(数字)
- Strings(字符串)
- List(列表)
- Tuple(元组)
- Set(集合)
- Dictionary(字典)



##### 变量赋值

Python 并不需要声明变量的类型，所说的"类型"是变量所指的内存中对象的类型。但每个变量使用前都必须赋值，然后才会创建变量。给变量赋值的方法是采用等号(=)，等号左边是变量名，右边是存储在变量中的值。

一个示例如下：

```python
counter = 100  # 赋值整型变量
miles = 1000.0  # 浮点型
name = "John"  # 字符串

print(counter)
print(miles)
print(name)
```

Python 还允许同时为多个变量赋值，有以下两种实现方式：

```python
# 创建一个整型对象，值为1，三个变量被分配到相同的内存空间上
n = m = k = 2
# 创建多个对象，然后指定多个变量
cc, mm, nn = 1, 3.2, 'abc'

print('n=m=k=', n, m, k)
print('cc=', cc)
print('mm=', mm)
print('nn=', nn)
```

其中同时给多个变量赋值的方式也是 Python 独特的一种变量赋值方法。

##### 数字

数字类型用于存储数值，它是不可改变的数据类型。Python 3 支持以下几种数字类型：

- **int** (整数)

- **float** (浮点型)
- **complex**(复数)
- **bool** (布尔)

数字类型的使用很简单，也很直观，如下所示：

```python
# int
q = 1
# float
w = 2.3
# bool
e = True
# complex
r = 1 + 3j
print(q, w, e, r) # 1 2.3 True (1+3j)

# 内置的 type() 函数可以用来查询变量所指的对象类型
print(type(q))  # <class 'int'>
print(type(w))  # <class 'float'>
print(type(e))  # <class 'bool'>
print(type(r))  # <class 'complex'>

# 也可以采用 isinstance()
# isinstance 和 type 的区别在于：type()不会认为子类是一种父类类型，isinstance()会认为子类是一种父类类型
print(isinstance(q, int)) # True
print(isinstance(q, float)) # False
```

对于数字的运算，包括基本的加减乘除，其中除法包含两个运算符，`/` 返回一个浮点数，而 `//` 则是得到整数，去掉小数点后的数值。而且在**混合计算的时候， Python 会把整数转换为浮点数**。

```python
# 加
print('2 + 3 =', 2 + 3)  # 2 + 3 = 5
# 减
print('3 - 2 =', 3 - 2)  # 3 - 2 = 1
# 乘
print('5 * 8 =', 5 * 8)  # 5 * 8 = 40
# 除
# 得到浮点数，完整的结果
print('5 / 2 =', 5 / 2)  # 5 / 2 = 2.5
# 得到一个整数
print('5 // 2 =', 5 // 2)  # 5 // 2 = 2
# 取余
print('5 % 2 =', 5 % 2)  # 5 % 2 = 1
# 乘方
print('5 ** 2 =', 5 ** 2)  # 5 ** 2 = 25
```



##### 字符串

字符串或串(String)是由数字、字母、下划线组成的一串字符。一般是用单引号 `''` 或者 `""` 括起来。

注意，Python 没有单独的字符类型，一个字符就是长度为 1 的字符串。并且，Python 字符串是不可变，向一个索引位置赋值，如 `strs[0]='m'` 会报错。

可以通过索引值或者切片来访问字符串的某个或者某段元素，注意索引值从 0 开始，例子如下所示：

![](http://www.runoob.com/wp-content/uploads/2013/11/o99aU.png)

切片的格式是 `[start:end]`，实际取值范围是 `[start:end)` ，即不包含 `end` 索引位置的元素。还会除了正序访问，还可以倒序访问，即索引值可以是负值。

具体示例如下所示：

```python
s1 = "talk is cheap"
s2 = 'show me the code'
print(s1)
print(s2)

# 索引值以 0 为开始值，-1 为从末尾的开始位置
print('输出 s1 第一个到倒数第二个的所有字符: ', s1[0:-1])  # 输出第一个到倒数第二个的所有字符
print('输出 s1 字符串第一个字符: ', s1[0])  # 输出字符串第一个字符
print('输出 s1 从第三个开始到第六个的字符: ', s1[2:6])  # 输出从第三个开始到第六个的字符
print('输出 s1 从第三个开始的后的所有字符:', s1[2:])  # 输出从第三个开始的后的所有字符

# 加号 + 是字符串的连接符
# 星号 * 表示复制当前字符串，紧跟的数字为复制的次数
str = "I love python "
print("连接字符串:", str + "!!!")
print("输出字符串两次:", str * 2)

# 反斜杠 \ 转义特殊字符
# 若不想让反斜杠发生转义，可以在字符串前面添加一个 r
print('I\nlove\npython')
print("反斜杠转义失效:", r'I\nlove\npython')
```

注意：

- 1、反斜杠可以用来转义，**使用 r 可以让反斜杠不发生转义**。
- 2、字符串可以用 + 运算符连接在一起，用 * 运算符重复。
- 3、Python 中的字符串有两种索引方式，从左往右以 0 开始，从右往左以 -1 开始。
- 4、Python 中的**字符串不能改变**。

字符串包含了很多内置的函数，这里只介绍几种非常常见的函数：

- **strip(x)**：当包含参数 `x` 表示删除句首或者句末 `x` 的部分，否则，就是删除句首和句末的空白字符，并且可以根据需要调用 `lstrip()` 和 `rstrip()` ，分别删除句首和句末的空白字符；
- **split()**：同样可以包含参数，如果不包含参数就是将字符串变为单词形式，如果包含参数，就是根据参数来划分字符串；
- **join()**：主要是将其他类型的集合根据一定规则变为字符串，比如列表；
- **replace(x, y)**：采用字符串 `y` 代替 `x`
- **index()**：查找指定字符串的起始位置
- **startswith() / endswith()**：分别判断字符串是否以某个字符串为开始，或者结束；
- **find()**：查找某个字符串；
- **upper() / lower() / title()**：改变字符串的大小写的三个函数

下面是具体示例代码：

```python
# strip()
s3 = " I love python "
s4 = "show something!"
print('输出直接调用 strip() 后的字符串结果: ', s3.strip())
print('lstrip() 删除左侧空白后的字符串结果: ', s3.lstrip())
print('rstrip() 删除右侧空白后的字符串结果: ', s3.rstrip())
print('输出调用 strip(\'!\')后的字符串结果: ', s4.strip('!'))
# split()
s5 = 'hello, world'
print('采用split()的字符串结果: ', s5.split())
print('采用split(\',\')的字符串结果: ', s5.split(','))
# join()
l1 = ['an', 'apple', 'in', 'the', 'table']
print('采用join()连接列表 l1 的结果: ', ''.join(l1))
print('采用\'-\'.join()连接列表 l1 的结果: ', '-'.join(l1))
# replace()
print('replace(\'o\', \'l\')的输出结果: ', s5.replace('o', 'l'))
# index()
print('s5.index(\'o\')的输出结果: ', s5.index('o'))
# startswith() / endswith()
print('s5.startswith(\'h\')的输出结果: ', s5.startswith('h'))
print('s5.endswith(\'h\')的输出结果: ', s5.endswith('h'))
# find()
print('s5.find(\'h\')的输出结果: ', s5.find('h'))
# upper() / lower() / title()
print('upper() 字母全大写的输出结果: ', s5.upper())
print('lower() 字母全小写的输出结果: ', s5.lower())
print('title() 单词首字母大写的输出结果: ', s5.title())
```

##### 列表

列表是 Python 中使用最频繁的数据类型，它可以完成大多数集合类的数据结构实现，可以包含不同类型的元素，包括数字、字符串，甚至列表（也就是所谓的嵌套）。

和字符串一样，可以通过索引值或者切片(截取)进行访问元素，索引也是从 0 开始，而如果是倒序，则是从 -1 开始。列表截取的示意图如下所示：

![](http://www.runoob.com/wp-content/uploads/2013/11/list_slicing1.png)

另外，还可以添加第三个参数作为步长：

![](http://www.runoob.com/wp-content/uploads/2013/11/python_list_slice_2.png)

同样，列表也有很多内置的方法，这里介绍一些常见的方法：

- **len(list)**：返回列表的长度
- **append(obj) / insert(index, obj) / extend(seq)**：增加元素的几个方法
- **pop() / remove(obj) / del list[index] / clear()**：删除元素
- **reverse() / reversed**：反转列表
- **sort() / sorted(list)**：对列表排序，注意前者会修改列表内容，后者返回一个新的列表对象，不改变原始列表
- **index()**：查找给定元素第一次出现的索引位置

初始化列表的代码示例如下：

```python
# 创建空列表，两种方法
list1 = list()
list2 = []
# 初始化带有数据
list3 = [1, 2, 3]
list4 = ['a', 2, 'nb', [1, 3, 4]]

print('list1:', list1)
print('list2:', list2)
print('list3:', list3)
print('list4:', list4)
print('len(list4): ', len(list4))
```

添加元素的代码示例如下：

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

删除元素的代码示例如下：

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

查找元素和修改、访问元素的代码示例如下：

```python
# index 根据数值查询索引
ind = list1.index(3)
print('list1.index(3)，index=', ind)
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

排序的代码示例如下：

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

反转列表的代码示例如下：

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

`reverse()` 方法会永久改变列表本身，而 `reversed()` 不会改变列表对象，它返回的是一个迭代对象，如例子输出的  `<list_reverseiterator object at 0x000001D0A17C5550>` , 要获取其排序后的结果，需要通过 `for` 循环，或者列表推导式，但需要注意，**它仅仅在第一次遍历时候返回数值**。

以及，一个小小的技巧，**利用切片实现反转**，即 `<list> = <list>[::-1]`。



##### 元组

元组和列表比较相似，不同之处是**元组不能修改，然后元组是写在小括号 `()` 里的**。

元组也可以包含不同的元素类型。简单的代码示例如下：

```python
t1 = tuple()
t2 = ()
t3 = (1, 2, '2', [1, 2], 5)
# 创建一个元素的元祖
t4 = (7, )
t5 = (2)
print('创建两个空元组：t1={}, t2={}'.format(t1, t2))
print('包含不同元素类型的元组：t3={}'.format(t3))
print('包含一个元素的元祖: t4=(7, )={}, t5=(2)={}'.format(t4, t5))
print('type(t4)={}, type(t5)={}'.format(type(t4), type(t5)))
print('输出元组的第一个元素：{}'.format(t3[0]))
print('输出元组的第二个到第四个元素：{}'.format(t3[1:4]))
print('输出元祖的最后一个元素: {}'.format(t3[-1]))
print('输出元祖两次: {}'.format(t3 * 2))
print('连接元祖: {}'.format(t3 + t4))
```

元祖和字符串也是类似，索引从 0 开始，-1 是末尾开始的位置，可以将字符串看作一种特殊的元组。

此外，从上述代码示例可以看到有个特殊的例子，创建一个元素的时候，必须在元素后面添加逗号，即如下所示:

```python
tup1 = (2,) # 输出为 (2,)
tup2 = (2)  # 输出是 2
print('type(tup1)={}'.format(type(tup1))) # 输出是 <class 'tuple'>
print('type(tup2)={}'.format(type(tup2))) # 输出是 <class 'int'>
```

还可以创建一个二维元组，代码例子如下：

```python
# 创建一个二维元组
tups = (1, 3, 4), ('1', 'abc')
print('二维元组: {}'.format(tups)) # 二维元组: ((1, 3, 4), ('1', 'abc'))
```

然后对于函数的返回值，如果返回多个，实际上就是返回一个元组，代码例子如下：

```python
def print_tup():
    return 1, '2'


res = print_tup()
print('type(res)={}, res={}'.format(type(res), res)) # type(res)=<class 'tuple'>, res=(1, '2')
```

元组不可修改，但如果元素可修改，那可以修改该元素内容，代码例子如下所示：

```python
tup11 = (1, [1, 3], '2')
print('tup1={}'.format(tup11)) # tup1=(1, [1, 3], '2')
tup11[1].append('123')
print('修改tup11[1]后，tup11={}'.format(tup11)) # 修改tup11[1]后，tup11=(1, [1, 3, '123'], '2')
```

因为元组不可修改，所以仅有以下两个方法：

- **count()**: 计算某个元素出现的次数
- **index()**: 寻找某个元素第一次出现的索引位置

代码例子：

```python
# count()
print('tup11.count(1)={}'.format(tup11.count(1)))
# index()
print('tup11.index(\'2\')={}'.format(tup11.index('2')))
```



##### 字典

字典也是 Python 中非常常用的数据类型，具有以下特点：

- 它是一种映射类型，用 `{}` 标识，是**无序**的 **键(key): 值(value)** 的集合；
- 键(key) 必须使用**不可变类型**；
- 同一个字典中，**键必须是唯一的**；

创建字典的代码示例如下，总共有三种方法：

```python
# {} 形式
dic1 = {'name': 'python', 'age': 20}
# 内置方法 dict()
dic2 = dict(name='p', age=3)
# 字典推导式
dic3 = {x: x**2 for x in {2, 4, 6}}
print('dic1={}'.format(dic1)) # dic1={'age': 20, 'name': 'python'}
print('dic2={}'.format(dic2)) # dic2={'age': 3, 'name': 'p'}
print('dic3={}'.format(dic3)) # dic3={2: 4, 4: 16, 6: 36}
```

常见的三个内置方法，`keys()`, `values()`, `items()` 分别表示键、值、对，例子如下：

```
print('keys()方法，dic1.keys()={}'.format(dic1.keys()))
print('values()方法, dic1.values()={}'.format(dic1.values()))
print('items()方法, dic1.items()={}'.format(dic1.items()))
```

其他对字典的操作，包括增删查改，如下所示：

```python
# 修改和访问
dic1['age'] = 33
dic1.setdefault('sex', 'male')
print('dic1={}'.format(dic1))
# get() 访问某个键
print('dic1.get(\'age\', 11)={}'.format(dic1.get('age', 11)))
print('访问某个不存在的键，dic1.get(\'score\', 100)={}'.format(dic1.get('score', 100)))
# 删除
del dic1['sex']
print('del dic1[\'sex\'], dic1={}'.format(dic1))
dic1.pop('age')
print('dic1.pop(\'age\'), dic1={}'.format(dic1))
# 清空
dic1.clear()
print('dic1.clear(), dic1={}'.format(dic1))
# 合并两个字典
print('合并 dic2 和 dic3 前, dic2={}, dic3={}'.format(dic2, dic3))
dic2.update(dic3)
print('合并后，dic2={}'.format(dic2))

# 遍历字典
dic4 = {'a': 1, 'b': 2}
for key, val in dic4.items():
    print('{}: {}'.format(key, val))
# 不需要采用 keys()
for key in dic4:
    print('{}: {}'.format(key, dic4[key]))
```

最后，因为字典的键必须是不可改变的数据类型，那么如何快速判断一个数据类型是否可以更改呢？有以下两种方法：

- **id()**：判断变量更改前后的 id，如果**一样**表示**可以更改**，**不一样表示不可更改**。
- **hash()**：如果不报错，表示**可以被哈希**，就**表示不可更改**；否则就是可以更改。

首先看下 `id()` 方法，在一个整型变量上的使用结果：

```python
i = 2
print('i id value=', id(i)) 
i += 3
print('i id value=', id(i)) 
```

输出结果，更改前后 id 是更改了，表明整型变量是不可更改的。

```
i id value= 1758265872
i id value= 1758265968
```

然后在列表变量上进行同样的操作：

```python
l1 = [1, 3]
print('l1 id value=', id(l1)) 
l1.append(4)
print('l1 id value=', id(l1))
```

输出结果，`id` 并没有改变，说明列表是可以更改的。

```
l1 id value= 1610679318408
l1 id value= 1610679318408
```

然后就是采用 `hash()` 的代码例子：

```python
# hash
s = 'abc'
print('s hash value: ', hash(s)) 
l2 = ['321', 1]
print('l2 hash value: ', hash(l2)) 
```

输出结果如下，对于字符串成功输出哈希值，而列表则报错 `TypeError: unhashable type: 'list'`，这也说明了字符串不可更改，而列表可以更改。

```
s hash value:  1106005493183980421
TypeError: unhashable type: 'list'
```

##### 集合

集合是一个**无序**的**不重复**元素序列，采用大括号 `{}` 或者 `set()` 创建，但空集合必须使用 `set()` ，因为 `{}` 创建的是空字典。

创建的代码示例如下：

```python
# 创建集合
s1 = {'a', 'b', 'c'}
s2 = set()
s3 = set('abc')
print('s1={}'.format(s1)) # s1={'b', 'a', 'c'}
print('s2={}'.format(s2)) # s2=set()
print('s3={}'.format(s3)) # s3={'b', 'a', 'c'}
```

注意上述输出的时候，每次运行顺序都可能不同，这是集合的**无序性**的原因。

利用集合可以去除重复的元素，如下所示：

```python
s4 = set('good')
print('s4={}'.format(s4)) # s4={'g', 'o', 'd'}
```

集合也可以进行增加和删除元素的操作，代码如下所示：

```python
# 增加元素，add() 和 update()
s1.add('dd')
print('s1.add(\'dd\'), s1={}'.format(s1)) # s1.add('dd'), s1={'dd', 'b', 'a', 'c'}
s1.update('o')
print('添加一个元素，s1={}'.format(s1)) # 添加一个元素，s1={'dd', 'o', 'b', 'a', 'c'}
s1.update(['n', 1])
print('添加多个元素, s1={}'.format(s1)) # 添加多个元素, s1={1, 'o', 'n', 'a', 'dd', 'b', 'c'}
s1.update([12, 33], {'ab', 'cd'})
print('添加列表和集合, s1={}'.format(s1)) # 添加列表和集合, s1={1, 33, 'o', 'n', 'a', 12, 'ab', 'dd', 'cd', 'b', 'c'}

# 删除元素, pop(), remove(), clear()
print('s3={}'.format(s3)) # s3={'b', 'a', 'c'}
s3.pop()
print('随机删除元素, s3={}'.format(s3)) # 随机删除元素, s3={'a', 'c'}
s3.clear()
print('清空所有元素, s3={}'.format(s3)) # 清空所有元素, s3=set()
s1.remove('a')
print('删除指定元素,s1={}'.format(s1)) # 删除指定元素,s1={1, 33, 'o', 'n', 12, 'ab', 'dd', 'cd', 'b', 'c'}
```

此外，还有专门的集合操作，包括求取两个集合的并集、交集

```python
# 判断是否子集, issubset()
a = set('abc')
b = set('bc')
c = set('cd')
print('b是否a的子集:', b.issubset(a)) # b是否a的子集: True
print('c是否a的子集:', c.issubset(a)) # c是否a的子集: False

# 并集操作，union() 或者 |
print('a 和 c 的并集:', a.union(c)) # a 和 c 的并集: {'c', 'b', 'a', 'd'}
print('a 和 c 的并集:', a | c) # a 和 c 的并集: {'c', 'b', 'a', 'd'}

# 交集操作，intersection() 或者 &
print('a 和 c 的交集:', a.intersection(c)) # a 和 c 的交集: {'c'}
print('a 和 c 的交集:', a & c) # a 和 c 的交集: {'c'}

# 差集操作，difference() 或者 - ，即只存在一个集合的元素
print('只在a中的元素:', a.difference(c)) # 只在a中的元素:: {'b', 'a'}
print('只在a中的元素:', a - c) # 只在a中的元素:: {'b', 'a'}

# 对称差集, symmetric_difference() 或者 ^, 求取只存在其中一个集合的所有元素
print('对称差集:', a.symmetric_difference(c)) # 对称差集: {'a', 'd', 'b'}
print('对称差集:', a ^ c) # 对称差集: {'a', 'd', 'b'}
```

##### 数据类型的转换

有时候我们需要对数据类型进行转换，比如列表变成字符串等，这种转换一般只需要将数据类型作为函数名即可。下面列举了这些转换函数：

- **int(x, [,base])**：将 x 转换为整数，`base` 表示进制，默认是十进制

- **float(x)**：将 x 转换为一个浮点数

- **complex(x, [,imag])**：创建一个复数, `imag` 表示虚部的数值，默认是0

- **str(x)**：将对象 x 转换为字符串  
- **repr(x)**： 将对象 x 转换为表达式字符串 
- **eval(str)**： 用来计算在字符串中的有效 Python 表达式,并返回一个对象
- **tuple(s)**： 将序列 s 转换为一个元组  
- **list(s)**： 将序列 s 转换为一个列表
- **set(s)**：转换为可变集合
- **dict(d)**： 创建一个字典。d 必须是一个序列 (key,value)元组
- **frozenset(s)**： 转换为不可变集合
- **chr(x)**：将一个整数转换为一个字符
- **ord(x)**：将一个字符转换为它的整数值 
- **hex(x)**：将一个整数转换为一个十六进制字符串
- **oct(x)**：将一个整数转换为一个八进制字符串                   







------

#### 参考

- 《Python 编程从入门到实践》
- [everything-about-python-from-beginner-to-advance-level](https://medium.com/fintechexplained/everything-about-python-from-beginner-to-advance-level-227d52ef32d2)
- [Python 基础教程](http://www.runoob.com/python/python-tutorial.html)
- [一天快速入门python](https://mp.weixin.qq.com/s/odBnvjO6dgc8HzV9N-aTzg)
- [廖雪峰老师的教程](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000)
- [超易懂的Python入门级教程，赶紧收藏！](https://mp.weixin.qq.com/s/ja8lZvEzZEuzC0C9kkXpag)