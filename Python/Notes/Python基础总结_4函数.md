

### 4. 函数

定义：函数是组织好的，可重复使用，用于实现单一或者相关联功能的代码段。

在 Python 中既有内建函数，比如 `print()`、`sum()` ，也可以用户自定义函数。

#### 4.1 定义函数

自定义一个函数需要遵守一些规则：

- 函数代码块必须以 **def** 关键词开头，然后是函数标识符名称(函数名)和圆括号 **()**；
- 圆括号内部用于定义参数，并且传入参数和自变量也是存放在圆括号内；
- 函数的第一行语句可以选择性地使用文档字符串—用于存放函数说明。
- 函数内容以冒号起始，并且缩进。
- **return [表达式]** 结束函数，选择性地返回一个值给调用方。不带表达式的 `return` 语句相当于返回 `None`。

一个函数的一般格式如下：

```python
def 函数名(参数列表):
	函数体
```

**默认情况下，参数值和参数名称是按照函数声明中定义的顺序匹配的**。

简单的定义和调用函数的例子如下所示：

```python
def hello():
    print("Hello, world!")
# 计算面积的函数
def area(width, height):
    return width * height

hello()
width = 2
height = 3
print('width={}, height={}, area={}'.format(width, height, area(width, height)))
```

输出结果：

```python
Hello, world!
width=2, height=3, area=6
```

上述例子定义了两个函数，第一个是没有参数的 `hello()`, 而第二个函数定义了两个参数。



#### 4.2 参数传递

在 python 中，**类型属于对象，变量是没有类型的**：

```python
a = [1, 2, 3]
a = "abc"
```

上述代码中，`[1,2,3]` 是 List 类型，`"abc"` 是 String 类型，但**变量 a 是没有类型的**，它仅仅是一个**对象的引用（一个指针）**，可以指向 List 类型，也可以指向 String 类型。

##### 可更改(mutable)与不可更改(immutable)对象

python 中，`strings, tuples, numbers` 是不可更改对象，而 `list, dict` 是可修改的对象。

- **不可变类型**：上述例子中 `a` 先赋值为 5，然后赋值为 10，实际上是生成一个新对象，赋值为 10，然后让 `a` 指向它，并且抛弃了 5，并非改变了 `a` 的数值；
- **可变类型**：对于 `list` 类型，变量 `la=[1,2,3]`，然后令 `la[2]=5` ，此时并没有改变变量 `la`，仅仅改变了其内部的数值。

在之前的第二节介绍变量类型中，介绍了如何判断数据类型是否可变，介绍了两种方法：

- **id()**
- **hash()**

这里用 `id()` 的方法来做一个简单的例子，代码如下：

```python
# 判断类型是否可变
a = 5
print('a id:{}, val={}'.format(id(a), a))
a = 3
print('a id:{}, val={}'.format(id(a), a))

la = [1, 2, 3]
print('la id:{}, val={}'.format(id(la), la))
la[2] = 5
print('la id:{}, val={}'.format(id(la), la))
```

输出结果，可以发现变量 `a` 的 `id` 是发生了变化，而列表变量 `la` 的 `id` 没有变化，这证明了 `a` 的类型 `int` 是不可变的，而 `list` 是可变类型。

```
a id:1831338608, val=5
a id:1831338544, val=3

la id:1805167229448, val=[1, 2, 3]
la id:1805167229448, val=[1, 2, 5]
```

然后在 Python 中进行函数参数传递的时候，根据传递的变量是否可变，也需要分开讨论：

- **不可变类型**：类似 `c++` 的**值传递**，如 整数、字符串、元组。如 `fun(a)`，传递的只是 `a` 的值，没有影响 `a` 对象本身。比如在 `fun(a)`内部修改 `a` 的值，只是修改另一个复制的对象，不会影响 `a` 本身。
- **可变类型**：类似 `c++` 的**引用传递，**如 列表，字典。如 `fun(la)`，则是将 `la` 真正的传过去，修改后 `fun` 外部的 `la`  也会受影响。

当然了，Python 中一切都是对象，这里应该说是传递可变对象和不可变对象，而不是引用传递和值传递，但必须**注意应该慎重选择传递可变对象的参数**，下面会分别给出传递两种对象的例子。

首先是传递不可变对象的实例：

```python
# 传递不可变对象的实例
def change_int(a):
    a = 10

b = 2
print('origin b=', b)
change_int(b)
print('after call function change_int(), b=', b)
```

输出结果，传递的变量 `b` 并没有发生改变。

```python
origin b= 2
after call function change_int(), b= 2
```

接着，传递可变对象的例子：

```python
# 传递可变对象的实例
def chang_list(la):
    """
    修改传入的列表参数
    :param la:
    :return:
    """
    la.append([2, 3])
    print('函数内部: ', la)
    return


la = [10, 30]
print('调用函数前, la=', la)
chang_list(la)
print('函数外取值, la=', la)
```

输出结果，可以看到在函数内部修改列表后，也会影响在函数外部的变量的数值。

```python
调用函数前, la= [10, 30]
函数内部:  [10, 30, [2, 3]]
函数外取值, la= [10, 30, [2, 3]]
```

当然，这里如果依然希望传递列表给函数，但又不希望修改列表本来的数值，可以采用传递列表的副本给函数，这样函数的修改只会影响副本而不会影响原件，最简单实现就是切片 `[:]` ，例子如下：

```python
# 不修改 lb 数值的办法，传递副本
lb = [13, 21]
print('调用函数前, lb=', lb)
chang_list(lb[:])
print('传递 la 的副本给函数 change_list, lb=', lb)
```

输出结果：

```python
调用函数前, lb= [13, 21]
函数内部:  [13, 21, [2, 3]]
传递 lb 的副本给函数 change_list, lb= [13, 21]
```



#### 4.3 参数类型

参数的类型主要分为以下四种类型：

- 位置参数
- 默认参数
- 可变参数
- 关键字参数
- 命名关键字参数

##### 位置参数

**位置参数须以正确的顺序传入函数。调用时的数量必须和声明时的一样。**其定义如下，`arg`  就是位置参数，`docstring` 是函数的说明，一般说明函数作用，每个参数的含义和类型，返回类型等；`statement` 表示函数内容。

```python
def function_name(arg):
	"""docstring"""
	statement
```

下面是一个例子，包括一个正确调用例子，和两个错误示例

```python
# 位置参数
def print_str(str1, n):
    """
    打印输入的字符串 n 次
    :param str1: 打印的字符串内容
    :param n: 打印的次数
    :return:
    """
    for i in range(n):
        print(str1)


strs = 'python '
n = 3
# 正确调用
print_str(strs, n)
# 错误例子1
print_str()
# 错误例子2
print_str(n, strs)
```

对于正确例子，输出：

```python
python python python 
```

错误例子1--`print_str()`，也就是没有传入任何参数，返回错误:

```python
TypeError: print_str() missing 2 required positional arguments: 'str1' and 'n'
```

错误例子1--`print_str(n, strs)`，也就是传递参数顺序错误，返回错误:

```python
TypeError: 'str' object cannot be interpreted as an integer
```

##### 默认参数

默认参数定义如下，其中 `arg2` 就是表示默认参数，它是在定义函数的时候事先赋予一个默认数值，调用函数的时候可以不需要传值给默认参数。

```python
def function_name(arg1, arg2=v):
	"""docstring"""
	statement
```

代码例子如下：

```python
# 默认参数
def print_info(name, age=18):
    '''
    打印信息
    :param name:
    :param age:
    :return:
    '''
    print('name: ', name)
    print('age: ', age)

print_info('jack')
print_info('robin', age=30)
```

输出结果：

```python
name:  jack
age:  18
name:  robin
age:  30
```

注意：**默认参数必须放在位置参数的后面**，否则程序会报错。

##### 可变参数

可变参数定义如下，其中 `arg3` 就是表示可变参数，顾名思义就是**输入参数的数量可以是从 0 到任意多个，它们会自动组装为元组**。

```python
def function_name(arg1, arg2=v, *arg3):
	"""docstring"""
	statement
```

这里是一个使用可变参数的实例，代码如下：

```python
# 可变参数
def print_info2(name, age=18, height=178, *args):
    '''
    打印信息函数2
    :param name:
    :param age:
    :param args:
    :return:
    '''
    print('name: ', name)
    print('age: ', age)
    print('height: ', height)
    print(args)
    for language in args:
        print('language: ', language)

print_info2('robin', 20, 180, 'c', 'javascript')
languages = ('python', 'java', 'c++', 'go', 'php')
print_info2('jack', 30, 175, *languages)
```

输出结果：

```python
name:  robin
age:  20
height:  180
('c', 'javascript')
language:  c
language:  javascript

name:  jack
age:  30
height:  175
('python', 'java', 'c++', 'go', 'php')
language:  python
language:  java
language:  c++
language:  go
language:  php
```

这里需要注意几点：

1. 首先如果要使用可变参数，那么传递参数的时候，默认参数应该如上述例子传递，不能如`print_info2('robin', age=20, height=180, 'c', 'javascript')`，这种带有参数名字的传递是会出错的；
2. 可变参数有两种形式传递：

- **直接传入函数**，如上述例子第一种形式，即 `print_info2('robin', 20, 180, 'c', 'javascript')`;
- 先组装为**列表或者元组**，再传入，并且**必须带有 `*`** ，即类似 `func(*[1, 2,3])` 或者 `func(*(1,2,3))`，之所以必须带 `*` ，是因为如果没有带这个，传入的可变参数会多嵌套一层元组，即 `(1,2,3)` 变为 `((1,2,3))`



##### 关键字参数

关键字参数定义如下，其中 `arg4` 就是表示关键字参数，关键字参数其实和可变参数类似，也是可以传入 0 个到任意多个，不同的是会自动组装为一个字典，并且是参数前 `**` 符号。 

```python
def function_name(arg1, arg2=v, *arg3, **arg4):
	"""docstring"""
	statement
```

一个实例如下：

```python
def print_info3(name, age=18, height=178, *args, **kwargs):
    '''
    打印信息函数3，带有关键字参数
    :param name:
    :param age:
    :param height:
    :param args:
    :param kwargs:
    :return:
    '''
    print('name: ', name)
    print('age: ', age)
    print('height: ', height)

    for language in args:
        print('language: ', language)
    print('keyword: ', kwargs)


# 不传入关键字参数的情况
print_info3('robin', 20, 180, 'c', 'javascript')
```

输出结果如下：

```python
name:  robin
age:  20
height:  180
language:  c
language:  javascript
keyword:  {}
```

传入任意数量关键字参数的情况：

```python
# 传入任意关键字参数
print_info3('robin', 20, 180, 'c', 'javascript', birth='2000/02/02')
print_info3('robin', 20, 180, 'c', 'javascript', birth='2000/02/02', weight=125)
```

结果如下：

```python
name:  robin
age:  20
height:  180
language:  c
language:  javascript
keyword:  {'birth': '2000/02/02'}

name:  robin
age:  20
height:  180
language:  c
language:  javascript
keyword:  {'birth': '2000/02/02', 'weight': 125}
```

第二种传递关键字参数方法--字典：

```python
# 用字典传入关键字参数
keys = {'birth': '2000/02/02', 'weight': 125, 'province': 'Beijing'}
print_info3('robin', 20, 180, 'c', 'javascript', **keys)
```

输出结果：

```python
name:  robin
age:  20
height:  180
language:  c
language:  javascript
keyword:  {'birth': '2000/02/02', 'province': 'Beijing', 'weight': 125}
```

所以，同样和可变参数相似，也是两种传递方式：

- 直接传入，例如 `func(birth='2012')`
- 先将参数组装为一个字典，再传入函数中，如 `func(**{'birth': '2000/02/02', 'weight': 125, 'province': 'Beijing'})`

##### 命名关键字参数

命名关键字参数定义如下，其中 `*, nkw` 表示的就是命名关键字参数，它是用户想要输入的关键字参数名称，定义方式就是在 `nkw` 前面添加 `*,` ，这个参数的作用主要是**限制调用者可以传递的参数名**。

```python
def function_name(arg1, arg2=v, *arg3, *,nkw, **arg4):
	"""docstring"""
	statement
```

一个实例如下：

```python
# 命名关键字参数
def print_info4(name, age=18, height=178, *, weight, **kwargs):
    '''
    打印信息函数4，加入命名关键字参数
    :param name:
    :param age:
    :param height:
    :param weight:
    :param kwargs:
    :return:
    '''
    print('name: ', name)
    print('age: ', age)
    print('height: ', height)

    print('keyword: ', kwargs)
    print('weight: ', weight)

print_info4('robin', 20, 180, birth='2000/02/02', weight=125)
```

输出结果如下：

```python
name:  robin
age:  20
height:  180
keyword:  {'birth': '2000/02/02'}
weight:  125
```

这里需要注意：

- **加入命名关键字参数后，就不能加入可变参数了**；
- 对于命名关键字参数，传递时候必须**指明该关键字参数名字**，否则可能就被当做其他的参数。

##### 参数组合

通过上述的介绍，Python 的函数参数分为 5 种，位置参数、默认参数、可变参数、关键字参数以及命名关键字参数，而介绍命名关键字参数的时候，可以知道它和可变参数是互斥的，是不能同时出现的，因此这些参数可以支持以下两种组合及其子集组合：

- **位置参数、默认参数、可变参数和关键字参数**
- **位置参数、默认参数、关键字参数以及命名关键字参数**

一般情况下，其实只需要位置参数和默认参数即可，通常并不需要过多的组合参数，否则函数会很难懂。

#### 4.4 匿名函数

 上述介绍的函数都属于同一种函数，即用 `def` 关键字开头的正规函数，Python 还有另一种类型的函数，用 `lambda` 关键字开头的**匿名函数**。

它的定义如下，首先是关键字 `lambda` ，接着是函数参数 `argument_list`，其参数类型和正规函数可用的一样，位置参数、默认参数、关键字参数等，然后是冒号 `:`，最后是函数表达式 `expression` ，也就是函数实现的功能部分。

```python
lambda argument_list: expression
```

一个实例如下：

```python
# 匿名函数
sum = lambda x, y: x + y

print('sum(1,3)=', sum(1, 3))
```

输出结果：

```python
sum(1,3)= 4
```

#### 4.5 变量作用域

Python 中变量是有作用域的，它决定了哪部分程序可以访问哪个特定的变量，作用域也相当于是变量的访问权限，一共有四种作用域，分别是：

- **L(Local)**：局部作用域
- **E(Enclosing)**：闭包函数外的函数中
- **G(Global)**：全局作用域
- **B(Built-in)**：内置作用域(内置函数所在模块的范围)

寻找的规则是 `L->E->G->B` ，也就是优先在局部寻找，然后是局部外的局部(比如闭包)，接着再去全局，最后才是内置中寻找。

下面是简单介绍这几个作用域的例子，除内置作用域：

```python
g_count = 0  # 全局作用域
def outer():
    o_count = 1  # 闭包函数外的函数中
    # 闭包函数 inner()
    def inner():
        i_count = 2  # 局部作用域
```

内置作用域是通过一个名为 `builtin` 的标准模块来实现的，但这个变量名本身没有放入内置作用域，需要导入这个文件才可以使用它，使用代码如下，可以查看预定义了哪些变量：

```python
import builtins
print(dir(builtins))
```

输出的预定义变量如下：

```python
['ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'BlockingIOError', 'BrokenPipeError', 'BufferError', 'BytesWarning', 'ChildProcessError', 'ConnectionAbortedError', 'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError', 'DeprecationWarning', 'EOFError', 'Ellipsis', 'EnvironmentError', 'Exception', 'False', 'FileExistsError', 'FileNotFoundError', 'FloatingPointError', 'FutureWarning', 'GeneratorExit', 'IOError', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError', 'InterruptedError', 'IsADirectoryError', 'KeyError', 'KeyboardInterrupt', 'LookupError', 'MemoryError', 'NameError', 'None', 'NotADirectoryError', 'NotImplemented', 'NotImplementedError', 'OSError', 'OverflowError', 'PendingDeprecationWarning', 'PermissionError', 'ProcessLookupError', 'RecursionError', 'ReferenceError', 'ResourceWarning', 'RuntimeError', 'RuntimeWarning', 'StopAsyncIteration', 'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError', 'TimeoutError', 'True', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning', 'ValueError', 'Warning', 'WindowsError', 'ZeroDivisionError', '__build_class__', '__debug__', '__doc__', '__import__', '__loader__', '__name__', '__package__', '__spec__', 'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'copyright', 'credits', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'exit', 'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'license', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'quit', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip']
```

注意：**只有模块(module)，类(class)以及函数(def, lambda)才会引入新的作用域**，其他代码块(比如 if/elif/else、try/except、for/while)是不会引入新的作用域，在这些代码块内定义的变量，外部也可以使用。

下面是两个例子，一个在函数中新定义变量，另一个在 `if` 语句定义的变量，在外部分别调用的结果：

```python
g_count = 0  # 全局作用域
def outer():
    o_count = 1  # 闭包函数外的函数中
    # 闭包函数 inner()
    def inner():
        i_count = 2  # 局部作用域
        
if 1:
    sa = 2
else:
    sa = 3
print('sa=', sa)
print('o_count=', o_count)
```

输出结果，对于在 `if` 语句定义的变量 `sa` 是可以正常访问的，但是函数中定义的变量 `o_count` 会报命名错误 `NameError` ，提示该变量没有定义。

```python
sa= 2
NameError: name 'o_count' is not defined
```

##### 全局变量和局部变量

**全局变量和局部变量的区别主要在于定义的位置是在函数内部还是外部**，也就是在函数内部定义的是局部变量，在函数外部定义的是全局变量。

**局部变量只能在其被声明的函数内部访问，而全局变量可以在整个程序范围内访问**。调用函数时，所有在函数内声明的变量名称都将被加入到作用域中。如下实例：

```python
# 局部变量和全局变量
total = 3  # 全局变量

def sum_nums(arg1, arg2):
    total = arg1 + arg2  # total在这里是局部变量.
    print("函数内是局部变量 : ", total)
    return total


# 调用 sum_nums 函数
sum_nums(10, 20)
print("函数外是全局变量 : ", total)
```

输出结果：

```python
函数内是局部变量 :  30
函数外是全局变量 :  3
```

##### global 和 nonlocal 关键字

如果在内部作用域想修改外部作用域的变量，比如函数内部修改一个全局变量，**那就需要用到关键字 `global` 和 `nonlocal`** 。

这是一个修改全局变量的例子：

```python
# 函数内部修改全局变量
a = 1

def print_a():
    global a
    print('全局变量 a=', a)
    a = 3
    print('修改全局变量 a=', a)


print_a()
print('调用函数 print_a() 后, a=', a)
```

输出结果：

```python
全局变量 a= 1
修改全局变量 a= 3
调用函数 print_a() 后, a= 3
```

而如果需要修改嵌套作用域，也就是闭包作用域，外部并非全局作用域，则需要用关键字 `nonlocal` ，例子如下：

```python
# 修改闭包作用域中的变量
def outer():
    num = 10

    def inner():
        nonlocal num  # nonlocal关键字声明
        num = 100
        print('闭包函数中 num=', num)

    inner()
    print('调用函数 inner() 后, num=',num)


outer()
```

输出结果：

```
闭包函数中 num= 100
调用函数 inner() 后, num= 100
```

#### 4.6 从模块中导入函数

一般我们会需要导入一些标准库的函数，比如 `os`、`sys` ，也有时候是自己写好的一个代码文件，需要在另一个代码文件中导入使用，导入的方式有以下几种形式：

```python
# 导入整个模块
import module_name
# 然后调用特定函数
module_name.func1()

# 导入特定函数
from module_name import func1, func2

# 采用 as 给函数或者模块指定别名
import module_name as mn
from module_name import func1 as f1

# * 表示导入模块中所有函数
from module_name import *
```

上述几种形式都是按照实际需求来使用，但最后一种方式并不推荐，原因主要是 Python 中可能存在很多相同名称的变量和函数，这种方式可能会覆盖相同名称的变量和函数。最好的导入方式还是导入特定的函数，或者就是导入整个模块，然后用句点表示法调用函数，即 `module_name.func1()` 。



本节的代码例子：https://github.com/ccc013/Python_Notes/blob/master/Practise/function_example.py

------

#### 参考

- 《Python 编程从入门到实践》
- [Python3 函数 | 菜鸟教程](https://www.runoob.com/python3/python3-function.html)
- [超易懂的Python入门级教程(下)，绝对干货！](https://mp.weixin.qq.com/s/WBKgcxfK66Io9oazrepDXA)







