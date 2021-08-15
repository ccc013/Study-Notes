

> 原文：https://medium.com/fintechexplained/everything-about-python-from-beginner-to-advance-level-227d52ef32d2
>
> 作者：[Farhad Malik](https://medium.com/@farhadmalik84)

本文主要的目标是介绍 Python 的所有关键知识点，并保证每个知识点都足够的简洁明了，清晰有意义。

阅读本文不需要你有任何的编程基础知识，本文会让你快速掌握所有必需的概念。

本文将包含 25 个关键的知识点。让我们开始吧！

### 1. Python 简介

#### 什么是 Python

- 这是一门解释型高级面向对象的动态类型的脚本语言；
- `Python` 解释器一次会读取一行代码，然后将它解释为低级的机器语言(字节代码)并执行；
- 结果就是经常会遭遇**运行时错误**。

#### Python 的优点

- `Python` 是目前最流行的语言，因为它易于理解和上手；
- 它是一门面向对象的编程语言，同时也可以写函数式的代码；
- 它是一门可以减小商业和开发者间差异的编程语言；
- 相比其他如 `C#`、`Java` 语言，`Python` 程序落地的时间更快；
- 目前有大量的机器学习和数据分析的 `Python` 库和框架；
- 还有大量的学习社区和书籍供 `Python` 开发者学习和交流；
- 可以实现几乎所有的应用，包括数据分析、UI 设计等；
- 它可以不需要声明变量类型，提高了开发一款应用的效率。

#### Python 的缺点

- 运行速度比 `C++`、`C#`、`Java` 慢。这是缺乏即时优化器；
- 空格缩减的句法限制会给初学者制造一些困难；
- 没有提供如同 `R` 语言的先进的统计产品；
- 不适合在低级系统和硬件上开发

#### Python 工作原理

下图展示了 `Python` 是如何工作的：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/Python_beginner_to_advance.png)

这里的关键就是**解释器**，它将高级的 `Python` 语言转换为低级的机器语言。

### 2. 变量--对象类型和领域

- 变量用于保存一些程序用到的信息，比如用户的输入、程序的局部状态信息等等；
- 在 `Python`，变量等同于它的变量名。

`Python` 中的标准数据类型分别是**数字、字符串、集合、列表、元组和字典**，总共六种。

#### 声明和赋值给变量

声明和赋值给变量的代码如下所示，这个操作在 `Python` 也被称为**绑定(binding)**。

```python
myFirstVariable = 1
mySecondVariable = 2
myFirstVariable = "Hello You"
```

注意，在上述代码中，先后将整数 `1` 和字符串 `Hello You` 赋值给了变量 `myFirstVariable` ，这主要是因为 `Python` 的数据类型是动态定义的。

#### 数字

- 支持整数、小数、浮点数
- 长整数会有一个后缀 `L`，比如 `9999999999999L`。

```python
value = 1 #integer
value = 1.2 #float with a floating point
```

#### 字符串

- 文本信息。字符串是单词的序列；
- 字符串是引号包围的一串字符；
- 字符串是可变的，初始后还可以修改；
- 字符串变量被赋予一个新数值时，`Python` 会创建一个新对象来保存该数值

```python
name = 'farhad'
a = 'me'
# 会报类型错误，a是一个字符串，而非列表
a[1]='y'
```

#### 变量的作用域

##### 局部作用域(Local Scope)

- 声明于 `for` / `while` 循环或者一个函数内的变量
- 在代码块外，就无法访问局部变量

```python
def some_funcion():
  TestMode = False
# 报错，因为变量 TestMode 只能在 some_function()内部访问  
print(TestMode) 
```

##### 全局作用域(Global Scope)

- 在任何函数中都可以访问的变量。它们保存在 `__main__` 结构下；
- 在函数外声明一个全局变量，但要修改全局变量，必须加上关键字 `global`，如下所示：

```python
TestMode = True
def some_function():
  global TestMode
  TestMode = False
some_function()
print(TestMode) # 返回 False
```

上述代码例子展示了如何修改全局变量，而如果删除 `global TestMode` ，那么打印的结果就是 `True`，因为此时 `some_function()` 中修改的只是局部变量，而非全局变量 `TestMode`。

注意，在后面的模块 `modules` 知识点中会介绍更多这部分内容，但如果希望在多个模块文件中共享一个全局变量，可以创建一个共享的模块文件，如 ` configuration.py`，然后在文件中声明需要共享的所有全局变量，然后在其他文件中导入该文件即可。

##### 获取变量类型

获取变量类型可以使用函数 `type`

```python
type('farhad')
# 返回 <type 'str'>
```

### 3. 运算符

#### 算术运算符

- 支持基本的加减乘除，`*, /, +, -`;
- 支持向下取整的除法
- 支持乘方操作 `**`
- 支持取模操作 `%` ，即返回除法的余数

```python
# 返回 0.333
1//3 
# 返回 0，向下取整
1/3 
# 乘方
2**3 = 2 * 2 * 2 = 8
# 取模
7%2 = 1
```

#### 字符串运算符

对于字符串，支持以下的操作：

```python
# 连接字符串
print('A' + 'B') 
# 重复字符
print('A'*3)
# 切片
y = 'Abc'
y[:2] = ab
y[1:] = c
y[:-2] = a
y[-2:] = bc
# 反转
x = 'abc'
x = x[::-1]
# 负数索引，访问最后一个字符
y = 'abc'
print(y[:-1])  # 输出 c
# 查找索引
name = 'farhad'
index = name.find('r') # 输出 2
name = 'farhad'
# 指定查找第二个 a
index = name.find('a', 2) # 输出 4

```

**转换**

- `str(x)`: 转为字符串
- `int(x)`: 转为整数
- `float(x)`:  转为浮点数

#### 集合运算符

集合支持的运算如下：

```python
# 定义一个集合
a = {1, 2, 3}
# 交集,也可以用 &
a = {1,2,3}
b = {3,4,5}
c = a.intersection(b)
# 集合差,也可以用 -
a = {1,2,3}
b = {3,4,5}
c = a.difference(b)
# 并集,也可以用 |
a = {1,2,3}
b = {3,4,5}
c = a.union(b)
```

**三元运算符**

定义为`[If True] if [Expression] Else [If False]` ，例子如下：

```python
Received = True if x = 'Yes' else False
```

### 4. 注释

#### 单行注释

```python
#this is a single line comment
```

#### 多行注释

~~~python
```this is a multi
line
comment```
~~~

### 5. 表达式

表达式可以实现布尔运算，如下所示：

- 相等：`==`
- 不等于：`!=`
- 大于：`>`
- 小于：`<`
- 大于等于：`>=`
- 小于等于：`<=`

### 6. 序列化

**序列化**：指将一个对象转换为字符串并将字符串存储到一个文件中；

反过来称为**反序列化(unpickling)**。











