标题：python-is-cool

原文：https://github.com/chiphuyen/python-is-cool

#### 导读

这篇文章主要是介绍一些 python 的技巧。

采用的python版本是 `3.6+`

#### 1. Lambda, map, filter, reduce

`lambda` 是创建匿名函数，下面是一个使用的例子，其中 `square_fn` 和 `square_ld` 这两个都是相同作用的函数：

```python
def square_fn(x):
    return x * x

square_ld = lambda x: x * x

for i in range(10):
    assert square_fn(i) == square_ld(i)
```

因为快速声明的特点使得 `lambda` 非常适合用于回调函数以及作为一个参数传入其他函数中。此外，它还可以很好的和 `map`, `filter` , `reduce` 这几个函数一起使用。

`map(fn, iterable)` 是将 `iterable` 参数的所有元素都传给函数`fn` ，这里可以作为`iterable`参数的有列表、集合、字典、元祖和字符串，返回的是一个 `map` 对象，例子如下所示：

```python
nums = [1/3, 333/7, 2323/2230, 40/34, 2/3]
nums_squared = [num * num for num in nums]
print(nums_squared)

==> [0.1111111, 2263.04081632, 1.085147, 1.384083, 0.44444444]
```

如果用 `map` 函数作为回调函数，则代码为：

```python
nums_squared_1 = map(square_fn, nums)
nums_squared_2 = map(lambda x: x * x, nums)
print(list(nums_squared_1))

==> [0.1111111, 2263.04081632, 1.085147, 1.384083, 0.44444444]
```

还可以使用多个迭代对象，例如，如果想计算一个简单的线性函数`f(x)=ax+b` 和真实标签 `labels` 的均方差，下面有两个相同作用的实现方法：

```python
a, b = 3, -0.5
xs = [2, 3, 4, 5]
labels = [6.4, 8.9, 10.9, 15.3]

# Method 1: using a loop
errors = []
for i, x in enumerate(xs):
    errors.append((a * x + b - labels[i]) ** 2)
result1 = sum(errors) ** 0.5 / len(xs)

# Method 2: using map
diffs = map(lambda x, y: (a * x + b - y) ** 2, xs, labels)
result2 = sum(diffs) ** 0.5 / len(xs)

print(result1, result2)

==> 0.35089172119045514 0.35089172119045514
```

需要注意的是，`map` 和 `filter` 返回的对象都是迭代器，也就是说它们的数值并没有被存储下来，只是在需要的时候生成，所以如果调用了`sum(diffs)` ，`diffs` 将变为空，如果想保存所有`diffs`的元素，需要转为列表的类型--`list(diffs)`。

`filter(fn, iterable)` 的使用方式和 `map` 一样，不同的是 `fn` 返回的是布尔类型的数值，然后 `filter` 函数返回的就是 `fn` 会返回 `True` 的元素，一个例子如下所示：

```python
bad_preds = filter(lambda x: x > 0.5, errors)
print(list(bad_preds))

==> [0.8100000000000006, 0.6400000000000011]
```

`reduce(fn, iterable, initializer)` 是在我们想对一个列表的元素都迭代地采用一个操作器的使用。比如，我们想计算一个列表的所有元素的乘积：

```python
product = 1
for num in nums:
    product *= num
print(product)

==> 12.95564683272412
```

这等价于：

```python
from functools import reduce
product = reduce(lambda x, y: x * y, nums)
print(product)

==> 12.95564683272412
```

**注意**：

`lambda` 函数的运算时间并不是很好，和用 `def` 定义的有名字函数相比，会稍微慢一些，因此更建议使用带名字的函数。

------

#### 2. 列表操作

`python` 中的列表也是有很多特别的技巧。

##### 2.1 Unpacking

对于展开列表的每个元素，可以这么实现：

```python
elems = [1, 2, 3, 4]
a, b, c, d = elems
print(a, b, c, d)

==> 1 2 3 4
```

也可以这么做：

```python
a, *new_elems, d = elems
print(a)
print(new_elems)
print(d)

==> 1
    [2, 3]
    4
```

##### 2.2 Slicing

反转一个列表可以通过切片方式实现--`[::-1]`：

```python
elems = list(range(10))
print(elems)

==> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(elems[::-1])

==> [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

这个语法 `[x:y:z]` 表示在一个列表中，从索引 `x` 到 `y` 中取出元素，步长是 `z` 。当 `z` 是负数，它表示从后往前，`x` 没有指定的时候，默认从第一个元素开始遍历列表。如果没有指定 `y` ，则默认采用最后一个元素。因此，如果我们希望每隔2个元素进行采样，可以采用 `[::2]`：

```python
evens = elems[::2]
print(evens)

reversed_evens = elems[-2::-2]
print(reversed_evens)

==> [0, 2, 4, 6, 8]
    [8, 6, 4, 2, 0]
```

也可以通过切片的方式来删除列表的元素：

```python
del elems[::2]
print(elems)

==> [1, 3, 5, 7, 9]
```

##### 2.3 Insertion

改变列表中一个元素的代码实现如下所示：

```python
elems = list(range(10))
elems[1] = 10
print(elems)

==> [0, 10, 2, 3, 4, 5, 6, 7, 8, 9]
```

而如果希望修改特定范围内的多个元素，比如用 3 个数值 `20,30,40` 来替换数值 `1` ，代码如下所示：

```python
elems = list(range(10))
elems[1:2] = [20, 30, 40]
print(elems)

==> [0, 20, 30, 40, 2, 3, 4, 5, 6, 7, 8, 9]
```

还可以在索引为0和索引为1之间插入 3 个数值 `[0.2, 0.3, 0.5]`：

```python
elems = list(range(10))
elems[1:1] = [0.2, 0.3, 0.5]
print(elems)

==> [0, 0.2, 0.3, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

##### 2.4 Flattening

通过采用 `sum` 方法来碾平一个嵌套列表的对象：

```python
list_of_lists = [[1], [2, 3], [4, 5, 6]]
sum(list_of_lists, [])

==> [1, 2, 3, 4, 5, 6]
```

但如果嵌套的层次太多，就需要递归的操作，这里介绍另一个通过 `lambda` 实现的方法：

```python
nested_lists = [[1, 2], [[3, 4], [5, 6], [[7, 8], [9, 10], [[11, [12, 13]]]]]]
flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
flatten(nested_lists)

# This line of code is from
# https://github.com/sahands/python-by-example/blob/master/python-by-example.rst#flattening-lists
```

##### 2.5 List vs generator

为了解释列表和生成器的区别，这里用一个创建一个列表的所有字符串的 n-grams 作为例子：

其中一个实现方法是采用滑动窗口：

```python
tokens = ['i', 'want', 'to', 'go', 'to', 'school']

def ngrams(tokens, n):
    length = len(tokens)
    grams = []
    for i in range(length - n + 1):
        grams.append(tokens[i:i+n])
    return grams

print(ngrams(tokens, 3))

==> [['i', 'want', 'to'],
     ['want', 'to', 'go'],
     ['to', 'go', 'to'],
     ['go', 'to', 'school']]
```

在上述例子中，我们需要同时存储所有的 `n-grams`，如果文本是有 `m` 个字符，那么内存大小就是 `O(nm)` ，这在 `m` 很大的时候问题会很大。

因此，可以考虑通过生成器在需要的时候才生成新的 `n-gram` ，所以我们可以创建一个函数 `ngrams` 通过关键词 `yield` 返回一个生成器，此内存只需要 `O(m+n)`：

```python
def ngrams(tokens, n):
    length = len(tokens)
    for i in range(length - n + 1):
        yield tokens[i:i+n]

ngrams_generator = ngrams(tokens, 3)
print(ngrams_generator)

==> <generator object ngrams at 0x1069b26d0>

for ngram in ngrams_generator:
    print(ngram)

==> ['i', 'want', 'to']
    ['want', 'to', 'go']
    ['to', 'go', 'to']
    ['go', 'to', 'school']
```

另外一种方式生成 `n-grams` 是通过切片方式来生成列表 `[0, 1, ..., -n]`, `[1, 2, ..., -n+1]`, ..., `[n-1, n, ..., -1]` ，然后通过 `zip` 来包装到一起：

```python
def ngrams(tokens, n):
    length = len(tokens)
    slices = (tokens[i:length-n+i+1] for i in range(n))
    return zip(*slices)

ngrams_generator = ngrams(tokens, 3)
print(ngrams_generator)

==> <zip object at 0x1069a7dc8> # zip objects are generators

for ngram in ngrams_generator:
    print(ngram)

==> ('i', 'want', 'to')
    ('want', 'to', 'go')
    ('to', 'go', 'to')
    ('go', 'to', 'school')
```

注意，这里生成切片的方法是 `(tokens[...] for i in range(n))` ，而不是 `[tokens[...] for i in range(n)]`，因为 `[]` 是列表生成式，而 `()` 会返回一个生成器。

------

#### 3. 类和魔法方法

 在 python 中，魔法方法是前缀和后缀都带有两个下划线的 `__`，最有名的一个魔法方法可能就是 `__init__` 了，下面是实现一个 `Node` 类，表示一个二叉树：

```python
class Node:
    """ A struct to denote the node of a binary tree.
    It contains a value and pointers to left and right children.
    """
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
```

如果我们要打印一个 `Node` 对象，不过输出结果并非很好解释：

```python
root = Node(5)
print(root) # <__main__.Node object at 0x1069c4518>
```

理想的情况是，可以打印一个节点的数值以及其包含的所有子节点，要实现这个功能，可以采用 `__repr__` 方法，它会返回一个可解释的对象，比如字符串。

```python
class Node:
    """ A struct to denote the node of a binary tree.
    It contains a value and pointers to left and right children.
    """
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self):
        strings = [f'value: {self.value}']
        strings.append(f'left: {self.left.value}' if self.left else 'left: None')
        strings.append(f'right: {self.right.value}' if self.right else 'right: None')
        return ', '.join(strings)

left = Node(4)
root = Node(5, left)
print(root) # value: 5, left: 4, right: None
```

接着，我们可能想进一步实现两个节点的比较数值的功能，这里通过 `__eq__` 实现相等 `==` ，`__lt__`实现小于 `<` ，`__ge__` 实现 大于等于 `>=` 。

```python
class Node:
    """ A struct to denote the node of a binary tree.
    It contains a value and pointers to left and right children.
    """
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __ge__(self, other):
        return self.value >= other.value


left = Node(4)
root = Node(5, left)
print(left == root) # False
print(left < root) # True
print(left >= root) # False
```

在下面这篇文章给出了所有的魔法方法列表：

https://www.tutorialsteacher.com/python/magic-methods-in-python

当然也可以查看官方文档的说明，不过阅读起来会有些难度：

https://docs.python.org/3/reference/datamodel.html#special-method-names

其中推荐以下这些方法：

- `__len__` ：重写 `len()` 方法
- `__str__`：重写`str()` 方法
- `__iter__`：如果想让对象可以迭代，可以继承这个方法，并且还可以调用 `next()` 方法

对于类似 `Node` 这样的类，即我们确定其支持的所有属性（比如对于 `Node` ，这里就是指 `value, left, right` 着三个属性），可以采用 `__slots__` 来表示这些数值，这有利于提升性能和节省内存空间。想更详细了解 `__slots__` ，可以看看这篇 Stackoverflow 上的回答：

https://stackoverflow.com/questions/472000/usage-of-slots/28059785#28059785

```python
class Node:
    """ A struct to denote the node of a binary tree.
    It contains a value and pointers to left and right children.
    """
    __slots__ = ('value', 'left', 'right')
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
```

------

#### 4. 本地命名空间，对象的属性

`locals()` 函数会返回一个字典，它包含了所有定义在本地命名空间的变量，例子如下所示：

```python
class Model1:
    def __init__(self, hidden_size=100, num_layers=3, learning_rate=3e-4):
        print(locals())
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

model1 = Model1()

==> {'learning_rate': 0.0003, 'num_layers': 3, 'hidden_size': 100, 'self': <__main__.Model1 object at 0x1069b1470>}
```

一个对象的所有属性都保存在它的 `__dict__` ：

```python
print(model1.__dict__)

==> {'hidden_size': 100, 'num_layers': 3, 'learning_rate': 0.0003}
```

注意手动将所有参数分配到对应的属性会非常麻烦，特别是在参数列表比较大的时候。为了避免这种情况，可以利用对象的 `__dict__`:

```python
class Model2:
    def __init__(self, hidden_size=100, num_layers=3, learning_rate=3e-4):
        params = locals()
        del params['self']
        self.__dict__ = params

model2 = Model2()
print(model2.__dict__)

==> {'learning_rate': 0.0003, 'num_layers': 3, 'hidden_size': 100}
```

如果对象是通过 `**kwargs` 来进行初始化，会更加的方便，不过`**kwargs` 应该尽量少使用：

```python
class Model3:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

model3 = Model3(hidden_size=100, num_layers=3, learning_rate=3e-4)
print(model3.__dict__)

==> {'hidden_size': 100, 'num_layers': 3, 'learning_rate': 0.0003}
```

------

#### 5. 疯狂的导入

通常会陷入这种疯狂的导入操作`*` 的例子如下所示：

在 `file.py`  文件中

```python
from parts import *
```

这个写法非常不负责任，它是将另一个模块的一切都导入到当前的模块，包括那个模块的导入的内容，比如说，`parts.py` 模块可能是这样的：

```python
import numpy
import tensorflow

class Encoder:
    ...

class Decoder:
    ...

class Loss:
    ...

def helper(*args, **kwargs):
    ...

def utils(*args, **kwargs):
    ...
```

由于 `parts.py` 没有指定 `__all__` ，所以 `file.py` 会导入 `Encoder, Decoder, Loss, utils, helper`，以及 `numpy` 和 `tensorflow` 。

如果我们只想让 `Encoder, Decoder, Loss` 被导入到另一个模块中使用，那么就需要指定 `__all__` 参数：

```python
 __all__ = ['Encoder', 'Decoder', 'Loss']
import numpy
import tensorflow

class Encoder:
    ...
```

通过上述代码，当有另一个文件也是直接采用 `from part import *` 的做法，那么只会导入给定的  `Encoder, Decoder, Loss` ，同时 `__all__` 也是对一个模块的一个概览。









------

### 参考

- https://www.tutorialsteacher.com/python/magic-methods-in-python
- https://docs.python.org/3/reference/datamodel.html#special-method-names
- https://stackoverflow.com/questions/472000/usage-of-slots/28059785#28059785























































