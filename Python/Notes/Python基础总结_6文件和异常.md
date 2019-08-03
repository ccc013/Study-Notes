### 6. 文件和异常

目录：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/6.%20%E6%96%87%E4%BB%B6%E5%92%8C%E5%BC%82%E5%B8%B8.png)

#### 1. 文件

##### 简介

Python 中读取、写入文件，都可以通过方法 `open()` 实现，该方法用于打开一个文件，然后返回文件对象，如果文件不存在或者无法打开，会报错 `OSError`。

`open` 方法的语法如下：

```python
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
```

几个参数说明:

- **file**：必需，文件路径，相对或者绝对路径皆可；
- **mode**：可选，文件打开的模式
- **buffering**：设置缓冲
- **encoding**：一般采用 `utf8` 
- **errors**：报错级别
- **newline**：区分换行符
- **closefd**：传入的 file 参数类型

常用的文件打开模式如下：

| 操作模式 |             具体含义             |
| :------: | :------------------------------: |
|    r     |      读取(默认文件打开模式)      |
|    w     |     写入（会截断前面的内容)      |
|    x     | 写入，如果文件已经存在会产生异常 |
|    a     |  追加，将内容写入到已有文件末尾  |
|    b     |            二进制模式            |
|    t     |          文本模式(默认)          |
|    +     |      更新(既可以读又可以写)      |

其中 `r` 、 `w`  、`a` 是三选一，表明读取或者写入，然后可以添加其他几种模型，即还存在：

- `rb` , `r+`, `rb+`
- `wb`, `w+`, `wb+`
- `ab`, `a+`, `ab+`

对于 `open` 方法返回的 `file` 文件对象，它常用函数有：

- **close()**：关闭文件
- **flush()**：将内部缓冲区数据立刻写入文件
- **read([size])**：从文件读取指定的字节数，如果没有或者是负数值，则读取所有
- **readline()**：读取整行，包含换行符 `\n` 字符
- **readlines([sizeint])**：读取所有行并返回列表，若给定 `sizeint>0`，返回总和大约为 `sizeint`字节的行, 实际读取值可能比 `sizeint` 较大, 因为需要填充缓冲区。
- **write(str)**：将字符串写入文件，返回的是写入字符的长度
- **writelines(sequence)**：向文件写入一个序列字符串列表，如果需要换行，需要自己添加每行的换行符
- **seek(offset[, whence])**：设置文件当前位置
- **tell()**：返回文件当前位置。
- **truncate([size]**：从文件的首行首字符开始截断，截断文件为 `size` 个字符，无 `size` 表示从当前位置截断；截断之后后面的所有字符被删除，其中 Windows 系统下的换行代表 2个字符大小。



##### 读取文本文件

读取文本文件，必须传入文件路径，然后打开模式指定为 `r` ，接着就就是通过 `encoding` 参数指定编码，当然不设置这个编码参数，它默认值是 `None` ，读取文件将采用操作系统默认的编码，通常如果文件内容不带有中文，这种方法是没问题的，如果带有中文内容，则必须指定 `encoding='utf8'` 才能正常打开文件。

以下是一个使用例子：

```python
# 方法1
f = open('test.txt', 'r')
print(f.read())
f.close()
```

输出结果：

```
life is short, I use Python.
Machine Learning
Computer Vision
```

这是第一种使用方法，这种方法的问题就是如果忘记调用 `close` 方法关闭文件，会出现错误，因此推荐使用上下文语法，通过 `with` 关键字指定**文件对象的上下文环境**并在离开上下文环境时**自动释放文件资源**，此外，读取文件内容，可以直接调用 `read()` 方法，也可以采用 `for-in` 循环读取：

```python
# 方法2
with open('test.txt', 'r') as fr:
    print(fr.read())
# 方法3 读取文件也可以采用 for-in 循环逐行读取
with open('test.txt', 'r') as fr:
    for line in fr:
        print(line.strip())
```

#### 2. 异常

Python 有两种错误很容易辨认：**语法错误和异常**。

语法错误也称为解析错，比如下面这个例子：

```python
while True
    print('hello')
```

会报错误，这里指出语法错误的地方，就是缺少非常重要的冒号 `:` 

```python
File "<ipython-input-41-b385275d6655>", line 1
    while True
              ^
SyntaxError: invalid syntax
```

而异常，是运行期检测到的错误，即解析成功后，开始运行时的错误，比如执行除法操作时候，除数是 0 的情况；读取文件的时候，文件路径错误；变量没有定义的情况等等。

##### 异常处理

异常的处理，就是采用 `try-exception` 语句，例子如下：

```python
def read_file(file_name):
    file_contents = None
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            file_contents = f.read()
    except FileNotFoundError:
        print('无法打开指定的文件!')
    except LookupError:
        print('指定了未知的编码!')
    except UnicodeDecodeError:
        print('读取文件时解码错误!')
    else:
        print('读取文件成功')
    finally:
        print('程序执行完毕')
        
    return file_contents
```

上述代码中，`try` 语句是这样执行的：

- 先执行 `try` 语句，即 `try` 和 `except` 之间的句子
- 如果没有异常发生，就忽略 `except` ，然后按顺序执行 `else` 语句，`finally` 语句
- 如果发生异常，那就忽略 `try` 语句中发生异常部分后面的代码，然后执行和异常类型一样的 `except` 语句，之后执行 `finally` 语句
- 如果一个异常没有与任何的 `except` 匹配，那么这个异常将会传递给上层的 `try` 中。

一个 `try` 语句可能包含多个 `except` 子句，分别来处理不同的特定的异常。**最多只有一个分支会被执行**。

**处理程序将只针对对应的 `try` 子句中的异常进行处理**，而不是其他的 try 的处理程序中的异常。

一个 `except` 子句可以**同时处理多个异常**，这些异常将被放在一个括号里成为一个元组，例如:

```python
except (FileNotFoundError, LookupError, UnicodeDecodeError) as e:
        print(e)
```

上述情况，可以添加一个 `except` 语句，忽略异常的名称，它将被当作通配符使用。你可以使用这种方法打印一个错误信息，然后再次把异常抛出，如下所示：

```python
import sys

def read_file2(file_name):
    file_contents = None
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            file_contents = f.read()
    except (LookupError, UnicodeDecodeError) as e:
        print(e)
    except:
        print('发生了未知的错误', sys.exc_info()[0])
    else:
        print('读取文件成功')
    finally:
        print('程序执行完毕')
        
    return file_contents
```

不过，其实还有一种比较偷懒的写法，直接用 `Exception` 捕获所有异常：

```python
import sys
def read_file3(file_name):
    file_contents = None
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            file_contents = f.read()
    except Exception as e:
        print(e)
    else:
        print('读取文件成功')
    finally:
        print('程序执行完毕')
        
    return file_contents
```

当然，推荐的写法还是前面几种写法，这样可以针对不同异常情况有不同的做法。

另外，这里的 `else` 语句是 `try` 语句执行成功后，继续执行的语句，而 `finally` 则是无论是否发生异常，都会执行的语句，它是定义了**无论任何情况都会执行的清理行为**。

有一些方法是有预定义的清理行为，比如说上述说到的关键词 `with` 语句，就定义了无论文件操作如何，都会执行关闭文件对象的行为

这两个语句是可选择的，不是使用的语句。

##### 抛出异常

上述的异常处理，在出现异常后，是可以继续执行后续的代码(`try-exception` 后面的语句)，即不会终止程序的执行，但如果希望发生异常就终止程序运行，可以采用 `raise` 关键字，如下代码所示：

```python
try:
    with open(file_name, 'r', encoding='utf-8') as f:
        file_contents = f.read()
except Exception as e:
    print(e)
    raise
```

或者如下所示，`raise` 后面加上指定的异常名称和参数

```python
raise NameError('HiThere')
```

最后，更多异常处理的例子可以查看 [《总结：Python中的异常处理》](https://segmentfault.com/a/1190000007736783)，文章给出了关于异常的最佳实践：

1. **只处理你知道的异常**，避免捕获所有异常然后吞掉它们。
2. 抛出的异常**应该说明原因**，有时候你知道异常类型也猜不出所以然。
3. 避免在 `catch` 语句块中干一些没意义的事情，**捕获异常也是需要成本的**。
4. **不要使用异常来控制流程**，那样你的程序会无比难懂和难维护。
5. 如果有需要，**切记使用 `finally` 来释放资源**。
6. 如果有需要，**请不要忘记在处理异常后做清理工作或者回滚操作**。



#### 3. 更多文件和异常的例子

介绍完文件和异常，接下来介绍更多的文件操作，搭配异常处理。

##### 写入文件

写入文件，只需要设置文件打开模式是写入模型，即 `w` ，代码例子如下所示，这里实现读取一个文件的内容，然后写入到一个新的文件中。

```python
def save_to_file(input_file, outputfile, write_mode='w'):
    file_contents = read_file(input_file)
    try:
        with open(outputfile, write_mode, encoding='utf-8') as fw:
            fw.write(file_contents)
    except IOError as ioerror:
        print(ioerror)
        print('写入文件出错')
input_file = 'test.txt'
output_file= 'new_test.txt'
save_to_file(input_file, output_file, write_mode='w')
```



##### 读取二进制文件

读写文本文件的例子都有了，接下来就是二进制文件的读取和写入，这里实现一个复制图片文件的功能，如下所示，读取和保存图片时候，采用的文件模型分别是 `rb` 和 `wb` 。

```python
def copy_image_file(input_image, output_image):
    try:
        with open(input_image, 'rb') as img1:
            data = img1.read()
            print(type(data))
        with open(output_image, 'wb') as out:
            out.write(data)
    except FileNotFoundError as e:
        print('指定的文件无法打开--{}'.format(input_image))
    except IOError as e:
        print('读写文件出现错误')
    finally:
        print('程序执行结束')
```

使用例子：

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
input_image = 'example.jpg'
output_image = 'copyed.jpg'
copy_image_file(input_image, output_image)
```

采用的用于复制的图片如下所示：



这里，在 `jupyter notebook` 中展示图片的代码如下：

```python
%matplotlib inline
img = Image.open(input_image)
print(np.asarray(img, dtype=np.uint8).shape)
plt.imshow(img);
```

这里最好一行代码 `plt.imshow(img)`，如果不加分号，会输出图片变量的信息，即 `<matplotlib.image.AxesImage at 0x1d0d1161dd8>`，然后再显示图片，而加分号，则直接显示图片。

##### 读写 JSON 文件

上述介绍了如何保存文本数据和二进制数据到文件中，但如果我们希望保存的是列表或者字典的数据，那么可以选择采用 `JSON` 格式。

`JSON` 是 `JavaScript Object Notion` 的缩写，现在广泛应用于跨平台跨语言的数据交换，因为它是纯文本，任何编程语言都可以狐狸纯文本。更多关于它的介绍，可以查看官网--http://json.org/

`JSON`  的数据类型和 Python 数据类型的对应关系分别如下面两张表所示：

JSON->Python:

|        JSON        |    Python    |
| :----------------: | :----------: |
|       object       |     dict     |
|       array        |     list     |
|       string       |     str      |
| number(int / real) | int / float  |
|    true / false    | True / False |
|        null        |     None     |

Python->JSON

|                 Python                 |     JSON     |
| :------------------------------------: | :----------: |
|                  dict                  |    object    |
|              list, tuple               |    array     |
|                  str                   |    string    |
| int, float, int- & float-derived Enums |    number    |
|              True / False              | true / false |
|                  None                  |     null     |

在 Python 中，使用 `json` 库就可以保存或者读取 JSON 格式的文件，代码例子如下：

```python
import json
# 先生成一个json文件
test_dict = {
    'name': 'python',
    'age': 40,
    'package': ['os', 'json', 'sys'],
    'features': {
        'data_analysis': ['pandas', 'matplotlib'],
        'deep_learning': ['scikit-learn', 'tensorflow', 'pytorch', 'keras']
    }
}
try:
    with open('test_data.json', 'w', encoding='utf-8') as fw:
        json.dump(test_dict, fw, indent=4, separators=(',', ': '))
except IOError as e:
    print(e)
finally:
    print('程序执行完毕')
```

这是一个保存为 JSON 文件的代码，调用 `dump()` 方法，然后读取代码如下所示：

```python
# 读取 json 文件
try:
    with open('test_data.json', 'r') as fr:
        contents = json.load(fr)
        for key, val in contents.items():
            print('{}: {}'.format(key, val))
except IOError as e:
    print(e)
print('读取完毕')
```

这里调用方法 `load()` 方法。

在 `json` 库中比较重要的是下面四个方法：

- `dump` ：将 Python 对象按照 JSON 格式序列化到文件中
- `dumps` ：将 Python 对象处理为 JSON 格式的字符串
- `load`：将文件中的 JSON 数据反序列化为 Python 对象
- `loads`：将字符串内容反序列化为 Python 对象

这里面，只要方法以 `s` 结尾，那就是和字符串有关系，而如果不带，那就是和文件有关系了。

这里的序列化和反序列化，其中序列化就是指将数据结果或者对象状态转换为可以存储或者传输的形式，也就是一系列字节的形式，而从这些字节提取数据结构的操作，就是反序列化。

在 Python 中，序列化和反序列化还可以采用 `pickle` 和 `shelve` 两个库，但它们仅适用于 Python，不能跨语言。

##### 删除文件或者文件夹

对于删除文件，可以采用这两种方法，分别用到 `os` 和 `pathlib`  两个模块

- `os.remove()` 
- `pathlib.Path.unlink()` ，这个方法也可以删除链接

删除文件夹有三种方法，不过前两者都只能删除空文件夹

- `os.rmdir()` 只能删除空的文件夹
- `pathlib.Path.rmdir()` 删除空文件夹 
- `shutil.rmtree()` 可以**删除非空文件夹**



#### 4. pathlib 模块

最后介绍一个在 Python 3.4 版本加入的新模块--`pathlib` ，这是一个可以很好处理文件相关操作的模块。

首先对于文件路径，最大的问题可能就是 `Unix` 系统和 `Windows` 系统采用的斜杠不同，前者是 `/` ，而后者是 `\` ，因此之前的处理文件路径拼接的方式，可以采用 `os.path.join` 方法，例如：

```python
data_folder = 'source_folder/python/'
file_path = os.path.join(data_folder, 'abc.txt')
print(file_path)
# 输出 source_folder/python/abc.txt
```

但现在有了一个更加简洁的方法，就是采用 `pathlib` 模块的 `Path` 方法，如下所示：

```python
from pathlib import Path
data_folder = Path('source_folder/python/')
file_path = data_folder / 'abc.txt'
print(file_path)
# 输出 source_folder\python\abc.txt
```

需要注意两件事情：

1. 采用 `pathlib` 的方法，**应该使用 `/` 斜杠**，`Path()` 方法会自动根据当前系统修改斜杠，比如上述例子就更改为 `Windows` 的 `\` 
2. 文件路径的拼接，**直接采用 `/` 即可**，不需要再写成 `os.path.join(a,b)` ，更加方便简洁。

当然，`pathlib` 并不只是有这个用途，它还可以完成更多事情：

读取文件内容不需要打开和关闭文件，如下所示，直接调用 `read_text` 方法即可读取内容

```python
data_folder = Path('./')
file_path = data_folder / 'test.txt'
print(file_path.read_text())
```

`pathlib` 可以更加快速容易实现大部分标准的文件操作，如下所示，分别执行打印文件路径，文件后缀，文件名（不包含文件后缀部分）、判断文件是否存在的四个操作。

```python
filename = Path('./test.txt')
print(filename.name) # 输出 test.txt
print(filename.suffix) # 输出 .txt
print(filename.stem)   # 输出 test

if not filename.exists():
    print('file does not exist!')
else:
    print('file exists')
```

可以显式将 `Unix` 路径转换为 `Windows` 形式的：

```python
from pathlib import PureWindowsPath
filename = Path('source/test.txt')
path_on_windows = PureWindowsPath(filename)
print(path_on_windows) # 输出 source\test.txt
```

当然，对于上述代码，如果希望安全的处理斜杠，可以显式定义为 `Windows` 格式的路径，然后 `pathlib` 会将其转换为可以在当前系统使用的形式：

```python
filename = PureWindowsPath('source/test.txt')
correct_path = Path(filename)
print(correct_path)
# 在 windows 输出 source\test.txt
# 在 Mac/Linux 输出 source/test.txt
```

最后，还可以利用 `pathlib` 将相对路径的文件解析生成如 `file://` 形式的 urls，如下所示：

```python
import webbrowser
filename = Path('./test.txt')
webbrowser.open(filename.absolute().as_uri())
```

当然，`pathlib` 的用法还有不少，更多的用法可以查看官网的介绍：

https://docs.python.org/3/library/pathlib.html

代码都上传到 Github：

https://github.com/ccc013/Python_Notes/blob/master/Practise/file_and_exception_example.ipynb

------

#### 参考

- [Python3 File(文件) 方法](https://www.runoob.com/python3/python3-file-methods.html)
- [Python3 OS 文件/目录方法](https://www.runoob.com/python3/python3-os-file-methods.html)
- [Python3 错误和异常](https://www.runoob.com/python3/python3-errors-execptions.html)
- [Day11--文件和异常](https://github.com/jackfrued/Python-100-Days/blob/master/Day01-15/11.%E6%96%87%E4%BB%B6%E5%92%8C%E5%BC%82%E5%B8%B8.md)
- [JSON 的官方网站](http://json.org/)
- [《总结：Python中的异常处理》](https://segmentfault.com/a/1190000007736783)
- https://stackoverflow.com/questions/6996603/delete-a-file-or-folder
- https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
- https://docs.python.org/3/library/pathlib.html