### 编写测试代码

Python基础入门系列最后一篇--如何编写测试。测试可以确保程序面对各种输入都可以按照要求进行工作。

这一部分内容主要是介绍利用 Python 模块 `unittest	` 中的工具来测试代码。

#### 测试函数

首先是给出用于测试的代码，如下所示，这是一个接收姓和名然后返回整洁的姓名的函数：

```python
def get_formatted_name(first, last):
    full_name = first + ' ' + last
    return full_name.title()
```

简单的测试代码：

```python
first = 'kobe'
last = 'bryant'
print(get_formatted_name(first, last)) # 输出 Kobe Bryant
```

在 Python 标准库中的模块 `unittest` 提供了代码测试工具。这里介绍几个名词的含义：

- **单元测试**：用于核实函数的某个方面没有问题；
- **测试用例**：一组单元测试，它们一起核实函数在各种情形下的行为符合要求。
- **全覆盖式测试用例**：包含一整套单元测试，涵盖了各种可能的函数使用方式。

通常，最初只需要对函数的重要行为编写测试即可，等项目被广泛使用时才考虑全覆盖。

接下来就开始介绍如何采用 `unittest` 对代码进行测试。

首先是需要导入 `unittest` 模块，然后创建一个继承 `unittest.TestCase` 的类，并编写一系列类方法对函数的不同行为进行测试，如下代码所示：

```python
import unittest

class NamesTestCase(unittest.TestCase):
    '''
    测试生成名字函数的类
    '''

    def test_first_last_name(self):
        formatted_name = get_formatted_name('kobe', 'bryant')
        self.assertEqual(formatted_name, 'Kobe Bryant')
        
unittest.main()
```

输出结果如下，显示运行的测试样例是 1 个，耗时是 0.001s。

```python
.
----------------------------------------------------------------------
Ran 1 test in 0.001s

OK
```

上述是给了一个可以通过的例子，而如果测试不通过，输出是怎样的呢，如下所示：

```python
# 添加中间名
def get_formatted_name(first, middel, last):
    full_name = first + ' ' + middle + ' ' + last
    return full_name.title()

class NamesTestCase(unittest.TestCase):
    '''
    测试生成名字函数的类
    '''
	# 不能通过的例子
    def test_first_name(self):
        formatted_name = get_formatted_name('kobe', 'bryant')
        self.assertEqual(formatted_name, 'Kobe Bryant')
                
unittest.main()
```

输出结果如下，这里会打印错误发生的地方和错误原因：

```python
E
======================================================================
ERROR: test_first_last_middle_name (__main__.NamesTestCase)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:/Python_Notes/Practise/unittest_practise.py", line 39, in test_first_last_middle_name
    formatted_name = get_formatted_name('kobe', 'bryant')
TypeError: get_formatted_name() missing 1 required positional argument: 'middle'

----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (errors=1)
```

很明显是因为缺少 `middle` 参数，如果希望通过测试，可以将原函数进行如下修改：

```python
def get_formatted_name(first, last, middle=''):
    '''
    接收姓和名然后返回完整的姓名
    :param first:
    :param last:
    :return:
    '''
    if middle:
        full_name = first + ' ' + middle + ' ' + last
    else:
        full_name = first + ' ' + last
    return full_name.title()
```

然后添加新的测试方法，继续运行，就可以测试通过。

```python
def test_first_last_middle_name(self):
    formatted_name = get_formatted_name('kobe', 'bryant', 'snake')
    self.assertEqual(formatted_name, 'Kobe Snake Bryant')
```

#### 测试类

上一小节介绍了给函数写测试的代码，接下来介绍如何编写针对类的测试。

##### 断言方法

在 `unitest.TestCase` 类中提供了很多断言方法，上一小节就采用了 `assertEqual` 这一个判断给定两个参数是否相等的断言方法，下面给出常用的 6 个断言方法：

|          方法           |          用途          |
| :---------------------: | :--------------------: |
|    assertEqual(a, b)    |      核实 a == b       |
|  assertNotEqual(a, b)   |      核实 a != b       |
|      assertTrue(x)      |     核实 x 是 True     |
|     assertFalse(x)      |    核实 x 是 False     |
|  assertIn(item, list)   |  核实 item 在 list 中  |
| assertNotIn(item, list) | 核实 item 不在 list 中 |

这些方法都只能在继承了 `unittest.TestCase` 的类中使用这些方法。

##### 编写针对类的测试

首先，编写用于进行测试的类，代码如下所示，这是一个用于管理匿名调查问卷答案的类：

```python
class AnonymousSurvey():
    '''
    收集匿名调查问卷的答案
    '''

    def __init__(self, question):
        '''

        :param question:
        '''
        self.question = question
        self.responses = []
    
    def show_question(self):
        '''
        显示问卷
        :return: 
        '''
        print(self.question)
    
    def store_response(self, new_response):
        '''
        存储单份调查问卷
        :param new_response: 
        :return: 
        '''
        self.responses.append(new_response)
    
    def show_results(self):
        '''
        显示所有答卷
        :return: 
        '''
        print('Survey results:')
        for response in self.responses:
            print('- ' + response)
   
```

这个类包含三个方法，分别是显示问题、存储单份问卷以及展示所有调查问卷，下面是一个使用例子：

```python
def use_anonymous_survey():
    question = "世上最好的语言是？"
    language_survey = AnonymousSurvey(question)
    # 显示问题
    language_survey.show_question()
    # 添加问卷
    language_survey.store_response('php')
    language_survey.store_response('python')
    language_survey.store_response('c++')
    language_survey.store_response('java')
    language_survey.store_response('go')
    # 展示所有问卷
    language_survey.show_results()


if __name__ == '__main__':
    use_anonymous_survey()
```

输出结果如下：

```
世上最好的语言是？
Survey results:
- php
- python
- c++
- java
- go
```

然后就开始编写对该类的测试代码，同样创建一个类，继承 `unittest.TestCase` ，然后类方法进行测试，代码如下所示：

```python
import unittest

class TestAnonmyousSurvey(unittest.TestCase):

    def test_store_single_response(self):
        '''
        测试保存单份问卷的方法
        :return:
        '''
        question = "世上最好的语言是？"
        language_survey = AnonymousSurvey(question)
        language_survey.store_response('php')

        self.assertIn('php', language_survey.responses)
unittest.main()
```

上述代码采用了 `assertIn` 断言方法来测试函数 `store_response()`。

这里还可以继续测试能否存储更多的问卷，如下所示，测试存储三份问卷：

```python
 def test_store_three_response(self):
     question = "世上最好的语言是？"
     language_survey = AnonymousSurvey(question)
     responses = ['c++', 'php', 'python']
     for response in responses:
     	language_survey.store_response(response)

    for response in responses:
    	self.assertIn(response, language_survey.responses)
```

最后，在 `unittest.TestCase` 中其实包含一个方法 `setUp()` ，它的作用类似类的初始化方法 `__init()__`，它会在各种以 `test_` 开头的方法运行前先运行，所以可以在这个方法里创建对象，避免在每个测试方法都需要创建一遍，所以上述代码可以修改为：

```python
class TestAnonmyousSurvey(unittest.TestCase):

    def setUp(self):
        '''
        创建一个调查对象和一组答案
        :return:
        '''
        question = "世上最好的语言是？"
        self.language_survey = AnonymousSurvey(question)
        self.responses = ['c++', 'php', 'python']

    def test_store_single_response(self):
        '''
        测试保存单份问卷的方法
        :return:
        '''
        self.language_survey.store_response(self.responses[1])

        self.assertIn('php', self.language_survey.responses)

    def test_store_three_response(self):
        for response in self.responses:
            self.language_survey.store_response(response)

        for response in self.responses:
            self.assertIn(response, self.language_survey.responses)

```

运行后，输出结果如下：

```python
..
----------------------------------------------------------------------
Ran 2 tests in 0.000s

OK
```

注意，这里运行成功，打印一个句号，因为是运行两个测试方法成功，所以打印了两个句号；如果运行出错，打印一个 `E` ；测试导致断言失败，打印一个 `F` 。

本文代码：

- https://github.com/ccc013/Python_Notes/blob/master/Practise/unittest_practise.py
- https://github.com/ccc013/Python_Notes/blob/master/Practise/unitest_class_example.py

------

#### 参考

- 《Python编程--从入门到实践》