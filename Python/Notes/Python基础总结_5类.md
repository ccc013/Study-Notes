### 5.面向对象

#### 5.1 简介

先简单介绍一些名词概念。

- **类**：用来描述具有**相同的属性和方法的对象的集合**。它定义了该集合中每个对象所共有的属性和方法。对象是类的实例。
- **类方法**：类中定义的函数。
- **类变量**：类变量在整个实例化的对象中是**公用的**。类变量**定义在类中且在函数体之外**。类变量**通常不作为实例变量使用**。
- **数据成员**：类变量或者实例变量用于处理类及其实例对象的相关的数据。
- **方法重写**：如果从父类继承的方法不能满足子类的需求，可以对其进行改写，这个过程叫**方法的覆盖（override）**，也称为**方法的重写**。
- **局部变量**：定义在方法中的变量，只作用于当前实例的类。
- **实例变量**：在类的声明中，属性是用变量来表示的。这种变量就称为实例变量，是在类声明的内部但是在类的其他成员方法之外声明的。
- **继承**：即一个**派生类（derived class）继承基类（base class）的字段和方法**。继承也允许把一个派生类的对象作为一个基类对象对待。例如，有这样一个设计：一个 Dog 类型的对象派生自 Animal 类，这是模拟"是一个（is-a）"关系（例图，Dog 是一个 Animal）。
- **实例化**：创建一个类的实例，类的具体对象。
- **对象**：通过类定义的数据结构实例。**对象包括两个数据成员（类变量和实例变量）和方法**。


Python中的类提供了面向对象编程的所有基本功能：**类的继承机制允许多个基类，派生类可以覆盖基类中的任何方法，方法中可以调用基类中的同名方法**。

对象可以包含任意数量和类型的数据。

#### 5.2 类定义

下面是简单定义一个类：

```python
# 定义一个动物类别
class Animal(object):
    # 类变量
    eat = True
    
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

     # 类方法
    def run(self):
        return 'Animal run!'
# 实例化类
anm = Animal('animal', 'male')

# 访问类的属性和方法
print("Animal 类的属性 eat 为：", anm.eat)
print("Animal 类的方法 run 输出为：", anm.run())
```

输出结果：

```
Animal 类的属性 eat 为： True
Animal 类的方法 run 输出为： Animal run!
```

上述是一个简单的类的定义，通常一个类需要有关键字 `class` ，然后接一个类的名字，然后如果是 `python2.7` 是需要如例子所示加上圆括号和 `object` ，但在 `python3` 版本中，其实可以直接如下所示：

```python
class Animal:
```



##### 构造方法和特殊参数 self 的表示

然后 `__init__` 是**构造方法**，即在进行**类实例化**的时候会调用该方法，也就是 `anm = Animal('animal', 'male')`。

此外，对于类的方法，第一个参数也是必须带上的参数，按照惯例名称是 `self` ，它代表的是类的实例，也就是指向实例本身的引用，让实例本身可以访问类中的属性和方法。如下代码所示：

```python
class Test:
    def prt(self):
        print(self)
        print(self.__class__)
 
t = Test()
t.prt()
```

输出结果：

```python
<__main__.Test object at 0x000002A262E2BA20>
<class '__main__.Test'>
```

可以看到 `print(self)` 的结果是输出当前对象的地址，而 `self.__class__` 表示的就是类。

刚刚说了 `self` 只是惯例取的名称，换成其他名称也可以，如下所示：

```python
# 不用 self 名称
class Test2:
    def prt(sss):
        print(sss)
        print(sss.__class__)

t2 = Test2()
t2.prt()
```

输出结果是一样的，类实例的地址改变了而已。

```
<__main__.Test2 object at 0x000001FB7644BBA8>
<class '__main__.Test2'>
```

##### 类方法

类方法和构造方法一样，首先是关键字 `def` ，接着就是参数第一个必须是 `self` ，表示类实例的参数。

```python
#类定义
class people:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))
 
# 实例化类
p = people('runoob',10,30)
p.speak()
```

输出结果

```
runoob 说: 我 10 岁。
```

#### 5.3 继承

继承的语法定义如下：

```python
class DerivedClassName(BaseClassName1,BaseClassName2,...):
    <statement-1>
    .
    .
    .
    <statement-N>
```

需要注意：

- 圆括号中**基类的顺序**，当基类含有相同方法名，子类没有指定(即类似 `BaseClass1.method1()`），python 会从左到右搜索继承的基类是否包含该方法；
- 基类和子类必须**定义在一个作用域内**；

下面给出一个代码例子，基类定义还是上一节中的 `people` 类别，这次定义一个子类 `student` 

```python
# 单继承示例
class student(people):
    grade = ''

    def __init__(self, n, a, w, g):
        # 调用父类的构造方法
        people.__init__(self, n, a, w)
        self.grade = g

    # 覆写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级" % (self.name, self.age, self.grade))


s = student('ken', 10, 60, 3)
s.speak()
```

输出结果

```
ken 说: 我 10 岁了，我在读 3 年级
```

这是一个单继承，即继承一个基类的示例，子类的构造方法必须先调用基类(父类)的构造方法：

```python
# 调用父类的构造方法
people.__init__(self, n, a, w)
```

另一种调用基类的构造方法，利用 `super()` 函数：

```python
super.__init__(self, n, a, w)
```



##### 方法重写

上述例子还重写了基类的方法 `speak()`。

方法重写是在基类的方法无法满足子类的需求时，在子类重写父类的方法。

##### 多继承

python 也支持多继承，下面是一个例子，继续沿用刚刚定义的一个类 `student` ，然后再重新定义一个基类 `speaker`

```python
#另一个类，多重继承之前的准备
class speaker():
    topic = ''
    name = ''
    def __init__(self,n,t):
        self.name = n
        self.topic = t
    def speak(self):
        print("我叫 %s，我是一个演说家，我演讲的主题是 %s"%(self.name,self.topic))
 
#多重继承
class sample(speaker,student):
    a =''
    def __init__(self,n,a,w,g,t):
        student.__init__(self,n,a,w,g)
        speaker.__init__(self,n,t)
 
test = sample("Tim",25,80,4,"Python")
test.speak()   #方法名同，默认调用的是在括号中排前地父类的方法
```

输出结果：

```
我叫 Tim，我是一个演说家，我演讲的主题是 Python
```

而如果想指定任意父类的方法，可以添加下面这段代码：

```python
# 显示调用 student 父类的 speak 方法
def speak(self):
    super(student, self).speak()
```

上面介绍过了， `super()` 函数是调用父类的一个方法，可以直接 `super().method()` ，但如果是多继承并且指定父类的话，就如上述所示，添加父类名字以及 `self` 来表示类实例。

另外，`python2.7` 调用 `super()` 方法，也需要传入父类名字和 `self` 两个参数。

#### 5.4 类属性与方法

属性和方法的**访问权限**，即可见性，有三种，**公开、受保护以及私有**，私有方法和私有属性如下定义：

- **类的私有属性**：两个下划线开头，声明该属性私有，不能在类的外部被使用或直接访问，而在类内部的方法中使用时：`self.__private_attrs` 

- **类的私有方法**：两个下划线开头，声明该方法为私有方法，只能在类的内部调用 ，不能在类的外部调用。**self.__private_methods**。

而如果是受保护的属性或者方法，则是一个下划线开头，例如 `_protected_attr` 。

下面是一个简单的示例：

```python
class JustCounter:
    __secretCount = 0  # 私有变量
    publicCount = 0  # 公开变量

    def count(self):
        self.__secretCount += 1
        self.publicCount += 1
        print(self.__secretCount)
        self.__count()

    def __count(self):
        print('私有方法')


counter = JustCounter()
counter.count()
counter.count()
print(counter.publicCount)
print(counter.__secretCount)  # 报错，实例不能访问私有变量
print(counter.__count())
```

输出结果

```python
1
私有方法
2
私有方法
2
```

调用私有属性会报错：

```python
AttributeError: 'JustCounter' object has no attribute '__secretCount'
```

调用私有方法会报错：

```python
AttributeError: 'JustCounter' object has no attribute '__count'
```

类的属性不仅可以是变量，也可以是类实例作为一个属性，例子如下所示：

```python
class TimeCounter:
    def __init__(self):
        print('timer')


class JustCounter:
    __secretCount = 0  # 私有变量
    publicCount = 0  # 公开变量

    def __init__(self):
        self.timer = TimeCounter()

    def count(self):
        self.__secretCount += 1
        self.publicCount += 1
        print(self.__secretCount)
        self.__count()

    def __count(self):
        print('私有方法')


counter = JustCounter()
counter.count()
counter.count()
print(counter.publicCount)
```

同样继续采用 `JustCounter` 类，只是新定义 `TimeCounter` ，并在 `JustCounter` 调用构造方法，实例化一个 `TimeCounter` 类，输出结果：

```
timer
1
私有方法
2
私有方法
2
```

#### 5.5 练习

最后是来自 [Python-100-Days--Day08面向对象基础](https://github.com/jackfrued/Python-100-Days/blob/master/Day01-15/08.%E9%9D%A2%E5%90%91%E5%AF%B9%E8%B1%A1%E7%BC%96%E7%A8%8B%E5%9F%BA%E7%A1%80.md) 的两道练习题：

##### 定义一个简单的数字时钟

这个例子将采用受保护的属性，即属性名字以单下划线开头，所以初始化的构造方法如下：

```python
from time import sleep


class Clock(object):
    """数字时钟"""

    def __init__(self, hour=0, minute=0, second=0):
        '''
        初始化三个基本属性，时，分，秒
        :param hour:
        :param minute:
        :param second:
        '''
        self._hour = hour
        self._minute = minute
        self._second = second
```

然后是模拟时钟的运行，这里只需要注意时钟运行过程边界问题，即秒和分都是每到 60 需要置零，并让分或者时加 1，而时是每隔 24 需要进行这样的操作

```python
 def run(self):
     '''
     模拟时钟的运行
     :return:
     '''
     self._second += 1
     if self._second == 60:
     	self._second = 0
     	self._minute += 1
     	if self._minute == 60:
     		self._minute = 0
     		self._hour += 1
    		 if self._hour == 24:
     			self._hour = 0
```

最后是显示时间，需要注意时、分和秒三个属性都是整数，如果采用 `%` 进行格式化，需要调用 `str()` 方法显示将它们从整数变成字符串类型，而如果用 `format()` 方法，就不需要。

```python
def show(self):
    '''
    显示时间
    :return:
    '''
    print("{:02d}:{:02d}:{:02d}".format(self._hour, self._minute, self._second))
```

简单的运用例子，这里调用 `time.sleep()` 方法，每显示一次时间休眠一秒，然后运行，设置循环次数是 5 次。

```python
# 简单时钟例子
clock = Clock(23, 59, 57)
i = 0
while i < 5:
    clock.show()
    sleep(1)
    clock.run()
    i += 1
```

输出结果：

```
23:59:57
23:59:58
23:59:59
00:00:00
00:00:01
```

##### 定义一个类描述点之间的移动和距离

第二个练习是定义一个类，描述平面上点之间的移动和距离计算

首先是基本的构造方法定义，这里作为平面上的点，需要定义的属性就是点的横纵坐标：

```python
# 定义描述平面上点之间的移动和计算距离的类
class Point(object):
    def __init__(self, x=0, y=0):
        '''
        初始的坐标
        :param x:横坐标
        :param y:纵坐标
        '''
        self._x = x
        self._y = y
```

接着，点的移动，可以有两种实现，第一种直接说明目标点的坐标：

```python
def move_to(self, new_x, new_y):
    '''
    移动到新的坐标
    :param new_x:新的横坐标
    :param new_y:新的纵坐标
    :return:
    '''
    self._x = new_x
    self._y = new_y
```

第二种则是只告诉分别在横、纵两个方向移动的距离：

```python
def move_by(self, dx, dy):
    '''
    移动指定的增量
    :param dx:横坐标的增量
    :param dy:纵坐标的增量
    :return:
    '''
    self._x += dx
    self._y += dy
```

然后计算点之间的距离方法，这里就需要用到刚刚从 `math` 库导入的方法 `sqrt` ，即求取平方根：

```python
def distance(self, other):
    '''
    计算与另一个点的距离
    :param other:
    :return:
    '''
    x_dist = self._x - other._x
    y_dist = self._y - other._y
    return sqrt(x_dist ** 2 + y_dist ** 2)
```

最后当然就是打印当前点的坐标信息了：

```python
 def __str__(self):
     '''
     显示当前点坐标
     :return:
     '''
     return '({},{})'.format(self._x, self._y)
```

简单的应用例子

```python
p1 = Point(10, 20)
p2 = Point(30, 5)
print('point1:', p1)
print('point2:', p2)
p1.move_to(15, 25)
print('after move to (15, 25), point1:', p1)
p1.move_by(20, 10)
print('move by (20, 10), point1:', p1)
dist = p1.distance(p2)
print('distance between p1 and p2: ', dist)
```

输出结果：

```
point1: (10,20)
point2: (30,5)
after move to (15, 25), point1: (15,25)
move by (20, 10), point1: (35,35)
distance between p1 and p2:  30.4138126514911
```



本章的代码都上传到 Github 了：

https://github.com/ccc013/Python_Notes/blob/master/Practise/class_example.py



------

#### 参考

- 《Python 编程从入门到实践》
- [Python3 面向对象](https://www.runoob.com/python3/python3-class.html)
- [Python-100-Days--Day08面向对象基础](https://github.com/jackfrued/Python-100-Days/blob/master/Day01-15/08.%E9%9D%A2%E5%90%91%E5%AF%B9%E8%B1%A1%E7%BC%96%E7%A8%8B%E5%9F%BA%E7%A1%80.md)