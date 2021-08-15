

原文：https://github.com/zedr/clean-code-python

python 版的代码整洁之道。目录如下所示：

1. 介绍
2. 变量
3. 函数



------

### 1. 介绍

软件工程的原则，来自 Robert C. Martin's 的书--《Clean Code》，而本文则是适用于 Python 版本的 clean code。这并不是一个风格指导，而是指导如何写出可读、可用以及可重构的 pyhton 代码。

并不是这里介绍的每个原则都必须严格遵守，甚至只有很少部分会得到普遍的赞同。下面介绍的都只是指导而已，但这都是来自有多年编程经验的 《Clean Code》的作者。

这里的 python 版本是 3.7+



------

### 2. 变量

#### 2.1 采用有意义和可解释的变量名

**糟糕的写法**

```python
ymdstr = datetime.date.today().strftime("%y-%m-%d")
```

**好的写法**

```python
current_date: str = datetime.date.today().strftime("%y-%m-%d")
```

#### 2.2 对相同类型的变量使用相同的词汇

**糟糕的写法**：这里对有相同下划线的实体采用三个不同的名字

```python
get_user_info()
get_client_data()
get_customer_record()
```

**好的写法**：如果实体是相同的，对于使用的函数应该保持一致

```python
get_user_info()
get_user_data()
get_user_record()
```

**更好的写法**：python 是一个面向对象的编程语言，所以可以将相同实体的函数都放在类中，作为实例属性或者是方法

```python
class User:
    info : str

    @property
    def data(self) -> dict:
        # ...

    def get_record(self) -> Union[Record, None]:
        # ...
```

#### 2.3 采用可以搜索的名字

我们通常都是看的代码多于写过的代码，所以让我们写的代码是可读而且可以搜索的是非常重要的，如果不声明一些有意义的变量，会让我们的程序变得难以理解，例子如下所示。

**糟糕的写法**

```python
# 86400 表示什么呢？
time.sleep(86400)
```

**好的写法**

```python
# 声明了一个全局变量
SECONDS_IN_A_DAY = 60 * 60 * 24

time.sleep(SECONDS_IN_A_DAY)
```

#### 2.4 采用带解释的变量

**糟糕的写法**

```python
address = 'One Infinite Loop, Cupertino 95014'
city_zip_code_regex = r'^[^,\\]+[,\\\s]+(.+?)\s*(\d{5})?$'
matches = re.match(city_zip_code_regex, address)

save_city_zip_code(matches[1], matches[2])
```

**还行的写法**

这个更好一点，但还是很依赖于正则表达式

```python
address = 'One Infinite Loop, Cupertino 95014'
city_zip_code_regex = r'^[^,\\]+[,\\\s]+(.+?)\s*(\d{5})?$'
matches = re.match(city_zip_code_regex, address)

city, zip_code = matches.groups()
save_city_zip_code(city, zip_code)
```

**好的写法**

通过子模式命名来减少对正则表达式的依赖

```python
address = 'One Infinite Loop, Cupertino 95014'
city_zip_code_regex = r'^[^,\\]+[,\\\s]+(?P<city>.+?)\s*(?P<zip_code>\d{5})?$'
matches = re.match(city_zip_code_regex, address)

save_city_zip_code(matches['city'], matches['zip_code'])
```

#### 2.5 避免让读者进行猜测

不要让读者需要联想才可以知道变量名的意思，显式比隐式更好。

**糟糕的写法**

```python
seq = ('Austin', 'New York', 'San Francisco')

for item in seq:
    do_stuff()
    do_some_other_stuff()
    # ...
    # item 是表示什么？
    dispatch(item)
```

**好的写法**

```python
locations = ('Austin', 'New York', 'San Francisco')

for location in locations:
    do_stuff()
    do_some_other_stuff()
    # ...
    dispatch(location)
```

#### 2.6 不需要添加额外的上下文

如果类或者对象名称已经提供一些信息来，不需要在变量中重复。

**糟糕的写法**

```python
class Car:
    car_make: str
    car_model: str
    car_color: str
```



**好的写法**

```python
class Car:
    make: str
    model: str
    color: str
```

#### 2.7 采用默认参数而不是条件语句

**糟糕的写法**

```python
def create_micro_brewery(name):
    name = "Hipster Brew Co." if name is None else name
    slug = hashlib.sha1(name.encode()).hexdigest()
    # etc.
```

这个写法是可以直接给 `name` 参数设置一个默认数值，而不需要采用一个条件语句来进行判断的。

**好的写法**

```python
def create_micro_brewery(name: str = "Hipster Brew Co."):
    slug = hashlib.sha1(name.encode()).hexdigest()
    # etc.
```

### 3. 函数

#### 3.1 函数参数（2个或者更少）

限制函数的参数个数是很重要的，这有利于测试你编写的函数代码。超过3个以上的函数参数会导致测试组合爆炸的情况，也就是需要考虑很多种不同的测试例子。

没有参数是最理想的情况。一到两个参数也是很好的，三个参数应该尽量避免。如果多于 3 个那么应该需要好好整理函数。通常，如果函数多于2个参数，那代表你的函数可能要实现的东西非常多。此外，很多时候，一个高级对象也是可以用作一个参数使用。

**糟糕的写法**

```python
def create_menu(title, body, button_text, cancellable):
    # ...
```



**很好的写法**

```python
class Menu:
    def __init__(self, config: dict):
        title = config["title"]
        body = config["body"]
        # ...

menu = Menu(
    {
        "title": "My Menu",
        "body": "Something about my menu",
        "button_text": "OK",
        "cancellable": False
    }
)
```

**另一种很好的写法**

```python
class MenuConfig:
    """A configuration for the Menu.

    Attributes:
        title: The title of the Menu.
        body: The body of the Menu.
        button_text: The text for the button label.
        cancellable: Can it be cancelled?
    """
    title: str
    body: str
    button_text: str
    cancellable: bool = False


def create_menu(config: MenuConfig):
    title = config.title
    body = config.body
    # ...


config = MenuConfig
config.title = "My delicious menu"
config.body = "A description of the various items on the menu"
config.button_text = "Order now!"
# The instance attribute overrides the default class attribute.
config.cancellable = True

create_menu(config)
```

**优秀的写法**

```python
from typing import NamedTuple


class MenuConfig(NamedTuple):
    """A configuration for the Menu.

    Attributes:
        title: The title of the Menu.
        body: The body of the Menu.
        button_text: The text for the button label.
        cancellable: Can it be cancelled?
    """
    title: str
    body: str
    button_text: str
    cancellable: bool = False


def create_menu(config: MenuConfig):
    title, body, button_text, cancellable = config
    # ...


create_menu(
    MenuConfig(
        title="My delicious menu",
        body="A description of the various items on the menu",
        button_text="Order now!"
    )
)
```

**更优秀的写法**

```python
rom dataclasses import astuple, dataclass


@dataclass
class MenuConfig:
    """A configuration for the Menu.

    Attributes:
        title: The title of the Menu.
        body: The body of the Menu.
        button_text: The text for the button label.
        cancellable: Can it be cancelled?
    """
    title: str
    body: str
    button_text: str
    cancellable: bool = False

def create_menu(config: MenuConfig):
    title, body, button_text, cancellable = astuple(config)
    # ...


create_menu(
    MenuConfig(
        title="My delicious menu",
        body="A description of the various items on the menu",
        button_text="Order now!"
    )
)
```

#### 3.2 函数应该只完成一个功能

这是目前为止软件工程里最重要的一个规则。函数如果完成多个功能，就很难对这个函数解耦、测试。如果可以对一个函数分离为仅仅一个动作，那么该函数可以很容易进行重构，并且代码也方便阅读。即便你仅仅遵守这一点建议，你也会比很多开发者更加优秀。

**糟糕的写法**

```python
def email_clients(clients: List[Client]):
    """Filter active clients and send them an email.
       筛选活跃的客户并发邮件给他们
    """
    for client in clients:
        if client.active:
            email(client)
```

**好的写法**

```python
def get_active_clients(clients: List[Client]) -> List[Client]:
    """Filter active clients.
    """
    return [client for client in clients if client.active]


def email_clients(clients: List[Client, ...]) -> None:
    """Send an email to a given list of clients.
    """
    for client in clients:
        email(client)
```

这里其实是可以使用生成器来改进函数的写法。



**更好的写法**

```python
def active_clients(clients: List[Client]) -> Generator[Client]:
    """Only active clients.
    """
    return (client for client in clients if client.active)


def email_client(clients: Iterator[Client]) -> None:
    """Send an email to a given list of clients.
    """
    for client in clients:
        email(client)
```



#### 3.3 函数的命名应该表明函数的功能

**糟糕的写法**

```python
class Email:
    def handle(self) -> None:
        # Do something...

message = Email()
# What is this supposed to do again?
# 这个函数是需要做什么呢？
message.handle()
```

**好的写法**

```python
class Email:
    def send(self) -> None:
        """Send this message.
        """

message = Email()
message.send()
```

#### 3.4 函数应该只有一层抽象

如果函数包含多于一层的抽象，那通常就是函数实现的功能太多了，应该把函数分解成多个函数来保证可重复使用以及更容易进行测试。

**糟糕的写法**

```python
def parse_better_js_alternative(code: str) -> None:
    regexes = [
        # ...
    ]

    statements = regexes.split()
    tokens = []
    for regex in regexes:
        for statement in statements:
            # ...

    ast = []
    for token in tokens:
        # Lex.

    for node in ast:
        # Parse.
```

**好的写法**

```python
REGEXES = (
   # ...
)


def parse_better_js_alternative(code: str) -> None:
    tokens = tokenize(code)
    syntax_tree = parse(tokens)

    for node in syntax_tree:
        # Parse.


def tokenize(code: str) -> list:
    statements = code.split()
    tokens = []
    for regex in REGEXES:
        for statement in statements:
           # Append the statement to tokens.

    return tokens


def parse(tokens: list) -> list:
    syntax_tree = []
    for token in tokens:
        # Append the parsed token to the syntax tree.

    return syntax_tree
```

#### 3.5 不要将标志作为函数参数

标志表示函数实现的功能不只是一个，但函数应该仅做一件事情，所以如果需要标志，就将多写一个函数吧。

**糟糕的写法**

```python
from pathlib import Path

def create_file(name: str, temp: bool) -> None:
    if temp:
        Path('./temp/' + name).touch()
    else:
        Path(name).touch()
```

**好的写法**

```python
from pathlib import Path

def create_file(name: str) -> None:
    Path(name).touch()

def create_temp_file(name: str) -> None:
    Path('./temp/' + name).touch()
```

#### 3.6 避免函数的副作用

函数产生副作用的情况是在它做的事情不只是输入一个数值，返回其他数值这样一件事情。比如说，副作用可能是将数据写入文件，修改全局变量，或者意外的将你所有的钱都写给一个陌生人。

不过，有时候必须在程序中产生副作用--比如，刚刚提到的例子，必须写入数据到文件中。这种情况下，你应该尽量集中和指示产生这些副作用的函数，比如说，保证只有一个函数会产生将数据写到某个特定文件中，而不是多个函数或者类都可以做到。

这条建议的主要意思是避免常见的陷阱，比如分析对象之间的状态的时候没有任何结构，使用可以被任何数据修改的可修改数据类型，或者使用类的实例对象，不集中副作用影响等等。如果你可以做到这条建议，你会比很多开发者都开心。

**糟糕的写法**

```python
# This is a module-level name.
# It's good practice to define these as immutable values, such as a string.
# However...
name = 'Ryan McDermott'

def split_into_first_and_last_name() -> None:
    # The use of the global keyword here is changing the meaning of the
    # the following line. This function is now mutating the module-level
    # state and introducing a side-effect!
    # 这里采用了全局变量，并且函数的作用就是修改全局变量，其副作用就是修改了全局变量，
    # 第二次调用函数的结果就会和第一次调用不一样了。
    global name
    name = name.split()

split_into_first_and_last_name()

print(name)  # ['Ryan', 'McDermott']

# OK. It worked the first time, but what will happen if we call the
# function again?
```

**好的写法**

```python
def split_into_first_and_last_name(name: str) -> list:
    return name.split()

name = 'Ryan McDermott'
new_name = split_into_first_and_last_name(name)

print(name)  # 'Ryan McDermott'
print(new_name)  # ['Ryan', 'McDermott']
```

**另一个好的写法**

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str

    @property
    def name_as_first_and_last(self) -> list:
        return self.name.split() 

# The reason why we create instances of classes is to manage state!
person = Person('Ryan McDermott')
print(person.name)  # 'Ryan McDermott'
print(person.name_as_first_and_last)  # ['Ryan', 'McDermott']
```

------

### 总结

原文的目录实际还有三个部分：

- 对象和数据结构

- 类
  - 单一职责原则（Single Responsibility Principle, SRP)
  - 开放封闭原则（Open/Closed principle，OCP）
  - 里氏替换原则（Liskov Substitution Principle ，LSP)
  - 接口隔离原则（Interface Segregation Principle ，ISP)
  - 依赖倒置原则（Dependency Inversion Principle ，DIP)

- 不要重复

不过作者目前都还没有更新，所以想了解这部分内容的，建议可以直接阅读《代码整洁之道》对应的这部分内容了。









































