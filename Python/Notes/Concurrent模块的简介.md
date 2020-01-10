**原题** | PYTHON: A QUICK INTRODUCTION TO THE CONCURRENT.FUTURES MODULE

**作者** | MASNUN

**原文** | http://masnun.com/2016/03/29/python-a-quick-introduction-to-the-concurrent-futures-module.html

`concurrent.futures` 是标准库里的一个模块，它提供了一个实现异步任务的高级 API 接口。本文将通过一些代码例子来介绍这个模块常见的用法。

#### Executors

`Executor` 是一个抽象类，它有两个非常有用的子类--`ThreadPoolExecutor` 和 `ProcessPoolExecutor` 。从命名就可以知道，前者采用的是多线程，而后者使用多进程。下面将分别介绍这两个子类，在给出的例子中，我们都会创建一个线程池或者进程池，然后将任务提交到这个池子，这个池子将会分配可用的资源（线程或者进程）来执行给定的任务。

#### ThreadPoolExecutor

首先，先看看代码：

```python
from concurrent.futures import ThreadPoolExecutor
from time import sleep
# 定义需要执行的任务--休眠5秒后返回传入的信息
def return_after_5_secs(message):
    sleep(5)
    return message
# 建立一个线程池，大小为 3
pool = ThreadPoolExecutor(3)

future = pool.submit(return_after_5_secs, ("hello"))
print(future.done())
sleep(5)
print(future.done())
print(future.result())
```

输出结果：

```
False
False
hello
```

这个代码中首先创建了一个 `ThreadPoolExecutor` 对象--`pool` ，通常这里默认线程数量是 5，但我们指定线程池的线程数量是 3。接着就是调用 `submit()` 方法来把需要执行的任务，也就是函数，以及需要传给这个函数的参数，然后会得到 `Future` 对象，这里调用其方法 `done()` 用于告诉我们是否执行完任务，是，就返回 `true` ，没有就返回 `false` 。

在上述例子中，第一次调用 `done()` 时候，并没有经过 5 秒，所以会得到 `false` ；之后进行休眠 5 秒后，任务就会完成，再次调用 `done()` 就会得到 `true` 的结果。如果是希望得到任务的结果，可以调用 `future` 的`result` 方法。

对 `Future` 对象的理解有助于理解和实现异步编程，因此非常建议好好看看官方文档的介绍：

https://docs.python.org/3/library/concurrent.futures.html

#### ProcessPoolExecutor

`ProcessPoolExecutor` 也是有相似的接口，使用方法也是类似的，代码例子如下所示：

```python
from concurrent.futures import ProcessPoolExecutor
from time import sleep
 
def return_after_5_secs(message):
    sleep(5)
    return message
 
pool = ProcessPoolExecutor(3)
 
future = pool.submit(return_after_5_secs, ("hello"))
print(future.done())
sleep(5)
print(future.done())
print("Result: " + future.result())
```

输出结果：

```
False
False
Result: hello
```

通常，我们会用**多进程 `ProcessPoolExecutor` 来处理 CPU 密集型任务**，**多线程 `ThreadPoolExecutor` 则更适合处理网络密集型 或者 I/O 任务**。

尽管这两个模块的接口相似，但 `ProcessPoolExecutor` 采用的是 `multiprocessing` 模块，并且不会被 GIL( Global Interpreter Lock) 所影响。不过对于这个模块，我们需要注意不能采用任何不能序列化的对象。

#### Executor.map()

上述两个模块都有一个共同的方法--`map()`。跟 Python 内建的 `map` 函数类似，该方法可以实现对提供的一个函数进行多次调用，并且通过给定一个可迭代的对象来将每个参数都逐一传给这个函数。另外，采用 `map()` 方法，提供的函数将是并发调用。

对于多进程，传入的可迭代对象将分成多块的数据，每块数据分配给每个进程。分块的数量可以通过调整参数 `chunk_size` ，默认是 1.

下面是官方文档给出的 `ThreadPoolExecutor`  的例子：

```python
import concurrent.futures
import urllib.request
 
URLS = ['http://www.baidu.com/',
        'http://www.163.com/',
        'http://www.126.com/',
        'http://www.jianshu.com/',
        'http://news.sohu.com/']
 
# Retrieve a single page and report the url and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()
 
# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page is %d bytes' % (url, len(data)))
```

输出结果：

```
'http://www.baidu.com/' page is 153759 bytes
'http://www.163.com/' page is 693614 bytes
'http://news.sohu.com/' page is 175707 bytes
'http://www.126.com/' page is 10521 bytes
'http://www.jianshu.com/' generated an exception: HTTP Error 403: Forbidden
```



而对于  `ProcessPoolExecutor` ，代码如下所示：

```python
import concurrent.futures
import math
 
PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]
 
def is_prime(n):
    if n % 2 == 0:
        return False
 
    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True
 
def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))
 
if __name__ == '__main__':
    main()
```

输出结果：

```
112272535095293 is prime: True
112582705942171 is prime: True
112272535095293 is prime: True
115280095190773 is prime: True
115797848077099 is prime: True
1099726899285419 is prime: False
```



#### as_completed() & wait()

`concurrent.futures` 模块中有两个函数用于处理进过 `executors` 返回的 `futures`，分别是 `as_completed()` 和 `wait()`。

 `as_completed()` 函数会获取 `Future` 对象，并且随着任务开始处理而返回任务的结果，也就是需要执行的函数的返回结果。它和上述介绍的 `map()` 的主要区别是 `map()` 方法返回的结果是按照我们传入的可迭代对象中的顺序返回的。而 `as_completed()` 返回的结果顺序则是按照任务完成的顺序，哪个任务先完成，先返回结果。

下面给出一个例子：

```python
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from time import sleep
from random import randint
 
def return_after_5_secs(num):
    sleep(randint(1, 5))
    return "Return of {}".format(num)
 
pool = ThreadPoolExecutor(5)
futures = []
for x in range(5):
    futures.append(pool.submit(return_after_5_secs, x))
 
for x in as_completed(futures):
    print(x.result())
```

输出结果

```
Return of 3
Return of 4
Return of 0
Return of 2
Return of 1
```

`wait()` 函数返回一个包含两个集合的带有名字的 tuple，一个集合包含已经完成任务的结果(任务结果或者异常)，另一个包含的就是还未执行完毕的任务。

同样，下面是一个例子：

```python
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from time import sleep
from random import randint
 
def return_after_5_secs(num):
    sleep(randint(1, 5))
    return "Return of {}".format(num)
 
pool = ThreadPoolExecutor(5)
futures = []
for x in range(5):
    futures.append(pool.submit(return_after_5_secs, x))
 
print(wait(futures))
```

输出结果：

```
DoneAndNotDoneFutures(done={<Future at 0x2474aa4fba8 state=finished returned str>, <Future at 0x2474a903048 state=finished returned str>, <Future at 0x2474aa4fa58 state=finished returned str>, <Future at 0x2474aa4fcf8 state=finished returned str>, <Future at 0x2474a8beda0 state=finished returned str>}, not_done=set())

```

我们可以通过指定参数来控制 `wait()` 函数返回结果的时间，这个参数是 `return_when`，可选数值有：`FIRST_COMPLETED`, `FIRST_EXCEPTION` 和 `ALL_COMPLETED`。默认结果是 `ALL_COMPLETED` ，也就是它会等待所有任务都执行完成才返回结果。

------

以上就是本次教程的所有内容，代码已经上传到：

https://github.com/ccc013/Python_Notes/blob/master/Tutorials/concurrent_futures_tutorial.py









