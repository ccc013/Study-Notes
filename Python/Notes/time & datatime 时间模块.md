


---
### time 模块知识点

#### sleep

sleep() 函数推迟调用线程的运行，可通过参数secs指秒数，表示进程挂起的时间。

##### 语法

```
time.sleep(t)
```

##### 参数

- t -- 推迟执行的秒数。

##### 返回值

该函数没有返回值

##### 实例


```
print("Start : %s" % time.ctime())
time.sleep(1)
print("End : %s" % time.ctime())
```
输出结果：

```
Start : Wed Mar  6 21:48:18 2019
End : Wed Mar  6 21:48:19 2019
```

#### localtime

localtime() 函数类似gmtime()，作用是**格式化时间戳为本地的时间**。 如果 sec 参数未输入，则以当前时间为转换标准。 DST (Daylight Savings Time) flag (-1, 0 or 1) 是否是夏令时。

##### 语法

```
time.localtime([ sec ])
```

##### 参数

- sec -- 转换为time.struct_time类型的对象的秒数。

##### 返回值

该函数没有返回值

##### 实例

```
print("time.localtime(): {}".format(time.localtime()))
```
输出结果如下：

```
time.localtime(): time.struct_time(tm_year=2019, tm_mon=3, tm_mday=6, tm_hour=21, tm_min=58, tm_sec=4, tm_wday=2, tm_yday=65, tm_isdst=0)
```

#### strftime

strftime() 函数接收以时间元组，并返回以可读字符串表示的当地时间，格式由参数 format 决定。

##### 语法

```
time.strftime(format[, t])
```

##### 参数

- format -- 格式字符串。
- t -- 可选的参数t是一个struct_time对象。

##### 返回值

返回以可读字符串表示的当地时间。

##### 说明

python中时间日期格式化符号：

- %y 两位数的年份表示（00-99）
- %Y 四位数的年份表示（000-9999）
- %m 月份（01-12）
- %d 月内中的一天（0-31）
- %H 24小时制小时数（0-23）
- %I 12小时制小时数（01-12）
- %M 分钟数（00=59）
- %S 秒（00-59）
- %a 本地简化星期名称
- %A 本地完整星期名称
- %b 本地简化的月份名称
- %B 本地完整的月份名称
- %c 本地相应的日期表示和时间表示
- %j 年内的一天（001-366）
- %p 本地A.M.或P.M.的等价符
- %U 一年中的星期数（00-53）星期天为星期的开始
- %w 星期（0-6），星期天为星期的开始
- %W 一年中的星期数（00-53）星期一为星期的开始
- %x 本地相应的日期表示
- %X 本地相应的时间表示
- %Z 当前时区的名称
- %% %号本身


##### 实例

```
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
time.sleep(1)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
```

输出结果如下：

```
2019-03-06 21:43:36
2019-03-06 21:43:37
```


---
### datetime
