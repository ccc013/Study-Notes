
主要安装过程参考[轻量级批量Omnitty工具安装和简单使用](https://www.cnblogs.com/lcj0703/p/6703970.html)

但是在安装 omnitty的时候，执行到采用

```
make
```
这步骤出现了这个错误：

```
/usr/bin/ld: help.o: undefined reference to symbol 'delwin'
/usr/bin/ld: note: 'delwin' is defined in DSO /lib64/libncurses.so.5 so try adding it to the linker command line
/lib64/libncurses.so.5: could not read symbols: Invalid operation
collect2: error: ld returned 1 exit status
make: *** [omnitty] Error 1
```

根据[UnderstandingDSOLinkChange](https://fedoraproject.org/wiki/UnderstandingDSOLinkChange) 这篇文章的例子介绍，明白这是需要将缺少的动态库添加到 Makefile 文件中，所以就是如下操作：

```
vim Makefile
```
然后在`LIBS= -L/usr/local/lib -lrote`后面添加缺少的库，即改为`LIBS= -L/usr/local/lib -lrote /lib64/libncurses.so.5 /lib64/libtinfo.so.5`

这里添加了后面两个库，是因为又报错，缺少了`/lib64/libtinfo.so.5`这个动态库，改完后，就能正常编译通过，然后继续往后面执行安装步骤。




