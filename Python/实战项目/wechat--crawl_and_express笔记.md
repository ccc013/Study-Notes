
这是参考下面的文章：

1. [Python 打造七夕最强表白程序](https://mp.weixin.qq.com/s/tIFbfKLxgAxz8WdES5allA)
2. [PhantomJS with Selenium error: Message: 'phantomjs' executable needs to be in PATH](https://stackoverflow.com/questions/37903536/phantomjs-with-selenium-error-message-phantomjs-executable-needs-to-be-in-pa)
3. 

首先，整体代码是根据参考 [1] 文章来实现的，但开始运行的时候，发现有个问题：`PhantomJS with Selenium error: Message: 'phantomjs' executable needs to be in PATH`，于是根据 [2] 文章来解决这个问题：

1. 首先从网站 http://phantomjs.org/download.html 下载安装包--phantomjs-2.1.1-windows.zip，这里我是下载 windows 版本的，如果是 linux 系统则下载对应 linux 版本的安装包；
2. 接着解压缩安装包
3. 然后是配置环境变量，具体步骤如下三张图所示。
4. 最后可能需要重启来完成配置。

这里也可以不配置环境变量，选择在代码中加入 phantomjs.exe 所在文件夹路径，即如：

```
browser = webdriver.PhantomJS(executable_path='E:\\phantomjs-2.1.1-windows\\bin\\phantomjs')
```



