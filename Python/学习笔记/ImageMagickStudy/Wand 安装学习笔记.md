
主要参考文章
1. [Wand--Installtion](http://docs.wand-py.org/en/0.4.2/guide/install.html)
2. [imagemagick home](http://www.imagemagick.org/script/index.php)
3. [Wand Documentation](http://docs.wand-py.org/en/0.4.4/)

### 简介

Wand 是基于 ctypes 库的适用于 Python 的 ImageMagick 的封装库。

相比其他对 ImageMagick 的封装库，Wand 有以下几个优势：

1. 符合 Python 习惯和现代化的接口
2. 有好的文档
3. 通过 ctypes 进行封装
4. 可以采用 pip 安装

### 安装教程

在 ubuntu下，可以直接按照下列命令安装：

```
$ apt-get install libmagickwand-dev
$ pip install Wand
```

#### 安装要求

**对 Python 版本要求：**
- Python 2.6+
- CPython 2.6+
- CPython 3.2+ or higher
- PyPy 1.5+ or higher

**MagickWand library**
- Debian/Ubuntu 系统：采用 apt-get 安装 libmagickwand-dev
- Mac 系统：用 MacPorts/Homebrew 安装 imagemagick
- CentOS 系统： 使用 yum 安装 ImageMagick-devel

#### Windows 注意事项

主要还是参照第一篇文章来安装，并且主要是在 Windows 下安装，其中下载 ImageMagick 的时候，在[下载地址](http://www.imagemagick.org/download/binaries/)中需要选择 6.9版本的 dll 的 exe 执行文件安装，而不能选择最新版本的 7.0+，否则在 Python 中调用的时候，会出现问题`ImportError: MagickWand shared library not found.`，原因根据[Python doesn't find MagickWand Libraries (despite correct location?)](https://stackoverflow.com/questions/25003117/python-doesnt-find-magickwand-libraries-despite-correct-location)中的说法是

> A few sources said that Image Magick 7.x is not compatible with magick Wand so make sure you're using 6.x. Additionally, "static" suffix versions do not work. The one that finally worked for me was "ImageMagick-6.9.8-10-Q8-x64-dll.exe"

也就是说  Image Magick 7.x 版本和 Wand 并不适配，所以只能采用 6+ 版本的。

### 使用教程

#### 图片读取

##### 读取文件

最常用的读取图片方式就是通过文件名读取，这里是通过采用`with`，并且需要添加关键词参数`filenamme`来制定读取的文件名，例子如下所示：

```
from __future__ import print_function
from wand.image import Image

with Image(filename='front.png') as img:
    print('width=', img.width)
    print('height=', img.height)
    print('size=', img.size)
```
结果如下：

```
width= 690
height= 368
size= (690L, 368L)
```




