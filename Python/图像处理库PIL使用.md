# 图像处理库PIL使用

参考：

- [Python图像处理库PIL的基本概念介绍](https://blog.csdn.net/icamera0/article/details/50647465)
- [Python图像处理库PIL中图像格式转换（一）](https://blog.csdn.net/icamera0/article/details/50843172)
- [Python图像处理库PIL中图像格式转换（二）](https://blog.csdn.net/icamera0/article/details/50843196)
- [python PIL 图像处理](https://www.jianshu.com/p/e8d058767dfa)
- [python中的深拷贝与浅拷贝](https://blog.csdn.net/jiandanjinxin/article/details/78214133)

安装：

```shell
$ pip install Pillow
```

<a name="IxvRQ"></a>
## 1. 基本使用
<a name="txGJc"></a>
### 读取图片

```python
from PIL import Image

img = Image.open('test.jpg')
# 保存
img.save('save.jpg')
```
<a name="xAgXL"></a>
### 尺寸
通过方法 `size` 返回一个二元数组

```python
weight, height = img.size
```

调整尺寸-- `resize` 

```python
new_img = img.resize((new_width, new_height))
```


<a name="SYjCz"></a>
### 模式

```python
mode = img.mode
```

关于模式：

- 1：1位像素，表示黑和白，但是存储的时候每个像素存储为8bit。
- L：8位像素，表示黑和白。
- P：8位像素，使用调色板映射到其他模式。
- RGB：3x8位像素，为真彩色。
- RGBA：4x8位像素，有透明通道的真彩色。
- CMYK：4x8位像素，颜色分离。
- YCbCr：3x8位像素，彩色视频格式。
- I：32位整型像素。
- F：32位浮点型像素。

模式转换通过方法 `convert` ，例子：

```python
new_img = img.convert('RGB')
```

<a name="YyEqw"></a>
### 通道
通过方法 `getbands()` 获取图片通道数量和名字

```python
from PIL import Image

img = Image.open('test.jpg')
im_bands = img.getbands()
print(im_bands)
print(len(im_bands))
print(im_bands[0])
```


---

<a name="RXqGq"></a>
## PNG 转 JPG

<a name="CsJLz"></a>
### L -> RGB
[http://www.voidcn.com/article/p-rbpllhah-btp.html](http://www.voidcn.com/article/p-rbpllhah-btp.html)

```python
from PIL import Image
im = Image.open("test.png")
bg = Image.new("RGB", im.size, (255,255,255))
bg.paste(im,im)
bg.save("test.jpg")
```


<a name="4StR0"></a>
### RGBA->RGB
[http://stackoverflow.com/a/9166671/284318](http://stackoverflow.com/a/9166671/284318)
```python
import io
import numpy as np
from os.path import join
from PIL import Image
try:
    from requests.utils import urlparse
    from requests import get as urlopen

    requests_available = True
except ImportError:
    requests_available = False
    if sys.version_info[0] == 2:
        from urlparse import urlparse  # noqa f811
        from urllib2 import urlopen  # noqa f811
    else:
        from urllib.request import urlopen
        from urllib.parse import urlparse

        
def alpha_composite(front, back):
    """Alpha composite two RGBA images.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object

    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result


def RGBA2RGB(image, color=(255, 255, 255)):
    """Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    result_rgba = alpha_composite(image, back)
    return result_rgba.convert('RGB')


def download_img(img_url, save_path):
    if sys.version_info[0] == 2:
    	image_data = urllib2.urlopen(img_url)
    	io_buffer = io.BytesIO(image_data.read())
    else:
        image_data = urlopen(img_url)
    	io_buffer = io.BytesIO(image_data.content)
    image = Image.open(io_buffer)
    image = RGBA2RGB(image)
    image_name = str('aaa') + '.jpg'
    image_path = join(save_path, image_name)
    image.save(image_path)
```

<a name="LnWti"></a>
### P->RGB
思路：

1. 先**转换为 RGBA 格式**
1. 然后按照 RGBA 转 RGB 的方法即可

```python
ori_img = Image.open(img_path)
rgba_img = ori_img.convert('RGBA')
rgb_img = RGBA2RGB(rgba_img)
```

---

<a name="kOfux"></a>
## PIL Image 图像互转 numpy 数组
<a name="J6PYF"></a>
### 将 PIL Image 图片转换为 numpy 数组

```python
im_array = np.array(im)
# 也可以用 np.asarray(im) 区别是 np.array() 是深拷贝，np.asarray() 是浅拷贝
```
更多细节见 [python中的深拷贝与浅拷贝](https://blog.csdn.net/jiandanjinxin/article/details/78214133)

**numpy image 查看图片信息，可用如下的方法**
```python
print(img.shape)  
print(img.dtype)
```

<a name="v5m0x"></a>
### 将 numpy 数组转换为 PIL 图片
_这里采用 _`_matplotlib.image_` _ 读入图片数组，注意这里读入的数组是 _`_ float32_` _ 型的，范围是 _`_0-1_` _，而 _`_PIL.Image_` _ 数据是 _`_uinit8_` _ 型的，范围是 _`_0-255_` ，所以要进行转换：
```python
import matplotlib.image as mpimg
from PIL import Image
lena = mpimg.imread('lena.png') # 这里读入的数据是 float32 型的，范围是0-1
im = Image.fromarray(np.uinit8(lena*255))
im.show()
```
**PIL image 查看图片信息，可用如下的方法**
```python
print type(img)
print img.size  #图片的尺寸
print img.mode  #图片的模式
print img.format  #图片的格式
print(img.getpixel((0,0))[0])#得到像素：
#img读出来的图片获得某点像素用getpixel((w,h))可以直接返回这个点三个通道的像素值
```


---

<a name="aCk3d"></a>
## 绘制
<a name="zUkMQ"></a>
### 图片加文字
代码例子如下，其中字体 `setFont` 是可以直接选择 `Windows` 系统中路径 `C:/windows/fonts/` 下的字体文件，如果采用默认字体，无法调整字体的大小

```python
from PIL import Image, ImageDraw, ImageFont

image = Image.Open('test.jpg')
#新建绘图对象
draw = ImageDraw.Draw(image)
#获取图像的宽和高
width, height = image.size
# ImageFont模块
#选择文字字体和大小
setFont = ImageFont.truetype('C:/windows/fonts/Dengl.ttf', 20)
# 设置文字颜色
fillColor = "#ff0000" # or fillColor='red'
#写入文字
draw.text((40, height - 100), 'python PIL', font=setFont, fill=fillColor)
```

<a name="Gxgi7"></a>
### 图片添加矩形框

```python
from PIL import Image, ImageDraw

image = Image.Open('test.jpg')
# 新建绘图对象
draw = ImageDraw.Draw(image)
# 矩形框左上角坐标
left, top = 0,0
# 获取图像的宽和高
width, height = image.size
# 绘制, 可选填充方式，outline 表示矩形框的边界线颜色，fill 表示填充满整个矩形框
draw.rectangle(((left, top), (left + width, top + height)), outline='red')
```

---

<a name="FDe8s"></a>
# 问题

<a name="OIUaN"></a>
## 1.OSError: image file is truncated

[https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images](https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images)

解决方案：

```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

