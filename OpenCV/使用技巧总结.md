Python版本的opencv使用

##### 1. 裁剪图片
总结自[How to crop an image in OpenCV using Python](http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python)

在python的PIL这个图像处理库中，要裁剪图片，代码如下：

```
im = Image.open('0.png').convert('L')
im = im.crop((1, 1, 98, 33))
im.save('_0.png')
```

而如果是使用opencv实现相同的功能，则是需要这样写：

```
import cv2
img = cv2.imread("lenna.png")
crop_img = img[200:400, 100:300] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
```
这是使用到了`numpy`的切片，即`slicing`，同时注意是`img[y1:y2,x1:x2]`。
