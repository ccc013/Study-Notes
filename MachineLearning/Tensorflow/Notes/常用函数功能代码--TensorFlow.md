
主要是记录学习别人的代码中比较好的函数功能实现或者更好的写法。

#### image check

检测图片是否符合规定。

检测图片是否是彩色图片：
```
import tensorflow as tf

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image
```

#### 保存和展示实验结果

对于输出图片，可以通过保存成网页形式展示实验输出的图片结果。


```
def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path
```


训练过程中，**设置打印当前训练结果，可以将判断条件写成一个函数**：


```
def should(freq):
    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)
```

如上函数所示，输入是间隔次数，判断条件就是只有间隔次数不为 0 且训练次数刚好是间隔次数的倍数或者是最后一次训练，返回为 True， 否则就是返回 False。


下面这个函数是用来**显示生成的图片**，打印其中的 25 张生成图片：
```
import matplotlib.gridspec as gridspec

def plot_images(images, save_dir=None):
    plt.figure(figsize=(5,5))
    gs1 = gridspec.GridSpec(5, 5)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(images.shape[0], 25, replace=False)
    for i in range(25):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = images[rand_index, :, :, :]
        image = (image + 1) * 127.5
        image = image.astype(np.uint8)
        fig = plt.imshow(image)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    if save_dir:
        plt.savefig("%s/%s.png" % (save_dir, str(time.time())), bbox_inches='tight', pad_inches=0)
    plt.show()
```

