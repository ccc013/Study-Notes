
参考：

- [在Centos 7下的OpenCV，CUDA，Caffe以及Tensorflow的安装](http://www.xiongfuli.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/2017-12/tensorflow-install-centos7.html)
- [在CentOS 7上安装Caffe](https://www.mtyun.com/library/how-to-install-caffe-on-centos7)
- [CentOS 7.x 安装Caffe GPU版本全过程](http://whatbeg.com/2018/03/15/caffeinstall.html)
- [Ubuntu16.04配置caffe（with cuda8.0）](https://www.jianshu.com/p/1d338f1c6bce)
- [我的AI之路(3)--安装Anaconda3 和Caffe](https://blog.csdn.net/XCCCCZ/article/details/80299784)
- [caffe编译错误总结记录](https://blog.csdn.net/jinman8199/article/details/78949805)
- [Caffe安装过程全记录 - CentOS，无GPU](https://blog.csdn.net/u010391437/article/details/73703200)
- [安装教程：使用Anaconda创建caffe和tensorflow共存环境](https://blog.csdn.net/primezpy/article/details/78819249)
- [五种方法利用Anaconda安装Caffe](https://blog.csdn.net/qq_33039859/article/details/80377356)


---
### 遇到的问题

#### 1. 编译caffe出现错误：make: *** [.build_release/src/caffe/common.o] Error 1

解决方法：[编译caffe出现错误：make: *** [.build_release/src/caffe/common.o] Error 1](https://blog.csdn.net/u011070171/article/details/52292680)

```
解决办法：

1.将./include/caffe/util/cudnn.hpp 换成最新版的caffe里的cudnn的实现，即相应的cudnn.hpp.

2. 将./include/caffe/layers里的，所有以cudnn开头的文件，例如cudnn_conv_layer.hpp。 都替换成最新版的caffe里的相应的同名文件。

3.将./src/caffe/layer里的，所有以cudnn开头的文件，例如cudnn_lrn_layer.cu，cudnn_pooling_layer.cpp，cudnn_sigmoid_layer.cu。

都替换成最新版的caffe里的相应的同名文件。
```

#### 2. fatal error: matio.h: No such file or directory

解决方法：[caffe matio 安装](https://blog.csdn.net/houqiqi/article/details/46469981)

#### 3. cannot find -lopencv_imgcodecs -lcblas -latlas


#### 4. caffe.pb.h丢失问题

参考[fatal error: caffe/proto/caffe.pb.h: No such file or directory #include "caffe/proto/caffe.pb.h" #3](https://github.com/muupan/dqn-in-the-caffe/issues/3), 需要生成`caffe.pb.h`文件

```
# In the directory you installed Caffe to
protoc src/caffe/proto/caffe.proto --cpp_out=.
mkdir include/caffe/proto
mv src/caffe/proto/caffe.pb.h include/caffe/proto
```



