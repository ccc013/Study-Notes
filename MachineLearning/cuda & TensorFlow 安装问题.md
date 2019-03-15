
参考文章：

1. [CentOS 7 卸载CUDA 9.1 安装CUDA8.0 并安装Tensorflow GPU版](http://whatbeg.com/2018/03/17/cudainstall.html)
2. [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#verify-kernel-packages)
3. [nvidia显卡驱动与编译器版本的查询命令](https://blog.csdn.net/s_sunnyy/article/details/64121826)
4. [升级Centos 7/6内核版本到4.12.4的方法](https://www.centos.bz/2017/08/upgrade-centos-7-6-kernel-to-4-12-4/)
5. [centos7安装nvidia Titan X CUDA9.0](https://www.jianshu.com/p/127e019ed2e8)
6. [centos7安装cuda-8.0报错modprobe: FATAL: Module nvidia-uvm not found.](https://blog.csdn.net/yijuan_hw/article/details/53439408)
7. [You do not appear to have the sources for the 3.10.0-327.el7.x86_64 kernel installed.](https://www.centos.org/forums/viewtopic.php?t=60468)
8. [TensorFlow官网安装文档](https://www.tensorflow.org/install/install_linux)
9. [Nvidia官网安装文档](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A)


---

初始问题：安装 cuda 成功，但是`执行nvidia-smi 时，总是报错：NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.`

一开始出现这个问题的原因估计是因为在已经安装了驱动的情况下，又在执行安装 cuda 的时候选择安装驱动`375.26`，这导致了出错，接着卸载了已经安装的驱动，然后根据[centos7安装cuda-8.0报错modprobe: FATAL: Module nvidia-uvm not found.](https://blog.csdn.net/yijuan_hw/article/details/53439408)的做法依次检查，在运行命令`dkms status`后，输出结果和文章一样，也是：

```
nvidia, 375.26: added
```
而执行命令：
```
dkms build -m nvidia 375.26
```
出现错误

```
 ERROR: Failed to run `/usr/sbin/dkms build -m nvidia -v 375.26 -k 3.10.0-327.28.3.el7.x86_64`: Error! echo
 56 Your kernel headers for kernel 3.10.0-327.28.3.el7.x86_64 cannot be found at
 57 /lib/modules/3.10.0-327.28.3.el7.x86_64/build or /lib/modules/3.10.0-327.28.3.el7.x86_64/source
```
当然这时候就走了比较多弯路，对内核进行了很多操作，包括升级源，导致安装了新的内核，本来内核根据命令

```
uname -r 

输出：3.10.0-327.28.3.el7.x86_64
```
但后面安装了新的内核，甚至根据[升级Centos 7/6内核版本到4.12.4的方法](https://www.centos.bz/2017/08/upgrade-centos-7-6-kernel-to-4-12-4/)升级到了最新的`4.18`版本的内核，通过命令`rpm -qa | grep kernel`可以查看系统的内核，然后发现就多了很多不同版本的内核，这里可以通过以下几个步骤选择系统使用的内核：

首先，根据命令查看默认启动顺序：

```
# awk -F\' '$1=="menuentry " {print $2}' /etc/grub2.cfg
CentOS Linux (3.10.0-862.3.2.el7.x86_64) 7 (Core)
CentOS Linux (3.10.0-862.el7.x86_64) 7 (Core)
CentOS Linux (3.10.0-693.el7.x86_64) 7 (Core)
CentOS Linux (3.10.0-327.28.3.el7.x86_64) 7 (Core)
CentOS Linux (3.10.0-327.el7.x86_64) 7 (Core)
CentOS Linux (0-rescue-a051bed3cf064e32b161377700fe2a6d) 7 (Core)
```
这里我们目前内核版本`3.10.0-327.28.3.el7.x86_64`是在位置3，那么需要修改内核启动顺序：

```
# vim /etc/default/grub
1 GRUB_TIMEOUT=5
2 GRUB_DISTRIBUTOR="$(sed 's, release .*$,,g' /etc/system-release)"
3 GRUB_DEFAULT=0
4 GRUB_DISABLE_SUBMENU=true
5 GRUB_TERMINAL_OUTPUT="console"
6 GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet"
7 GRUB_DISABLE_RECOVERY="true"
```
这里将第三行的`GRUB_DEFAULT=0`改为`GRUB_DEFAULT=3`，然后运行`grub2-mkconfig`命令来重新创建内核配置，如下：

```
# grub2-mkconfig -o /boot/grub2/grub.cfg
```
接着就是重启，执行命令

```
# reboot
```
重启后，再执行命令
```
# uname -r
3.10.0-327.28.3.el7.x86_64
```
就是选择的内核版本。

此外，要解决刚刚的问题，其实需要`kernel, kernel-devel, kernel-headers, kernel-tool`都一致，即

```
# rpm -qa | grep kernel | sort

kernel-devel-3.10.0-327.28.3.el7.x86_64
kernel-headers-3.10.0-327.28.3.el7.x86_64
kernel-tools-3.10.0-327.28.3.el7.x86_64
kernel-tools-libs-3.10.0-327.28.3.el7.x86_64
```
如上所示，必须这几个一致，接着就可以继续运行：

```
# dkms build -m nvidia -v 375.26

# dkms install -m nvidia -v 375.26
```
运行无误后，执行下面的命令确认结果：

```
# modinfo nvidia

# modinfo nvidia-uvm

# nvidia-smi
```

---
#### 安装 Tensorflow1.5+

参考文章：

1. [【解决】Centos7 安装NVDIA GPU+TensorFlow-gpu1.5.0](https://blog.csdn.net/jiede1/article/details/81062552)
2. [Install CUDA 9.1 and cuDNN 7 for TensorFlow 1.5.0](https://medium.com/@xinh3ng/install-cuda-9-1-and-cudnn-7-for-tensorflow-1-5-0-cda36239bc68)
3. [Ubuntu16.04配置tensorflow-gpu 1.5](https://blog.csdn.net/yslTech/article/details/79739357)



---
#### cudnn 安装配置

官网选择合适的版本下载，根据安装的 CUDA 版本下载，注意需要登陆才能下载 cudnn

下载路径：https://developer.nvidia.com/rdp/cudnn-download

如果下载的文件是如`cudnn-9.0-linux-x64-v7.solitairetheme8`，以`solitairetheme8`作为后缀名，可以通过下述命令修改后缀名：

```
cp cudnn-9.0-linux-x64-v7.solitairetheme8 cudnn-9.0-linux-x64-v7.tgz
```

然后，继续执行下列命令：

```
$ tar -xvf cudnn-9.0-linux-x64-v7.1.tgz
cuda/include/cudnn.h
cuda/NVIDIA_SLA_cuDNN_Support.txt
cuda/lib64/libcudnn.so
cuda/lib64/libcudnn.so.7　
cuda/lib64/libcudnn.so.7.1.4
cuda/lib64/libcudnn_static.a

```
上述是解压缩，以及得到的文件夹文件，接着只需要将这些文件移动到安装了 CUDA 的文件夹即可，如下命令：

```
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

```




---

#### nvidia显卡驱动与编译器版本的查询命令

1,首先验证你是否有nvidia的显卡（http://developer.nvidia.com/cuda-gpus这个网站查看你是否有支持gpu的显卡）：


```
$ lspci | grep -i nvidia
```



2,查看你的linux发行版本（主要是看是64位还是32位的）：

```
$ uname -m && cat /etc/*release
```

3,看一下gcc的版本：

```
$ gcc –version
```

4，查看NVIDIA显卡的驱动版本


```
$cat /proc/driver/nvidia/version
```

5，查看nvcc编译器的版本

```
nvcc -V i
```

6,/dev/nvidia* 这里的文件代表了本机的NVIDIA显卡，如：


```
foo@bar-serv2:/dev$ ls -l nvidia*
crw-rw-rw- 1 root root 195, 0 Oct 24 18:51 nvidia0
crw-rw-rw- 1 root root 195, 1 Oct 24 18:51 nvidia1
crw-rw-rw- 1 root root 195, 255 Oct 24 18:50 nvidiactl
```

表示本机有两块NVIDIA显卡

7，查看显卡名称以及驱动版本


```
nvidia-smi
nvidia-smi -a
```

8. ==**卸载驱动**==


```
To uninstall the NVIDIA Driver, run nvidia-uninstall:
$ sudo /usr/bin/nvidia-uninstall
```


---
#### 查看内核

##### 1. 查看正在使用的内核


```
uname -a
或
uname -r
```

##### 2. 查看系统中的全部内核


```
rpm -qa | grep kernel
```

---
### 问题

##### 安装 cuda9 遇到 `Not enough space on parition mounted at /.` 即`/`存储空间不足

可以先在空间比较充足的建立一个文件夹，如下命令：

```
# 在根目录下的opt文件夹中新建tmp文件夹，用作安装文件的临时文件夹
$ sudo mkdir /opt/tmp
# 再次执行安装的时候，加上 --tmpdir, 指定刚刚建立的文件夹
$ sudo sh cuda_8.0.44_linux.run --tmpdir=/opt/tmp/
```

##### make samples时候的错误

问题描述：

执行
```
~/NVIDIA_CUDA-9.0_Samples$ make
```
出现问题：

```
/usr/bin/ld: cannot find -lglut
collect2: error: ld returned 1 exit status
make[1]: *** [Mandelbrot] Error 1
```

但是，继续执行下列命令：

```
# cd 1_Utilities/deviceQuery
# ./deviceQuery
```
输出结果正常：

```
...
deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.0, CUDA Runtime Version = 9.0, NumDevs = 4
Result = PASS

```

##### python3.6的问题

采用 conda 安装一个 python3.6的环境后，使用 pip 想安装 tf 的时候，出现问题

```
Traceback (most recent call last):
  File "/root/anaconda3/envs/py36/bin/pip", line 4, in <module>
    import re
  File "/root/anaconda3/envs/py36/lib/python3.6/re.py", line 142, in <module>
    class RegexFlag(enum.IntFlag):
AttributeError: module 'enum' has no attribute 'IntFlag'
```



