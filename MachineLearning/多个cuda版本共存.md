
参考文章：

1. [ubuntu 安装cuda9.0+cudnn7.1－与cuda8.0共存](https://blog.csdn.net/lovebyz/article/details/80704800)
2. [ubuntu 同时安装cuda8.0与cuda9.0,cuda9.1](https://blog.csdn.net/weixin_32820767/article/details/80421913)


---
主要参考文章 [ubuntu 安装cuda9.0+cudnn7.1－与cuda8.0共存](https://blog.csdn.net/lovebyz/article/details/80704800)

关键操作如下：

```
rm –rf /usr/local/cuda  
  
ln -s /usr/local/cuda-9.0 /usr/local/cuda  
```
上述是安装一个`cuda-9.0`版本的操作，这样是创建新的软链接。

另外，需要注意驱动的安装，驱动是可以向下兼容的，所以不需要担心更换回低版本的`cuda`的时候，驱动的兼容问题，
但需要先卸载旧版本的驱动，命令如下：
```
To uninstall the NVIDIA Driver, run nvidia-uninstall:
$ sudo /usr/bin/nvidia-uninstall

或者如下，如果是384版本的驱动

sudo apt-get remove --purge nvidia-384 nvidia-modprobe nvidia-settings
```

在安装`cuda`的时候，如下设置：

```
Install the CUDA 9.0 Toolkit?
(y)es/(n)o/(q)uit: y

Enter Toolkit Location
 [ default is /usr/local/cuda-9.0 ]: 

/usr/local/cuda-9.0 is not writable.
Do you wish to run the installation with 'sudo'?
(y)es/(n)o: y

Please enter your password: 
Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: n　　<--(链接可在完成安装之后软cuda-9.0链接过来

Install the CUDA 9.0 Samples?
(y)es/(n)o/(q)uit: y

Enter CUDA Samples Location
 [ default is /home/byz ]: /home/byz/app/cuda_9   <--(刚才新建的文件夹

```

