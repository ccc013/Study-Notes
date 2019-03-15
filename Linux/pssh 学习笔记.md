
参考文章：
- [批量管理工具-pdsh|pssh](https://blog.opskumu.com/pdsh-pssh.html)

服务器多了，有一个烦恼就是如何批量快速操作一堆服务器。这里我推荐一下经常使用利器 pssh。这个工具给我的工作带来了莫大的帮助。

#### 简介

pssh是一款开源的软件，使用python实现。用于批量ssh操作大批量机器。pssh的项目地址 

https://code.google.com/p/parallel-ssh/

#### 安装
在pssh的项目主页找到相应的版本，下载到我们的服务器上，解压后执行python setup.py安装。下面以pssh-2.3的安装为例


```
wget 'https://parallel-ssh.googlecode.com/files/pssh-2.3.tar.gz'
 
#如果上面链接无法下载（被墙）可以换我这个链接
wget 'http://files.opstool.com/files/pssh-2.3.tar.gz' 

tar -xzvf pssh-2.3.tar.gz
cd pssh-2.3
python setup.py install
```

#### 常用的方法


```
# pssh使用帮助

pssh --help

# pssh查看所有服务器的uptime
# -h list 指定了执行命令的机器列表
# -A表示提示输入密码（如果机器都是ssh key打通的则无需加-A）

pssh -i -A -h list 'uptime'
例子：pssh -i -A -h hosts.txt "cd /home/ && ls -l"

# 使用pscp向一堆机器分发文件
pscp -h list  localfile   remote_dir
例：pscp -h hosts.txt -l root -Av test.zip /home/

# 例如：远程主机名是 tecmint, 传送压缩包 wine-1.7.55.tar.bz2 到远程服务器的 /tmp/ 文件夹
pscp -h hosts.txt -l tecmint -Av wine-1.7.55.tar.bz2 /tmp/

# 传送文件夹
pscp -h myscphosts.txt -l tecmint -Av -r Android\ Games/ /tmp/
OR
pscp.pssh -h myscphosts.txt -l tecmint -Av -r Android\ Games/ /tmp/


# 从一堆机器中拷贝文件到中心机器
pslurp -h list /etc/hosts local_dir
```
一般用一个文本文件保存远程服务器的用户名和 ip 地址、端口地址信息，格式如'username@ip:port'，例如 `root@10.10.12.1:80`，

#### 常见问题

1. 如果你遇到这样的错误：`IOError: [Errno 4] Interrupted system call `  建议升级python版本到python2.7

2. 执行多句shell指令时候 用 && 不然带着路径执行 sh 或者其他: `pssh -i -h -A ip.txt "cd /home/ && sh run.sh"`



