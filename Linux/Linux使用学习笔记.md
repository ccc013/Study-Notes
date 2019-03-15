

---
#### 命令

##### 修改密码


```
[root@lc ~]# passwd
```

##### 查看命令用法

man 命令可以查看命令的用法，如：

```
man tail
```
打印出如下信息：

```
TAIL(1)                                             User Commands                                             TAIL(1)

NAME
       tail - output the last part of files

SYNOPSIS
       tail [OPTION]... [FILE]...

DESCRIPTION
       Print  the last 10 lines of each FILE to standard output.  With more than one FILE, precede each with a header
       giving the file name.  With no FILE, or when FILE is -, read standard input.

       Mandatory arguments to long options are mandatory for short options too.

       -c, --bytes=K
              output the last K bytes; or use -c +K to output bytes starting with the Kth of each file

       -f, --follow[={name|descriptor}]
              output appended data as the file grows;

              an absent option argument means 'descriptor'
 Manual page tail(1) line 1 (press h for help or q to quit)

```
输入回车键是往下移动一行，空格键是滚屏移动，q是退出


##### 文件操作

###### cat

cat 命令可以打印出文件内容到屏幕上
```
cat filename
```

###### tail

tail 用于输出文件的末尾 n 行信息

```
# 默认输出最后10行
tail test.txt
# 设置输出行数
tail -n 5 test.txt
# 实时追踪文件新的信息
tail -f test.txt

# 查看命令用法
man tail
```

##### 复制 cp


```
copy命令的功能是将给出的文件或目录拷贝到另一文件或目录中，同MSDOS下的copy命令一样，功能十分强大。
语法： cp [选项] 源文件或目录 目标文件或目录
说明：该命令把指定的源文件复制到目标文件或把多个源文件复制到目标目录中。
该命令的各选项含义如下：
- a 该选项通常在拷贝目录时使用。它保留链接、文件属性，并递归地拷贝目录，其作用等于dpR选项的组合。
- d 拷贝时保留链接。
- f 删除已经存在的目标文件而不提示。
- i 和f选项相反，在覆盖目标文件之前将给出提示要求用户确认。回答y时目标文件将被覆盖，是交互式拷贝。
- p 此时cp除复制源文件的内容外，还将把其修改时间和访问权限也复制到新文件中。
- r 若给出的源文件是一目录文件，此时cp将递归复制该目录下所有的子目录和文件。此时目标文件必须为一个目录名。
- l 不作拷贝，只是链接文件。

需要说明的是，为防止用户在不经意的情况下用cp命令破坏另一个文件，如用户指定的目标文件名已存在，用cp命令拷贝文件后，这个文件就会被新源文件覆盖，因此，建议用户在使用cp命令拷贝文件时，最好使用i选项。
```

假设复制源目录 为 dir1 ,目标目录为dir2。怎样才能将dir1下所有文件复制到dir2下了
如果dir2目录不存在，则可以直接使用

```
cp -r dir1 dir2
```

如果dir2目录已存在，则需要使用

```
cp -r dir1/. dir2
```

如果这时使用`cp -r dir1 dir2`,则也会将dir1目录复制到dir2中，明显不符合要求。
ps:dir1、dir2改成对应的目录路径即可。


```
cp -r /home/www/xxx/statics/. /home/www/statics
```

如果存在文件需要先删除

```
rm -rf /home/www/statics/*
```

否则会一个个文件提示你确认，使用`cp -rf `也一样提示




##### 查看文件/文件夹/分区大小


```
查看系统每个分区大小：
df -hl

查看当前文件夹大小: 
du -h --max-depth=0

查看指定文件夹：
du -h --max-depth=1 /path

要显示一个目录树及其每个子树的磁盘使用情况：
du /path
```

##### watch 监控命令


```
每隔一秒高亮显示网络链接数的变化情况
watch -n 1 -d netstat -ant

每隔两秒监视 python 的进程
watch -n 2 -d "ps -ef | grep python"
```

##### 压缩/解压缩

###### zip

```
压缩
zip –qr 压缩包名字 待打包文件夹名字

解压缩, -q 参数表示安静解压缩，不会显示解压缩过程
unzip -q 压缩包名字

解压缩到指定文件夹
unzip -q 压缩包名字 -d 文件夹名字


```

###### tar

```
.tar 
解包：tar xvf FileName.tar
打包：tar cvf FileName.tar DirName
（注：tar是打包，不是压缩！）
———————————————
.gz
解压1：gunzip FileName.gz
解压2：gzip -d FileName.gz
压缩：gzip FileName

.tar.gz 和 .tgz
解压：tar zxvf FileName.tar.gz
压缩：tar zcvf FileName.tar.gz DirName
```

###### 7z文件到解压缩实例

```
安装：Redhat、Fedora、Centos安装命令：yum install p7zip
安装：Debian、Ubuntu安装命令：apt-get install p7zip
```
解压实例
```
$ 7za x manager.7z -r -o /home/xx

x 代表解压缩文件，并且是按原始目录解压（还有个参数 e 也是解压缩文件，但其会将所有文件都解压到根下，而不是自己原有的文件夹下）manager.7z 是压缩文件，这里大家要换成自己的。如果不在当前目录下要带上完整的目录
-r 表示递归所有的子文件夹
-o 是指定解压到的目录，这里大家要注意-o后是没有空格的直接接目录
```
压缩文件：

```
$ 7z a -t7z -r manager.7z /home/manager/*

解释如下：
a 代表添加文件／文件夹到压缩包
-t 是指定压缩类型 一般我们定为7z
-r 表示递归所有的子文件夹，manager.7z 是压缩好后的压缩包名，/home/manager/* 是要压缩的目录，＊是表示该目录下所有的文件
```






##### 查看文件/文件夹个数


```
查看某个文件夹下文件的个数
ls -l|grep "^-"| wc -l

查看某个文件夹下文件的个数，包括子文件夹下的文件个数
ls -lR|grep "^-"| wc -l

查看某个文件夹下文件夹的个数
ls -l|grep "^d"| wc -l

查看某个文件夹下文件夹的个数，包括子文件夹下的文件夹个数
ls -lR|grep "^d"| wc -l

查看文件夹下所有的文件和文件夹。也就是统计ls -l命令所输出的行数
ls -l| wc -l
```

##### 进程相关


```
杀死python进程
ps -ef | grep python | cut -c 9-15 | xargs kill -s 9

查询python进程
ps -ef | grep python
```

##### 请求服务


```
curl -X POST 服务名 --data-binary @test.json
```

##### 服务器

```
跳转
ssh 服务器ip

传文件到指定ip的服务器
scp 要传的文件 root@目标ip:路径
例子: scp test.txt root@127.0.0.1:/home/

ssh 连接服务器后，通过rz和sz上传和下载文件
上传到服务器并且覆盖文件
rz -y 

下载文件到本地
sz filename
```

##### 查看端口

###### lsof

```
端口号查看某个端口是否被占用
lsof -i
或者
lsof -i:端口号
```

###### netstat


```
查看 80 端口使用情况
netstat -anp | grep 80
```







