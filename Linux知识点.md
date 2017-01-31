### Linux知识点

* MMU 是Memory Manage Unit的缩写，即是存储管理单元，其功能是和物理内存之间进行地址转换 在CPU和物理内存之间进行地址转换，将地址从逻辑空间映映射到物理地址空间。

------

##### 文件、目录与磁盘格式

* **/dev** 是用来挂载外设的映射目录

* Linux进程间通信方式有：**消息队列，命名管道，信号量，共享内存，Berkeley套接字**等。临界区是每个进程中访问临界资源的那段代码称，每次只准许一个进程进入临界区，进入后不允许其他进程进入。不论是硬件临界资源，还是软件临界资源，多个进程必须互斥地对它进行访问。**它可以作为线程间通信方式而不能作为进程间通信方式，因为进程间内存是相互隔离的。**

* 使用pthread库的多线程程序编译时需要加**-pthread**连接参数。

* 属于网络操作系统的是`Unix, Linux, WINDOWS DT, NETWARE 4.11, LAN MANGER 4.0`。

* `fopen`和`exit`函数需要进入内核才能完成调用。

* WEB服务器配置文件  http.conf

  启动脚本配置文件   initd.conf

  samba脚本          rc.samba

  samba服务配置文件  smb.conf

* OSI七层，每层包含的协议如下：
  1. 物理层： RJ45 、 CLOCK 、 IEEE802.3 （中继器，集线器，网关） - 
  2. 数据链路： PPP 、 FR 、 HDLC 、 VLAN 、 MAC （网桥，交换机） - 
  3. 网络层： IP 、 ICMP 、 ARP 、 RARP 、 OSPF 、 IPX 、 RIP 、 IGRP 、 （路由器） - 
  4. 传输层： TCP 、 UDP 、 SPX - 
  5. 会话层： NFS 、 SQL 、 NETBIOS 、 RPC - 
  6. 表示层： JPEG 、 MPEG 、 ASII - 
  7. 应用层： FTP 、 DNS 、 Telnet 、 SMTP 、 HTTP 、 WWW 、 NFS

* **文件默认权限666     目录默认权限777 。 实际权限则减去umas**

* **hosts文件是Linux系统上一个负责ip地址与域名快速解析的文件**，以ascii格式保存在/etc/目录下。hosts文件包含了ip地址与主机名之间的映射，还包括主机的别名。**在没有域名解析服务器的情况下，系统上的所有网络程序都通过查询该文件来解析对应于某个主机名的ip地址，否则就需要使用dns服务程序来解决**。通过可以将常用的域名和ip地址映射加入到hosts文件中，实现快速方便的访问

* ​

  ​

------

##### shell与shell script

*   bash中有两个内置的命令declare 和 typeset 可用于创建变量。除了使用内置命令来创建和设置变量外，还可以直接赋值，格式为：**变量名=变量值**

    * 变量名前面不应加美元“\$”符号。
    * 等号“=”前后不可以有空格。
    * Shell中不需要显式的语法来声明变量。
    * 变量名不可以直接和其他字符相连，如果想相连，必须用括号：` echo “this is $(he)llo!”`

*   **`＄＄`表示当前命令的进程标识数。** 

*   **`＄*`表示所有位置参量，例如`＄1`、`＄2`等。** 

*   **`＄@`与`＄*`类似，但当用双引号进行转义时，"`＄@`"能够分解多个参数，而"`＄*`"合并成一个参数。** 

*   **`＄#`包括位置参数的个数，但是不包括命令名。**

*   csh:调用 C shell。

      Tcsh是csh的增强版，并且完全兼容csh。它不但具有csh的全部功能，还具有命令行编辑、拼写校正、可编程字符集、历史纪录、 [作业控制](http://baike.baidu.com/view/4509410.htm) 等功能，以及C语言风格的语法结构。

      AWK 是一种优良的文本处理工具， [Linux](http://baike.baidu.com/view/1634.htm) 及 [Unix](http://baike.baidu.com/view/8095.htm) 环境中现有的功能最强大的数据处理引擎之一, AWK 提供了极其强大的功能：可以进行样式装入、 [流控制](http://baike.baidu.com/view/1292763.htm) 、数学 [运算符](http://baike.baidu.com/view/425996.htm) 、进程 [控制语句](http://baike.baidu.com/view/1359886.htm) 甚至于内置的变量和函数。

      SED: Stream EDitor

*   ​


------

##### 命令

* tar是操作.tar的命令

  gzip是压缩.gz压缩包的命令

  compress：压缩.Z文件

  uncompress：解压缩.Z文件

* 为了将当前目录下的归档文件myftp. tgz解压缩到/tmp目录下，用户可以使用命令`tar xvzf myftp. tgz –C /tmp`，通常情况下**解压.tar.gz和.tgz等格式的归档文件就可以直接使用`tar xvzf`**；因为要**解压到指定目录下，所以还应在待解压文件名后加上-C（change to directory）参数**

* 关机命令有`halt init 0`, `poweroff `  `shutdown -h 时间`，其中shutdown是最安全的

* 重启命令有`reboot`, `init 6`,` shutdow -r 时间`

* `grep "牛客" fileName |wc –l` /*统计有多少个牛客信息*

* tar 主要命令：

  -c 创建包
  -x 解包
  -t 列出包中的内容
  -r 增加文件到指定包中
  -u 更新包中的文件

  这五个是独立的命令，压缩解压都要用到其中一个，可以和别的命令连用但只能用其中一个。下面的参数是根据需要在压缩或解压档案时可选的。

  可选命令：

  -j 创建或解开包时 使用bzip2 进行压缩或解压

  -z 创建或解开包时 使用gzip 进行压缩或解压

  -Z 创建或解开包时 使用compress 进行压缩或解压

  -f 后面跟指定的包文件名

  -v 显示打包/解包过程

  -C 指定解包后的路径

  因此，文件aaa打包为bak.tar的命令是`tar -cf bak.tar aaa`

* 统计日志中ip登陆次数的命令有：

  * `cat catalina.log  | awk -F ' ' '{print $3}' | sort | uniq -c | wc -l`
  * `cat catalina.log  | awk '{print $3}' | sort -klnr | uniq -c | wc -l`

  知识点有：

  1、`awk -F ' ' '{print $3}' `指定空格是分隔符进行分割，取第三个。（不指定**默认分隔符也是空格**）

  2、`uniq -c`（uniq命令可以去除排序过的文件中的重复行，因此uniq经常和sort合用。也就是说，**为了使uniq起作用，所有的重复行必须是相邻的。**参数 - c ：进行计数）

  3、`wc -l `行计数。

* ​

  ​

  ​





