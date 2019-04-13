一文总结 Python 基础知识，目录如下：

1. 简介和环境配置
2. 变量和简单的数据类型
3. 列表和元组
4. 字典
5. if 条件语句
6. for / while循环语句
7. 函数
8. 类
9. 文件和异常
10. 测试代码

------

### 1. 简介和环境配置

#### 1.1 简介

> Python 是由 Guido van Rossum 在八十年代末和九十年代初，在荷兰国家数学和计算机科学研究所设计出来的。目前是最常用也是最热门的一门编程语言之一，应用非常广泛。

Python 是一个**高层次的结合了解释性、编译性、互动性和面向对象**的脚本语言。

Python 的设计具有很强的可读性，相比其他语言经常使用英文关键字，其他语言的一些标点符号，它具有比其他语言更有特色语法结构。

优点：

- **Python 是一种解释型语言：** 这意味着开发过程中没有了编译这个环节。类似于PHP和Perl语言。
- **Python 是交互式语言：** 这意味着，您可以在一个 Python 提示符 >>> 后直接执行代码。
- **Python 是面向对象语言:** 这意味着Python支持面向对象的风格或代码封装在对象的编程技术。
- **Python 是初学者的语言：**Python 对初级程序员而言，是一种伟大的语言，它支持广泛的应用程序开发，从简单的文字处理到 WWW 浏览器再到游戏。

缺点：

- 运行速度比 `C++`、`C#`、`Java` 慢。这是缺乏即时优化器；
- 空格缩减的句法限制会给初学者制造一些困难；
- 没有提供如同 `R` 语言的先进的统计产品；
- 不适合在低级系统和硬件上开发

#### 1.2 环境搭建

##### Python 下载

Python 官网可以查看最新的源码、入门教程、文档，以及和 Python 相关的新闻资讯，链接如下：

https://www.python.org/

官方文档下载地址：

https://www.python.org/doc/

##### Python 安装

Python 在多个平台上都可以使用，不同的平台有不同的安装方式，下面是不同平台上安装的方法：

**Unix & Linux 平台安装 Python **

在 Unix & Linux 平台安装 Python 的简单步骤如下：

- 打开 WEB 浏览器访问 https://www.python.org/downloads/source/
- 选择适用于 `Unix/Linux `的源码压缩包。
- 下载及解压压缩包。
- 如果你需要自定义一些选项修改 `Modules/Setup`
- **执行** ./configure 脚本
- `make`
- `make install`

执行完上述步骤后，Python 会安装在 `/usr/local/bin` 目录中，Python 库安装在 `/usr/local/lib/pythonXX`，XX 为你使用的 Python 的版本号。

**Window 平台安装 Python**

安装步骤如下：

- 打开 WEB 浏览器访问 https://www.python.org/downloads/windows/
- 在下载列表中选择Window平台安装包，包格式为：`python-XYZ.msi` 文件 ， XYZ 为你要安装的版本号。
- 要使用安装程序 `python-XYZ.msi`, Windows 系统必须支持 `Microsoft Installer 2.0` 搭配使用。只要保存安装文件到本地计算机，然后运行它，看看你的机器支持 MSI。Windows XP 和更高版本已经有 MSI，很多老机器也可以安装 MSI。
- 下载后，双击下载包，进入 Python 安装向导，安装非常简单，你只需要使用默认的设置一直点击"下一步"直到安装完成即可。

**MAC 平台安装 Python**

MAC 系统一般都自带有 `Python2.x`版本 的环境，你也可以在链接 <https://www.python.org/downloads/mac-osx/> 上下载最新版安装。

##### 环境变量配置

环境变量是由操作系统维护的一个命名的字符串，这些变量包含可用的命令行解释器和其他程序的信息。`path`(路径)存储在环境变量中。

Unix 或 Windows 中路径变量为`PATH`（UNIX 区分大小写，Windows 不区分大小写）。

在 Mac OS 中，安装程序过程中改变了 python 的安装路径。如果你需要在其他目录引用Python，你必须在 path 中添加 Python 目录。

**Unix/Linux 设置环境变量**

有以下三种方法：

- **在 `csh shell` 中输入**：

```shell
setenv PATH "$PATH:/usr/local/bin/python"
```

- **在 bash shell (Linux)输入:** 

```shell
export PATH="$PATH:/usr/local/bin/python" 
```

- **在 sh 或者 ksh shell:** 输入 

```shell
PATH="$PATH:/usr/local/bin/python" 
```

**注意:** ·/usr/local/bin/python· 是 Python 的安装目录。

**Window 设置环境变量**

两种方法设置环境变量。

第一种是**在命令提示框中(cmd) :** 输入 

```shell
path=%path%;C:\Python 
```

**注意:** `C:\Python` 是Python的安装目录。

也可以通过以下方式设置：

- 右键点击"计算机"，然后点击"属性"
- 然后点击"高级系统设置"
- 选择"系统变量"窗口下面的 "Path",双击即可！
- 然后在 "Path" 行，添加 python 安装路径即可，所以在后面，添加该路径即可。 **ps：记住，路径直接用分号"；"隔开！**
- 最后设置成功以后，在`cmd`命令行，输入命令`"python"`，就可以有相关显示。

##### Anaconda 安装

目前 Python 有两个版本，Python 2 和 Python 3，并且两个版本还有比较大的差异，所以推荐使用 `Anaconda` 库来管理不同的环境。

官网地址：

https://www.anaconda.com/

以下安装步骤参考 [Anaconda介绍、安装及使用教程](https://zhuanlan.zhihu.com/p/32925500)

**1.Linux 安装**

1.前往[官方下载页面](http://link.zhihu.com/?target=https%3A//www.anaconda.com/download/%23linux)下载。有两个版本可供选择：Python 3.6 和 Python 2.7。

2. 启动终端，在终端中输入命令 **\*md5sum /path/filename*** 或 **\*sha256sum /path/filename***

- 注意：将该步骤命令中的 **\*/path/filename*** 替换为文件的实际下载路径和文件名。其中，path是路径，filename为文件名。
- 强烈建议：

① 路径和文件名中不要出现空格或其他特殊字符。

② 路径和文件名最好以英文命名，不要以中文或其他特殊字符命名。

3. 根据 Python 版本的不同有选择性地在终端输入命令：

▫ Python 3.6： **bash ~/Downloads/Anaconda3-5.0.1-Linux-x86_64.sh**

▫ Python 2.7： **bash ~/Downloads/Anaconda2-5.0.1-Linux-x86_64.sh**

- 注意：

① 首词 `bash` 也需要输入，无论是否用的 Bash shell。

② 如果你的下载路径是自定义的，那么把该步骤路径中的 `~/Downloads` 替换成你自己的下载路径。

③ 除非被要求使用 root 权限，否则均选择“Install Anaconda as a user”。

4. 安装过程中，看到提示“In order to continue the installation process, please review the license agreement.”（“请浏览许可证协议以便继续安装。”），点击“Enter”查看“许可证协议”。

5. 在“许可证协议”界面将屏幕滚动至底，输入“yes”表示同意许可证协议内容。然后进行下一步。

6. 安装过程中，提示“Press Enter to accept the default install location, CTRL-C to cancel the installation or specify an alternate installation directory.”（“按回车键确认安装路径，按'CTRL-C'取消安装或者指定安装目录。”）如果接受默认安装路径，则会显示**PREFIX=/home/<user>/anaconda<2 or 3>** 并且继续安装。安装过程大约需要几分钟的时间。

- 建议：直接接受默认安装路径。

7. 安装器若提示“Do you wish the installer to prepend the Anaconda<2 or 3> install location to PATH in your /home/<user>/.bashrc ?”（“你希望安装器添加Anaconda安装路径在 **/home/<user>/.bashrc** 文件中吗？”），建议输入“yes”。

- 注意：

① 路径 **/home/<user>/.bash_rc** 中 **“<user>”** 即进入到家目录后你的目录名。

② 如果输入“no”，则需要手动添加路径，否则conda将无法正常运行。

8. 当看到“Thank you for installing Anaconda<2 or 3>!”则说明已经成功完成安装。

9. 关闭终端，然后再打开终端以使安装后的 Anaconda 启动。或者直接在终端中输入 `source ~/.bashrc` 也可完成启动。

10. 验证安装结果。可选用以下任意一种方法：

① 在终端中输入命令 `condal list` ，如果 Anaconda 被成功安装，则会显示已经安装的包名和版本号。

② 在终端中输入`python`。这条命令将会启动 Python 交互界面，如果 Anaconda 被成功安装并且可以运行，则将会在 Python 版本号的右边显示“Anaconda custom (64-bit)”。退出 Python 交互界面则输入 `exit()` 或 `quit()` 即可。

③ 在终端中输入 **anaconda-navigator** 。如果 Anaconda 被成功安装，则 Anaconda Navigator 将会被启动。



**2.Window 安装**

1. 前往[官方下载页面](http://link.zhihu.com/?target=https%3A//docs.anaconda.com/anaconda/install/windows)下载。有两个版本可供选择：Python 3.6 和 Python 2.7，选择之后根据自己操作系统的情况点击“64-Bit Graphical Installer”或“32-Bit Graphical Installer”进行下载。

2. 完成下载之后，双击下载文件，启动安装程序。

- 注意：

① 如果在安装过程中遇到任何问题，那么暂时地关闭杀毒软件，并在安装程序完成之后再打开。

② 如果在安装时选择了“为所有用户安装”，则卸载 Anaconda 然后重新安装，只为“我这个用户”安装。

3. 选择“Next”。

4. 阅读许可证协议条款，然后勾选“I Agree”并进行下一步。

5. 除非是以管理员身份为所有用户安装，否则仅勾选“Just Me”并点击“Next”。

6. 在“Choose Install Location”界面中选择安装 Anaconda 的目标路径，然后点击“Next”。

- 注意：

① 目标路径中**不能**含有**空格**，同时不能是**“unicode”**编码。

② 除非被要求以管理员权限安装，否则不要以管理员身份安装。

7. 在“Advanced Installation Options”中**不要**勾选“Add Anaconda to my PATH environment variable.”（“添加Anaconda至我的环境变量。”）。因为如果勾选，则将会影响其他程序的使用。如果使用 Anaconda，则通过打开 Anaconda Navigator或者在开始菜单中的“Anaconda Prompt”（类似macOS中的“终端”）中进行使用。

除非你打算使用多个版本的 Anaconda 或者多个版本的 Python，否则便勾选“Register Anaconda as my default Python 3.6”。

然后点击“Install”开始安装。如果想要查看安装细节，则可以点击“Show Details”。

8. 点击“Next”。

9. 进入“Thanks for installing Anaconda!”界面则意味着安装成功，点击“Finish”完成安装。

- 注意：如果你不想了解“Anaconda云”和“Anaconda支持”，则可以**不勾选**“Learn more about Anaconda Cloud”和“Learn more about Anaconda Support”。

10. 验证安装结果。可选以下任意方法：

① “开始 → Anaconda3（64-bit）→ Anaconda Navigator”，若可以成功启动Anaconda Navigator则说明安装成功。

② “开始 → Anaconda3（64-bit）→ 右键点击Anaconda Prompt → 以管理员身份运行”，在Anaconda Prompt中输入 **conda list** ，可以查看已经安装的包名和版本号。若结果可以正常显示，则说明安装成功。



**3.Mac 安装**

两种安装方法，第一种是**图形界面安装**：

1. 前往[官方下载页面](http://link.zhihu.com/?target=https%3A//www.anaconda.com/downloads%23macos)下载。有两个版本可供选择：Python 3.6 和 Python 2.7，目前推荐选择前者，也可以根据自己学习或者工作需求选择不同版本。选择版之后点击“64-Bit Graphical Installer”进行下载。

2. 完成下载之后，双击下载文件，在对话框中“Introduction”、“Read Me”、“License”部分可直接点击下一步

3. “Destination Select”部分选择“Install for me only”并点击下一步。

- 注意：若有错误提示信息“You cannot install Anaconda in this location”则重新选择“Install for me only”并点击下一步。

4.“Installation Type”部分，可以点击“Change Install Location”来改变安装位置。标准的安装路径是在用户的家目录下。若选择默认安装路径，则直接点击“Install”进行安装。

5.等待“Installation”部分结束，在“Summary”部分若看到“The installation was completed successfully.”则安装成功，直接点击“Close”关闭对话框。

6.在 mac 的 Launchpad 中可以找到名为 “Anaconda-Navigator” 的图标，点击打开。

7.若“Anaconda-Navigator”成功启动，则说明真正成功地安装了Anaconda；如果未成功，请务必仔细检查以上安装步骤。

8.完成安装

第二种方法，**命令行安装**：

1.前往[官方下载页面](http://www.anaconda.com/downloads%23macos)下载。有两个版本可供选择：Python 3.6 和 Python 2.7，目前推荐选择前者，也可以根据自己学习或者工作需求选择不同版本。选择版之后点击“64-Bit Graphical Installer”进行下载。

2.完成下载之后，在mac的Launchpad中找到“其他”并打开“终端”。

▫ 安装Python 3.6： `bash ~/Downloads/Anaconda3-5.0.1-MacOSX-x86_64.sh`

▫ 安装Python 2.7： `bash ~/Downloads/Anaconda2-5.0.1-MacOSX-x86_64.sh`

如果下载路径是自定义，将路径中的`~/Downloads` 替换为你下载的路径，此外如果更改过下载的文件名，那么也将 `Anaconda3-5.0.1-MacOSX-x86_64.sh` 更改为你修改的文件名。

**ps**：强烈建议不要修改文件名，如果重命名，也要采用**英文**进行命名。

3.安装过程中，看到提示“In order to continue the installation process, please review the license agreement.”（“请浏览许可证协议以便继续安装。”），点击“Enter”查看“许可证协议”。

4. 在“许可证协议”界面将屏幕滚动至底，输入“yes”表示同意许可证协议内容。然后进行下一步。

5. 安装过程中，提示“Press Enter to confirm the location, Press CTRL-C to cancel the installation or specify an alternate installation directory.”（“按回车键确认安装路径，按'CTRL-C'取消安装或者指定安装目录。”）如果接受默认安装路径，则会显示 **PREFIX=/home/<user>/anaconda<2 or 3>**  并且继续安装。安装过程大约需要几分钟的时间。

- 建议：直接接受默认安装路径。

6. 安装器若提示“Do you wish the installer to prepend the Anaconda install location to PATH in your /home/<user>/.bash_profile ?”（“你希望安装器添加Anaconda安装路径在**/home/<user>/.bash_profile** 文件中吗？”），建议输入“yes”。

- 注意：

① 路径 **/home/<user>/.bash_profile** 中 **<user>** 即进入到家目录后你的目录名。

②如果输入“no”，则需要手动添加路径。添加 **export PATH="/<path to anaconda>/bin:$PATH"** 在 **.bashrc** 或者 **.bash_profile** 中。其中， **<path to anaconda>**替换为你真实的Anaconda安装路径。

7. 当看到“Thank you for installing Anaconda!”则说明已经成功完成安装。

8. 关闭终端，然后再打开终端以使安装后的 Anaconda 启动。

9. 验证安装结果。可选用以下任意一种方法：

- 在终端中输入命令 **condal list** ，如果 Anaconda 被成功安装，则会显示已经安装的包名和版本号。
- 在终端中输入 **python** 。这条命令将会启动 Python 交互界面，如果 Anaconda 被成功安装并且可以运行，则将会在Python版本号的右边显示“Anaconda custom (64-bit)”。退出 Python 交互界面则输入 **exit()** 或 **quit()** 即可。
- 在终端中输入 **anaconda-navigator** 。如果 Anaconda 被成功安装，则 Anaconda Navigator 的图形界面将会被启动。

##### Anaconda 使用

简单介绍几个 Anaconda 的基本使用命令：

1.查看版本

```shell
conda --version
```

2.创建环境

```shell
# 基本命令
conda create --name <env_name> <package_names>
# 例子：创建一个 python3.6 的环境, 环境名字为 py36
conda create -n py36 python=3.6
```
3.删除环境

```shell
conda remove -n py36 --all
```

4.激活环境

```shell
source activate py36
```

5.退出环境

```shell
source deactivate
```

##### Jupyter Notebook 安装

**1.简介**

`Jupyter Notebook` 是一个开源的 Web 应用程序，允许用户创建和共享包含代码、方程式、可视化和文本的文档。它的用途包括：数据清理和转换、数值模拟、统计建模、数据可视化、机器学习等等。它具有以下优势：

- 可选择语言：支持超过40种编程语言，包括 Python、R、Julia、Scala等。
- 分享笔记本：可以使用电子邮件、Dropbox、GitHub 和 Jupyter Notebook Viewer 与他人共享。
- 交互式输出：代码可以生成丰富的交互式输出，包括 HTML、图像、视频、LaTeX 等等。
- 大数据整合：通过 Python、R、Scala 编程语言使用 Apache Spark 等大数据框架工具。支持使用 pandas、scikit-learn、ggplot2、TensorFlow 来探索同一份数据。

**2.安装**

有两种安装的方式，分别是通过 `Anaconda` 安装和命令行安装。

第一种方式就是安装 `Anaconda` ，它附带 Jupyter Notebook 等常用的科学计算和数据科学软件包。

第二种通过命令行安装，命令如下，根据安装的 Python 选择对应的命令安装即可。

```shell
# Pyhton 3
python3 -m pip install --upgrade pip
python3 -m pip install jupyter

# Python 2
python -m pip install --upgrade pip
python -m pip install jupyter
```

**3.运行和使用**

运行 Jupyter Notebook 的方法很简单，只需要在系统的终端(Mac/Linux 的 Terminal，Window 的 cmd) 运行以下命令即可：

```shell
jupyter notebook
```

官方文档地址如下：

https://jupyter.org/documentation

##### Pycharm 安装

Pycharm 是 Python 的一个 IDE，配置简单，功能强大，而且对初学者友好，下面介绍如何安装和简单配置 Pycharm。

**1.安装**

**Pycharm** 提供 **免费的社区版** 与 **付费的专业版**。专业版额外增加了一些功能，如项目模板、远程开发、数据库支持等。个人学习 **Python** 使用免费的社区版已足够。
pycharm社区版：[PyCharm :: Download Latest Version of PyCharm](http://link.zhihu.com/?target=http%3A//www.jetbrains.com/pycharm/download/)

安装过程照着提示一步步操作就可以了。注意安装路径尽量不使用带有 **中文或空格** 的目录，这样在之后的使用过程中减少一些莫名的错误。

**2.配置**

**Pycharm** 提供的配置很多，这里讲几个比较重要的配置

编码设置：

**Python** 的编码问题由来已久，为了避免一步一坑，**Pycharm** 提供了方便直接的解决方案

![img](https://pic1.zhimg.com/80/v2-7b7bf6cb827f253a21257c187c047fa4_hd.jpg)

在 **IDE Encoding** 、**Project Encoding** 、**Property Files** 三处都使用 **UTF-8** 编码，同时在文件头添加

```python
#-*- coding: utf-8 -
```

这样在之后的学习过程中，或多或少会避免一些编码坑。

解释器设置：

当有多个版本安装在电脑上，或者需要管理虚拟环境时，**Project Interpreter** 提供方便的管理工具。


![img](https://pic3.zhimg.com/80/v2-29679fb60fcf0d0d4f948d5c85726e86_hd.jpg)

在这里可以方便的切换 **Python** 版本，添加卸载库等操作。

修改字体：

在 **Editor** → **Font** 选项下可以修改字体，调整字体大小等功能。

![img](https://pic2.zhimg.com/80/v2-5128036e99620e4c20f37a58eca347e1_hd.jpg)

快捷键设置：

在 windows 下一些最常用的默认快捷键：

![img](https://pic3.zhimg.com/80/v2-2c95f7f722a4342d1db875c03ef45daa_hd.jpg)

**Pycharm** 也为不同平台的用户提供了定制的快捷键方案，习惯了用**emacs**、**vim**、**vs**的同学，可以直接选择对应的方案。

![img](https://pic2.zhimg.com/80/v2-61a55f048193449a4026c81ff236eb99_hd.jpg)

同时，**Pycharm** 也提供了自定义快捷键的功能。

![img](https://pic2.zhimg.com/80/v2-53f98fdefa195d1179c108408f24180d_hd.jpg)

修改完成之后就去试试效果吧！

**3.调试**

强大的 **Pycharm** 为我们提供了方便易用的断点调试功能，步骤如下图所示：

![img](https://pic2.zhimg.com/80/v2-2e0306477e8ab7d354b05a09b475729d_hd.jpg)

简单介绍一下调试栏的几个重要的按钮作用：

![img](https://pic4.zhimg.com/80/v2-0353997a0ba329f1211451ef5028ed13_hd.jpg)

**Resume Program**：断点调试后，点击按钮，继续执行程序；

![img](https://pic4.zhimg.com/80/v2-a8c0d6061d0a68efaf22f680c18385fb_hd.jpg)

**Step Over** ：在单步执行时，在函数内遇到子函数时不会进入子函数内单步执行，而是将子函数整个执行完再停止，也就是把子函数整个作为一步。有一点,经过我们简单的调试,在不存在子函数的情况下是和**Step Into**效果一样的（简而言之，越过子函数，但子函数会执行）；

![img](https://pic3.zhimg.com/80/v2-7cda50d4e2f7db7b2754f02a2344e432_hd.jpg)

**Step Into**：单步执行，遇到子函数就进入并且继续单步执行（简而言之，进入子函数）；

![img](https://pic2.zhimg.com/80/v2-82ce1dc84514744a8fd46b9226454655_hd.jpg)

**Step Out** ： 当单步执行到子函数内时，用step out就可以执行完子函数余下部分，并返回到上一层函数。

如果程序在某一步出现错误，程序会自动跳转到错误页面，方便我们查看错误信息
更详细的关于调试的知识参考之前的一篇文章：

[如何在 Python 中使用断点调试 - Crossin的编程教室 - 知乎专栏](https://zhuanlan.zhihu.com/p/21304838)

另外，PyCharm 还提供了一个方便调试的小功能，但隐藏得比较深，参见：

[pycharm 如何程序运行后，仍可查看变量值？ - 知乎专栏](https://zhuanlan.zhihu.com/p/27062841)

**4.Python 控制台**

为了方便用户，**Pycharm** 提供了另一个贴心的功能，将 **Python shell** 直接集成在软件中，调出方法如下：

![img](https://pic3.zhimg.com/80/v2-b3dac053584f6746e2b5d96e0eeb6a92_hd.jpg)



------

#### 参考

- 《Python 编程从入门到实践》
- [everything-about-python-from-beginner-to-advance-level](https://medium.com/fintechexplained/everything-about-python-from-beginner-to-advance-level-227d52ef32d2)
- [Python 基础教程](http://www.runoob.com/python/python-tutorial.html)
- [Anaconda介绍、安装及使用教程](https://zhuanlan.zhihu.com/p/32925500)
- [最详尽使用指南：超快上手Jupyter Notebook](https://zhuanlan.zhihu.com/p/32320214)
- [喏，你们要的 PyCharm 快速上手指南](https://zhuanlan.zhihu.com/p/26066151)
- [一天快速入门python](https://mp.weixin.qq.com/s/odBnvjO6dgc8HzV9N-aTzg)
- [廖雪峰老师的教程](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000)
- [超易懂的Python入门级教程，赶紧收藏！](https://mp.weixin.qq.com/s/ja8lZvEzZEuzC0C9kkXpag)

