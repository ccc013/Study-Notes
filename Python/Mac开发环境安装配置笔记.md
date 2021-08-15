# 前言

记录下Mac电脑的开发环境安装配置，主要包括：

- 安装&使用Homebrew
- 安装使用 git
- 安装 anaconda，配置 python3 环境
- 安装 jupyter notebook
- 安装 pycharm
- 安装常用的 python 库，包括 numpy、sklearn、pandas等



------

# 安装 & 使用Homebrew

首先是先安装 Home-brew, 它是一款**软件包管理工具**，通过它可以很方便的安装/卸载软件工具等，类似于 Linux 下的 apt-get，node 的npm等包管理工具。

Homebrew将工具安装在自己创建的 `/usr/local/Cellar`目录下，并在`/usr/local/bin`建立这些工具的符号链接。

安装方法有两种方式，参考文章：

https://blog.csdn.net/zzq900503/article/details/80404314

## 官网安装

mac自带 ruby 环境，在终端下输入下面的指令即可完成安装：

```
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

这个命令分两步执行：

1. `curl`命令下载安装文件，文件地址为：`https://raw.githubusercontent.com/Homebrew/install/master/install`，下载到本地，文件名为 `install`

2. 接着执行 `ruby -e install`，不过这一步因为是连接国外的网，速度会比较慢

安装完成的结果如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/brew_install.png)

我们输入命令 `brew help`就可以查看可以使用的一些命令：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/brew_help.png)

具体含义如下：

|                  命令                  |                  说明                   |
| :------------------------------------: | :-------------------------------------: |
|      brew search [TEXT\|/REGEX/]       |                搜索软件                 |
|       brew (info\|home\|options)       |              查询软件信息               |
|        brew install FORMULA...         |                安装软件                 |
|              brew update               |          更新brew和所有软件包           |
|        brew update [FORMULA...]        |               更新软件包                |
|       brew uninstall FORMULA...        |               卸载软件包                |
|         brew list [FORMULA...]         |         罗列所有已安装的软件包          |
|              brew config               |         查询brew命令的使用手册          |
|              brew doctor               |           检查系统的潜在问题            |
| brew install --verbose --debug FORMULA | 安装软件包，打印详细信息并开启debug功能 |
|     brew create [URL [--no-fetch]]     |               创建软件包                |
|         brew edit [FORMULA...]         |             编辑软件包源码              |



## 第三方安装

主要思路为，替换brew的镜像源。

1.先把https://raw.githubusercontent.com/Homebrew/install/master/install 文件下载下来(使用浏览器打开，另存为也可以)，把文件命名为install.txt

接下来，修改install.txt文件：
应该在第7行左右，(记住这里的原来的url，后面你可能需要还原回来)

```
HOMEBREW_REPO = 'https://github.com/Homebrew/homebrew'
```

改为：

```
HOMEBREW_REPO = 'git://mirrors.ustc.edu.cn/homebrew.git'
```

这里就是把Homebrew的 原始镜像 替换为别的镜像（见最下面的参考镜像） 这样就差不多，最后继续执行ruby命令(注意：shell当前路径最好为 install.txt所在路径)
```shell
rm -rf /usr/local/Cellar /usr/local/.git && brew cleanup
ruby  install.txt文件的绝对路径
```

第一行的`rm`命令，是为了防止之前你安装Homebrew失败而残留的文件，导致这次安装失败
ruby install.txt 执行之后，安装命令行提示安装，应该会安装成功。
安装成功后，执行以下命令：

```
brew doctor
```

这个命令是Homebrew的自我检测命令，看看有没有配置不对的地方。

由于我们使用别的镜像，所以会提示镜像为认证，如果你觉得不安全，可以把镜像替换为原来的，不过替换回原始镜像，那么`brew update`可能会很慢，甚至是失败。

下面是修改为原始镜像连接的方法

```
cd /usr/local && git remote set-url origin https://github.com/Homebrew/homebrew1
```

再执行brew doctor看看，应该就没有这个警告了。

完毕！

brew的镜像：

```
git://mirrors.ustc.edu.cn/homebrew.git （中科大的）
https://gitcafe.com/ban-ninja/homebrew.git （gitcafe）
https://git.coding.net/homebrew/homebrew.git （coding.net）
```



## 卸载brew

终端执行命令：

```shell
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall)"
```



------

# Git 安装&使用

## 安装

安装了 brew 后，就可以直接通过这个命令来安装 git：

```shell
brew install git
```

安装完成后，输入下列命令，验证是否安装成功：

```shell
git --version
```

接着就是配置 Git 账号，这需要和你在 Github 使用的用户名和邮箱一致：

```
$ git config --global user.name "Your Name Here"
$ git config --global user.email "your_email@youremail.com"
```

配置信息将写入`~/.gitconfig`文件中。

## 使用

安装配置好git后，可以开始创建本地 git 仓库，然后推送到远程仓库。

基本的使用方式如下，新建一个文件夹，然后进入文件夹，打开终端，依次输入以下命令，完成建立本地Git仓库，提交文件到Github上的操作：

```shell
# 在当前目录新建一个Git代码库
$ git init
# 添加当前目录的所有文件到暂存区
$ git add .
# 提交暂存区到仓库区
$ git commit -m "message"
# 关联一个远程仓库
$ git remote add origin git@server-name:path/repo-name.git(直接复制Github上所建立的仓库的SSH地址即可)
# 第一次推送本地master分支内容到Github上
$ git push -u origin master
# 非第一次推送到Github上
$ git push origin master
```

另外，推送到GitHub有两种方式：

- http
- ssh

前者的话，需要每次都输入用户名和密码，所以可以考虑用 SSH 的方式，使用方法参考：

https://help.github.com/articles/generating-ssh-keys

**1.生成ssh密钥**

首先是判断是否存在密钥，打开终端，输入下列命令：

```shell
$ ls -al ~/.ssh
```

如果存在密钥，那么上述命令会展示所有 `./ssh` 文件夹里的文件，如果没有，就是显示错误

```
ls: /Users/luocai/.ssh: No such file or directory
```

**2.生成密钥**

如果不存在ssh密钥，那么就需要生成密钥了，命令如下：

```shell
$ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

这里输入刚刚配置的邮箱地址，回车后，出现下列提示信息，表示开始生成密钥：

```shell
> Generating public/private rsa key pair.
```

接着会询问存放ssh密钥的位置，直接回车，安装到默认位置即可：

```shell
> Enter a file in which to save the key (/Users/you/.ssh/id_rsa): [Press enter]
```

然后就是设置密码，这里可以不需要设置密码，直接连续按两次回车即可：

```shell
> Enter passphrase (empty for no passphrase): [Type a passphrase]
> Enter same passphrase again: [Type passphrase again]
```

执行成功后，就会在 `~/.ssh`目录下生成两个文件--`id_rsa`私钥文件；`id_rsa.pub`公钥文件。



**3.添加密钥信息到github仓库**

最后一步就是在远程仓库github上添加 `id_rsa.pub`公钥文件的内容，输入下列命令将该文件的内容进行复制：

```shell
$ pbcopy < ~/.ssh/id_rsa.pub
```

如果命令没有起作用，可以手动打开文件，进行复制；

接着，在Github的设置，右上角点开账户头像，选择 “**setting**”：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/ssh_setting1.png)

接着选择 “**SSH and GPG keys**”：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/ssh_setting2.png)

然后在这个界面选择右上方的 “**New SSH key**:

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/ssh_setting3.png)

然后在  Title 填写一个标签说明这个 ssh 密钥的来源，比如来自mac系统或者是windows等，然后在Key里面粘贴刚刚复制的ssh密钥，最后点击下方绿色按钮完成添加。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/ssh_setting4.png)

点击添加后，会需要你输入github账户的密码进行确认。

确认完后，以后就可以通过ssh的方式将本地仓库的修改推送到github上，不需要每次都输入账户名字和密码了。

------

# Anaconda 安装&使用

## 安装

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

1.前往[官方下载页面](http://www.anaconda.com/downloads%23macos)下载。有两个版本可供选择：Python 3.6 和 Python 2.7，目前推荐选择前者，也可以根据自己学习或者工作需求选择不同版本。选择版之后点击“64-Bit Command Line Installer”进行下载。

2.完成下载之后，在mac的Launchpad中找到“其他”并打开“终端”。

▫ 安装Python 3.6： `bash ~/Downloads/Anaconda3-5.0.1-MacOSX-x86_64.sh`

▫ 安装Python 2.7： `bash ~/Downloads/Anaconda2-5.0.1-MacOSX-x86_64.sh`

如果下载路径是自定义，将路径中的`~/Downloads` 替换为你下载的路径，此外如果更改过下载的文件名，那么也将 `Anaconda3-5.0.1-MacOSX-x86_64.sh` 更改为你修改的文件名。

**ps**：强烈建议不要修改文件名，如果重命名，也要采用**英文**进行命名。

3.安装过程中，看到提示“In order to continue the installation process, please review the license agreement.”（“请浏览许可证协议以便继续安装。”），点击“Enter”查看“许可证协议”。

4.在“许可证协议”界面将屏幕滚动至底，输入“yes”表示同意许可证协议内容。然后进行下一步。

5.安装过程中，提示“Press Enter to confirm the location, Press CTRL-C to cancel the installation or specify an alternate installation directory.”（“按回车键确认安装路径，按'CTRL-C'取消安装或者指定安装目录。”）如果接受默认安装路径，则会显示 **PREFIX=/home//anaconda<2 or 3>**  并且继续安装。安装过程大约需要几分钟的时间。

- 建议：直接接受默认安装路径。

6.安装器若提示“Do you wish the installer to prepend the Anaconda install location to PATH in your /home//.bash_profile ?”（“你希望安装器添加Anaconda安装路径在**/home//.bash_profile** 文件中吗？”），建议输入“yes”。

- 注意：

① 路径 **/home//.bash_profile** 中  即进入到家目录后你的目录名。

②如果输入“no”，则需要手动添加路径。添加 **export PATH="//bin:$PATH"** 在 **.bashrc** 或者 **.bash_profile** 中。其中， 替换为你真实的Anaconda安装路径。



7.当看到“Thank you for installing Anaconda!”则说明已经成功完成安装。

8.关闭终端，然后再打开终端以使安装后的 Anaconda 启动。

9.验证安装结果。可选用以下任意一种方法：

- 在终端中输入命令 **condal list** ，如果 Anaconda 被成功安装，则会显示已经安装的包名和版本号。
- 在终端中输入 **python** 。这条命令将会启动 Python 交互界面，如果 Anaconda 被成功安装并且可以运行，则将会在Python版本号的右边显示“Anaconda custom (64-bit)”。退出 Python 交互界面则输入 **exit()** 或 **quit()** 即可。
- 在终端中输入 **anaconda-navigator** 。如果 Anaconda 被成功安装，则 Anaconda Navigator 的图形界面将会被启动。

## 使用

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

------

# Jupyter Notebook 

## 安装

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

另外，这里也推荐安装一个更好使用的 Jupyter lab，安装方法如下：

```shell
pip install jupyterlab
```

它的使用界面如下，其功能会更加强大，具体可以查看文档：

https://jupyterlab.readthedocs.io/en/stable/

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/jupyter_lab.png)





## 使用

运行 Jupyter Notebook 的方法很简单，只需要在系统的终端(Mac/Linux 的 Terminal，Window 的 cmd) 运行以下命令即可：

```shell
jupyter notebook
```

官方文档地址如下：

https://jupyter.org/documentation

使用 jupyter lab 的命令如下：

```shell
jupyter lab
```



------

# Pycharm

## 安装

Pycharm 是 Python 的一个 IDE，配置简单，功能强大，而且对初学者友好，下面介绍如何安装和简单配置 Pycharm。

**Pycharm** 提供 **免费的社区版** 与 **付费的专业版**。专业版额外增加了一些功能，如项目模板、远程开发、数据库支持等。个人学习 **Python** 使用免费的社区版已足够。

pycharm社区版：[PyCharm :: Download Latest Version of PyCharm](http://link.zhihu.com/?target=http%3A//www.jetbrains.com/pycharm/download/)

安装过程照着提示一步步操作就可以了。注意安装路径尽量不使用带有 **中文或空格** 的目录，这样在之后的使用过程中减少一些莫名的错误。



## 使用

**Pycharm** 提供的配置很多，这里讲几个比较重要的配置

**编码设置**：

**Python** 的编码问题由来已久，为了避免一步一坑，**Pycharm** 提供了方便直接的解决方案

![img](https://cdn.nlark.com/yuque/0/2019/jpeg/308996/1558771422814-1f71ad67-1f44-4ae6-87cc-146d87ad6b58.jpeg)

在 **IDE Encoding** 、**Project Encoding** 、**Property Files** 三处都使用 **UTF-8** 编码，同时在文件头添加

```
#-*- coding: utf-8 -
```

这样在之后的学习过程中，或多或少会避免一些编码坑。

**解释器设置**：

当有多个版本安装在电脑上，或者需要管理虚拟环境时，**Project Interpreter** 提供方便的管理工具。

![img](https://cdn.nlark.com/yuque/0/2019/jpeg/308996/1558771422591-5a30180c-298f-4dce-86f5-d5bc42fd1d46.jpeg)

在这里可以方便的切换 **Python** 版本，添加卸载库等操作。

**修改字体**：

在 **Editor** → **Font** 选项下可以修改字体，调整字体大小等功能。

![img](https://cdn.nlark.com/yuque/0/2019/jpeg/308996/1558771422678-60e8aba8-17bb-4f29-9121-1c1531d90d7c.jpeg)



------

# 常用第三方库的安装

常用的第三方库安装，包括：

- numpy
- pandas
- Scikit-learn
- matplotlib
- requests
- tqdm
- scipy
- PIL
- opencv-python
- json

安装命令如下：

```shell
$ pip install numpy pandas scikit-learn
$ pip install tqdm requests opencv-python
$ pip install matplotlib scipy json pillow
```

------

# MySQL安装&使用

## 安装

安装命令如下：

```shell
$ brew update # 这是一个好习惯
$ brew install mysql
```

在使用 MySQL 前，我们需要做一些设置:

```
$ unset TMPDIR
$ mkdir /usr/local/var
$ mysql_install_db --verbose --user=`whoami` --basedir="$(brew --prefix mysql)" --datadir=/usr/local/var/mysql --tmpdir=/tmp
```



## 使用





## MySQLdb安装





------

# Vim 安装&使用



------

# 参考

- [macOS安装Homebrew(https://blog.csdn.net/zzq900503/article/details/80404314)
- [Mac OS下brew的安装和使用(https://www.jianshu.com/p/ab50ea8b13d6)
- [Mac开发配置手册](https://aaaaaashu.gitbooks.io/mac-dev-setup/content/Git/index.html)
- [Anaconda介绍、安装及使用教程](https://zhuanlan.zhihu.com/p/32925500)
- [最详尽使用指南：超快上手Jupyter Notebook](https://zhuanlan.zhihu.com/p/32320214)
- [喏，你们要的 PyCharm 快速上手指南](https://zhuanlan.zhihu.com/p/26066151)
- [MySQL安装](https://aaaaaashu.gitbooks.io/mac-dev-setup/content/MySql/index.html)