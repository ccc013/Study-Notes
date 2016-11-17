
学习和使用Git有一段时间了，所谓好记性不如烂笔头，结合当初学习的教程和网上其他文章，以及自己日常使用的经验，进行总结。

#目录
一. [日常使用流程](#流程)

二. [使用总结](#使用总结)

三.[遇到的问题和解决方法](#问题)


<h3 id="流程">一. 日常使用流程</h3>
日常使用只要记住下图中6个命令即可:

![Git 使用流程图](https://github.com/ccc013/Study-Notes/raw/master/images/Git使用流程图.png)

总结下，新建一个仓库到将其推送到远程仓库，即Github上，可以根据以下命令：

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

这里要注意，在github上建立一个远程仓库的时候，可能会添加READ.md或者是一个.gitgnore等文件，并且这些文件是本地仓库中没有的时候，在第一次推送本地master分支内容前，需要使用命令`git pull <remote> <branch>`，这里的remote，也就是github上对应仓库的地址，可以使用在关联仓库时使用的SSH地址，而branch，初次创建的分支都是master，只有执行了这个操作，才可以成功push本地的文件，否则会出错。

一些专有名词的解释：
* Workspace: 工作区
* Index / Stage: 暂存区
* Repository: 仓库区(或本地仓库)
* Remote: 远程仓库

<h3 id="使用总结">二. 使用总结</h3>
#### 1.删除文件
在Git中删除文件的命令是`git rm text.txt`,当使用了这个命令后，工作区和版本库就不一致了，可以使用`git status`这个命令显示做出的修改是什么，比如这里就是显示删除了哪些文件。

这个时候有两个选择，第一个是确实需要从版本库中删除该文件，那么就使用`git commit`，同时再使用`git push`推送到远程仓库Github后，在Github上该文件也会被删除。

第二个选择是删除错误，那么可以使用命令`git checkout -- text.txt`,命令`git checkout`就是使用版本库里的版本替换工作区的版本，无论工作区是修改还是删除，都可以“一键还原”。


<h3 id="问题">三.遇到的问题和解决方法</h3>
####1）如何删除github上的一个分支
有时候在github上的一个仓库建立了一个不想要的分支，应该如何删除呢，参考了网上的文章后，发现可以通过新建一个空的版本库，然后直接将其推送到需要删除分支的github仓库上，这个分支就会被自动删除了。

代码如下:

    # 新建一个空的文件夹
    $ mkdir git-empty
    # 进入该文件夹
    $ cd git-empty
    # 新建一个仓库
    $ git init
    # 推送到github上要某个仓库要删除的分支上
    $ git push remote_repo :remote_branch 
    (这里的冒号不能删除，remote_repo就是github上这个仓库的https地址，remote_branch就是要删除的分支名字)

####2） push到github时，每次都要输入用户名和密码的问题
在github.com上 建立了一个小项目，可是在每次push的时候，都要输入用户名和密码，很是麻烦

**原因是使用了https方式 push**

在Git bash里边输入 `git remote -v `,可以看到形如一下的返回结果

    origin https://github.com/yuquan0821/demo.git (fetch)
    origin https://github.com/yuquan0821/demo.git (push)

下面把它换成ssh方式的。

    1. git remote rm origin
    2. git remote add origin SSH地址
    3. git push origin 

#### 3） 出现如`remote: Invalid username or password`的错误信息

在将本地代码提交到`github`时，即输入`git push origin master`后，出现错误信息如下：

```shell
remote: Invalid username or password.
fatal: Authentication failed for 'https://github.com/ccc013/CodingPractise.git/'
```

这里参考了[git invalid username or password](http://stackoverflow.com/questions/29297154/git-invalid-username-or-password)上的答案，输入如下命令：

```shell
git remote set-url origin git@github.com:ccc013/CodingPractise.git
```

然后再次输入命令`git push origin master`即可成功提交。

#### 4） 没有在github上添加公钥

这个错误的具体信息如下：

```shell
Warning: Permanently added the RSA host key for IP address '192.30.253.113' to the list of known hosts. Permission denied (publickey). fatal: Could not read from remote repository. Please make sure y
```

这里参考[如何在github上添加公钥](http://www.cnblogs.com/qcwblog/p/5709720.html)这篇文章。具体步骤如下所示：

1 可以用 ssh -T git@github.com去测试一下

![img](http://images2015.cnblogs.com/blog/923829/201607/923829-20160726173600653-822778976.png)

图上可以明显看出缺少了公钥

2 直接上图

![img](http://images2015.cnblogs.com/blog/923829/201607/923829-20160727084851341-167270095.png)

\3. cat 一下  把出现的key 复制下来

![img](http://images2015.cnblogs.com/blog/923829/201607/923829-20160727085024091-1309310232.png)

4 .在github上添加刚刚生成的公钥

![img](http://images2015.cnblogs.com/blog/923829/201607/923829-20160727085229481-635311296.png)

![img](http://images2015.cnblogs.com/blog/923829/201607/923829-20160727085251559-1389913670.png)

![img](http://images2015.cnblogs.com/blog/923829/201607/923829-20160727085332997-1227342030.png)

![img](http://images2015.cnblogs.com/blog/923829/201607/923829-20160727085538497-118378956.png)

正常完成这步后，再次输入向`github`提交代码应该是会成功的。

#### 5）克隆某一个特定的远程分支

正常克隆某个仓库的命令如下：

```shell
git clone [remote repository address] (file_name)
```

其中`file_name`是代码保存在本地电脑的文件夹名字。

如果需要指定某个分支的代码，可以添加`-b <branch name>`，也就是命令如下：

```shell
git clone -b <branch name> [remote repository address] (file_name)
```





参考的教程和文章有:

1. [廖雪峰的Git教程](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/001373962845513aefd77a99f4145f0a2c7a7ca057e7570000)
2. [常用 Git 命令清单](http://www.ruanyifeng.com/blog/2015/12/git-cheat-sheet.html?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)
3. [日常使用 Git 的 19 个建议](http://blog.jobbole.com/96088/)
4. [push到github时，每次都要输入用户名和密码的问题](http://blog.csdn.net/yuquan0821/article/details/8210944)
5. [删除github.com上的一个分支](http://www.linuxso.com/linuxrumen/2752.html)
