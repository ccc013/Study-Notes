
学习和使用Git有一段时间了，所谓好记性不如烂笔头，结合当初学习的教程和网上其他文章，以及自己日常使用的经验，进行总结。

#目录
一. [日常使用流程](#流程)

[遇到的问题和解决方法](#问题)


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

<h3 id="问题">遇到的问题和解决方法</h3>
####1.如何删除github上的一个分支
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

####2. push到github时，每次都要输入用户名和密码的问题
在github.com上 建立了一个小项目，可是在每次push的时候，都要输入用户名和密码，很是麻烦

**原因是使用了https方式 push**

在Git bash里边输入 `git remote -v `,可以看到形如一下的返回结果

    origin https://github.com/yuquan0821/demo.git (fetch)
    origin https://github.com/yuquan0821/demo.git (push)

下面把它换成ssh方式的。

    1. git remote rm origin
    2. git remote add origin SSH地址
    3. git push origin 

参考的教程和文章有:

1. [廖雪峰的Git教程](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/001373962845513aefd77a99f4145f0a2c7a7ca057e7570000)
2. [常用 Git 命令清单](http://www.ruanyifeng.com/blog/2015/12/git-cheat-sheet.html?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)
3. [日常使用 Git 的 19 个建议](http://blog.jobbole.com/96088/)
4. [push到github时，每次都要输入用户名和密码的问题](http://blog.csdn.net/yuquan0821/article/details/8210944)
5. [删除github.com上的一个分支](http://www.linuxso.com/linuxrumen/2752.html)
