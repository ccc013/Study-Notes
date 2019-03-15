
参考文章：

http://man.linuxde.net/alias


---

alias 命令用来设置指令的别名。我们可以使用该命令可以将一些较长的命令进行简化。使用 alias 时，**用户必须使用单引号''将原来的命令引起来，防止特殊字符导致错误**。

alias 命令的作用只局限于该次登入的操作。若要每次登入都能够使用这些命令别名，则可将相应的 alias 命令存放到 bash 的初始化文件`/etc/bashrc`中。


**实例**


```
alias 新的命令='原命令 -选项/参数'
```

例如：`alias l=‘ls -lsh'`将重新定义`ls`命令，现在只需输入`l`就可以列目录了。直接输入 `alias` 命令会列出当前系统中所有已经定义的命令别名。

要删除一个别名，可以使用 unalias 命令，如 `unalias l`。

**查看系统已经设置的别名：**


```
alias -p
alias cp='cp -i'
alias l.='ls -d .* --color=tty'
alias ll='ls -l --color=tty'
alias ls='ls --color=tty'
alias mv='mv -i'
alias rm='rm -i'
alias which='alias | /usr/bin/which --tty-only --read-alias --show-dot
```


