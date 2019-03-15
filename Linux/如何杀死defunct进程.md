
参考文章[如何杀死defunct进程（译）](http://blog.xiayf.cn/2016/02/18/kill-defunct/)

首先是通过下面的命令找到父进程的PID

```
$ ps -ef | grep defunct | more
```
输出结果为：

```
UID PID PPID ...
---------------------------------------------------------------

root     40428 40427  0 Sep22 pts/0    00:01:57 [python] <defunct>
root     40733 40727  0 Sep22 pts/0    00:01:58 [python] <defunct>
root     41796 22740  0 11:02 pts/0    00:00:00 grep --color=auto defunct
root     42073 42067  0 Sep22 pts/0    00:01:58 [python] <defunct>
root     42319 42313  0 Sep22 pts/0    00:02:01 [python] <defunct>
```
其中：
- UID：用户ID
- PID：进程ID
- PPID：父进程ID

如果你使用命令 “kill -9 40428” 尝试杀死 ID 为 40428 的进程，可能会没效果。要想成功杀死该进程，需要对其父进程（ID 为 40427）执行 kill 命令（$ kill -9 40427）。对所有这些进程的父进程 ID 应用 kill 命令，并验证结果（$ ps -A | grep defunct）。

当然如果上述结果不成功，那么就需要重启电脑了。

