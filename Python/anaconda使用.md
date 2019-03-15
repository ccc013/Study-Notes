

---

#### 环境设置

创建环境

```
# 创建了一个 python3.6 的环境
conda create -n py36 python=3.6
```
删除环境

```
conda remove -n py36 --all
```

激活环境

```
source activate py36
```

退出环境

```
source deactivate
```

---

删除 anaconda

[python-anaconda-how-to-safely-uninstall](https://stackoverflow.com/questions/22585235/python-anaconda-how-to-safely-uninstall)

> To uninstall Anaconda open a terminal window and remove the entire anaconda install directory: `rm -rf ~/anaconda`. You may also edit  `~/.bash_profile` and remove the anaconda directory from your `PATH` environment variable, and remove the hidden `.condarc` file and  `.conda` and `.continuum` directories which may have been created in the home directory with `rm -rf ~/.condarc ~/.conda ~/.continuum`.

Further notes:

- Python3 installs may use a `~/anaconda3` dir instead of `~/anaconda`.
- You might also have a `~/.anaconda` hidden directory that may be removed.
- Depending on how you installed, it is possible that the `PATH` is modified in one of your runcom files, and not in your shell profile. So, for example if you are using bash, be sure to check your `~/.bashrc` if you don't find the PATH modified in `~/.bash_profile`.


