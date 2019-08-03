
原文--https://jupyter-notebook.readthedocs.io/en/latest/public_server.html#notebook-server-security

Jupyter notebook[^1]是一个基于服务器-客户端结构的网络应用，其服务器端是采用一个基于 `ZeroMQ`[^2]和`Tornado`[^3]双进程核心结构[^4]，其中后者 `Tornado` 负责处理 HTTP 的请求服务。

> **注意**：默认 `notebook` 的服务器运行在本地的 IP 地址是 `127.0.0.1:8888`，并且也只能通过 `localhost` 进行访问，也就是可以在浏览器中输入 `http://127.0.0.1:8888` 进行访问

但本教程将介绍如何访问一个 `notebook` 的服务器，并且采用一个公开的接口。

> 这里提醒，这不是应用于多人服务器的教程，仅供用于只有一个人使用的服务器的情况，如果是希望多人使用的情况，可以采用 `JupyterHub`[^5]，如果要应用 `JupyterHub` ，需要一台 `Unix` （通常就是 Linux）的服务器，然后可以连接网络。

接下来就介绍如何实现远程访问服务器的 Jupyter notebook 的方法。

------

### 配置 notebook 服务器

首先可以通过设置密码来保护你的 `notebook` 服务器，在 5.0 版本的 `notebook` ，这一步可以自动化实现了。当然要手动设置密码的话，需要在文件 `jupyter_notebook_config.py` 中修改 `NotebookApp.password` 这里的内容。

#### 1. 前置条件：一个配置文件

第一步就是先找到或者生成配置文件  `jupyter_notebook_config.py` ，默认的配置文件是在 `Jupyter` 文件夹中的，不同系统位置如下：

- **Windows**：`C:\Users\USERNAME\.jupyter\jupyter_notebook_config.py`
- **OS X**：`/Users/USERNAME/.jupyter/jupyter_notebook_config.py`
- **Linux**：`/home/USERNAME/.jupyter/jupyter_notebook_config.py`

如果上述位置没有这个文件，可以通过下列命令生成：

```shell
$ jupyter notebook --generate-config
```

如果是 root 用户执行上面的命令，会发生一个问题：

```shell
Running as root it not recommended. Use --allow-root to bypass.
```


提示信息很明显，root 用户执行时需要加上 `--allow-root` 选项。

```shell
jupyter notebook --generate-config --allow-config
```

执行成功后，会出现下面的信息：

```shell
Writing default config to: /root/.jupyter/jupyter_notebook_config.py
```

#### 2. 生成密码

##### 自动生成

**从 `notebook` 5.3 版本开始**，当第一次通过令牌(token)登录的时候，`notebook` 服务器会让用户有机会在用户界面设置一个密码，这将通过一个表单来询问当前的令牌以及新的密码，输入并点击 `Login and setup new password` 。

下次登录时候就可以直接选择输入密码而不需要令牌。如果没有设置密码，也可以按照下面的操作通过命令行设置密码。此外，可以在配置文件中设置 `--NotebookApp.allow_password_change=False` 来禁止第一次登录时候修改密码。

在 `notebook` 5.0 版本开始，可以通过一个命令 `jupyter notebook password` 设置密码并保存到文件 `jupyter_notebook_config.json`，如下所示：

```shell
$ jupyter notebook password
Enter password:  ****
Verify password: ****
[NotebookPasswordApp] Wrote hashed password to /Users/you/.jupyter/jupyter_notebook_config.json
```

##### 手动生成

除了上述方法，也可以手动生成密码，方法如下所示：

```shell
In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password:
Verify password:
Out[2]: 'sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed'
```

`passwd()` 方法在没有传入参数时候，会如上所示提示输入和验证密码，它也可以传入一个字符串作为密码，即 `passwd('mypassword')` ，但不建议这种做法，因为本来输入命令都会被保存起来，直接输入密码，相当于密码以明文方式保存在输入历史。

##### 添加到配置文件

最后一步，就是需要将哈希密码添加到配置文件 `jupyter_notebook_config.py`，也就是刚刚手动生成的例子中的 `sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed`，如下所示：

```shell
c.NotebookApp.password = u'sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed'
```

#### 3. 采用 SSL 加密通信

采用密码的时候，配合带有网站证书的 SSL 是一个好办法，可以避免哈希的密码被非加密的形式发送给浏览器。

可以通过设置参数 `certfile` 来开启 `notebook` 服务器，进行一次安全协议模式的通信，其中 `mycert.perm` 是自签(self-signed)证书。

```shell
$ jupyter notebook --certfile=mycert.pem --keyfile mykey.key
```

自签证书可以通过 `openssl` 生成，如下所示，生成一个有效期为 365 天，将 `key` 和 证书数据都保存在同个文件中：

```shell
$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
```

当打开浏览器连接服务器的时候，浏览器会提示你的自签证书是不安全或者无法识别的，如果你希望获取不会产生警告的自签证书，可以根据教程[^6]所说来操作。此外，也可以通过采用 `Let's Encrypt` [^7]来得到免费的 SSL 证书，然后根据教程[^8]来配置服务器。

### 运行 notebook 服务器

上述步骤介绍了如何进行配置，接下来就是开始运行服务器，然后远程访问。这里首先还是需要进行如下操作，也就是修改配置文件 `jupyter_notebook_config.py` ，找到下面几个信息修改并去掉注释：

```python
# 证书的信息
c.NotebookApp.certfile = u'/absolute/path/to/your/certificate/mycert.pem'
c.NotebookApp.keyfile = u'/absolute/path/to/your/certificate/mykey.key'
# ip 设置为 *
c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:bcd259ccf...<your hashed password here>'
c.NotebookApp.open_browser = False

# 设置一个固定的接口
c.NotebookApp.port = 80
```

这里官方教程是建议 `c.NotebookApp.ip` 设置为 `*` ，但实际上这样操作可能会连接失败，所以可以选择设置为 `0.0.0.1` 或者就是服务器的 IP

我这边最终主要配置如下：


```python
c.NotebookApp.ip = '服务器IP'
c.NotebookApp.password = u'sha:ce...刚才复制的那个密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port =80 #可自行指定一个端口, 访问时使用该端口
```

接着运行命令，如果是 `root` 用户，需要再加上 `--allow-root` ，

```shell
$ jupyter notebook
```

然后本地浏览器输入 `服务器IP:80`，接着就是输入刚刚设置的密码，即可访问 Jupyter notebook，然后就和在本地电脑操作 Jupyter notebook 一样，创建文件，运行。

需要注意的是不能在隐藏目录 (以 . 开头的目录)下启动 Jupyter notebook, 否则无法正常访问文件。

如果访问失败了，则有可能是服务器防火墙设置的问题，此时最简单的方法是在本地建立一个 `ssh` 通道：
 在本地终端中输入

```shell
$ ssh username@address_of_remote -L 127.0.0.1:1234:127.0.0.1:8888
```


 便可以在 `localhost:1234` 直接访问远程的 `jupyter notebook`了。

关于 ssh 通道的知识，可以查看：

https://www.zsythink.net/archives/2450



---
参考：

[^1]: https://jupyter-notebook.readthedocs.io/en/latest/notebook.html 
[^2]: http://zeromq.org/
[^3]: http://www.tornadoweb.org/ 
[^4]: https://ipython.readthedocs.io/en/stable/overview.html#ipythonzmq
[^5]: https://jupyterhub.readthedocs.io/en/latest/
[^6]: https://arstechnica.com/information-technology/2009/12/how-to-get-set-with-a-secure-sertificate-for-free/
[^7]: https://letsencrypt.org/
[^8]: https://jupyter-notebook.readthedocs.io/en/latest/public_server.html#using-lets-encrypt 

- https://blog.csdn.net/simple_the_best/article/details/77005400
- https://www.jianshu.com/p/7cac2e5ec35b
