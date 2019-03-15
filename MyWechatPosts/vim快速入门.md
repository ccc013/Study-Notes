

> 本文大约 5000 字， 阅读大约需要 10 分钟



在 Linux 下最常使用的文本编辑器就是 vi 或者 vim 了，如果能很好掌握这个编辑器，非常有利于我们更好的在 Linux 下面进行编程开发。

#### vim 和 vi

Vim是从 vi 发展出来的一个文本编辑器。代码补完、编译及错误跳转等方便编程的功能特别丰富，在程序员中被广泛使用。

简单的来说， vi 是老式的字处理器，不过功能已经很齐全了，但是还是有可以进步的地方。 vim 则可以说是程序开发者的一项很好用的工具。

下面是 vim 快捷键盘图：

![vim 快捷键](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/vi-vim-cheat-sheet-sch.gif)

#### vi/vim 的工作模式

基本上 vi/vim 共分为三种模式，分别是命令模式（Command mode），输入模式（Insert mode）和底线命令模式（Last line mode）。 这三种模式的作用分别是：

##### 命令模式

当使用 vi/vim 打开一个文件就进入了命令模式（也可称为一般模式），这是默认的模式。在这个模式，你可以采用『上下左右』按键来移动光标，你可以使用『删除字符』或『删除整行』来处理档案内容，也可以使用『复制、贴上』来处理你的文件数据。

##### 输入模式

在命令模式并不能编辑文件，需要输入如『i, I, o, O, a, A, r,R』等任何一个字母之后才会进入输入模式（也称为编辑模式）。注意了！通常在 Linux 中，按下这些按键时，在画面的左下方会出现『 INSERT 或 REPLACE 』的字样，此时才可以进行编辑。而如果要回到命令模式时，则必须要按下『Esc』这个按键即可退出编辑模式。

##### 底线命令模式

在命令模式下，按下『:,/,?』中任意一个，就可以将光标移动到最底下那一行，进入底线命令模式（也称为指令列命令模式）。在这个模式当中， 可以提供你『搜寻资料』的动作，而读取、存盘、大量取代字符、退出、显示行号等等的动作则是在此模式中达成的！同理，必须按下『Esc』按钮才可以退出该模式，返回命令模式

三种模式的切换和功能可以用下图来总结：

![vi/vim 工作模式](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/vim-vi-workmodel.png)


#### 简易示例

###### 1. 使用 vim 打开文件

在命令行中输入如下命令：

```
$ vim test.txt
```
采用 `vi 文件名` 或者 `vim 文件名` 就可以打开文件并且进入了命令模式。这里文件名是必须添加的，当文件不存在的时候，也能打开，并且进行编辑保存后就是创建一个新的文件。打开后的界面如下图所示：

![vim1.png](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/vim1.png)

整个界面可以分为两个部分，最底下一行和上面的部分，最底下一行主要是显示当前文件名和文件的行数、列数，上图是一个新的文件，所以最底下显示的是文件名，而且后面括号也说是新文件，而下图是一个已经有内容的文件，那么上面部分就显示文件内容，最底下一行显示了文件名，文件的行数和列数，并且在最右侧部分会显示当前坐标的位置，比如图中是显示 (4,1) 表示当前坐标在第四行第一列的位置。

![vim2.png](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/vim2.png)

###### 2. 进入编辑模式

接下来就是开始对文件进行编辑，也就是需要进入编辑模式。只要按下『i, I, o, O, a, A, r,R』等字符就可以进入编辑模式了！在编辑模式当中，你可以发现在左下角状态栏中会出现 –INSERT- 的字样，那就是可以输入任意字符的提示啰！这个时候，键盘上除了 [Esc] 这个按键之外，其他的按键都可以视作为一般的输入按钮了，所以你可以进行任何的编辑！

如下图所示：

![vim3.png](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/vim3.png)

注意：在 vim/vi 中 [Tab] 键是向右移动 8 个空格字符。


###### 3. 按下 [ESC] 按钮回到命令模式

如果对文件编辑完毕了，那么应该要如何退出呢？此时只需要按下 [Esc] 这个按钮即可！马上你就会发现画面左下角的 – INSERT – 不见了！并且返回了命令模式了

###### 4. 退出

最后就是存盘并离开，指令很简单，输入『:wq』即可存档离开！ (注意了，按下 : 该光标就会移动到最底下一行去！) ，如下图所示：

![vim4.png](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/vim4.png)



#### 更多按键说明

上述简易示例只是使用了简单的几个按键，但是从 vim 快捷键图可以知道 vim 是有很多快捷键的。

<table border=0 cellpadding=0 cellspacing=0 width=1026 style='border-collapse:
 collapse;table-layout:fixed;width:772pt'>
 <col width=69 span=3 style='width:52pt'>
 <col width=181 span=3 style='mso-width-source:userset;mso-width-alt:5781;
 width:136pt'>
 <col width=69 span=4 style='width:52pt'>
 <tr height=30 style='height:22.5pt'>
  <td colspan=6 height=30 class=xl65 width=750 style='height:22.5pt;width:564pt'>移动光标的方法</td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
 </tr>
 <tr height=26 style='mso-height-source:userset;height:19.5pt'>
  <td colspan=3 height=26 class=xl97 width=207 style='border-right:1.0pt solid gray;
  height:19.5pt;width:156pt'>h 或 向左方向鍵(←)</td>
  <td colspan=3 class=xl94 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>光标向左移动一个字符</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=26 style='mso-height-source:userset;height:19.5pt'>
  <td colspan=3 height=26 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:19.5pt;width:156pt'>j 或 向下方向鍵(↓)</td>
  <td colspan=3 class=xl103 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>光标向下移动一个字符</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=26 style='mso-height-source:userset;height:19.5pt'>
  <td colspan=3 height=26 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:19.5pt;width:156pt'>k 或 向上方向鍵(↑)</td>
  <td colspan=3 class=xl94 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>光标向上移动一个字符</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=26 style='mso-height-source:userset;height:19.5pt'>
  <td colspan=3 height=26 class=xl100 width=207 style='border-right:1.0pt solid gray;
  height:19.5pt;width:156pt'>l 或 向右方向鍵(→)</td>
  <td colspan=3 class=xl103 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>光标向右移动一个字符</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=19 style='mso-height-source:userset;height:14.0pt'>
  <td colspan=6 rowspan=6 height=114 class=xl85 width=750 style='border-right:
  1.0pt solid gray;border-bottom:.5pt solid black;height:84.0pt;width:564pt'>如果你将右手放在键盘上的话，你会发现
  hjkl 是排列在一起的，因此可以使用这四个按钮来移动光标。 如果想要进行多次移动的话，例如向下移动 30 行，可以使用 "30j"或 "30↓" 的组合按键， 亦即加上想要进行的次数(数字)后，按下动作即可！</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=19 style='mso-height-source:userset;height:14.0pt'>
  <td height=19 colspan=4 style='height:14.0pt;mso-ignore:colspan'></td>
 </tr>
 <tr height=19 style='mso-height-source:userset;height:14.0pt'>
  <td height=19 colspan=4 style='height:14.0pt;mso-ignore:colspan'></td>
 </tr>
 <tr height=19 style='mso-height-source:userset;height:14.0pt'>
  <td height=19 colspan=4 style='height:14.0pt;mso-ignore:colspan'></td>
 </tr>
 <tr height=19 style='mso-height-source:userset;height:14.0pt'>
  <td height=19 colspan=4 style='height:14.0pt;mso-ignore:colspan'></td>
 </tr>
 <tr height=19 style='mso-height-source:userset;height:14.0pt'>
  <td height=19 colspan=4 style='height:14.0pt;mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='mso-height-source:userset;height:16.5pt'>
  <td colspan=3 height=22 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>[Ctrl] + [f]</td>
  <td colspan=3 class=xl106 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>屏幕『向下』移动一页，相当于 [Page Down]按键</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='mso-height-source:userset;height:16.5pt'>
  <td colspan=3 height=22 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>[Ctrl] + [b]</td>
  <td colspan=3 class=xl109 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>屏幕『向上』移动一页，相当于 [Page Up]按键</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>[Ctrl] + [d]</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>向下滚动（移动半页）</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>[Ctrl] + [u]</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>向上滚动（移动半页）</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>+</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>光标移动到非空格符的下一行</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>-</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>光标移动到非空格符的上一行</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>n&lt;space&gt;</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>向右移动 n 个字符，n 是数量</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>0 或功能鍵[Home]</td>
  <td colspan=3 class=xl109 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>数字0，移动到当前行最前面字符处<span
  style='mso-spacerun:yes'>&nbsp;</span></td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>$ 或功能鍵[End]</td>
  <td colspan=3 class=xl106 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>移动到这一行的最后字符处</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>H</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>光标移动到这个屏幕最上方一行的第一个字符处</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>M</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>光标移动到这个屏幕中央一行的第一个字符处</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>L</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>光标移动到这个屏幕最下方一行的第一个字符处</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>G</td>
  <td colspan=3 class=xl106 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>移动到这个档案的最后一行</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>nG</td>
  <td colspan=3 class=xl109 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>移动到这个档案的第 n 行，n是数字（可配合 :set nu)</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>gg</td>
  <td colspan=3 class=xl106 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>移动到这个档案的第一行，效果等同于 1G 啊！<span
  style='mso-spacerun:yes'>&nbsp;</span></td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>n&lt;Enter&gt;</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>向下移动 n 行</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=30 style='height:22.5pt'>
  <td colspan=6 height=30 class=xl65 style='height:22.5pt'>搜索</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=67 style='mso-height-source:userset;height:50.5pt'>
  <td colspan=3 height=67 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:50.5pt;width:156pt'>/word</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>向光标之下寻找一个名称为 word
  的字符串。例如要在档案内搜寻 vbird 这个字符串，就输入 /vbird 即可！<span
  style='mso-spacerun:yes'>&nbsp;</span></td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=23 style='mso-height-source:userset;height:17.0pt'>
  <td colspan=3 height=23 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:17.0pt;width:156pt;min-width: 24px'>?word</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>向光标之上寻找一个字符串名称为 word 的字符串。</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=23 style='mso-height-source:userset;height:17.0pt'>
  <td colspan=3 height=23 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:17.0pt;width:156pt;min-width: 24px'>n</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>重复前一个搜寻的动作。</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=23 style='mso-height-source:userset;height:17.0pt'>
  <td colspan=3 height=23 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:17.0pt;width:156pt;min-width: 24px'>N</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>与 n 刚好相反，为『反向』进行前一个搜寻动作。<span
  style='mso-spacerun:yes'>&nbsp;</span></td>
  <td class=xl66 width=69 style='border-left:none;width:52pt'>　</td>
  <td class=xl67 width=69 style='width:52pt'>　</td>
  <td class=xl68 width=69 style='width:52pt'>　</td>
  <td></td>
 </tr>
 <tr height=53 style='mso-height-source:userset;height:39.5pt'>
  <td colspan=6 height=53 class=xl74 width=750 style='border-right:.5pt solid black;
  height:39.5pt;width:564pt;min-width: 24px'>使用 /word 配合 n 及 N
  是非常有帮助的！可以让你重复的找到一些你搜寻的关键词！</td>
  <td colspan=2 style='mso-ignore:colspan'></td>
  <td class=xl73 style='border-top:none'>　</td>
  <td></td>
 </tr>
 <tr height=30 style='mso-height-source:userset;height:22.5pt'>
  <td colspan=6 height=30 class=xl65 style='height:22.5pt'>替换</td>
  <td colspan=2 style='mso-ignore:colspan'></td>
  <td class=xl73 style='border-top:none'>　</td>
  <td></td>
 </tr>
 <tr height=41 style='mso-height-source:userset;height:30.5pt'>
  <td colspan=3 height=41 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:30.5pt;width:156pt;min-width: 24px'>:n1,n2s/word1/word2/g</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>在第 n1 与 n2 行之间寻找 word1
  这个字符串，并将该字符串取代为 word2 ！</td>
  <td class=xl72 width=69 style='width:52pt'>　</td>
  <td class=xl72 width=69 style='border-left:none;width:52pt'>　</td>
  <td class=xl72 width=69 style='border-top:none;border-left:none;width:52pt'>　</td>
  <td class=xl72 width=69 style='border-left:none;width:52pt'>　</td>
 </tr>
 <tr height=48 style='mso-height-source:userset;height:36.0pt'>
  <td colspan=3 height=48 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:36.0pt;width:156pt;min-width: 24px'>:1,$s/word1/word2/g</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>从第一行到最后一行寻找 word1 字符串，并将该字符串取代为
  word2</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=67 style='mso-height-source:userset;height:50.0pt'>
  <td colspan=3 height=67 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:50.0pt;width:156pt;min-width: 24px'>:1,$s/word1/word2/gc</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>从第一行到最后一行寻找 word1 字符串，并将该字符串取代为
  word2 ！且在取代前显示提示字符给用户确认 (confirm) 是否需要取代</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=30 style='height:22.5pt'>
  <td colspan=6 height=30 class=xl65 style='height:22.5pt'>删除</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=67 style='mso-height-source:userset;height:50.5pt'>
  <td colspan=3 height=67 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:50.5pt;width:156pt'>x, X</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>在一行字当中，x 为向后删除一个字符 (相当于 [del]
  按键)， X 为向前删除一个字符(相当于 [backspace] 亦即是退格键)<span
  style='mso-spacerun:yes'>&nbsp;</span></td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>nx</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>连续向后删除 n 个字符。</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>dd</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>删除游标所在的那一整行</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>ndd</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>删除光标所在的向下 n 行</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>d1G</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>删除光标所在到第一行的所有数据</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>dG</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>删除光标所在到最后一行的所有数据</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>d$</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>删除游标所在处，到该行的最后一个字符</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>d0</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>删除游标所在处，到该行的最前面一个字符</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=30 style='height:22.5pt'>
  <td colspan=6 height=30 class=xl65 style='height:22.5pt'>复制</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>yy</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>复制游标所在的那一行</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>nyy</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>复制光标所在的向下 n 行</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>y1G</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>复制游标所在行到第一行的所有数据</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>yG</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>复制游标所在行到最后一行的所有数据</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>y0</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>复制光标所在的那个字符到该行行首的所有数据</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>y$</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>复制光标所在的那个字符到该行行尾的所有数据</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=30 style='height:22.5pt'>
  <td colspan=6 height=30 class=xl65 style='height:22.5pt'>粘贴</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=43 style='mso-height-source:userset;height:32.0pt'>
  <td colspan=3 height=43 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:32.0pt;width:156pt;min-width: 24px'>p, P</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>p 为将已复制的数据在光标下一行贴上，P
  则为贴在游标上一行！<span style='mso-spacerun:yes'>&nbsp;</span></td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl69 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>J</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>将光标所在行与下一行的数据结合成同一行</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=30 style='height:22.5pt'>
  <td colspan=6 height=30 class=xl65 style='height:22.5pt'>其他</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl112 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>c</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>重复删除多个数据，任意方向，并且进入编辑模式</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>u</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>复原前一个动作。(常用)</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>[Ctrl]+r</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>重做上一个动作。(常用)</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=48 style='mso-height-source:userset;height:36.0pt'>
  <td colspan=6 height=48 class=xl83 width=750 style='border-right:1.0pt solid gray;
  height:36.0pt;width:564pt;min-width: 24px'>这个 u 与 [Ctrl]+r
  是很常用的指令！一个是复原，另一个则是重做一次～ 利用这两个功能按键，你的编辑，嘿嘿！很快乐的啦！</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=47 style='mso-height-source:userset;height:35.0pt'>
  <td colspan=3 height=47 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:35.0pt;width:156pt;min-width: 24px'>.</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>重复前一个动作，比如重复删除、重复贴上等等动作，按下小数点『.』</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=30 style='height:22.5pt'>
  <td colspan=6 height=30 class=xl65 style='height:22.5pt'>编辑模式</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=66 style='mso-height-source:userset;height:49.5pt'>
  <td colspan=3 height=66 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:49.5pt;width:156pt'>i, I</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'><font class="font11">进入输入模式</font><font
  class="font12">(Insert mode)</font><font class="font11">：<br></font><font class="font12">i </font><font class="font11">为『从目前光标所在处输入』，</font><font
  class="font12"> I </font><font class="font11">为『在目前所在行的第一个非空格符处开始输入』</font></td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=64 style='mso-height-source:userset;height:48.0pt'>
  <td colspan=3 height=64 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:48.0pt;width:156pt;min-width: 24px'>a, A</td>
  <td colspan=3 class=xl115 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'><font class="font11">进入输入模式</font><font
  class="font12">(Insert mode)</font><font class="font11">：<br></font><font class="font12">a </font><font class="font11">为『从目前光标所在的下一个字符处开始输入』，</font><font
  class="font12"> A </font><font class="font11">为『从光标所在行的最后一个字符处开始输入』</font></td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=85 style='mso-height-source:userset;height:63.5pt'>
  <td colspan=3 height=85 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:63.5pt;width:156pt;min-width: 24px'>o, O</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>进入输入模式(Insert mode)：<br>
这是英文字母 o 的大小写。o 为『在目前光标所在的下一行处输入新的一行』； O 为在目前光标所在处的上一行输入新的一行</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=66 style='mso-height-source:userset;height:49.5pt'>
  <td colspan=3 height=66 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:49.5pt;width:156pt;min-width: 24px'>r, R</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>进入取代模式(Replace mode)：<br>
r 只会取代光标所在的那一个字符一次；R会一直取代光标所在的文字，直到按下 ESC 为止</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=96 style='mso-height-source:userset;height:72.0pt'>
  <td colspan=6 height=96 class=xl85 width=750 style='height:72.0pt;width:564pt;
  min-width: 24px'>上面这些按键中，在 vi 画面的左下角处会出现『--INSERT--』或『--REPLACE--』的字样。
  由名称就知道该动作了吧！！特别注意的是，我们上面也提过了，你想要在档案里面输入字符时， 一定要在左下角处看到 INSERT 或 REPLACE
  才能输入喔！</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl80 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>[Esc]</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>退出编辑模式，回到一般模式中</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=30 style='height:22.5pt'>
  <td colspan=6 height=30 class=xl65 style='height:22.5pt'>底线命令模式</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>:w</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>将编辑的数据写入硬盘档案中</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=38 style='mso-height-source:userset;height:28.5pt'>
  <td colspan=3 height=38 class=xl109 width=207 style='border-right:1.0pt solid gray;
  height:28.5pt;width:156pt;min-width: 24px'>:w!</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>若文件属性为『只读』时，强制写入该档案。不过，到底能不能写入，
  还是跟你对该档案的档案权限有关啊！</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl77 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>:q</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>离开<span
  style='mso-spacerun:yes'>&nbsp;</span></td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl109 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>:q!</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>若曾修改过档案，又不想储存，使用 ! 为强制离开不储存档案。</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=36 style='mso-height-source:userset;height:27.0pt'>
  <td colspan=6 height=36 class=xl85 width=750 style='height:27.0pt;width:564pt;
  min-width: 24px'>注意一下啊，那个惊叹号 (!) 在 vi/vim 当中，常常具有『强制』的意思～</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl106 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>:wq</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>储存后离开，若为 :wq! 则为强制储存后离开<span
  style='mso-spacerun:yes'>&nbsp;</span></td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=43 style='mso-height-source:userset;height:32.5pt'>
  <td colspan=3 height=43 class=xl109 width=207 style='border-right:1.0pt solid gray;
  height:32.5pt;width:156pt;min-width: 24px'>ZZ</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>这是大写的 Z
  喔！若档案没有更动，则不储存离开，若档案已经被更动过，则储存后离开！</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl106 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>:w [filename]</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>将编辑的数据储存成另一个档案（类似另存新档）</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=41 style='mso-height-source:userset;height:31.0pt'>
  <td colspan=3 height=41 class=xl109 width=207 style='border-right:1.0pt solid gray;
  height:31.0pt;width:156pt;min-width: 24px'>:r [filename]</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>在编辑的数据中，读入另一个档案的数据。亦即将
  『filename』 这个档案内容加到游标所在行后面</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl106 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>:n1,n2 w [filename]</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>将 n1 到 n2 的内容储存成 filename 这个档案。</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=38 style='mso-height-source:userset;height:28.5pt'>
  <td colspan=3 height=38 class=xl109 width=207 style='border-right:1.0pt solid gray;
  height:28.5pt;width:156pt;min-width: 24px'>:! command</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'><font class="font11">暂时离开</font><font
  class="font12"> vi </font><font class="font11">到指令行模式下执行</font><font
  class="font12"> command </font><font class="font11">的显示结果！例如『</font><font
  class="font12">:! ls /home</font><font class="font11">』即可在</font><font
  class="font12"> vi/vim </font><font class="font11">当中察看</font><font class="font12">
  /home </font><font class="font11">底下以</font><font class="font12"> ls </font><font
  class="font11">输出的档案信息！</font></td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=30 style='height:22.5pt'>
  <td colspan=6 height=30 class=xl65 style='height:22.5pt'>vim/vi 环境改变</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl106 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>:set nu</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>显示行号，设定之后，会在每一行的前缀显示该行的行号</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl109 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt;min-width: 24px'>:set nonu</td>
  <td colspan=3 class=xl112 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt;min-width: 24px'>与 set nu 相反，为取消行号！</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
 <tr height=22 style='height:16.5pt'>
  <td colspan=3 height=22 class=xl106 width=207 style='border-right:1.0pt solid gray;
  height:16.5pt;width:156pt'>:set tabstop=n</td>
  <td colspan=3 class=xl69 width=543 style='border-right:1.0pt solid gray;
  border-left:none;width:408pt'>设置 Tab 键间隔的空格符数量</td>
  <td colspan=4 style='mso-ignore:colspan'></td>
 </tr>
</table>

vim 更多快捷键可以如下思维导图所示：

![vim 快捷键.png](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/vim%20%E5%BF%AB%E6%8D%B7%E9%94%AE.png)

#### 练习

题目是来自[vim 程序编辑器](http://cn.linux.vbird.org/linux_basic/0310vi.php)的练习，如下所示，使用的操作文件 man_db.conf 可以在 http://linux.vbird.org/linux_basic/0310vi/man_db.conf
处获取。

```
1. 請在 /tmp 這個目錄下建立一個名為 vitest 的目錄；
2. 進入 vitest 這個目錄當中；
3. 將 /etc/man_db.conf 複製到本目錄底下(或由上述的連結下載 man_db.conf 檔案)；
4. 使用 vi 開啟本目錄下的 man_db.conf 這個檔案；
5. 在 vi 中設定一下行號；
6. 移動到第 43 列，向右移動 59 個字元，請問你看到的小括號內是哪個文字？
7. 移動到第一列，並且向下搜尋一下『 gzip 』這個字串，請問他在第幾列？
8. 接著下來，我要將 29 到 41 列之間的『小寫 man 字串』改為『大寫 MAN 字串』，並且一個一個挑選是否需要修改，如何下達指令？如果在挑選過程中一直按『y』， 結果會在最後一列出現改變了幾個 man 呢？
9. 修改完之後，突然反悔了，要全部復原，有哪些方法？
10. 我要複製 66 到 71 這 6 列的內容(含有MANDB_MAP)，並且貼到最後一列之後；
11. 113 到 128 列之間的開頭為 # 符號的註解資料我不要了，要如何刪除？
12. 將這個檔案另存成一個 man.test.config 的檔名；
13. 去到第 25 列，並且刪除 15 個字元，結果出現的第一個單字是什麼？
14. 在第一列新增一列，該列內容輸入『I am a student...』；
15. 儲存後離開吧！
```

那么，整体步骤应该如下所示：

1. mkdir vitest
2. cd vitest
3. mv /etc/man_db.conf .
4. vi man_db.conf
5. :set nu
6. 43G -> 59l ->括号内是 as 这个单词
7. gg 或 1G -> /gzip -> 在第 93 列
8. 输入命令 [:29,41s/man/MAN/gc] -> 然后一直点击 y ，总共需要替换 13 个
9. 一直按 u 键即可复原；更加简单粗暴的就是强制退出，也就是输入 :q!
10. 66G 跳到 66 行 -> 6yy 复制 6 行内容(输入后，屏幕最后一行会显示 6 lines yanked) -> G 跳到最后一行，输入 p 复制到最后一行的后面
11. 113G 跳到 113 行 -> 总共需要删除 16 行内容，所以输入 16dd，删除后光标所在行开头就是 ‘#Flags’
12. 输入 [:w man.test.config] 实现保存操作，接着可以输入 [:! ls -l]，即显示查看当前文件夹内文件内容的命令 ls -l 显示的内容在 vim 内，再次按下回车键即回到 vim 命令模式
13. 输入 25G 到 25 行 -> 15x 删除 15 个字符，然后显示的是 tree
14. gg / 1G 到 第一行 -> O 在上方新增一行，然后输入 『I am a student...』-> Esc 键返回命令模式
15. [:wq] 或者 ZZ 保存离开文件




本文参考文章如下：

- [vim 程序编辑器](http://cn.linux.vbird.org/linux_basic/0310vi.php)
- [Linux vi/vim](http://www.runoob.com/linux/linux-vim.html)

------



以上就是本文的主要内容和总结，因为我还没有开通留言功能，所以欢迎关注我的微信公众号--机器学习与计算机视觉或者扫描下方的二维码，和我分享你的建议和看法，指正文章中可能存在的错误，大家一起交流，学习和进步！

公众号后台回复“vim快捷键”可以获取 vim 思维导图！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)

**推荐阅读**
1.[机器学习入门系列(1)--机器学习概览(上)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483667&idx=1&sn=c6b6feb241897ede16bd745d595cef92&chksm=fe3b0f66c94c86701e9b071e62750d189c254fd3ebe9bb6251505162139efefdf866093b38c3&token=2134085567&lang=zh_CN#rd)
2.[机器学习入门系列(2)--机器学习概览(下)](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483672&idx=1&sn=34b6687030db92fd3e04dcdebd09fffc&chksm=fe3b0f6dc94c867b2a72c427ebb90e2a683e6ad97ea2c5fbdc3a3bb86a8b159b8e5f107d2dcc&token=2134085567&lang=zh_CN#rd)
3.[[实战] 图片转素描图](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483679&idx=1&sn=229eaae83f0fad327d4ae419dc6bf865&chksm=fe3b0f6ac94c867cf72992dd2ec118d165c3990818ddd45d5a87736bac907b8871e8a006e9ab&token=2134085567&lang=zh_CN#rd)

如果你觉得我写得还不错，可以给我点个赞！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/0.jpg)

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/02.gif)