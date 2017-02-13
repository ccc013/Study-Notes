### C++知识点总结

------

#### 1. 基础知识

* `endl`操纵符的效果是结束当前行，并将与设备关联的**缓冲区中的内容刷到设备中**。**缓冲刷新操作**可以保证到目前为止程序所产生的所有输出都真正写入输出流中，而不是仅停留在内存中等待写入流。

* **注释界定符不能嵌套**。它是以`/*`开始，以`*/`结束的。因此，一个注释不能嵌套在另一个注释之内。

##### 1.1 基本类型

###### 1.1.1 算术类型

**算术类型**所能表示的数据范围如下：

![这里写图片描述](http://img.blog.csdn.net/20170213132504102?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

基本的字符类型是**char**,一个**char**的空间应确保可以存放机器基本字符集中任意字符对应的数字值。也就是说**一个char的大小和一个机器字节一样**。

其他字符类型用于扩展字符集，如**wchar_t, char16_t, char32_t**。其中**wchar_t**确保可以存放机器最大拓展字符集中的任意一个字符。而后两种字符类型则是为**Unicode**字符集服务。

C++标准指定了一个浮点数有效位数的最小值，但是大多数编译器都实现了更高的精度。通常，**float**以**1个字(32比特)**来表示，**double**以**2个字(64比特)**来表示，**long double**以**3或4个字(96或128比特)**来表示。此外，一般**float和double**分别有**7和16**个有效位。

除了布尔型和扩展的字符型之外，其他整型可以分为**带符号的和无符号的**。类型**int、short、long和long long**都是**带符号的**，在它们前面加上**unsigned**则可以得到无符号类型。其中类型**unsigned int**可以缩写为**unsigned**。

字符型则分成3种：**char、signed char 和 unsigned char**。并且，**char和signed char**并不一样，**而且字符的表现形式同样是两种，带符号和无符号，因为char类型会表现为这两种形式中的一种，具体是由编译器决定具体形式。**

具体类型的选择建议如下：

![这里写图片描述](http://img.blog.csdn.net/20170213133902606?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

###### 1.1.2 类型转换

类型转换的过程如下：

![这里写图片描述](http://img.blog.csdn.net/20170213134501678?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**注意，不能混合使用有符号数和无符号数，如果一个表达式中即包含有符号数和无符号数，那么有符号数会转换成无符号数来进行计算，如果这个有符号数还是负数，那么会得到异常结果。**如下例子所示：

```c++
unsigned u = 10;
int i = -42;
// 输出-84
std::cout << i+i << std::endl;   
// 混合了无符号数和有符号数，如果int占32位，输出4294967264
std::cout << u+i << std::endl;  
```

上述例子最后一个输出说明了一个负数和一个无符号数相加是有可能得到异常结果的。32位的无符号数范围是**0到4294967295**。

###### 1.1.3 字面值常量

**定义：**形如42的值被称为**字面值常量**。

我们可以将整型字面值写作十进制数、八进制数和十六进制数。其中以0开头的代表八进制数，以0x或者0X开头的代表十六进制数。**默认情况下，十进制字面值是带符号数，而八进制和十六进制可以是带符号也可以是无符号数。它们的类型都是选择可以使用的类型中尺寸最小的，并且可以容纳下当前数值的类型。**如十进制可以使用的是**int, long和long long**，而八进制和十六进制还可以使用无符号类型的**unsigned int，unsigned long 和unsigned long long**。

有两类字符是程序员不能直接使用的：一类是**不可打印**的字符，如退格或其他控制字符，因为它们没有可视的图符；另一类是在C++语言中有特殊含义的字符，如单引号、双引号、问号、反斜线，这些情况下需要用到**转义序列**，转义序列均以反斜线开始，C++语言规定的转义序列包括：

![这里写图片描述](http://img.blog.csdn.net/20170213140817117?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

转义序列被当做一个字符使用。

对于字面值类型可以通过添加一些前缀和后缀来改变其默认类型，如下例子所示：

![这里写图片描述](http://img.blog.csdn.net/20170213141100512?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGMwMTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

注意 ，**指定一个长整型字面值时使用大写字母L来标记，这是由于小写字母l和数字1容易混淆。**

##### 1.2 变量

