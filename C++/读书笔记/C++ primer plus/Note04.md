##第6章 分支语句和逻辑运算符
###1. if语句
if语句的语法如下:

    if(test-condition)
        statement
test-condition(测试条件)为真，则程序会执行statement(语句)，否则就跳过语句；

**条件运算符和错误防范**

一般将更直观的表达式`variable == value`反转为`value == variable`，以此来捕获将相等运算符误写为赋值运算符的错误

###2. 逻辑表达式
C++提供了3种逻辑运算符，分别是OR(||)、AND(&&)、NOT(!)。

#####(1) 逻辑OR运算符：||
* ||的优先级比关系运算符低；
* C++规定，||运算符是个顺序点，也就是先修改左侧的值，再对右侧的值进行判定；
* 对于||，如果左侧表达式为true，则C++不会去判定右侧的表达式，因为只要一个表达式为true，整个逻辑表达式也是true

#####(2) 逻辑AND运算符：&&
* &&的优先级也是低于关系运算符
* &&也是一个顺序点，所以首先判定左侧，如果左侧为false，则整个逻辑表达式必定为false，在这种情况下，C++将不会对右侧进行判定
* 使用&&来设置取值范围，如`age > 17 && age < 35`

#####(3) 逻辑NOT运算符：！
* ！运算符将它后面的表达式的真值取反
* ！运算符的优先级高于所有的关系运算符和算术运算符，因此，要对表达式求反，必须用括号将其括起
* 逻辑运算符也可以用另一种表示方式--and、or和not,这些都是C++保留字，这意味着不能将它们用作变量名等。

###3. 字符函数库cctype
在头文件cctype中定义了一些可以简化诸如确定字符是否为大写字母、数字、标点符号等工作的函数，具体函数如下所示:

![cctype中的字符函数](https://raw.githubusercontent.com/ccc013/Study-Notes/master/images/cctype%E4%B8%AD%E7%9A%84%E5%AD%97%E7%AC%A6%E6%95%B0.png "cctype中的字符函数")

###4. ?:运算符
?:运算符是可以常用来代替if else语句的运算符，被称为条件运算符，其通用格式如:`expression1 ? expression2 : expression3`。

使用条件运算符会比if else语句更加简洁，它是生成一个表达式，因此是一个值，可以将其赋给变量或将其放到一个更大的表达式中；但如果代码变得更复杂时，使用if else语句可能会让表达更为清晰。

###5. switch 语句
switch的通用格式如下：

    switch (integer-expression)
    {
        case label1: statement(s)
                    break;
        case label2: statement(s)
                    break;
        ...
        default:    statement(s)
    }
integer-expression必须是一个结果为整数值的表达式，而每个标签都必须是整数常量表达式，最常见标签是int或char常量(如1或'q'),也可以是枚举量。

* 通常cin无法识别枚举类型，而switch语句将int值和枚举量标签进行比较时，将枚举量提升为int；另外在while循环测试条件中，也会将枚举量提升为int类型
* switch并不是为处理取值范围而设计的，它的每一个case标签都必须是一个单独的值，而且是整数值，并且是常量；如果选项设计取值范围、浮点测试或两个变量的比较，应使用if else语句

###6. 读取数字的循环
对于如下代码:

    int n;
    cin >> n;
如果用户输入一个单词，而不是一个数字，则将发生4种情况:
    
  * n的值保持不变
  * 不匹配的输入将被留在输入队列中
  * cin对象中的错误标记被设置
  * 对cin方法的调用将返回false

对于输入错误后，错误处理代码如下:

    while (!(cin >> golf[i])){
        cin.clear();
        while(cin.get() != '\n')
            continue;
        cout << "Please enter a number: ";
    }
golf是一个保存数字的数组，cin.clear()方法是用来重置输入，没有这句话，程序将拒绝继续读取输入，然后再通过一个while循环，使用cin.get()来读取行尾之前的所有输入，从而删除这一行中的错误输入。

###7. 简单文本输入/输出
#####(1) 写入到文本文件
* 必须包含头文件fstream,头文件fstream定义了一个用于处理输出的ofstream
* 需要声明一个或多个ofstream变量(对象)，并命名
* 必须指明名称空间std
* 需将ofstream对象和文件关联起来，方法之一是使用open()方法，而使用完文件需要用close()方法将其关闭

使用例子如下:

    ofstream outFile;   // 声明一个ofstream对象
    outFile.open("fish.txt");   // 关联一个文件
    double t = 12.5;
    outFile << t;   // 将一个浮点数写入文件
    outFile.close();    // 关闭文件
通过这个例子，对于写入到文件的操作是与cout的用法非常相似

* outFile可使用cout可使用的任何方法，即除了运算符<<外，还可以使用各种格式化方法，如self()和precision()
* 对于open()方法，可以接受C-风格字符串，对于输入的文件名，如果文件不存在则会新建一个文件；如果文件存在，默认情况下，该方法是首先截断该文件，即丢掉原有的内容，再将新的输出加入到该文件中

#####(2) 读取文本文件
* 必须包含头文件fstream,该头文件定义了一个用于处理输入的ifstream类
* 需要声明一个或多个ifstream变量(对象),必须使用名称空间std；
* 必须将ifstream对象和文件关联起来，方法之一也是使用open()方法，使用完文件也是需要close()方法
* 可结合使用ifstream对象和运算符>>来读取各种类型的数据
* 可以使用ifstream对象和get()方法来读取一个字符，使用ifstream对象和getline()来读取一行字符
* 可以结合使用ifstream和eof()、fail()等方法来判断输入是否成功
* ifstream对象本身被用作测试条件时，如果最后一个读取操作成功，它将被转换为布尔值true，否则转换为false

使用例子如下:

    ifstream inFile;    // 声明一个ifstream对象
    inFile.open("b.txt");
    double t;
    inFile >> t;
检查文件是否被成功打开的首先方法是使用方法is_open()方法，如下所示:

    inFile.open("b.txt");
    if(!inFile.is_open()){
        exit(EXIT_FAILURE);
    }
函数exit()是在头文件cstdlib中定义的，在该头文件中，还定义了一个用于同操作系统通信的参数值EXIT_FAILURE，函数exit()终止程序。

* 方法is_open()是C++中相对较新的内容，也可以使用较老的方法good()来代替它，但是该方法在检查可能存在的问题方面，没有前者那么广泛。
* 读取文件时，首先不应超过EOF，如果最后一次读取数据遇到EOF，方法eof()将返回true；
* 其次如果是遇到类型不匹配，则是fail()方法会返回true(如果遇到EOF，该方法也会返回true)；
* 最后可能出现意外的问题，如文件受损或硬件故障，则bad()方法会返回true；
* 可以不用分别检查这些情况，使用good()方法，它会在没有发生任何错误时返回true。

