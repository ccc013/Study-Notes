## 《大话数据结构》 第四章 栈与队列

#### 1. 栈（Stack）

    栈是限定仅在表尾进行插入和删除操作的线性表。
    
    允许插入和删除的一端称为栈顶(top)，另一端称为栈底(bottom)，不含任何数据元素的栈称为空栈。栈又称为后进先出(Last In First Out)的线性表，简称LIFO结构。
    
###### 栈的抽象数据类型

```
ADT 栈(Stack)
Data
    同线性表。元素具有相同的类型，相邻元素具有前驱和后继关系。
Operation
    InitStack(*S): 初始化操作，建立一个空栈。
    DestroyStack(*S): 若栈存在，则销毁它。
    ClearStack(*S): 将栈清空。
    StackEmpty(S): 若栈为空，返回true，否则返回false。
    GetTop(S,*e): 若栈存在且非空，用e返回S的栈顶元素。
    Push(*S,e): 若栈存在，插入新元素e到栈S中并成为栈顶元素。
    Pop(*S,*e): 删除栈S中栈顶元素，并用e返回其值。
    StackLength(S): 返回栈S的元素个数
endADT
```

###### 栈的顺序存储结构及实现

```
typedef int SElemType;
typedef struct
{
    SElemType data[MAXSIZE];
    int top;
}SqStack;
```

###### 两栈共享空间

对于两个相同类型的栈，可以使用一个数组来存储。

其结构代码如下

```
typedef struct
{
    SElemType data[MAXSIZE];
    int top1;   // 栈1栈顶指针
    int top2;   // 栈2栈顶指针
}SqDoubleStack;
```

这种情况，栈1空就是`top1 = -1`,栈2空就是`top2 = n`；而栈满的情况，栈1满，而栈2空，就是`top1 = n-1`，栈2满，栈1空，则是`top2 = 0`。而两个栈都满的条件是`top1 + 1 = top2`。

插入操作，此时需要一个参数`stackNumber`判断是栈1还是栈2

```
Status Push(SqDoubleStack *S, SElemType e, int stackNumber){
    if(S->top1 +1 == S->top2)   // 栈已经满了
        return ERROR;
    if(stackNumber == 1)
        S->data[++S->top1] = e;
    if(stackNumber == 2)
        S->data[--S->top2] = e;
    return OK;
}
```

删除操作

```
Status Pop(SqDoubleStack *S, SElemType *e, int stackNumber){
    if(stackNumber == 1){
        if(S->top1 == -1)   // 栈1是空的
            return ERROR;
        *e = data[S->top1--];
        
    }
    else if(stackNumber == 2){
        if(S->top2 == MAXSIZE)
            return ERROR;
        *e = data[S->top2++];
    }
    return OK;
}
```


###### 栈的链式存储结构及实现
链栈的结构如下

```
typedef struct StackNode
{
    SElemType data;
    struct StackNode *next;
}StackNode,*LinkStackPtr;

// 栈顶
typedef struct LinkStack
{
    LinkStackPtr top;
    int count;
}LinkStack;
```

代码例子看[链栈](https://github.com/ccc013/Study-Notes/blob/master/DataStructe%20%26%20Algorithm/CodeExample/LinkStack.md),[顺序结构的栈](https://github.com/ccc013/Study-Notes/blob/master/DataStructe%20%26%20Algorithm/CodeExample/OrderStack.md)


#### 2. 栈的应用

###### (1) 递归

斐波那契数列(Fibonacci)的实现


```
//斐波那契的递归函数
int Fbi(int i){
    if(i<2)
        return i == 0 ? 0:1;
    return Fbi(i-1) + Fbi(i-2);
}

int main(){
    int i;
    for(int i=0;i<40;i++)
        cout<<Fbi(i)<<endl;
    return 0;

}
```

    递归函数是一个直接调用自己或通过一系列的调用语句间接地调用自己的函数。每个递归定义必须至少有一个条件，满足时递归不再进行，即不再引用自身而是返回值退出。
    
    迭代使用的是循环结构，而递归使用的是选择结构。递归能使程序的结构更清晰、更简洁、更容易让人理解，从而减少读懂代码的时间，但是大量的递归调用会建立函数的副本，会耗费大量的时间和内存。迭代则不需要反复调用函数和占用额外的内存。
    
###### (2) 四则运算表达式求值

**后缀表达式**

  对于表达式`"9+(3-1)*3+10/2"`，如果使用后缀表达式表示，应该是`"9 3 1 - 3 * + 10 2 / +"`，称为后缀的原因是所有的符号都是在要运算数字的后面出现。
  
  使用后缀表达式计算出结果的规则如下：
  

  从左到右遍历表达式的每个数字和符号，遇到是数字就进栈，遇到是符号，就将处于栈顶的两个数字出栈，进行运算，运算结果进栈，一直到最终获得结果。


代码例子看[后缀表达式](https://github.com/ccc013/Study-Notes/blob/master/DataStructe%20%26%20Algorithm/CodeExample/postfix%20expression.md)


**中缀表达式转后缀表达式**

  中缀表达式就是平时使用的标准四则运算表达式，即`"9+(3-1)*3+10/2"`。
  
  中缀表达式变成后缀表达式的规则如下：
  
  从左到右遍历表达式的每个数字和符号，若是数字就输出，即成为后缀表达式的一部分；若是符号，则判断其与栈顶符号的优先级，是右括号或优先级不高于栈顶符号，则栈顶元素依次出栈并输出，并将当前符号进栈，一直到最终输出后缀表达式为止。
  
  注意，如果是右括号，则是将左括号出栈就为止，并且也不需要将右括号进栈。

代码例子看[中缀表达式](https://github.com/ccc013/Study-Notes/blob/master/DataStructe%20%26%20Algorithm/CodeExample/infix%20expression.md)


#### 2. 队列
    
    队列是只允许在一端进行插入操作，而在另一端进行删除操作的线性表。
    
    队列是一种先进先出的线性表，简称FIFO。允许插入的一端是队尾，允许删除的一端是队头。

队列的抽象数据类型如下

```
ADT 队列(Queue)
Data
    同线性表。元素具有相同的类型，相邻元素具有前驱和后继关系。
Operation
    InitQueue(*Q): 初始化操作，建立一个空队列Q。
    DestroyQueue(*Q): 若队列存在，则销毁它。
    ClearQueue(*Q): 将队列清空。
    QueueEmpty(Q): 若队列为空，返回true，否则返回false。
    GetHead(Q,*e): 若队列存在且非空，用e返回队列Q的队头元素。
    EnQueue(*Q,e): 若队列存在，插入新元素e到队列中并成为队尾元素。
    DeQueue(*Q,*e): 删除队列Q中队头元素，并用e返回其值。
    QueueLength(Q): 返回队列Q中的元素个数。
```

##### (1) 循环队列
    头尾相接的顺序存储结构的队列被称为循环队列。
    
    循环队列中，空队列的判断条件是`front == rear`，而队列满的判断方法是有两种，第一种是设置一个标志标量flag，当`rear == front`且`flag=0`时，队列空，而`rear == front`,且`flag = 1`时，队列满。
    
    第二种方法就是队列满时，要保留一个空闲元素，此时队列满的条件是`(rear+1) % QueueSize == front;`,通用的计算队列长度是`(rear - front + QueueSize) % QueueSize`。

循环队列的顺序存储结构如下

```
typedef int QElemType;
typedef struct
{
    QElemType data[MAXSIZE];
    int front;
    int rear;
}SqQueue;
```

代码例子：[循环队列](https://github.com/ccc013/Study-Notes/blob/master/DataStructe%20%26%20Algorithm/CodeExample/CirculateQueue.md)

##### (2) 队列的链式结构

链队列的结构如下：

```
typedef int QElemType;
typedef struct QNode
{
    QElemType data;
    struct QNode *next;
}QNode, * QueuePtr;

typedef struct
{
    QueuePtr front, rear;   // 队头，队尾指针
}
```

代码例子看：[队列的链式结构实现](https://github.com/ccc013/Study-Notes/blob/master/DataStructe%20%26%20Algorithm/CodeExample/LinkQueue.md)




