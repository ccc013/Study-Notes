栈的顺序结构实现

类声明

```
#ifndef ORDERSATCK_H_
#define ORDERSTACK_H_

typedef int OSElemType;
#define MAXSIZE 100

class OrderStack{
private:
	OSElemType data[MAXSIZE];
	int top;	// 栈顶指针
public:
	OrderStack();
	~OrderStack();
	bool Push(OSElemType e);
	bool Pop(OSElemType* e);
	bool GetTop(OSElemType *e);
	bool isEmpty() const;
	int length() const;
	void show() const;
};

#endif
```


类方法定义

```
#include<iostream>
#include"OrderStack.h"

OrderStack::OrderStack(){
	// 初始是一个空栈，栈顶指针为-1
	top = -1;
}

OrderStack::~OrderStack(){

}

// 插入新元素e到栈顶
bool OrderStack::Push(OSElemType e){
	if (top == MAXSIZE - 1){
		std::cout << "This stack is full\n";
		return false;
	}
	top++;
	data[top] = e;
	return true;
}

// 删除栈顶元素，并用e返回其值
bool OrderStack::Pop(OSElemType* e){
	if (isEmpty()){
		std::cout << "This stack is empty!\n";
		return false;
	}

	*e = data[top];
	top--;
	return true;
}

// 返回栈顶元素
bool OrderStack::GetTop(OSElemType *e){
	if (isEmpty()){
		std::cout << "This stack is empty!\n";
		return false;
	}
	*e = data[top];
	return true;
}

// 判断栈是否为空
bool OrderStack::isEmpty() const{
	return top == -1;
}

// 返回栈中元素个数
int OrderStack::length() const{
	return (top + 1);
}

void OrderStack::show() const{
	if (isEmpty()){
		std::cout << "This stack is empty!\n";
		return;
	}
	int i = 0;
	while (i <= top){
		std::cout << data[i] << ", ";
		i++;
		if (i % 10 == 0)
			std::cout << "\n";
	}
	std::cout << "\n";
}
```
