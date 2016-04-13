链栈的实现例子

类声明

```
#ifndef LINKSTACK_H_
#define LINKSTACK_H_

typedef int LSElemType;

class LinkStack{
private:
	// 结点
	struct StackNode
	{
		LSElemType data;
		struct StackNode *next;
	};
	// 栈顶
	StackNode* top;
	int count;
public:
	LinkStack();
	~LinkStack();
	bool isEmpty();
	int length()const;
	void show()const;
	bool Push(LSElemType e);
	bool Pop(LSElemType *e);
	bool GetTop(LSElemType *e);
};

#endif
```

类方法定义
大部分实现方法与顺序结构相同，区别是在插入和删除的操作有所不同。

```
#include<iostream>
#include"LinkStack.h"

LinkStack::LinkStack(){
	top = NULL;
	count = 0;
}

LinkStack::~LinkStack(){

}

// 判断是否为空
bool LinkStack::isEmpty(){
	return count == 0;
}

// 返回元素个数
int LinkStack::length() const{
	return count;
}

// 插入元素
bool LinkStack::Push(LSElemType e){
	StackNode * s = new StackNode[sizeof(StackNode)];
	s->data = e;
	// 将当前栈顶元素赋给新结点的后继
	s->next = top;
	top = s;
	count++;
	return true;
}

// 删除元素
bool LinkStack::Pop(LSElemType *e){
	if (isEmpty()){
		std::cout << "This stack is empty!\n";
		return false;
	}

	*e = top->data;
	StackNode* p = top;
	top = top->next;	// 将栈顶指针往后移动一位，指向后一个结点
	delete p;	// 释放结点p
	count--;
	return true;
}

// 获得栈顶元素
bool LinkStack::GetTop(LSElemType *e){
	if (isEmpty()){
		std::cout << "This stack is empty!\n";
		return false;
	}

	*e = top->data;
	return true;
}

void LinkStack::show()const{
	if (count == 0){
		std::cout << "This stack is empty!\n";
		return;
	}
	StackNode*p = top;
	while (p){
		std::cout << p->data << ", ";
		p = p->next;

	}
	std::cout << "\n";
}
```
