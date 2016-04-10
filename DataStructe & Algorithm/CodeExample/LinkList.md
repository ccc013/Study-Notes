##### 单链表
根据《大话数据结构》第三章的线性表，顺序存储结构实现的代码，封装成一个类。

类声明

```
#ifndef LINKLISTL_H_
#define LINKLISTL_H_

typedef int ElemTypeL;

class LinkListL
{
private:
	struct Node
	{
		ElemTypeL data;
		struct Node * next;
	};
	// 头指针
	Node*front;
	// 当前链表的元素个数
	int items;
	// 使用伪私有方法，一是避免了本来将自动生成的默认方法定义，二是由于这些方法是私有的，所以不能被广泛使用
	LinkListL(const LinkListL& l){};
	LinkListL & operator=(const LinkListL & l){ return *this; }
public:
	LinkListL();
	LinkListL(const ElemTypeL elem);
	~LinkListL();
	bool isempty() const;
	int length() const;
	void createListTail(int n);
	void clearList();
	bool getElem(int i, ElemTypeL * e);
	bool insertElem(int i, ElemTypeL e);
	bool deleteElem(int i, ElemTypeL * e);
	void show()const;
};

#endif
```

类定义，除了析构函数，构造函数外，还有创建单链表，使用尾插法，随机生成n个元素；清空链表；获取某个特定位置元素，插入和删除元素，以及显示整个链表的元素

```
#include<iostream>
#include<time.h>
#include"LinkListL.h"

LinkListL::LinkListL(){
	front = NULL;
	items = 0;
}

LinkListL::LinkListL(const ElemTypeL elem){
	Node* p = new Node;
	p->data = elem;
	p->next = NULL;
	
	front = p;
	items = 1;
}

LinkListL::~LinkListL(){
	Node * temp;
	while (front != NULL){
		// 当链表还不是空的时候,释放内存
		temp = front;
		front = front->next;
		delete temp;
	}
}

// 创建一个链表，随机产生n个元素的值，建立带表头结点的单链表（尾插法）
void LinkListL::createListTail(int n){
	int i;
	Node*p, *r;
	srand(time(0));
	front = new Node[sizeof(Node)];
	r = front;
	for (i = 0; i < n; i++){
		p = new Node[sizeof(Node)];
		p->data = rand() % 100 + 1;		// 随机生成100以内的数字
		r->next = p;
		r = p;											// 将当前的新结点定义为表尾终端结点
	}
	r->next = NULL;
	items = n;
}

// 清空整个链表
void LinkListL::clearList(){
	Node*p, *q;
	p = front->next;			// p指向第一个结点
	while (p){
		q = p->next;
		delete p;
		p = q;
	}
	front->next = NULL;
}

bool LinkListL::isempty()const{
	return items == 0;
}

int LinkListL::length()const{
	return items;
}

// 获取第i个位置的元素，并赋给e, 成功返回true，失败返回false
bool LinkListL::getElem(int i, ElemTypeL*e){
	int j;							// 计数器
	Node* p;
	p = front->next;		// 指向链表上的第一个结点
	j = 1;
	while (p && j < i){	// 在p不为空且j不等于i的时候，循环继续，遍历链表
		p = p->next;
		++j;
	}
	if (!p || j>i)
		return false;				// 第i个结点不存在
	*e = p->data;			// 取出第i个结点的数据
	return true;
}

// 在链表中第i个位置之前插入新的数据元素e
bool LinkListL::insertElem(int i, ElemTypeL e){
	int j;
	Node * p;
	p = front->next;
	j = 1;
	while (p && j < i-1){
		p = p->next;
		++j;
	}
	
	if (!p || j>i)
		return false;

	Node * s = new Node;	// 生成新的结点
	s->data = e;					
	s->next = p->next;		// 必须先将p的后继点赋值给s的后继
	p->next = s;					// 将s赋给p的后继
	items++;							// 长度加1
	return true;
}

// 删除链表的第i个结点，并用e返回其值
bool LinkListL::deleteElem(int i, ElemTypeL* e){
	int j;
	Node * p, * q;				// 声明为指针，必须都带有 * 
	p = front->next;
	j = 1;
	while (p && j < i-1){
		p = p->next;
		++j;
	}

	if (!p || j>i)
		return false;

	q = p->next;
	p->next = q->next;
	*e = q->data;
	delete q;						// 释放被删除的结点
	return true;
}

// 显示链表中的所有元素
void LinkListL::show() const{
	Node*p;
	int j;
	if (front == NULL){
		std::cout << "no elem in linkList.\n";
		return;
	}else
	{
		p = front->next;
	}
	j = 1;
	while (p){
		std::cout << j << ": " << p->data << std::endl;
		p = p->next;
		++j;
	}
}
```
