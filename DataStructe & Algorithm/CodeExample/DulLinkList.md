双向链表，带循环。

这是类声明

```
#ifndef DULLINKLIST_H_
#define DULLINKLIST_H_

typedef int DElemType;

// 创建循环的双向链表
class DulLinkList
{
private:
	struct DulNode
	{
		DElemType data;
		DulNode *prior;
		DulNode *next;
	};
	DulNode * front;
	int items;
public:
	DulLinkList();
	~DulLinkList();
	bool isempty() const;
	int length() const;
	void createListTail(int n);
	void clearList();
	bool getElem(int i, DElemType * e);
	bool insertElem(int i, DElemType e);
	bool deleteElem(int i, DElemType * e);
	void show()const;
};



#endif
```

类定义
双向链表在查询操作，求长度跟单链表还是一样的，区别是在插入和删除的操作，因为涉及到前驱和后驱两个指针。

```
#include"DulLinkList.h"
#include<iostream>
#include<time.h>

using std::cout;
using std::end;

DulLinkList::DulLinkList(){
	front = NULL;
	items = 0;
}

DulLinkList::~DulLinkList(){
	DulNode* temp;
	while (front != NULL){
		temp = front;
		front = front->next;
		delete temp;
	}
}

bool DulLinkList::isempty() const{
	return items == 0;
}

int DulLinkList::length()const{
	return items;
}

// 创建一个长度为n的双向链表
void DulLinkList::createListTail(int n){
	int i;
	DulNode*p, *r;
	srand(time(0));
	front = new DulNode[sizeof(DulNode)];
	r = front;
	for (i = 0; i < n; i++){
		p = new DulNode[sizeof(DulNode)];
		p->data = rand() % 100 + 1;		// 随机生成100以内的数字
		p->prior = r;		// 设置新结点的前驱指针
		r->next = p;
		r = p;											// 将当前的新结点定义为表尾终端结点
	}
	r->next = front;
	items = n;
}

// 清空整个链表
void DulLinkList::clearList(){
	DulNode*p, *q;
	p = front->next;			// p指向第一个结点
	while (p != front){
		q = p->next;
		delete p;
		p = q;
	}
	front->next = NULL;
	items = 0;
}

// 获取第i个位置的元素，并赋给e, 成功返回true，失败返回false
bool DulLinkList::getElem(int i, DElemType*e){
	int j;							// 计数器
	DulNode* p;
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

// 在第i个位置插入元素
bool DulLinkList::insertElem(int i, DElemType e){
	int j;
	DulNode * p;
	p = front->next;
	j = 1;
	while (p && j <(i - 1)){
		p = p->next;
		++j;
	}

	if (!p || j>i)
		return false;

	DulNode *s = new DulNode;
	s->data = e;
	// 插入操作，先搞定s的前驱和后继
	s->prior = p;
	s->next = p->next;
	// 再搞定s的后结点的前驱
	p->next->prior = s;
	// 最后是前结点的后驱
	p->next = s;
	items++;
	return true;
}

// 删除第i个结点
bool DulLinkList::deleteElem(int i, DElemType * e){
	int j;
	DulNode* p;
	p = front->next;
	j = 1;
	while (p&&j < i){
		p = p->next;
		j++;
	}

	if (!p || j>i)
		return false;

	p->prior->next = p->next;
	p->next->prior = p->prior;
	*e = p->data;
	delete p;
	items--;
	return true;
}

void DulLinkList::show()const{
	DulNode*p;
	int j;
	if (items==0){
		std::cout << "no elem in DulLinkList.\n";
		return;
	}
	else
	{
		p = front->next;
	}
	j = 1;
	while (p != front){
		std::cout << j << ": " << p->data << std::endl;
		p = p->next;
		++j;
	}
}
```

