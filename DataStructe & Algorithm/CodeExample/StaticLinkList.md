
静态链表的例子

类声明
```
#ifndef STATICLINKLIST_H_
#define STATICLINKLIST_H_

#define MAXSIZE 1000
typedef int SElemType;

class StaticLinkList{
private:
	struct staticNode
	{
		SElemType data;
		int cur;
	};
	staticNode datas[MAXSIZE];
public:
	StaticLinkList();
	~StaticLinkList();
	// 获得备用链表的第一个结点的下标
	int getSLL();
	int insertItem(int i, SElemType e);
	int length() const;
	int deleteItem(int i);
	void freeSLL(int k);
	void show() const;
};

#endif
```

类方法的定义，主要实现了插入和删除操作，返回静态链表的长度，显示静态链表的所有元素，获得第一个空闲的元素，以及回收被删除的空闲结点。

```
#include"StaticLinkList.h"
#include<iostream>

StaticLinkList::StaticLinkList(){
	// 初始化数组的状态，此时静态链表为空
	int i;
	for (i = 0; i < MAXSIZE - 1; i++)
		datas[i].cur = i + 1;
	datas[MAXSIZE - 1].cur = 0;	// 目前静态链表为空，最后一个元素的cur为0
}

StaticLinkList::~StaticLinkList(){

}

// 获得备用链表的第一个结点的下标
int StaticLinkList::getSLL(){
	int i = datas[0].cur;		// 数组第一个元素的cur存放的值就是备用链表的第一个元素的下标

	if (datas[0].cur)
		datas[0].cur = datas[i].cur;			// 将备用链表的第二个元素下标赋给数组第一个元素

	return i;
}

// 在第i个元素之前插入新的数据e
int StaticLinkList::insertItem(int i, SElemType e){
	int j, k, l;
	k = MAXSIZE - 1;
	if (i < 1 || i >length() + 1)
		return -1;
	j = getSLL();
	if (j){
		datas[j].data = e;
		for (l = 1; l <= i - 1; l++)
			k = datas[k].cur;
		datas[j].cur = datas[k].cur;
		datas[k].cur = j;
		return 1;
	}
	return -1;
}

// 返回长度
int StaticLinkList::length() const{
	int j = 0;
	int k = datas[MAXSIZE - 1].cur;
	while (k){
		k = datas[k].cur;
		j++;
	}
	return j;
}

// 删除第i个元素e
int StaticLinkList::deleteItem(int i){
	int j, k;
	if (i<1 || i>length() + 1)
		return -1;
	k = MAXSIZE - 1;
	for (j = 1; j <= i - 1; j++)
		k = datas[k].cur;
	
	j = datas[k].cur;
	datas[k].cur = datas[j].cur;
	freeSLL(j);
	return 1;
}

// 将下标为k的空闲结点回收到备用链表中
void StaticLinkList::freeSLL(int k){
	datas[k].cur = datas[0].cur;
	datas[0].cur = k;
}

void StaticLinkList::show() const{
	if (length() <= 0){
		std::cout << "staticLinkList is empty.\n";
		return;
	}
	int k, j;
	j = 0;
	k = MAXSIZE - 1;
	int i = datas[k].cur;
	std::cout << "StaticLinkList contains:\n";
	while (i){
		std::cout << datas[i].data << ", ";
		i = datas[i].cur;
		j++;
		if (j % 10 == 0)
			std::cout << "\n";
	}
	std::cout << "\n";
}

```

