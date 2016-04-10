根据《大话数据结构》第三章的线性表，顺序存储结构实现的代码，封装成一个类。


这是头文件，类声明。
```
#ifndef LINEARLIST_H_
#define LINEARLIST_H_

typedef int ElemType;
const int OK = 1;
const int ERROR = 0;

class LinearList_order
{
private:
	static const int MAXSIZE = 20;		// 设置线性表的存储空间
	ElemType data[MAXSIZE];		// 存储数据元素的数组
	int length;		// 当前线性表的长度
public:
	LinearList_order();
	LinearList_order(ElemType * d,int l);
	LinearList_order(const LinearList_order & L);
	~LinearList_order();
	ElemType getElem(int i, ElemType * e);
	ElemType insertElem(int i, ElemType e);
	int getLength(){ return length; }
	ElemType deleteElem(int i, ElemType *e);
};

#endif
```

这是类方法的定义，除了实现构造函数和析构函数外，实现了获得元素的方法，插入和删除方法，以及返回当前线性表元素个数的方法。
```
#include<iostream>
#include"LinearList.h"

LinearList_order::LinearList_order(){
	data[0] = { 0 };
	length = 0;
}

LinearList_order::LinearList_order(ElemType* d,int l){
	if (l > MAXSIZE){
		std::cout << "the length is too large!\n";
	}
	else{
		for (int i = 0; i < l; i++){
			data[i] = d[i];
		}
		length = l;
	}

}

LinearList_order::LinearList_order(const LinearList_order & L){
	length = L.length;
	for (int i = 0; i < length; i++){
		data[i] = L.data[i];
	}
}

LinearList_order::~LinearList_order(){
}

ElemType LinearList_order::getElem(int i, ElemType* e){
	if (length == 0 || i<1 || i>length)
		return ERROR;
	*e = data[i - 1];
	return OK;
}

ElemType LinearList_order::insertElem(int i, ElemType e){
	int k;
	// 线性表已经满了
	if (length == MAXSIZE)
		return ERROR;
	// i不在范围内时
	if (i<1 || i>(length + 1))
		return ERROR;
	if (i <= length){
		for (k = length - 1; k >= (i - 1); k--)		// 将要插入的位置后数据元素向后移动一位
			data[k + 1] = data[k];
	}
	// 插入新元素
	data[i - 1] = e;
	length++;
	return OK;
}

ElemType LinearList_order::deleteElem(int i, ElemType * e){
	int k;
	// 线性表为空
	if (length == 0)
		return ERROR;
	// 删除位置不正确
	if (i<1 || i>length)
		return ERROR;
	*e = data[i - 1];
	if (i < length){
		// 将删除位置后继元素前移
		for (k = i; k < length; k++)
			data[k - 1] = data[k];
	}
	length--;
	return OK;

}
```
