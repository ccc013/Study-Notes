队列的链式结构实现

代码如下，同样也是使用了模板类
```
#ifndef LINKQUEUE_H_
#define LINKQUEUE_H_
#include<iostream>

template<class T>
class LinkQueue
{
private:
	// 结点定义
	struct QNode{
		T data;
		QNode* next;
	};
	// 队头和队尾指针
	QNode* front;
	QNode* rear;
	int counts;
public:
	LinkQueue();
	~LinkQueue();
	int length() const{ return counts; }
	bool empty() { return counts == 0; }
	bool EnQueue(T e);
	bool DeQueue(T &e);
	void show() const;
};

template<class T>
LinkQueue<T>::LinkQueue(){
	counts = 0;
	// 创建头结点
	QNode* top = new QNode[sizeof(QNode)];
	front = rear = top;
}

template <class T>
LinkQueue<T>::~LinkQueue(){
	QNode* temp;
	while (front != NULL){
		temp = front;
		front = front->next;
		delete temp;
	}
}

template<class T>
bool LinkQueue<T>::EnQueue(T e){
	QNode* s = new QNode[sizeof(QNode)];
	if (!s)
		return false;		// 存储分配失败
	s->data = e;
	s->next = NULL;
	rear->next = s;	// 将新结点赋给原队尾结点的后继
	rear = s;				// 将新结点设置为队尾结点
	counts++;
	return true;
}

template<class T>
bool LinkQueue<T>::DeQueue(T &e){
	QNode * p;
	// 判断队列是否为空
	if (front == rear)
		return false;
	p = front->next;
	e = p->data;
	front->next = p->next;
	// 若队头是队尾，则删除后将rear指向头结点
	if (rear == p)
		rear = front;
	delete p;
	counts--;
	return true;
}

template<class T>
void LinkQueue<T>::show() const{
	if (front == rear){
		std::cout << "The queue is empty.\n";
		return;
	}
	QNode* p = front->next;
	int j = 1;
	while (j<=counts){
		std::cout << j << ": " << p->data << "\n";
		p = p->next;
		++j;
	}
	std::cout << "finsh showing.\n";
	
}

#endif
```
