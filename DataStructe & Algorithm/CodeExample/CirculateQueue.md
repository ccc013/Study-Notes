循环队列的简单实现，这次使用了模板类，需要注意的是要将类方法的定义和类声明放在同一个文件中。

所以完整的类方法和声明如下：

```
#ifndef CIRCULEQUEUE_H_
#define CIRCULEQUEUE_H_

// 使用类模板
template <class T>
class CirculateQueue{
private:
	enum {MAX = 100};
	T data[MAX];
	int front;	// 头指针
	int rear;	// 队尾指针
public:
	CirculateQueue();
	~CirculateQueue();
	int Length() const;
	bool EnQueue(T e);
	bool DeQueue(T & e);
	bool empty();
	void show() const;
};

template <class T>
CirculateQueue<T>::CirculateQueue(){
	front = rear = 0;
}

template <class T>
CirculateQueue<T>::~CirculateQueue(){

}

template <class T>
int CirculateQueue<T>::Length() const{
	return (rear - front + MAX) % MAX;
}

template <class T>
bool CirculateQueue<T>::empty(){
	return front == rear;
}

template <class T>
bool CirculateQueue<T>::EnQueue(T e){
	// 判断队列是否满
	if ((rear + 1) % MAX == front){
		return false;
	}
	data[rear] = e;
	rear = (rear + 1) % MAX;	// 队尾指针后移一位，若到最后则转到数组头部
	return true;
}

template<class T>
bool CirculateQueue<T>::DeQueue(T & e){
	// 判断队列是否空
	if (front == rear)
		return false;
	e = data[front];
	front = (front + 1) % MAX;
	return true;
}

template<class T>
void CirculateQueue<T>::show() const{
	if (front == rear){
		std::cout << "Queue is empty.\n";
		return;
	}
	int i = front;
	while (i != rear){
		std::cout << i << ": " << data[i] << "\n";
		i = (i + 1) % MAX;
	}

}

#endif
```
