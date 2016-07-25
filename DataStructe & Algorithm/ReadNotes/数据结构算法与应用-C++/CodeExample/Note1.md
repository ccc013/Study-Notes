第一章部分练习题的代码

1. 
```
/*
它要求用户输入一个非负数，并负责验证用户所输入的数是否真的大于或等于 0，如果不是，
它将告诉用户该输入非法，需要重新输入一个数。在函数非成功退出之前，应给用户三次机会。
如果输入成功，函数应当把所输入的数作为引用参数返回。
输入成功时，函数应返回true, 否则返回f a l s e。
*/
template <class T>
bool testInput(T & output){
	std::cout << "Please input a non-negative number:\n";
	int wrongNumbers = 0;
	int input;
	std::cin >> input;
	while (input < 0){
		wrongNumbers++;
		if (wrongNumbers > 3){
			std::cout << "your input 3 fault input!\n";
			return false;
		}
		std::cout << "your input is illegal,please input again:\n";
		std::cin >> input;
	}
	output = input;

	return true;
}
```

2.

```
/*用来测试数组 a中的元素是否按升序排列（即a [ i ]≤a [ i + 1 ] ,
其中0≤i＜n - 1）。如果不是，函数应返回f a l s e，否则应返回t r u e。*/
template<class T>
bool isSorted(const T a[], int n){
	for (int i = 0; i < n - 1; i++){
		if (a[i] > a[i + 1])
			return false;
	}
	return true;
}
```

3.

```
/*创建二维数组*/
template<class T>
bool make2DArray(T ** &x, int rows, int cols){
	try{
		// 创建行指针
    	x = new T *[rows];

		// 为每一行分配空间
		for (int i = 0; i < rows; i++){
			x[i] = new int[cols];
		}
		return true;
	}
	catch (...){
		return false;
	}
}

/*释放空间*/
template<class T>
void Delete2DArray(T ** &x, int rows){
	// 释放为每一行所分配的空间
	for (int i = 0; i < rows; i++){
		delete[] x[i];
	}
	// 删除行指针
	delete[] x;
	x = 0;
}
```


