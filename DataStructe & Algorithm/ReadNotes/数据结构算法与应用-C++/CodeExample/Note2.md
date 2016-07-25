第二章部分练习及例子的代码

1.在未排序的数组 a [ 0 : n-1 ]中搜索 x，
如果找到，则返回所在位置，否则返回- 1

```

template<class T>
int SequentialSearch(T a[], const T& x, int n)
{
	int i;
	for (i = 0; i < n && a[i] != x; i++);
	if (i == n) return -1;
	return i;
}
```

2. 多项式求值

```
/*多项式求值*/
template<class T>
T PolyEval(T coeff[], int n, const T&x){
	// 计算n次多项式的值，coeff[0:n]为多项式的系数
	T y = 1, value = coeff[0];
	for (int i = 1; i < n; i++){
		y *= x;
		value += y*coeff[i];
	}
	return value;
}

/*使用Horner法则求解多项式*/
template<class T>
T Horner(T coeff[], int n, const T&x){
	T value = coeff[n-1];
	for (int i = 2; i <= n; i++){
		value = value*x + coeff[n - i];
	}
	return value;
}

```

3. 计数排序

```
/*[计算名次] 元素在队列中的名次（ r a n k）可定义为队列中所有比它小的元素数目加上在
它左边出现的与它相同的元素数目。
例如，给定一个数组 a=[4, 3, 9, 3, 7]作为队列，则各元素
的名次为r =[2, 0, 4, 1, 3]*/
template<class T>
void Rank(T a[], int n, int r[]){
	// 计算a[0:n-1]中n个元素的排名
	for (int i = 0; i < n; i++)
		r[i] = 0;		// 初始化排名数组
	// 逐对比较所有的元素
	for (int i = 0; i < n; i++){
		for (int j = i+1; j < n; j++){
			if (a[i] <= a[j])
				r[j]++;
			else
				r[i]++;
		}
	}
}

/*利用Rank函数中得到的每个元素的名次，对数组重新排序*/
template<class T>
void Rearrange(T a[], int n, int r[]){
	// 按顺序重排数组a中的元素，使用附加数组u
	T *u = new T[n];
	// 在u中移动到正确的位置
	for (int i = 0; i < n; i++)
		u[r[i]] = a[i];
	// 移回到a中
	for (int i = 0; i < n; i++)
		a[i] = u[i];
	delete[] u;
}

```

不需要一个附加数组的方法
```
template<class T>
void Rearrange2(T a[], int n, int r[])
{// 原地重排数组元素
	for (int i = 0; i < n; i++)
		// 获取应该排在 a [ i ]处的元素
	while (r[i] != i) {
		int t = r[i];
		Swap(a[i], a[t]);
		Swap(r[i], r[t]);
	}
}

/*交换两个数组元素*/
template<class T>
void Swap(T& a, T&  b){
	T temp = a;
	a = b;
	b = temp;
}
```


4. 选择排序

```
/*选择排序，按照递增次序排列*/
template<class T>
void selectionSort(T a[], int n){
	for (int size = n; size > 1; size--){
		int max = Max(a, size);
		Swap(a[max], a[size - 1]);
	}
}

/*寻找数组中最大元素*/
template<class T>
int Max(T a[], int n){
	int pos = 0;
	for (int i = 1; i < n; i++){
		if (a[pos] < a[i])
			pos = i;
	}
	return pos;
}
/*交换两个数组元素*/
template<class T>
void Swap(T& a, T&  b){
	T temp = a;
	a = b;
	b = temp;
}
```

及时终止的选择排序，可以避免不必要的循环。
```
/*及时终止的选择排序*/
template<class T>
void selectionSort2(T a[], int n){
	bool isSorted = false;		// 加入终止循环的条件
	for (int size = n; !isSorted && (size > 1); size--){
		int pos = 0;
		isSorted = true;
		// 找最大元素
		for (int i = 1; i < size; i++){
			if (a[pos] <= a[i])
				pos = i;
			else
				// 没有按序排列
				isSorted = false;
		}
		Swap(a[pos], a[size - 1]);
	}
}
```

5. 冒泡排序


```
/*一次冒泡*/
template<class T>
void Bubble(T a[], int n){
	for (int i = 0; i < n-1; i++){
		if (a[i]>a[i + 1])
			Swap(a[i], a[i + 1]);
	}
}

/*对数组a[0:n - 1]中的n个元素进行冒泡排序*/
template <class T>
void BubbleSort(T a[], int n)
{
	for (int i = n; i>1; i--)
			Bubble(a, i);
}
```

及时终止的冒泡排序
```
/*一次冒泡*/
template<class T>
bool Bubble(T a[], int n){
	// 没有发生交换
	bool isSwaped = false;
	for (int i = 0; i < n - 1; i++){
		if (a[i]>a[i + 1]){
			Swap(a[i], a[i + 1]);
			// 发生了交换
			isSwaped = true;
		}
	}
	return isSwaped;
}

/*对数组a[0:n - 1]中的n个元素进行冒泡排序*/
template <class T>
void BubbleSort(T a[], int n)
{
	for (int i = n; i > 1 && Bubble(a, i); i--){}
}
```

6. 插入排序
```
/*向一个有序数组中插入元素,假定a的大小超过n*/
template<class T>
void Insert(T a[], int& n, const T& x){
	int i;
	for (i = n - 1; i >= 0 && x < a[i]; i--)
		a[i + 1] = a[i];
	a[i + 1] = x;
}

template<class T>
void InsertionSort(T a[], int n){
	for (int i = 1; i < n; i++){
		T t = a[i];
		Insert(a, i, t);
	}
}
```

7. 折半搜索
```
/*折半搜索*/
template<class T>
int BinarySearch(T a[], const T& x, int n){
	int left = 0, right = n - 1;
	while (left <= right){
		int middle = (left + right) / 2;
		if (x == a[middle])
			return middle;
		if (x > a[middle])
			left = middle + 1;
		else
			right = middle - 1;
	}
	// 未找到x
	return -1;
}
```

