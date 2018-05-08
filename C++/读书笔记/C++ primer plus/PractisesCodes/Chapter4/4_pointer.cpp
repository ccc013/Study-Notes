#include<iostream>

using namespace std;

int main(){
	int updates = 6;
	int * p_updates;   // declare a pointer to an int
	p_updates = &updates; // assign address of int to pointer

	// use new 
	int nights = 1001;
	int* pt = new int;
	*pt = 1001;
	double * pd = new double;
	*pd = 10001201.01;

	cout << "nights value = ";
	cout << nights << ": location " << &nights << endl;
	cout << "int ";
	cout << "value = " << *pt << ": location = " << pt << endl;
	cout << "size of pt = " << sizeof(pt);
	cout << ": size of *pt = " << sizeof(*pt) << endl;

	cout << "double ";
	cout << "value = " << *pd << ": location = " << pd << endl;
	cout << "size of pd = " << sizeof(pd);
	cout << ": size of pd = " << sizeof(*pd) << endl;

	// express values two ways
	cout << "Values: updates = " << updates;
	cout << ", *p_updates = " << *p_updates << endl;

	// express address two ways
	cout << "Addresses: &updates = " << &updates;
	cout << ", p_updates = " << p_updates << endl;

	// use pointer to change value
	*p_updates = *p_updates + 1;
	cout << "Now updates = " << updates << endl;

	system("pause");
	return 0;
}