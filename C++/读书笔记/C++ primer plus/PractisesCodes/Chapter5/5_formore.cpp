#include<iostream>
using namespace std;

// more looping with for
const int ArSize = 16;

int main(){
	/*long long factorials[ArSize];
	factorials[1] = factorials[0] = 1LL;
	for (int i = 2; i < ArSize; i++){
		factorials[i] = i * factorials[i - 1];
	}
	for (int i = 0; i < ArSize; i++){
		cout << i << "! = " << factorials[i] << endl;
	}*/

	double arr[3] = { 1.2, 23.4, 3.4 };
	double * pd = arr;
	double x = *++pd;
	double x2 = *pd++;
	cout << x << endl;
	cout << x2 << endl;
	cout << *pd << endl;

	system("pause");
	return 0;
}