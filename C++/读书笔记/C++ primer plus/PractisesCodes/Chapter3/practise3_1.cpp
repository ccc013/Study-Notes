#include<iostream>
using namespace std;

int main(){
	int inches;
	const float trans = 12.0;

	cout << "Enter your height(inches)__";
	cin >> inches;
	cout << "You are " << inches << " inches heigh, "
		<< "that is " << inches / trans << " feets." << endl;

	system("pause");
	return 0;
}