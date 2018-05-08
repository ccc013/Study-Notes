#include<iostream>

using namespace std;

int main()
{
	cout << "Enter your name: ";
	char name[10], address[20];
	cin >> name;
	cout << "Enter your address: ";
	cin >> address;

	cout << "ok! So you are " << name
		<< " your address is " << address << endl;

	system("pause");
	return 0;
}