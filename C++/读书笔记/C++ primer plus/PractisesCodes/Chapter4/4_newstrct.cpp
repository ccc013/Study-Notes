#include<iostream>
using namespace std;

struct inflatable
{
	char name[20];
	float volume;
	double price;
};

int main(){
	inflatable * ps = new inflatable;  // allot memory for structure
	cout << "Enter name of inflatable item: ";
	cin.get(ps->name, 20);			// method 1 for member access
	cout << "Enter volume in cubic feet: ";
	cin >> (*ps).volume;
	cout << "Enter price: $";
	cin >> ps->price;

	cout << "Name: " << (*ps).name << endl;
	cout << "Volume: " << ps->volume << " cubic feet" << endl;
	cout << "Price: $" << ps->price << endl;
	delete ps;
	system("pause");
	return 0;
}