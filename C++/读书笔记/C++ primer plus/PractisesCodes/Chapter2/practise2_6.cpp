#include<iostream>

using namespace std;

double lightToAstron(double light){
	double astron = light * 63240;
	return astron;
}

int main(){

	double light;
	cout << "Enter the number of light years: ";
	cin >> light;
	cout << light << " light years = " << lightToAstron(light)
		<< " astronomical units." << endl;

	system("pause");
	return 0;
}

