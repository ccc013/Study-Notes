#include<iostream>
using namespace std;

int main(){
	long long popurWor, popurUs;
	double percent;
	const double per = 100.0;

	cout << "Enter the world's population: ";
	cin >> popurWor;
	cout << "Enter the population of the US: ";
	cin >> popurUs;

	percent = (popurUs * 1.0) / popurWor * per;	
	cout << "The population of the US is " << percent << "\% of the world population." << endl;

	system("pause");
	return 0;
}