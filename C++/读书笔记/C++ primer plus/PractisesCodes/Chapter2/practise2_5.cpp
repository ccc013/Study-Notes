#include<iostream>

using namespace std;

double celsiusToFahrenheit(int cels){
	int Fahrenheit = 1.8 * cels + 32.0;
	return Fahrenheit;
}

int main(){

	int cels;
	cout << "Please enter a Celsius value: ";
	cin >> cels;
	cout << cels << " degrees Celsius is " << celsiusToFahrenheit(cels)
		<< " degrees Fahrenheit." << endl;

	system("pause");
	return 0;
}

