#include<iostream>
using namespace std;

int main(){
	int inches, feets;
	double bmiValues, pounds, meters, kilos;
	const float trans = 12.0;
	const double inToMeter = 0.0254;
	const double poundToKilo = 2.2;

	cout << "Enter your height in inches,feets: ";
	cout << "\nEnter the feets: ";
	cin >> feets;
	cout << "Enter the inches: ";
	cin >> inches;
	cout << "Enter the weight in pounds: ";
	cin >> pounds;

	inches = inches + feets * trans;

	cout << "You are " << inches << " inches heigh.	" 
		 << "\nYou are " << pounds << " pounds." << endl;

	meters = inches * inToMeter;
	kilos = pounds / poundToKilo;
	bmiValues = kilos / (meters * meters);
	cout << "Your BMI is " << bmiValues << endl;
	system("pause");
	return 0;
}