#include<iostream>
using namespace std;

int main(){
	int days, hours, minutes, seconds;
	long inputSeconds;
	const int transHour = 3600;
	const int trans = 60;

	cout << "Enter the number of seconds: ";
	cin >> inputSeconds;
	cout << inputSeconds;
	days = inputSeconds / (24 * transHour);
	inputSeconds -= days * 24 * transHour;
	hours = inputSeconds / transHour;
	inputSeconds -= hours * transHour;
	minutes = inputSeconds / trans;
	inputSeconds -= minutes * trans;
	seconds = inputSeconds;

	cout << " secondes = " << days << " days, " << hours << " hours, "
		<< minutes << " minutes, " << seconds << " seconds." << endl;
		  
	system("pause");
	return 0;
}