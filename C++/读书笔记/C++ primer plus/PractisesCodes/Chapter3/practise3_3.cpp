#include<iostream>
using namespace std;

int main(){
	int degrees, minutes, seconds;
	const float trans = 60.0;
	double latitude;

	cout << "Enter a latitude in degrees, minutes, and seconds:" << endl;
	cout << "First, enter the degrees: ";
	cin >> degrees;
	cout << "Next, enter the minutes of arc: ";
	cin >> minutes;
	cout << "Finally, enter the seconds of arc: ";
	cin >> seconds;

	latitude = degrees + minutes / trans + seconds / (trans * trans);

	cout << degrees << " degrees, " << minutes << " minutes, "
		<< seconds << " seconds = " << latitude << " degrees" << endl;

	system("pause");
	return 0;
}