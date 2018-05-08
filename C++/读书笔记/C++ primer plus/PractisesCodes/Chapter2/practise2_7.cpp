#include<iostream>
using namespace std;

void getTime(int hour, int mins){
	cout << "Time: " << hour << ":" << mins << endl;
}

int main(){
	int hours, mins;
	cout << "Enter the number of hours: ";
	cin >> hours;
	cout << "Enter the number of minutes: ";
	cin >> mins;
	getTime(hours, mins);

	system("pause");
	return 0;
}