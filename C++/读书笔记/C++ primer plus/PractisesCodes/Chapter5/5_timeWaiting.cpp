#include<iostream>
#include<ctime>		// describes clock() function, clock_t type

//	using clock() in a time-delay loop
using namespace std;

int main(){
	cout << "Enter the delay time, in seconds: ";
	float secs;
	cin >> secs;
	clock_t delay = secs * CLOCKS_PER_SEC;	// convert to clock ticks;
	cout << "staring\a\n";
	clock_t start = clock();
	while (clock() - start < delay){

	}
	cout << "done \a\n";
	system("pause");
	return 0;
}