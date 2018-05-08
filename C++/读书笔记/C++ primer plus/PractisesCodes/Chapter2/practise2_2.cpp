#include<iostream>

using namespace std;
int longToMa(int);

int main()
{
	int longNum;
	cout << "Enter a nums: ";
	cin >> longNum;
	cout << longNum << " Ma = " << longToMa(longNum) << endl;

	system("pause");
	return 0;
}

int longToMa(int l)
{
	int Ma = 220 * l;
	return Ma;
}