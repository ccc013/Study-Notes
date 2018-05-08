#include<iostream>

using namespace std;

// using the if statement to count the spaces and chars from input

int main(){
	char ch;
	int spaces = 0;
	int total = 0;
	cin.get(ch);
	while (ch != '.'){
		if (ch == ' ')
		{
			++spaces;
		}
		++total;
		cin.get(ch);
	}
	cout << spaces << " spaces, " << total;
	cout << " characters total in sentence\n";

	/*cout << "Enter a char:\n";
	cin.get();
	cin.get(ch);
	char ch2 = ch; 
	cout << ch << endl;
	cout << "++ch: " << ++ch << endl;
	cout << "ch + 1 : " << ch2 + 1 <<endl;*/

	system("pause");
	return 0;
}