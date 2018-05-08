#include<iostream>
#include<string>	// make string class available
#include<cstring>
using namespace std;

int main(){
	char charr1[20];
	char charr2[20] = "jaguar";
	string str1;
	string str2 = "panther";
	string str;

	cout << "Length of string in charr1 before input: "
		<< strlen(charr1) << endl;
	cout << "Length of string in str before input: "
		 << str.size() << endl;
	cout << "Enter a line of text:\n";
	cin.getline(charr1, 20);	// indicate maximum length
	cout << "You entered: " << charr1 << endl;
	cout << "Enter another line of text:\n";
	getline(cin, str);	// cin now an argument; no length specifier
	cout << "You entered: " << str << endl;
	cout << "Length of string in charr1 after input:"
		<< strlen(charr1) << endl;
	cout << "Length of string in str after input:"
		<< str.size() << endl;

	// cout raw string
	cout << R"(Jim "King" Tutt uses "\n" instead of endl.)" << '\n';

	/*cout << "Enter a kind of feline: ";
	cin.get(charr1, 20).get();
	cout << "Enter another kind of feline: ";
	cin >> str1;
	cout << "Here are some felines:\n";
	cout << charr1 << " " << charr2 << " "
	<< str1 << " " << str2
		<< endl;
	cout << "The third letter in " << charr2 << " is "
		<< charr2[2] << endl;
	cout << "The third letter in " << str2 << " is "
		<< str2[2] << endl;*/

	// assignment for string objects and character arrays
	str1 = str2;
	//strcpy(charr1, charr2);  // copy charr2 to charr1

	// appending for string objects and character arrays
	str1 += " paste";
	//strcat(charr1, " juice");

	// finding the length of a string object and a C-style string
	int len1 = str1.size();
	int lens = str1.length();
	int len2 = strlen(charr1);

	cout << "The string " << str1 << " contains "
		<< len1 << " characters." << endl;
	/*cout << "The string " << charr1 << " contains "
		<< len2 << " characters." << endl;*/
	cout << "lens = str1.length() = " << lens << endl;

	system("pause");
	return 0;
}