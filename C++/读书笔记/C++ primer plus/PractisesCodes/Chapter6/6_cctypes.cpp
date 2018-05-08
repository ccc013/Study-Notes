#include<iostream>
#include<cctype>
using namespace std;

int main(){
	cout << "Enter text for analysis, and type @"
		" to terminate input.\n";
	char ch;
	int whitespace = 0;
	int digits = 0;
	int chars = 0;
	int punct = 0;
	int others = 0;

	cin.get(ch);
	while (ch != '@'){
		if (isalpha(ch))
			++chars;
		else if (isspace(ch))
			++whitespace;
		else if (isdigit(ch))
			++digits;
		else if (ispunct(ch))
			++punct;
		else
			++others;
		cin.get(ch);
	}
	cout << chars << " letters, "
		<< whitespace << " whitespaces, "
		<< digits << " digits, "
		<< punct << " punctuations, "
		<< others << " others.\n";

	system("pause");
	return 0;
}