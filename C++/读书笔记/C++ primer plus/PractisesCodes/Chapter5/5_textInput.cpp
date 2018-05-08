#include<iostream>
using namespace std;
// reading chars with a while loop
void readFunction1();
// reading chars with cin.get(ch)
void readFunction2();
// test for EOF
void readFunction3();
// optimilization readFuntion3()
void readFunction4();

int main(){

	//readFunction1();
	//readFunction2();
	//readFunction3();
	readFunction4();

	system("pause");
	return 0;
}

void readFunction1(){
	char ch;
	int count = 0;	// use basic input
	cout << "Enter characters; enter # to quit:\n";
	cin >> ch;
	while (ch != '#'){
		cout << ch;
		count++;
		cin >> ch;
	}
	cout << endl << count << " characters read\n";
}

void readFunction2(){
	char ch;
	int count = 0;	// use basic input
	cout << "Enter characters; enter # to quit:\n";
	cin.get(ch);
	while (ch != '#'){
		cout << ch;
		count++;
		cin.get(ch);
	}
	cout << endl << count << " characters read\n";
}

void readFunction3(){
	char ch;
	int count = 0;	// use basic input
	cin.get(ch);
	while (cin.fail() == false){	// test for EOF
		cout << ch;
		count++;
		cin.get(ch);
	}
	cout << endl << count << " characters read\n";
}

void readFunction4(){
	int ch;
	int count = 0;
	
	while ((ch = cin.get()) != EOF){
		cout.put(char(ch));
		count++;
	}
	cout << endl << count << " characters read\n";
}