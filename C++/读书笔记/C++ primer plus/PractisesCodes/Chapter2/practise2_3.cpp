#include<iostream>

using namespace std;

void print1();
void print2();

int main(){

	for (int i = 0; i < 2; i++){
		print1();
		cout << endl;
	}
	for (int j = 0; j < 2; j++){
		print2();
		cout << endl;
	}

	system("pause");
	return 0;
}

void print1(){
	cout << "Three blind mice";
}

void print2(){
	cout << "See how they run";
}