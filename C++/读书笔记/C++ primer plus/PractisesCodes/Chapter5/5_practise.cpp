#include<iostream>
#include<array>
#include<string>
#include<cstring>
using namespace std;
// practise 5_1
void function5_1();
// practise 5_2
void function5_2();
// practise 5_3
void function5_3();
// practise 5_4
void function5_4();
// practise 5_5, 5_6
void function5_5();
// practise 5_7
const int Size = 20;
void function5_7();
// practise 5_8 5_9
void function5_8();
// practise 5_10
void function5_10();

struct car
{
	char names[Size];
	int years;
};

int main(){

	function5_10();


	system("pause");
	return 0;
}

void function5_1(){
	int start, end;
	cout << "Enter two numbers(the first must less than the second one):\n ";
	cin >> start >> end;
	int sum = 0;
	for (int i = start; i <= end; i++){
		sum += i;
	}
	cout << "The sum between " << start
		<< " and " << end << " is " << sum << endl;
}

void function5_2(){
	const int ArSize = 101;
	array<long double, ArSize> factorials;
	factorials[1] = factorials[0] = 1;
	for (int i = 2; i < ArSize; i++){
		factorials[i] = i * factorials[i - 1];
	}
	for (int i = 0; i < ArSize; i++){
		cout << i << "! = " << factorials[i] << endl;
	}
}

void function5_3(){
	int num;
	int sum = 0;
	cout << "Enter a number(0 to quit):\n";
	cin >> num;
	while (num != 0){
		sum += num;
		cout << "The sum now is " << sum << endl;
		cin >> num;
	}
	cout << "The final sum is " << sum << endl;
}

void function5_4(){
	double sum_Daphne = 100;
	double sum_Cleo = 100;
	int years = 0;

	for (; sum_Cleo <= sum_Daphne; years++)
	{
		sum_Daphne += 10;
		sum_Cleo += sum_Cleo * 0.05;
	}
	cout << "After " << years << " years, "
		<< " Cleo has " << sum_Cleo << ", Daphne has "
		<< sum_Daphne << ".\n";

}

void function5_5(){
	const int Months = 12;
	const int Years = 3;
	int nums;
	char * books[Months] = {
		"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
	};
	//int bookNums[Months];
	int bookNums[Years][Months];
	int sum[Years] = { 0 };

	cout << "Input the numbers of books every month:\n";
	for (int j = 0; j < Years; j++){
		cout << "The year " << (j + 1) << ":\n";
		for (int i = 0; i < Months; i++){
				cin >> nums;
				cout << books[i] << ": " << nums << endl;
				bookNums[j][i] = nums;
				sum[j] += nums;
				
			}
	}
	int sums = 0;
	for (int s : sum){
		sums += s;
	}
	cout << "The first year sell " << sum[0] << " books.\n";
	cout << "The second year sell " << sum[1] << " books.\n";
	cout << "The third year sell " << sum[2] << " books.\n";
	cout << "3 years totally sell " << sums << " books.\n";

}

void function5_7(){
	
	int cars;
	cout << "How many cars do you wish to catalog? ";
	(cin >> cars).get();

	car * pc = new car[cars];
	for (int i = 0; i < cars; i++){
		cout << "Car #" << i+1 << ": \n";
		cout << "Please enter the make: ";
		//getline(cin, pc[i].names);
		cin.get(pc[i].names, Size).get();
		//cin >> pc[i].names;
		cout << "Please enter the year made: ";
		cin >> pc[i].years;
		cin.get();
	}
	cout << "Here is your collection:\n";
	cout << pc[0].years << " " << pc[0].names << endl;
	cout << pc[1].years << " " << pc[1].names << endl;
}

void function5_8(){
	//char words[20];
	string words;
	int sum = 0;

	cout << "Enter words (to stop, type the word done):\n";
	cin >> words;
	//while (strcmp(words, "done"))
	while(words != "done"){
		cin >> words;
		sum++;
	}

	cout << "You entered a total of " << sum << " words.\n";
}

void function5_10(){
	int lines;
	cout << "Enter number of rows: ";
	cin >> lines;
	
	for (int line = 0; line < lines; line++){
		for (int i = 0; i < lines - line; i++){
			cout << ".";
		}
		for (int i = 0; i <= line; i++){
			cout << "*";
		}
		cout << endl;
	}
}