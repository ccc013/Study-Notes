#include<iostream>
#include<string>
#include<array>
using namespace std;
// practise 4-1 function
void function4_1();
// practise 4-2 function
void function4_2();
// practise 4-4 function
void function4_4();
// practise 4-5 and 4-6, 4-9
void function4_5();
// practise 4-7, 4-8
void function4_7();
void function4_8();
// parctise 4-10
void function4_10();

struct CandyBar	 // struct used in function4_5
{
	string band;
	double weight;
	int calories;
};

struct Pizzas	// struct for function4_7
{
	string name;
	double diameter;
	double weight;
};

int main(){
	
	//function4_1();
	//function4_2();
	//function4_4();
	//function4_5();
	//function4_7();
	//function4_8();
	function4_10();

	system("pause");
	return 0;
}

void function4_1(){
	const int Size = 20;
	char firstName[20];
	char lastName[20];
	char grade;
	int age;

	cout << "What is your first name? ";
	cin.get(firstName, Size).get();
	cout << "What is your last name? ";
	cin.get(lastName, Size).get();
	cout << "What letter grade do you deserve? ";
	cin >> grade;
	cout << "What is your age? ";
	cin >> age;

	cout << "Name: " << lastName << ", " << firstName << endl;
	cout << "Grade: ";
	cout.put(grade + 1); // cout char not int
	cout << endl;
	cout << "Age: " << age << endl;
}

void function4_2(){
	string name;
	string dessert;

	cout << "Enter your name:\n";
	getline(cin, name);
	cout << "Enter your favorite dessert:\n";
	// cin >> dessert  // 只能接受一个单词，遇到空白就结束了
	getline(cin, dessert);
	cout << "I have some delicious " << dessert;
	cout << " for you, " << name << endl;
}

void function4_4(){
	string fName;
	string lName;

	cout << "Enter your first name: ";
	getline(cin, fName);
	cout << "Enter your last name: ";
	getline(cin, lName);
	cout << "Here's the information in a single string: "
		<< lName + ", " + fName << endl;
}

void function4_5(){
	CandyBar snack =
	{
		"Mocha Munch",
		2.3,
		350
	};

	/*CandyBar sBars[3] =
	{
		{ "chocalate cake", 1.5, 250 },
		{ "chocalate mousse", 2.5, 300 },
		{ "chocalate", 3.5, 450 }
	};*/
	CandyBar * sBars = new CandyBar [3];
	sBars[0] = { "chocalate cake", 1.5, 250 };
	sBars[1] = { "chocalate mousse", 2.5, 300 };
	*(sBars + 2) = { "chocalate", 3.5, 450 };

	cout << "snack Band: " << snack.band << endl;
	cout << "snack weight: " << snack.weight << endl;
	cout << "snack calories: " << snack.calories << endl;
	for (int i = 0; i < 3; i++){
		cout << i << ": band = " << sBars[i].band << ", weight = "
			<< sBars[i].weight << ", calories = " << sBars[i].calories << endl;
	}

}

void function4_7(){
	Pizzas pizza;
	cout << "Enter pizza's name: ";
	getline(cin, pizza.name);
	cout << "Enter the diameter: ";
	cin >> pizza.diameter;
	cout << "Enter the weight: ";
	cin >> pizza.weight;
	cout << "The pizza's name is " << pizza.name << ", "
		<< "diameter is " << pizza.diameter
		<< ", weight is " << pizza.weight << ".\n";
}

void function4_8(){
	Pizzas * pizza1 = new Pizzas;
	cout << "Enter pizza's name: ";
	getline(cin, pizza1->name);
	cout << "Enter the diameter: ";
	cin >> (*pizza1).diameter;
	cout << "Enter the weight: ";
	cin >> pizza1->weight;
	cout << "The pizza's name is " << pizza1->name << ", "
		<< "diameter is " << pizza1->diameter
		<< ", weight is " << pizza1->weight << ".\n";
}

void function4_10(){
	array<double, 3> grades;

	cout << "Enter the first time grade in 40 meters running(minutes): ";
	cin >> grades[0];
	cout << "Enter the second time grade in 40 meters running(minutes): ";
	cin >> grades[1];
	cout << "Enter the third time grade in 40 meters running(minutes): ";
	cin >> grades[2];

	cout << "You have input " << grades.size() << " times grades." << endl;
	cout << "The average grade is " << (grades[0] + grades[1] + grades[2]) / 3.0 << " minutes.\n";
}