#include<iostream>
#include<cctype>
#include<string>
#include<fstream>
using namespace std;

// practse6_1
void function6_1();
// practise6_2
void function6_2();
//
void function6_3();
void function6_5();
void function6_6();
void function6_7();
void function6_8();

// practise 6_6
const int strSize = 20;
struct donations
{
	char name[strSize];
	double donation;
};


int main(){

	function6_8();

	system("pause");
	return 0;
}

// 回显键盘输入（除数字外），并转化大小写，遇到@退出
void function6_1(){
	char ch;

	cout << "Enter some characters(@ to quit):\n";
	while (cin.get(ch) && ch != '@'){
		if (!isdigit(ch)){
			ch = (isupper(ch)) ? tolower(ch) : toupper(ch);
			cout << ch;
		}
	}
	cout << endl;

}

// 输入最多10个数字到double数组，并计算平均值及大于平均值的个数
void function6_2(){
	const int Size = 10;
	double donations[Size];

	double sum = 0.0;
	double average;
	int i = 0;

	cout << "First value: ";
	// 非数字输入结束输入
	while (cin >> donations[i] ){
		sum += donations[i];
		++i;
		/*if (i < Size){
			cout << "Next value: ";
			cin >> temp;
		}*/
		cout << "Next value: ";
	}
	cout << endl;

	if (i == 0)
		cout << "No data store in donations.\n";
	else {
		average = sum / i;
		int count = 0;
		for (int j = 0; j < i; ++j){
			if (donations[j] > average)
				++count;
		}
		cout << "You have totally input " << i << " numbers, "
			<< "the average is " << average
			<< ", " << count << " numbers are bigger than average.\n";
		
	}
}

// 显示一个菜单，使用switch语句
void function6_3(){
	char ch;

	cout << "Please enter one of the following choices: \n"
		<< "c) carnivore           p) pianist\n"
		<< "t) tree                g) game\n";
	while (cin.get(ch) && ch != 'q'){
		cin.get();
		switch (ch)
		{
		case 'c':cout << "carnivore\n";
			break;
		case 'p':cout << "He is the best pianist in this country.\n";
			break;
		case't':cout << "A maple is a tree.\n";
			break;
		case'g':cout << "This is just a game,don't be serious,guy!\n";
			break;
		default:cout << "Please enter a c, p, t, or g: ";
			break;
		}
	}
	
}

// 计算工资税收
void function6_5(){
	int earns;
	const double fax1 = 0.1;
	const double fax2 = 0.15;
	const double fax3 = 0.2;
	double tax = 0.0;

	cout << "Enter the earnings:\n";
	while (cin >> earns && earns >= 0){
		if (earns <= 5000)
			cout << "The tax is 0 tvarp.\n";
		else if (earns <= 15000){
			tax = (earns - 5000) * fax1;
			cout << "The tax is " << tax << " tvarps.\n";
		}
		else if (earns <= 35000){
			tax = 10000 * fax1 + (earns - 15000) * fax2;
			cout << "The tax is " << tax << " tvarps.\n";
		}
		else{
			tax = 10000 * fax1 + 20000 * fax2 + (earns - 35000) * fax3;
			cout << "The tax is " << tax << " tvarps.\n";
		}
		cout << "Enter the earnings:\n";
	}

}

// 纪录捐助人名字及捐款，按捐款多少显示两种列表
void function6_6(){
	int nums;
	int count_1 = 0, count_0 = 0;
	cout << "Enter the numbers of donators:\n";
	cin >> nums;
	cin.get();
	donations * pd = new donations[nums];
	cout << "Enter the first donator's name and donation:\n";
	int count = 0;
	while (count < nums){
		cin.get(pd[count].name, strSize).get();
		(cin >> pd[count].donation).get();
		if (pd[count].donation > 10000)
			++count_1;
		else
			++count_0;
		++count;
		if (count < nums)
			cout << "Enter next donator's name and donations:\n";
	}
	cout << "\nFinish input.\n";
	cout << "\n\tGrand Patrons\n";
	if (count_1 > 0){
		for (int j = 0; j < nums; ++j){
			if (pd[j].donation > 10000)
				cout << pd[j].name << "--" << pd[j].donation << endl;
		}
	}else
		cout << "none" << endl;

	cout << "\n\tPartrons\n";
	if (count_0 > 0){
		for (int j = 0; j < nums; ++j){
			if (pd[j].donation <= 10000)
				cout << pd[j].name << endl;
		}
	}
	else
		cout << "none" << endl;

	delete[] pd;
}

// 读取单词并统计开头字母分别是元音还是辅音的个数
void function6_7(){
	char ch;
	string word;
	int yuan_count = 0;
	int fu_count = 0;
	int others = 0;
	
	cout << "Enter words (q to quit):\n";
	cin >> word;
	
	while (word[0] != 'q' || word.size() != 1){		// 满足单词首字母不是q，且并非只是一个字母
		ch = word[0];
		if (isalpha(ch)){
			switch (ch)
			{
			case'a':++yuan_count;
				break;
			case'o':++yuan_count;
				break;
			case'u':++yuan_count;
				break;
			case'e':++yuan_count;
				break;
			case'i':++yuan_count;
				break;
			default:++fu_count;
				break;
			}
		}
		else{
			++others;	
		}
		cin >> word;
	}
	cout << yuan_count << " words beginning with vowels\n"
		<< fu_count << " words beginning with consonants\n"
		<< others << " others\n";

}

void function6_8(){
	string filename = "prac.txt";
	char name[20];
	int age;
	double height, weight;

	ofstream outFile;
	outFile.open(filename);
	cout << "Enter your name:\n";
	cin.get(name, 20).get();
	cout << "Enter your age:\n";
	(cin >> age).get();
	cout << "Enter your height:\n";
	(cin >> height);
	cout << "Enter your weight:\n";
	cin >> weight;

	// write to File
	outFile << fixed;
	outFile.precision(2);
	outFile.setf(ios_base::showpoint);
	outFile << "Name: " << name << endl;
	outFile << "Age: " << age << endl;
	outFile << "Height: " << height << endl;
	outFile << "Weight:	" << weight << endl;
	outFile.close();

	// read File
	ifstream inFile;
	inFile.open(filename);
	if (!inFile.is_open()){
		cout << "Could not open the file " << filename << endl;
		cout << "Program terminating.\n";
		exit(EXIT_FAILURE);
	}

	int count = 0;
	char ch;
	string content;
	inFile >> ch;
	while (inFile.good()){
		++count;
		inFile >> ch;
	}
	if (inFile.eof()){
		cout << "End of file reached.\n";
	}
	else if (inFile.fail()){
		cout << "Input terminated by data mismatch.\n";
	}
	else{
		cout << "Input terminated by unknown reason.\n";
	}
	cout << "There are " << count << " characters in the file "
		<< filename << endl;
}