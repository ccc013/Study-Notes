这里包括后缀表达式求值


### 后缀表达式求值

使用标准库中的栈来实现
```
#include<iostream>
#include<stack>

int main(){
	using std::cout;
	using std::endl;
	using std::cin;

	std::stack<int> stack;
	

	// 实现后缀表达式,输入为 "9 3 1 - 3 * + 10 2  /  +"
	char ch;	
	// 用来判断是否连续两个字符都是数字
	int isnum = 0;
	cout << "Please input: ";	
	cin.get(ch);
	while (ch != '\n'){
		if (isdigit(ch)){
			// 数字就进栈
			int item = ch - '0';	// 字符转成数字的方法
			if (isnum){
				int last;
				last = stack.top();
				stack.pop();
				item = last * 10 + item;
			}
			stack.push(item);
			isnum++;
		}
		else if (ch != ' '){
			isnum = 0;
			int a, b,c;
			b = stack.top();
			stack.pop();
			a = stack.top();
			stack.pop();
			switch (ch)
			{
			case '+':
				c = a + b;
				break;
			case '-':
				c = a - b;
				break;
			case '*':
				c = a * b;
				break;
			case '/':
				c = a / b;
				break;
			default:
				cout << "invalid input!\n";
				break;
			}
			stack.push(c);
		}
		else{
			isnum = 0;
		}
		cin.get(ch);
	}
	int s;
	s = stack.top();
	stack.pop();
	cout << "The result is " << s << endl;

	system("pause");
	return 0;
}
```

### 中缀表达式转后缀表达式

