中缀表达式转后缀表达式


注意输入必须是英文字符，特别是对于括号的输入，如果是中文字符会出错。
```
#include<iostream>
#include<stack>

int main(){
	using std::cout;
	using std::endl;
	using std::cin;

	std::stack<char> stack;
	

	// 实现中缀表达式转后缀表达式,例如输入为"9+(3-1)*3+10/2",输出为 "9 3 1 - 3 * + 10 2  /  +"
	char ch;	
	char output[50];
	int i = 0;
	// 用来判断是否连续两个字符都是数字
	int isnum = 0;
	cout << "Please input: ";	
	cin.get(ch);
	while (ch != '\n'){
		if (isdigit(ch)){
			// 是数字就输出			
			if (isnum){
				i--; 
			}
			output[i++] = ch;
			output[i++] = ' ';
			isnum++;
		}
		else if (ch != ' '){
			// 是符号就进栈
			isnum = 0;
			char top;
			if (stack.empty()){
				stack.push(ch);
			}
			else{
				top = stack.top();
				if ((ch == '+' || ch == '-') && (top == '*' || top == '/')){
					// 乘除优先级高于加减,栈顶元素符号优先级更高的时候，输出
					while (!stack.empty()){
						top = stack.top();
						stack.pop();
						output[i++] = top;
						output[i++] = ' ';
					}
					// 将当前的符号进栈
					stack.push(ch);
				}
				else if (ch == ')'){
					while (top != '('){
						top = stack.top();
						stack.pop();
						if (top == '(' || top == ')')
							continue;
						output[i++] = top;
						output[i++] = ' ';
					}
				}else{
				stack.push(ch);
			}
		}	
		}
		else{
			isnum = 0;
		}
		cin.get(ch);
	}
	while (!stack.empty()){
		char top = stack.top();
		output[i++] = top;
		output[i++] = ' ';
		stack.pop();
	}
	output[i] = '\0';
	cout << "result: ";
	for (int j = 0; j < i; j++){
		cout << output[j];
	}
	cout << endl;

	system("pause");
	return 0;
}
```
