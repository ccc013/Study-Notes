## 《大话数据结构》 第五章 串

#### 1. 串

> 串(string)是由零个或多个字符组成的有限序列，又名叫字符串。
    
串的比较是通过组成串的字符之间的编码来进行的，而字符的编码是指字符在对应字符集中的符号。
    
计算机常用字符是使用标准的`ASCII`编码，由7位二进制数表示一个字符，总共可以表示128个字符。后来又拓展为8位二进制数表示，总共可以表示256个字符。但是256个字符是不能表示完世界上那么多的语言和文字的，所以就有了`Unicode`编码。比较常用的是用16位的二进制数表示一个字符。
    
##### 串的抽象数据类型

```
ADT 串(string)
Data
    串中元素仅由一个字符组成，相邻元素有前驱和后继的关系。
Opration
    StrAssign(T, *chars): 生成一个其值等于字符常量chars的串T；
    StrCopy(T,S): 串S存在，由串S复制得到串T；
    ClearString(S): 若串存在，将串清空；
    StringEmpty(S): 若串为空，返回true，否则返回false；
    StrLength(S): 返回串S的元素个数，即串的长度；
    StrCompare(S,T): 若S>T,返回值>0, 若S=T, 返回0，若S<T, 返回值<0;
    Concact(T, S1, S2): 用T返回由S1和S2连接而成的新串；
    SubString(Sub, S, pos, len): 串S存在， 1<= pos <= StrLength(S), 且0 <= len <= StrLength(S)-pos+1,用Sub返回串S的第pos个字符起长度为len的子串。
    Index(S, T, pos): 串S和T存在，T是非空串，1<= pos <= StrLength(S),若主串S中存在和串T值相同的子串，则返回它在主串S中第pos个字符之后第一次出现的位置，否则返回0。
    Replace(S,T,V): 串S、T和V存在，T是非空串，用V替换主串S中出现的所有与T相等的不重叠的子串。
    StrInsert(S,pos,T): 串S和T存在， 1<= pos <= StrLength(S)+1，在串的第pos个字符之前插入串T。
    StrDelete(S,pos,len): 串S存在， 1<= pos <= StrLength(S)-len+1,。从串S中删除第pos个字符起长度为len的子串。
```

##### 朴素的模式匹配算法
    子串的定位操作通常称做串的模式匹配。

朴素的模式匹配算法实现的代码如下：
```
// 返回子串T在主串S中第pos个字符后的位置，不存在则返回0
int Index(std::string S, std::string T, int pos){
	int i = pos;
	int j = 0;
	while (i <= S.size() && j <= T.size()){
		if (S[i] == T[j]){
			++i;
			++j;
		}
		else{
			i = i - j + 1;
			j = 0;
		}
	}
	if (j > T.size())
		return i - T.size();
	else
		return 0;
}
```
简单的说，就是对主串的每一个字符作为子串开头，与要匹配的字符串进行匹配，对主串做大循环，每个字符开头做T的长度的小循环，直到匹配成功或全部遍历完成为止。

上述算法的问题就是效率不高，当遇到最坏的情况，需要花费的时间就很多。

##### KMP模式匹配算法
  在朴素匹配算法中，主串的i值是会不断地回溯，如主串S=`abcdefgab`,要匹配的子串T=`abcdex`，第一次匹配结束时`i=6`，然后i又会从2开始，一直回到6，但是由于因为子串T中每个字符都不相等，而第一次匹配的时候可以知道两个字符串的前5位都是分别相等的，因此可以直接跳到i=6，j=1来继续进行匹配，这样可以省略了中间的几步；而如果子串T=`abcabx`,主串S=`abcababca`，子串中是有相等的字符，也可以发现两个字符串的前5位是相等的，同样是可以减少一些步骤的。简而言之，主串的i值是可以不需要回溯的。
  
  KMP模式匹配算法就是为了不让主串的i值发生回溯。这个时候需要考虑的就是子串T的j值的变化了。经过研究发现，j值的变化与主串没有什么关系，有关系的是子串T中是否有重复的问题。比如在T=‘++ab++c++ab++x’中，前缀的`ab`与`x`的后缀`ab`是相等的，所以j值是从6变成3。因此，j值的多少取决于当前字符之前的前后缀的相似度。
  
  将T串各个位置的j值的变化定义为一个数组next，那么next的长度就是T串的长度，可以得到如下函数定义：
  
  ![函数定义](https://raw.githubusercontent.com/ccc013/Study-Notes/master/DataStructe%20%26%20Algorithm/images/KMP.png)
    
  上述定义中，根据经验，如果前后缀一个字符相等，k=2，两个字符相等则k=3，所以`n个相等，则k=n+1`.
    
  因此代码实现如下：
  
```
// 通过计算返回子串T的next数组
void get_next(string T, int *next){
	int i, j;
	i = 0;
	j = -1;
	next[0] = -1;
	while (i < T.size()){
		if (j == -1 ||T[i] == T[j]){
			// T[i]表示后缀的单个字符，T[j]表示前缀的单个字符
			++i;
			++j;
			next[i] = j;
		}
		else{
			j = next[j];	// 若字符不相同，则j值回溯
		}
	}
}

// 返回子串T在主串S中第pos个字符后的位置，不存在则返回0
int Index_KMP(std::string S, std::string T, int pos){
	int i = pos;
	int j = 0;
	int next[255] = { 0 };	// 定义一个next数组
	get_next(T, next);
	std::cout << "next array show as :\n";
	for (int k = 0; k < T.size(); k++){
		std::cout << k << ": " << next[k] << std::endl;
	}
	while (i <= S.size() && j <= T.size()){
		if ( j==0 || S[i] == T[j]){
			++i;
			++j;
		}
		else{
			j = next[j];
		}
	}
	if (j > T.size())
		return i - T.size();
	else
		return 0;
}
```
跟书中的代码有所不同，因为使用的是标准库中的`string`类，而书中是自定义的一个`String`类，其第一个空间是存放的是长度，而`string`类则调用其`size()`方法就可以得到长度了，所以每个值都是需要再减一，那么其实next数组的定义中，j是从0开始，然后`next[0]=-1`，然后返回的是该子串在主串中首个字符的位置，该位置减一就是其在主串中的索引值。


##### KMP模式匹配算法的改进

  如果主串S=`aaaabcde`，子串T=`aaaaax`，其next数组值是012345,那么第一次匹配结束时是i=5，j=5的时候，然后j=next[5]=4，但此时主串中是b，子串是a，依然不相同，然后依次根据next数组调整j值，然后就会发现知道j=next[1]=0时，根据算法让i和j各自加一，这个过程其实都是可以省略的，所以可以使用首位next[1]的值去取代与它相等的字符后续next[j]的值，所以改进的算法的代码实现如下：
  
```
void get_nextval(string T, int *nextval){
	int i, j;
	i = 0;
	j = -1;
	nextval[0] = -1;
	while (i < T.size()){
		if (j == -1 ||T[i] == T[j]){
			// T[i]表示后缀的单个字符，T[j]表示前缀的单个字符
			++i;
			++j;
			// next[i] = j;
			
			// 增加的内容
			if (T[i] != T[j]){
				// 当前字符与前缀字符不相等,则nextval在i位置的值是当前的j值
				nextval[i] = j;
			}
			else{
				// 如果与前缀字符相等，则将前缀字符的nextval值赋值给nextval在i位置的值
				nextval[i] = nextval[j];
			}
		}
		else{
			j = nextval[j];	// 若字符不相同，则j值回溯
		}
	}
}
```

---
#### 2. 总结
  关于串的内容，其实刚开始看有关KMP的匹配算法的时候还是不能太完全理解，看了第二遍后才完成这篇笔记，当然现在可能也就是理解多一点，但感觉好像还是没有完全能理解，可能还是需要实践，多做题吧，这个匹配算法可以加深对`string`类的`Index()`方法实现的理解。

