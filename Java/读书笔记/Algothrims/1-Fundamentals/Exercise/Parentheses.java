package chapter1.Exercise;

import StandardLibrary.DataStructure.Stack;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

/**
 * Created by cai on 2016/5/21 11:45.
 * Email: 429546420@qq.com
 */
public class Parentheses {

    /*
    *从标准输入中读取一个文本流并使用栈判定其中的括号是否配对完整
    *例如，对于[()]{}{[()()]()}程序应返回true，对于[(])则打印false
    */

    /**
     * 对输入的内容判断其中的括号是否匹配
     *
     * @param test
     * @return
     */
    public static boolean isBalance(String test) {
        Stack<Character> s = new Stack<Character>();
        int N = test.length();
        for (int i = 0; i < N; i++) {
            char ch = test.charAt(i);

            if (ch == '(' || ch == '[' || ch == '{') {
                s.push(ch);
            } else if (ch == ')' && (s.pop() != '(' || s.isEmpty())) {
                return false;
            } else if (ch == ']' && (s.pop() != '[' || s.isEmpty())) {
                return false;
            } else if (ch == '}' && (s.pop() != '{' || s.isEmpty())) {
                return false;
            }
        }
        return s.isEmpty();
    }

    /**
     * Test
     *
     * @param args
     */
    public static void main(String[] args) throws IOException {
        Stack<String> s = new Stack<String>();
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        String str = null;

        System.out.println("please input(exit to quit): ");
        ArrayList<String> strs = new ArrayList<>();
        while (!(str = bufferedReader.readLine()).equals("exit")) {
            strs.add(str);
        }
        bufferedReader.close();
        int i = 0;
        for (String string : strs) {
            i++;
            System.out.println("Line " + i + ": " + isBalance(string));
        }
    }
}
