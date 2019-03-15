package chapter1.Exercise;

/**
 * Created by cai on 2016/5/18 21:31.
 * Email: 429546420@qq.com
 */
public class Exercises_1_2 {

    /**
     * 判断两个字符串是否相互为循环变位(circular rotation)
     * 如果字符串s中的字符循环移动任意位置之后能够得到字符串t，则称s是t的回环变位
     *
     * @param s
     * @param t
     * @return
     */
    public static boolean isCircularRotation(String s, String t) {
        return ((s.length() == t.length()) && (s.concat(s).indexOf(t) >= 0));
    }

    /**
     * 实现反转字符串s的功能
     *
     * @param s
     * @return
     */
    public static String mystery(String s) {
        int N = s.length();
        if (N <= 1)
            return s;
        String a = s.substring(0, N / 2);
        String b = s.substring(N / 2, N);
        return mystery(b) + mystery(a);
    }

    /**
     * 判断字符串是否是一条回文
     *
     * @param s
     * @return
     */
    public static boolean isPalindrome(String s) {
        int N = s.length();
        for (int i = 0; i < N / 2; i++)
            if (s.charAt(i) != s.charAt(N - 1 - i))
                return false;
        return true;
    }

    public static void main(String[] args) {
        String s = "ACTGACG";
        String t = "TGACGAC";

        System.out.println("Is s is t's circular rotation? " + isCircularRotation(s, t));
        System.out.println(mystery("Hello"));
        System.out.println(isPalindrome("ACBC"));
    }
}
