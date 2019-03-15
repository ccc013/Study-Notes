阅读《算法》中第一章第三节有关于栈部分内容所做的笔记。

##### 泛型定容栈
```
public class FixedCapacityStackOfStrings<Item> {
    private Item[] a; // stack entries
    private int N;

    public FixedCapacityStackOfStrings(int cap) {
        a = (Item[]) new Object[cap];
    }

    public boolean isEmpty() {
        return N == 0;
    }

    public int size() {
        return N;
    }

    public boolean isFull() {
        return N == a.length;
    }

    public void push(Item item) {
        
        a[N++] = item;
    }

    public Item pop() {
        return a[--N];
    }

    // Test
    public static void main(String[] args) throws Exception {
        FixedCapacityStackOfStrings<String> s;
        s = new FixedCapacityStackOfStrings<String>(100);
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        String str = null;

        System.out.println("please input(exit to quit): ");
        ArrayList<String> strs = new ArrayList<>();
        while (!(str = bufferedReader.readLine()).equals("exit")) {
            strs.add(str);
        }

        for (String string : strs) {
            String[] subString = string.split(" ");
            for (String st : subString) {
                String item = st;
                if (!item.equals("-"))
                    s.push(item);
                else if (!s.isEmpty())
                    System.out.print(s.pop() + " ");
            }
        }


        System.out.println("(" + s.size() + " left on stack)");

    }
}

```
---
##### 下压栈(能够动态调整数组大小的实现)

```
// 算法1.1 下压(LIFO)栈(能够动态调整数组大小的实现)
public class ResizingArrayStack<Item> implements Iterable<Item> {
    private Item[] a = (Item[]) new Object[1];  // stack element
    private int N = 0;                          // elements numbers

    public boolean isEmpty() {
        return N == 0;
    }

    public int size() {
        return N;
    }

    public boolean isFull() {
        return N == a.length;
    }

    /**
     * create a larger array to get a more capacity
     *
     * @param max
     */
    private void resize(int max) {
        Item[] temp = (Item[]) new Object[max];
        for (int i = 0; i < N; i++)
            temp[i] = a[i];
        a = temp;
    }

    public void push(Item item) {
        if (isFull())
            resize(2 * a.length);
        a[N++] = item;
    }

    public Item pop() {
        Item item = a[--N];
        a[N] = null;    // 避免对象游离
        if (N > 0 && N == a.length / 4)
            resize(a.length / 2);
        return item;
    }

    public Iterator<Item> iterator() {
        return new ReverseArrayIterator();
    }

    private class ReverseArrayIterator implements Iterator<Item> {
        // 支持后进先出的迭代
        private int i = N;

        @Override
        public boolean hasNext() {
            return i > 0;
        }

        @Override
        public Item next() {
            return a[--i];
        }

        @Override
        public void remove() {

        }
    }
    
    // Test
    public static void main(String[] args) throws IOException{
        ResizingArrayStack<String> s = new ResizingArrayStack<String>();
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        String str = null;

        System.out.println("please input(exit to quit): ");
        ArrayList<String> strs = new ArrayList<>();
        while (!(str = bufferedReader.readLine()).equals("exit")) {
            strs.add(str);
        }

        for (String string : strs) {
            String[] subString = string.split(" ");
            for (String st : subString) {
                String item = st;
                if (!item.equals("-"))
                    s.push(item);
                else if (!s.isEmpty())
                    System.out.print(s.pop() + " ");
            }
        }
        bufferedReader.close();

        System.out.println("(" + s.size() + " left on stack)");
        for(String elem: s){
            System.out.print(elem+", ");
        }
        System.out.println("\nthat's all.");
    }
}

```


这里设置了一个可以调整数组大小的函数，在进行`push()`和`pop()`的时候都可以随时调整数组的大小。

**对象游离**

  Java的垃圾收集策略是回收所有无法被访问的对象的内存。而在上例中，使用`pop()`的实现中，被弹出的元素的引用仍然存在于数组中，这个元素实际上已经是一个**孤儿**——它永远也不会再被访问了，但Java的GC没法知道这一点，除非该引用被覆盖。即使用例已经不再需要这个元素了，数组中的引用仍然可以让它继续存在。这种情况(保存一个不需要对象的引用)称为**游离**。在这里，避免对象游离很容易，只需要将被弹出的数组元素的值设为null即可，这将覆盖无用的引用并使系统可以在用例使用完被弹出的元素后回收它的内存。
  
**迭代**

> 集合类数据类型的基本操作之一就是能够使用Java的`foreach`语句通过**迭代**遍历并处理集合中的每个元素。

这里的`foreach`语句只是while语句的一种简写方式，其等价于下面的代码

```
Iterator<String> i = collection.iterator();
while(i.haxNext()){
    String s = i.next();
    System.out.println(s);
}
```
这个例子表示任意可迭代的集合数据类型都需要实现的东西：
- 必须实现一个`iterator()`方法并返回一个`Iterator`对象；
- `Iterator`类必须包含两个方法:`haxNext()`(返回一个布尔值)和`next()`(返回集合中的一个泛型元素)。

所以第一步就是继承一个接口`Iterable`，也就是添加`implements Iterable<Item>`。

然后就是在类中添加一个方法`iterator()`方法并返回一个迭代器`Iterator<Item>`,而迭代器都是泛型的，而在本例中，我们将迭代器命名为`ReverseArrayIterator`,并添加了这段代码:

```
 public Iterator<Item> iterator() {
        return new ReverseArrayIterator();
    }
```

> 迭代器是一个实现了`hasNext()`和`next()`的方法的类的对象，由以下接口所定义(即java.util.Iterator):

```
public interface Iterator<Item>{
    boolean hasNext();
    Item next();
    void remove();
}
```
在本例中，使用一个内部类来实现迭代器，代码如下：

```
 private class ReverseArrayIterator implements Iterator<Item> {
        // 支持后进先出的迭代
        private int i = N;

        @Override
        public boolean hasNext() {
            return i > 0;
        }

        @Override
        public Item next() {
            return a[--i];
        }

        @Override
        public void remove() {

        }
    }
```

对于下压栈，它几乎(但还没有)达到了任意集合类型数据类型的实现的最佳性能：
- 每项操作的用时都与集合大小无关；
- 空间需求总是不超过集合大小乘以一个常数。

其缺点就在于`push()`和`pop()`在需要调整数组大小的时候，其耗时和栈大小成正比。

