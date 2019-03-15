使用定长数组实现队列的抽象，然后添加一个可以调整数组大小的方法。


```
/**
 * Created by cai on 2016/5/22 10:54.
 * Email: 429546420@qq.com
 */
public class ResizingArrayQueue<Item> implements Iterable<Item> {
    /**
     * 使用定长数组实现队列，添加调整数组大小的方法
     */
    private int first;
    private int last;
    private int N;
    private Item[] a;


    public ResizingArrayQueue() {
        a = (Item[]) new Object[2];
        first = 0;
        last = 0;
        N = 0;
    }

    public boolean isEmpty() {
        return N == 0;
    }

    public int size() {
        return N;
    }

    private void resize(int max) {
        assert max >= N;
        Item[] temp = (Item[]) new Object[max];
        for (int i = 0; i < max; i++) {
            temp[i] = a[(first + i) % a.length];
        }
        a = temp;
        first = 0;
        last = N;
    }

    public void enqueue(Item item) {
        if (N == a.length)
            resize(2 * a.length);
        a[last++] = item;
        N++;
        if (last == a.length)
            last = 0;
    }

    public Item dequeue() {
        if (isEmpty())
            throw new NoSuchElementException("Queue is empty");
        Item item = a[first];
        a[first] = null;
        first++;
        N--;
        if (first == a.length)
            first = 0;
        if (N > 0 && N == a.length / 4)
            resize(a.length / 2);
        return item;
    }

    public Item peek() {
        if (isEmpty())
            throw new NoSuchElementException("Queue is empty");
        return a[first];
    }

    public Iterator<Item> iterator() {
        return new ListIterator();
    }

    private class ListIterator implements Iterator<Item> {
        private int i = 0;

        @Override
        public boolean hasNext() {
            return i < N;
        }

        @Override
        public void remove() {

        }

        @Override
        public Item next() {
            if (!hasNext())
                throw new NoSuchElementException("Queue is empty");
            Item item = a[(i + first) % a.length];
            i++;
            return item;
        }
    }

    // Test
    public static void main(String[] args) throws IOException {
        ResizingArrayQueue<String> s = new ResizingArrayQueue<>();
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
                    s.enqueue(item);
                else if (!s.isEmpty())
                    System.out.print(s.dequeue() + " ");
            }
        }
        bufferedReader.close();

        System.out.println("(" + s.size() + " left on stack)\nQueue-- ");
        for (String elem : s) {
            System.out.print(elem + ", ");
        }
        System.out.print("\nthe top element now is " + s.peek());
        System.out.println("\nthat's all.");
    }
}


```
