
#### 单链表
Java实现简单的单链表
```
/**
 * Created by cai on 2016/5/24 15:15.
 * Email: 429546420@qq.com
 */
public class LinkList<Item> {
    private Node first;
    private int N;

    private class Node {
        Item item;
        Node next;
    }

    public LinkList() {
        N = 0;
        first = null;
    }

    public boolean isEmpty() {
        return first == null;
    }

    public int size() {
        return N;
    }

    /**
     * 删除第k个位置的元素
     *
     * @param k
     * @return
     */
    public Item delete(int k) {
        if (isEmpty())
            throw new NoSuchElementException("LinkList is empty!");
        if (k > N || k < 0)
            throw new NoSuchElementException("error!");

        Node current = first;
        Node prev = first;
        for (int i = 1; i < k; i++) {
            prev = current;
            current = current.next;
        }

        Item item = current.item;
        prev.next = current.next;
        N--;
        return item;
    }

    /**
     * 在指定结点前插入一个值
     *
     * @param k
     * @param data
     */
    public void insertBefore(int k, Item data) {
        if (k > N || k < 0)
            throw new NoSuchElementException("error!");

        Node current = first;
        Node temp = new Node();
        temp.item = data;
        for (int i = 1; i < k - 1; i++) {
            current = current.next;
        }
        temp.next = current.next;
        current.next = temp;

        N++;
    }

    /**
     * 在指定结点位置后插入
     *
     * @param k
     * @param data
     */
    public void insertAfter(int k, Item data) {
        if (k > N || k < 0)
            throw new NoSuchElementException("error!");

        Node current = first;
        Node temp = new Node();
        temp.item = data;
        for (int i = 1; i < k; i++) {
            current = current.next;
        }
        temp.next = current.next;
        current.next = temp;

        N++;
    }

    /**
     * 返回第k个位置的元素值
     *
     * @param k
     * @return
     */
    public Item get(int k) {
        if (isEmpty())
            throw new NoSuchElementException("LinkList is empty!");
        if (k > N || k < 0)
            throw new NoSuchElementException("error!");

        Node temp = first;
        for (int i = 1; i < size(); i++) {
            if (i == k)
                return temp.item;
            else
                temp = temp.next;
        }
        return null;
    }

    /**
     * 根据提供的值返回其在链表中的位置，如果没有则返回-1
     *
     * @param key
     * @return
     */
    public int find(Item key) {
        if (isEmpty())
            throw new NoSuchElementException("LinkList is empty!");

        Node temp = first;
        for (int i = 0; i < size(); i++) {
            if (temp.item == key) {
                return i + 1;
            } else {
                temp = temp.next;
            }
        }

        return -1;
    }

    /**
     * 在表头插入结点
     *
     * @param item
     */
    public void insertFirst(Item item) {
        Node temp = first;
        first = new Node();
        first.item = item;
        first.next = temp;
        N++;
    }

    /**
     * 删除表头第一个结点
     *
     * @return
     */
    public Item deleteFirst() {
        Item item = first.item;
        first = first.next;
        N--;
        return item;
    }

    /**
     * 展示链表中所有的值
     */
    public void display() {
        Node temp = first;
        while (temp != null) {
            System.out.print(temp.item + " ");
            temp = temp.next;
        }
        System.out.println();
    }

    /**
     * 假设item是Integer类型下，输出链表中最大值
     *
     * @return
     */
    public int max() {
        if (isEmpty())
            return 0;
        int max = (Integer) first.item;
        Node current = first;
        for (int i = 1; i < size(); i++) {
            current = current.next;
            max = max > (Integer) current.item ? max : (Integer) current.item;
        }
        return max;
    }

    // Test
    public static void main(String[] args) {
        int N = 10;
        LinkList<Integer> linkList = new LinkList<>();
        for (int i = 0; i < N; i++)
            linkList.insertFirst(2 * i);
        linkList.display();
//        System.out.println("delete first element");
//        linkList.deleteFirst();
//        linkList.display();
//        System.out.println("delete a element");
//        int elem = linkList.delete(5);
//        System.out.println(elem);
//        System.out.println("get a element");
//        System.out.println(linkList.get(2));
//        linkList.display();
        System.out.println("insert a node before:");
        linkList.insertBefore(linkList.size(), 33);
        linkList.display();
        System.out.println("insert a node after:");
        linkList.insertAfter(linkList.size(), 55);
        linkList.display();

//        System.out.println("find 8 in linklist:");
//        System.out.println(linkList.find(8));
//        linkList.display();
        System.out.println("the max element in linklist is " + linkList.max());
    }

}

```
