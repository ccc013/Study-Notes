## 《大话数据结构》第六章 树

#### 1. 树的定义
> 树(Tree)是n(n》0)个结点的有限集。n=0时称为空树。在任何一棵非空树中：(1) 有且仅有一个特定的称为根(Root)的结点；(2) 当n\>1时，其余结点可分为m(m\>0)个互不相交的有限集T1，T2，...,Tm,其中每一个集合本身又是一棵树，并且称为根的子树(SubTree)，如下图1.1(图片均截取书本配图)

![树的定义](https://raw.githubusercontent.com/ccc013/Study-Notes/master/DataStructe%20%26%20Algorithm/images/Tree.png)    

##### (1) 结点分类
> 结点拥有的子树数称为结点的度(Degree)。度为0的结点称为叶结点(Leaf)或终端结点；度不为0的结点称为非终端结点或分支结点。除根结点之外，分支结点也称为内部结点。树的度是树内各结点的度的最大值。如下图，树中结点的度的最大值是3，所以树的度是3。

![Degree](http://7xrluf.com1.z0.glb.clouddn.com/Tree_Degree.png)

##### (2) 结点间关系
> 结点的子树的根称为该结点的孩子(Child),相应地，该结点称为孩子的双亲(Parent)。同一个双亲的孩子之间互称兄弟(Sibling)。结点的祖先是从根到该结点所经分支上的所有结点，以某结点为跟的子树中的任一结点都称为该结点的子孙。

  具体如下图所示，其中对于H结点，其祖先是D、B、A，而对于B，其子孙有D、G、H、I。
  ![Tree_relationship](http://7xrluf.com1.z0.glb.clouddn.com/Tree_relationship.png)
  
##### (3) 树的其他相关概念
> 结点的层次(Level)从根开始定义起来，根为第一层，根的孩子为第二层；双亲在同一层的结点互为堂兄弟；树中结点的最大层次称为树的深度(Depth)或高度。

 具体如下图所示。
 ![Tree_Depth](http://7xrluf.com1.z0.glb.clouddn.com/Tree_Depth.png)
 
> 如果将树中结点的各子树看成从左至右是有次序的，不能互换的，则称该树为有序树，否则称为无序树。

> 森林(Forest)是m(m》0)棵互不相交的树的集合。


---
#### 2. 树的抽象数据类型

```
ADT 树(tree)
Data
    树是由一个根结点和若干棵子树构成。树中结点具有相同数据类型及层次关系。
Operation
    InitTree(*T): 构造空树T。
    DestroyTree(*T): 销毁树T。
    CreateTree(*T, definition): 按definition中给出的树的定义来构造树。
    ClearTree(*T): 若树T存在，则将树T清为空树。
    TreeEmpty(T): 若T为空树，返回true，否则返回false。
    TreeDepth(T): 返回T的深度。
    Root(T): 返回T的根结点。
    Value(T, cur_e): cur_e是树T中的一个结点，返回该结点的值。
    Assign(T,cur_e,value): 给树T的结点cur_e赋值为value。
    Parent(T,cur_e): 若cur_e是树的非根结点，则返回它的双亲，否则返回空。
    LeftChild(T,cur_e): 若cur_e是树T的非叶结点，则返回它的最左孩子，否则返回空。
    RightSibling(T,cur_e): 若cur_e有右兄弟，则返回它的右兄弟，否则返回空。
    InsertChild(*T, *p, i, c): 其中p是指向树T的某个结点，i是所指结点p的度加上1，非空树c与T不相交，操作结果是插入c为树T中p所指结点的第i棵子树。
    DeleteChild(*T, *p, i): 其中p指向树T的某个结点，i是p结点的度，操作结果是删除树T中p所指结点的第i棵子树。

endADT
```

---
#### 3. 树的存储结构
  由于树的某个结点可以有多个孩子，简单的顺序存储结构是不能够满足树的实现要求的。所以需要充分利用顺序存储和链式存储结构的特点。这里会介绍3种不同的表示方法：双亲表示法、孩子表示法以及孩子兄弟表示法。
  
##### (1) 双亲表示法
双亲表示法的结点结构定义

```
#define MAX_TREE_SIZE 100
typedef int TElemTpye;
typedef struct PTNode
{
    TElemType data; // 结点数据
    int parent;     // 双亲位置
} PTNode;

typedef struct                      // 树结构
{
    PTNode nodes[MAX_TREE_SIZE];    // 结点数组
    int r,n;                        // 跟的位置和结点数
}
```
  由于根结点是没有双亲的，所以我们可以约定根结点的位置域设置为-1.
  
  上述存储结构可以很快地查找结点的双亲位置，但是对于结点的孩子位置，必须要遍历整个结构才可以。那么如果需要知道孩子位置，可以在结构中增加一个结点最左边孩子的域，而如果需要知道兄弟位置，则可以对应增加一个右兄弟域来获得右兄弟的位置。
  
> 存储结构的设计是一个非常灵活的过程。一个存储结构设计得是否合理，取决于基于该存储结构的运算是否适合、是否方便，时间复杂度好不好等。

##### (2) 孩子表示法
> 由于树中每个结点可能有多棵子树，可以考虑用多重链表，即每个结点有多个指针域，其中每个指针指向一棵子树的根结点，我们把这种方法叫做多重链表示法。

  实现多重链表示法有两种方案，分别如下所述。
  
###### 方案一
  指针域的个数等于树的度。其结构如下图所示：
  ![image](http://7xrluf.com1.z0.glb.clouddn.com/Tree_struct1.png)
  对图1.1的树来说，其实现效果如下图所示：
  ![image](http://7xrluf.com1.z0.glb.clouddn.com/Tree_struct2.png)

  这种方法由于树中各结点的度相差很大，所以显然是会浪费空间的。只有当树中结点的度相差不大的时候，才适合使用这种方案。
  
###### 方案二
  每个结点指针域的个数等于该结点的度，即专门增加一个位置存储结点指针域的个数。结构如下图所示：
  ![image](http://7xrluf.com1.z0.glb.clouddn.com/Tree_struct3.png)
  对图1.1的树来说，其实现效果如下图：
  ![image](http://7xrluf.com1.z0.glb.clouddn.com/Tree_struct4.png)

这种方法克服了浪费空间的缺点，对空间利用率是很高了，但是由于各个结点的链表是不同的结构，加上要维护结点的度的数值，在运算上就会带来时间上的损耗。

> 因此，这里展示真正的孩子表示法。具体办法是，把每个结点的孩子结点排列起来，以单链表作存储结构，则n个结点有n个孩子链表。如果是叶结点，则其链表为空。然后n个头指针又组成一个线性表，采用顺序存储结构，存放进一个一维数组中。

  对图1.1的树来说，其具体实现如下图所示
  ![image](http://7xrluf.com1.z0.glb.clouddn.com/Tree_struct5.png)
  
  根据我的理解，就是树的结构是一个顺序存储的结构，用一个数组表示，而数组存储的是每个结点，结点的结构则是由数据域和指针域组成，指针域是一个指向该结点的孩子链表的头指针，然后孩子链表中的孩子结点同样是数据域加指针域，指针域就是指向其右边兄弟的指针。
  
结构定义代码如下：
```
#define MAX_TREE_SIZE 100
typedef struct CTNode   // 孩子结点
{
    int child;
    struct CTNode* next;
} *ChildPtr;

typedef struct          // 表头结构
{
    TElemType data;
    ChildPtr firstchild;
} CTBox;

typedef struct          // 树结构
{
    CTBox nodes[MAX_TREE_SIZE];     // 结点数组
    int r,n;                        // 根的位置和结点数
} CTree;
```
在这种结构中，如果需要知道结点的双亲，可以在表头结构中增加一个指针域，用来指向其双亲结点的位置。

##### (3) 孩子兄弟表示法
> 任意一棵树，它的结点的第一个孩子如果存在就是唯一的，它的右兄弟如果存在也是唯一的。因此，我们设置两个指针，分别指向该结点的第一个孩子和此结点的右兄弟。

结点结构如下：
![image](http://7xrluf.com1.z0.glb.clouddn.com/Tree_struct6.png)
对图1.1的树来说，其实现的示意图如下：
![image](http://7xrluf.com1.z0.glb.clouddn.com/Tree_struct7.png)

结构定义代码如下：
```
typedef struct CSNode
{
    TElemType data;
    struct CSNode * firstchild, *rightsib;
} CSNode, *CSTree;
```

---
#### 4. 二叉树的定义
> 二叉树(Binary Tree)是n(n》0)个结点的有限集合，该集合或者为空集(称为空二叉树)，或者由一个根结点和两棵互不相交的、分别称为根结点的左子树和右子树的二叉树组成。

二叉树的例子如下图所示：
![image](http://7xrluf.com1.z0.glb.clouddn.com/BinaryTree.png)

##### (1) 特点
* 每个结点最多有两棵子树，所以二叉树不存在度大于2的结点。
* 左右子树是有顺序的，次序不能颠倒。
* 即使树中某结点只有一棵子树，也要区分它是左子树，还是右子树。

二叉树具有五种基本形态：
* 空二叉树
* 只有一个根结点
* 根结点只有左子树
* 根结点只有右子树
* 根结点既有左子树也有右子树

##### (2) 特殊二叉树

###### 斜树
> 所有的结点只有左子树的二叉树叫左斜树。所有结点只有右子树的二叉树的叫右斜树。这两者统称为斜树。

###### 满二叉树
> 在一棵二叉树中，如果所有结点都存在左子树和右子树，并且所有叶子都在同一层上，这样的二叉树称为满二叉树。

一棵满二叉树如下图所示
![image](http://7xrluf.com1.z0.glb.clouddn.com/FullBinaryTree.png)

满二叉树的特点有：

- 叶子只能出现在最下一层。出现在其它层就不可能达成平衡。
- 非叶子结点的度一定是2.
- 在同样深度的二叉树中，满二叉树的结点个数最多，叶子数最多。

###### 完全二叉树
> 对一棵具有n个结点的二叉树按层序编号，如果编号为i(1<=i<=n)的结点与同为深度的满二叉树中编号为i的结点在二叉树的位置完全相同，则这棵二叉树称为完成二叉树。

如下图所示
![image](http://7xrluf.com1.z0.glb.clouddn.com/CompleteBinaryTree.png)

完全二叉树的特点有：
- 叶子结点只能出现在最下两层
- 最下层的叶子一定集中在左部连续位置
- 倒数第二层，若有叶子结点，一定都在右边连续位置
- 如果结点度为1，则该结点只有左子树，即不存在只有右子树的情况
- 同样结点数的二叉树，完全二叉树的深度最小


---
#### 5.二叉树的性质
- 性质1：在二叉树的第i层上最多有如下结点数(i>=1)
```math
2^{i-1}
```
这个结论可以通过归纳法得到。

- 性质2：深度为k的二叉树最多有如下结点树(k>=1)
```math
2^k-1
```
这个结论是在第一个性质的基础上，就是一个求和

- 性质3：对任意一棵二叉树T，如果其终端结点数位n0，度为2的结点树为n2，则`n0 = n2 + 1`

推导如下：

> 假设度为1的结点数是n1，则树T的总结点数`n=n0+n1+n2`，而对于树中的分支线，只有跟结点是没有分支进来，只有分支出去的，所以分支线总数是`n-1 = n1+2n2`，代入总结点数的公式有`n0+n1+n2-1 = n1+2n2`，因此结论就是`n0 = n2 + 1`.
  
- 性质4：具有n个结点的完全二叉树的深度为
```math
[log_2n] + 1
```
([x]表示不大于x的最大整数)

推导如下：

> 对于满二叉树，深度是k的满二叉树的结点数为
```math
n = 2^k-1
```
> 由此得到满二叉树的深度为
```math
k = log_2(n+1)
```
![image](http://7xrluf.com1.z0.glb.clouddn.com/BinaryTree1.png)

最后一个性质如下：
![image](http://7xrluf.com1.z0.glb.clouddn.com/BinaryTree2.png)

可以根据下图来验证上述性质
![image](http://7xrluf.com1.z0.glb.clouddn.com/BinaryTree3.png)

---
#### 6. 二叉树的存储结构
