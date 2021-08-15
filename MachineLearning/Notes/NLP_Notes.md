# NLP学习笔记

参考：

- [2020深度学习面试题总结（更新中）](https://zhuanlan.zhihu.com/p/72680338)



简单记录整理看到的 NLP 相关的知识点，后续再整理和分类；

------

## 网络模型

### Bert

**Bert的微创新：**

- the “masked language model” (*MLM*) 随机mask输入中的一些tokens，然后在预训练中对它们进行预测。

- 增加句子级别的任务：“next sentence prediction”





### FastText与CBOW的相同点与不同点

相同点：

- 两种模型都是基于 Hierarchical Softmax，都是三层架构：输入层、 隐藏层、输出层。

不同点：

（1）CBOW 模型基于N-gram模型和BOW模型，此模型将W(t−N+1)…..W(t−1)W(t−N+1)……W(t−1)作为输入，去预测W(t) ,fastText的模型则是将整个文本作为特征去预测文本的类别。

（2）CBOW是词袋模型，而fastText 还加入了 N-gram 特征。

（3）word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是分类的label。

（4）word2vec的输入层，是context window内的term；而fasttext 对应的整个sentence的内容，包括term，也包括 n-gram的内容；



### RCNN

**基于RCNN的文本分类原理**

- [自然语言处理 | (23) 基于RCNN的文本分类原理](https://blog.csdn.net/sdu_hao/article/details/88099535)





## 工具框架

### Jieba

**Jieba分词过程：**

**(1)如何构建DAG**

```python
   def get_DAG(self, sentence):
        self.check_initialized()
        DAG = {}
        N = len(sentence)
        # k,i 相当于双指针，k遍历每个位置,i遍历k到该sentence结尾，如果这个词出现在词汇表中
        # 将这个词的结尾下标添加到以k为键值的列表中，循环往复。
        for k in xrange(N):
            tmplist = []
            i = k
            frag = sentence[k]
            # self.FREQ存储词和词对应的词频，可以由dict.txt构建
            while i < N and frag in self.FREQ:
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG
```

如果sentence是'我来到北京清华大学‘，那么DAG为

```text
{0: [0], 1: [1, 2], 2: [2], 3: [3, 4], 4: [4], 5: [5, 6, 8], 6: [6, 7], 7: [7, 8], 8: [8]}
```

DAG[5]=[5,6,8]的意思就是，以’清‘开头的话，分别以5、6、8结束时，可以是一个词语，即’清‘、’清华‘、’清华大学‘

**（2）全模式切词：**

特点：把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义。

```python
    def __cut_all(self, sentence):
        dag = self.get_DAG(sentence)
        old_j = -1
        for k, L in iteritems(dag):
            if len(L) == 1 and k > old_j:
                yield sentence[k:L[0] + 1]
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k:j + 1]
                        old_j = j
```

**（3）不是用HMM分词：**

```python
    def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        if buf:
            yield buf
            buf = ''
```



------

## 相关问题

### 1. 拼写检查

**经常在网上搜索东西的朋友知道，当你不小心输入一个不存在的单词时，搜索引擎会提示你是不是要输入某一个正确的单词，比如当你在Google中输入“Julw”时，系统会猜测你的意图：是不是要搜索“July”**。

根据谷歌一员工写的文章How to Write a Spelling Corrector显示，Google**的拼写检查基于贝叶斯方法**。

用户输入一个单词时，可能拼写正确，也可能拼写错误。如果把拼写正确的情况记做c（代表correct），拼写错误的情况记做w（代表wrong），那么”拼写检查”要做的事情就是：**在发生w的情况下，试图推断出c**。

换言之：已知w，然后在若干个备选方案中，找出可能性最大的那个c，也就是求P(c|w) 的最大值。而根据贝叶斯定理，有：
$$
P(c|w)=\frac{P(w|c)P(c)}{P(w)}
$$
由于对于所有备选的c来说，对应的都是同一个w，所以它们的P(w)是相同的，因此我们只要最大化P(w|c)P(c)即可。其中：

P(c)表示某个正确的词的出现”概率”，它可以用”频率”代替。如果我们有一个足够大的文本库，那么这个文本库中每个单词的出现频率，就相当于它的发生概率。某个词的出现频率越高，P(c)就越大。比如在你输入一个错误的词“Julw”时，系统更倾向于去猜测你可能想输入的词是“July”，而不是“Jult”，因为“July”更常见。

P(w|c)表示在试图拼写c的情况下，出现拼写错误w的概率。为了简化问题，假定两个单词在字形上越接近，就有越可能拼错，P(w|c)就越大。举例来说，相差一个字母的拼法，就比相差两个字母的拼法，发生概率更高。你想拼写单词July，那么错误拼成Julw（相差一个字母）的可能性，就比拼成Jullw高（相差两个字母）。值得一提的是，一般把这种问题称为“编辑距离”，参见程序员编程艺术第二十八~二十九章：最大连续乘积子串、字符串编辑距离。

http://blog.csdn.net/v_july_v/article/details/8701148#t4

所以，我们比较所有拼写相近的词在文本库中的出现频率，再从中挑出出现频率最高的一个，即是用户最想输入的那个词。具体的计算过程及此方法的缺陷请参见How to Write a Spelling Corrector。

http://norvig.com/spell-correct.html



