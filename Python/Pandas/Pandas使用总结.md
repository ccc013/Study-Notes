
记录下`pandas`库的一些函数的使用方法。

##### .ix()函数

参考[官方文档用法](http://pandas-docs.github.io/pandas-docs-travis/indexing.html#indexing-deprecate-ix)，或者是[pandas iloc vs ix vs loc explanation?](http://stackoverflow.com/questions/31593201/pandas-iloc-vs-ix-vs-loc-explanation)。

它有两个参数，第一参数表示哪个元素，而第二个表示哪一列。

##### 如何融合两个不含相同列的表

参考[pandas cross join no columns in common](http://stackoverflow.com/questions/35265613/pandas-cross-join-no-columns-in-common),

代码例子如下：

```
import pandas as pd

df1 = pd.DataFrame({'fld1': ['x', 'y'],
                'fld2': ['a', 'b1']})


df2 = pd.DataFrame({'fld3': ['y', 'x', 'y'],
                'fld4': ['a', 'b1', 'c2']})

print df1
  fld1 fld2
0    x    a
1    y   b1

print df2
  fld3 fld4
0    y    a
1    x   b1
2    y   c2

df1['tmp'] = 1
df2['tmp'] = 1

df = pd.merge(df1, df2, on=['tmp'])
df = df.drop('tmp', axis=1)
print df
  fld1 fld2 fld3 fld4
0    x    a    y    a
1    x    a    x   b1
2    x    a    y   c2
3    y   b1    y    a
4    y   b1    x   b1
5    y   b1    y   c2
```


