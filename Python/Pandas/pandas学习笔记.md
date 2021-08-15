# pandas学习笔记

# 基础操作
## 读取数据
参考：

- [pandas.read_csv参数详解](https://www.cnblogs.com/datablog/p/6127000.html)

`read_csv` 即可用于读取数据，基本操作如下：

```python
df = pd.read_csv('file_name.csv')
```
如果数据太大，可以添加 `nrow` 参数来只加载前几行数据，例子如下所示：

```python
# 仅读取前5行数据
df2 = pd.read_csv('file_name.csv', nrows=5)
```

### txt to csv
读取 txt 文本文件，并保存为 csv

```
data = pd.read_csv('output_list.txt', sep=" ", header=None)
data.columns = ["a", "b", "c", "etc."]
```

保存为csv

```
df.to_csv(file_name, encoding='utf-8', index=False)
or
df.to_csv(file_name, sep='\t')
```

---

## 编码问题
参考：

- [https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python](https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python)

遇到类似编码问题：

```python
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xda in position 6:
```

解决的办法就是读取的时候，指定编码：

```python
data = pd.read_csv('file_name.csv', encoding='utf-8')
```

可替代的编码方式还有：

```python
encoding = "cp1252"
encoding = "ISO-8859-1"
```

## 读取大文件的数据
对于文件很大的，比如5，6G，甚至几十G的，内存无法一次读取完成，可以分段读取，例子如下，来自[https://stackoverflow.com/questions/25962114/how-to-read-a-6-gb-csv-file-with-pandas](https://stackoverflow.com/questions/25962114/how-to-read-a-6-gb-csv-file-with-pandas)

```python
chunksize = 10 ** 6
for chunk in pd.read_csv(filename, chunksize=chunksize):
    process(chunk)
```

## 选取/排除指定行
参考：

- [https://blog.csdn.net/yuanxiang01/article/details/79285769](https://blog.csdn.net/yuanxiang01/article/details/79285769)

找到指定的列等于某些数值的行：

```python
>>> df = pd.DataFrame([['GD', 'GX', 'FJ'], ['SD', 'SX', 'BJ'], ['HN', 'HB', 'AH'], ['HEN', 'HEN', 'HLJ'], ['SH', 'TJ', 'CQ']], columns=['p1', 'p2', 'p3'])
>>> df
    p1   p2   p3
0   GD   GX   FJ
1   SD   SX   BJ
2   HN   HB   AH
3  HEN  HEN  HLJ
4   SH   TJ   CQ
# 筛选p1列中值为'SD'和'HN'的行：
>>> df[df.p1.isin(['SD','HN'])]
   p1  p2  p3
1  SD  SX  BJ
2  HN  HB  AH
```

排除指定行的方法，这个方法是将列转为列表，然后移除需要排除的数值，再来作选择指定行的操作：

```python
# 将p1转换为列表，再从列表中移除特定的行：
>>> ex_list = list(df.p1)
>>> ex_list.remove('SD')
>>> ex_list.remove('HN')
>>> df[df.p1.isin(ex_list)]
    p1   p2   p3
0   GD   GX   FJ
3  HEN  HEN  HLJ
4   SH   TJ   CQ
```

## 修改某列的特定数值
采用 `loc` 方法，例子如下所示

```python
>>> import pandas as pd
>>> a = pd.DataFrame({'a':[1,2,3], 'b': ['a', 'b', 'c']})
>>> a
   a  b
0  1  a
1  2  b
2  3  c
>>> a.loc[a['b']=='b', 'b'] = 'bb'
>>> a
   a   b
0  1   a
1  2  bb
2  3   c
>>>
```

## 选择特定数据类型的列
主要方法是 `select_dtypes` ，首先是查看每列的数据类型分布：

```python
# 查看每列数据类型的分布
df.dtypes.value_counts()
```
比如输出结果是如下所示：

```python
float64    5
object     1
int64      1
dtype: int64
```


然后选择特定的数据类型：

```python
# 选择特定类型，select_dtypes
df3 = df.select_dtypes(include=['int64', 'float64'])
```

## 转换为其他数据类型
### dataframe<->dict
参考文章：[https://stackoverflow.com/questions/26716616/convert-a-pandas-dataframe-to-a-dictionary](https://stackoverflow.com/questions/26716616/convert-a-pandas-dataframe-to-a-dictionary)

主要是通过 `to_dict` 方法，它可以接受一个参数，表示得到的字典的数值是什么数据类型，默认是字典，例子如下：
先生成一个 `dataframe` 
```python
>>> df = pd.DataFrame({'a': ['red', 'yellow', 'blue'], 'b': [0.5, 0.25, 0.125]})
>>> df
        a      b
0     red  0.500
1  yellow  0.250
2    blue  0.125
```

默认参数： `dict` ，键名是列的名字，而值则是 索引和数据 对
```python
>>> df.to_dict('dict')
{'a': {0: 'red', 1: 'yellow', 2: 'blue'}, 
 'b': {0: 0.5, 1: 0.25, 2: 0.125}}
```
其他可选参数的例子：

```python
# list 不带索引
>>> df.to_dict('list')
{'a': ['red', 'yellow', 'blue'], 
 'b': [0.5, 0.25, 0.125]}

# series
>>> df.to_dict('series')
{'a': 0       red
      1    yellow
      2      blue
      Name: a, dtype: object, 

 'b': 0    0.500
      1    0.250
      2    0.125
 
 # splits，将列、索引和数据分别作为键名
 >>> df.to_dict('split')
{'columns': ['a', 'b'],
 'data': [['red', 0.5], ['yellow', 0.25], ['blue', 0.125]],
 'index': [0, 1, 2]}
 
 # records，每一行作为一个字典，实际得到的是一个list，里面是一个个字典元素
 >>> df.to_dict('records')
[{'a': 'red', 'b': 0.5}, 
 {'a': 'yellow', 'b': 0.25}, 
 {'a': 'blue', 'b': 0.125}]
 
 # index ,index 作为键名，也就是每一行就是一个字典
 >>> df.to_dict('index')
{0: {'a': 'red', 'b': 0.5},
 1: {'a': 'yellow', 'b': 0.25},
 2: {'a': 'blue', 'b': 0.125}}
```

## 如何融合两个不含相同列的表
参考：
[pandas cross join no columns in common](http://stackoverflow.com/questions/35265613/pandas-cross-join-no-columns-in-common),

给两个表添加一个临时的相同数值的列，调用 `merge` 融合后，删除临时添加的列，代码例子如下：

```python
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



---


# 数据分析

## 查看某列的数值分布
采用函数 `value_counts()` 可以统计不同数值的样本数量，官方例子如下：

```python
>>> index = pd.Index([3, 1, 2, 3, 4, np.nan])
>>> index.value_counts()
3.0    2
4.0    1
2.0    1
1.0    1
dtype: int64
```

### normalize
添加参数 `normalize` ，数量将做一个简单的归一化，每个数量会除以样本总数：

```python
>>> s = pd.Series([3, 1, 2, 3, 4, np.nan])
>>> s.value_counts(normalize=True)
3.0    0.4
4.0    0.2
2.0    0.2
1.0    0.2
dtype: float64
```
### bins
可以将连续性变量变成类别型变量，通过设置 `bins` 的数值，如下表示分为 3 个区间显示，每个区间都是半开区间，左开右闭的区间。

```python
>>> s.value_counts(bins=3)
(2.0, 3.0]      2
(0.996, 2.0]    2
(3.0, 4.0]      1
dtype: int64
```

### dropna
如果想显示 NaN 的统计：

```python
>>> s.value_counts(dropna=False)
3.0    2
NaN    1
4.0    1
2.0    1
1.0    1
dtype: int64
```

### ascending
默认是根据数量进行降序显示，可以设置 `ascending=True`  实现升序显示：

```python
>>> s.value_counts(ascending=True)
1.0    1
2.0    1
4.0    1
3.0    2
dtype: int64
```
 
## 查看某列的unique数值
函数： `unique()` ，官方例子：

```python
>>> pd.unique(pd.Series([2, 1, 3, 3]))
array([2, 1, 3])
```


## 对类别特征的处理
### 问题描述

一般特征可以分为两类特征，连续型和离散型特征，而离散型特征既有是数值型的，也有是类别型特征，也可以说是字符型，比如说性别，是男还是女；职业，可以是程序员，产品经理，教师等等。

本文将主要介绍一些处理这种类别型特征的方法，分别来自 pandas 和 sklearn 两个常用的 python 库给出的解决方法，这些方法也并非是处理这类特征的唯一答案，通常都需要具体问题具体分析。

### 数据准备

参考文章：[https://mlln.cn/2018/09/18/pandas%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE%E8%BD%AC%E6%95%B4%E6%95%B0%E5%88%86%E7%B1%BB%E7%BC%96%E7%A0%81%E7%9A%84%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5/](https://mlln.cn/2018/09/18/pandas%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE%E8%BD%AC%E6%95%B4%E6%95%B0%E5%88%86%E7%B1%BB%E7%BC%96%E7%A0%81%E7%9A%84%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5/)

采用 UCI 机器学习库的一个汽车数据集，它包括类别型特征和连续型特征，首先是简单可视化这个数据集的部分样本，并简单进行处理。

首先导入这次需要用到的 python 库：

```python
import pandas as pd
import numpy as np
import pandas_profiling
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
```

接着加载数据：

```python
# 定义数据的列名称, 因为这个数据集没有包含列名称
headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

# 读取在线的数据集, 并将?转换为缺失NaN
df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/autos/imports-85.data",
                  header=None, names=headers, na_values="?" )
df.head()[df.columns[:10]]
```

展示的前 10 列的 5 行数据结果如下：

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1572769832457-41622a46-e11e-46d1-8e13-d2c94093897e.png#align=left&display=inline&height=334&originHeight=334&originWidth=1802&size=0&status=done&style=none&width=1802)

这里介绍一个新的数据分析库--`pandas_profiling`，这个库可以帮我们先对数据集做一个数据分析报告，报告的内容包括说明数据集包含的列数量、样本数量，每列的缺失值数量，每列之间的相关性等等。

安装方法也很简单：

```python
pip install pandas_profiling
```

使用方法也很简单，用 `pandas`读取数据后，直接输入下列代码：

```python
df.profile_report()
```

显示的结果如下，概览如下所示，看右上角可以选择有 5 项内容，下面是概览的内容，主要展示数据集的样本数量，特征数量（列的数量）、占用内存、每列的数据类型统计、缺失值情况等：



这是一个很有用的工具，可以让我们对数据集有一个初步的了解，更多用法可以去查看其 github 上了解：

[https://github.com/pandas-profiling/pandas-profiling](https://github.com/pandas-profiling/pandas-profiling)

加载数据后，这里我们仅关注类别型特征，也就是 `object` 类型的特征，这里可以有两种方法来获取：

方法1:采用 pandas 提供的方法 `select_dtypes`:

```python
df2 = df.select_dtypes('object').copy()
df2.head()
```

方法2: 通过 `bool` 型的 mask 获取 object 类型的列

```python
category_feature_mask = df.dtypes == object
category_cols = df.columns[category_feature_mask].tolist()
df3 = df[category_cols].copy()
df3.head()
```

输出结果如下：

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1572769832454-c08c7eb9-e02d-4312-82dd-51925f07c561.png#align=left&display=inline&height=318&originHeight=318&originWidth=1768&size=0&status=done&style=none&width=1768)

因为包含一些缺失值，这里非常简单的选择丢弃的方法，但实际上应该如何处理缺失值也是需要考虑很多因素，包括缺失值的数量等，但这里就不展开说明了：

```python
# 简单的处理缺失值--丢弃
df2.dropna(inplace=True)
```

### 标签编码

第一种处理方法是标签编码，其实就是直接将类别型特征从字符串转换为数字，有两种处理方法：

- 直接替换字符串
- 转为 `category` 类型后标签编码

直接替换字符串，算是手动处理，实现如下所示，这里用 `body_style` 这列特征做例子进行处理，它总共有 5 个取值方式，先通过 `value_counts`方法可以获取每个数值的分布情况，然后映射为数字，保存为一个字典，最后通过 `replace` 方法进行转换。

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1572769832459-46f21645-90cb-4fdd-ba8b-5580a91d2fa0.png#align=left&display=inline&height=934&originHeight=934&originWidth=1522&size=0&status=done&style=none&width=1522)

第二种，就是将该列特征转化为 `category` 特征，然后再用编码得到的作为数据即可：

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1572769832438-0112c328-4430-4db1-991e-6392dd715a54.png#align=left&display=inline&height=752&originHeight=752&originWidth=1352&size=0&status=done&style=none&width=1352)

### 自定义二分类

第二种方法比较特别，直接将所有的类别分为两个类别，这里用 `engine_type` 特征作为例子，假如我们仅关心该特征是否为 `ohc` ,那么我们就可以将其分为两类，包含 `ohc` 还是不包含，实现如下所示：

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1572769833298-70f5aa55-f94d-448b-9807-418f9d54a108.png#align=left&display=inline&height=614&originHeight=614&originWidth=1208&size=0&status=done&style=none&width=1208)

### One-hot 编码

前面两种方法其实也都有各自的局限性

- 第一种标签编码的方式，类别型特征如果有3个以上取值，那么编码后的数值就是 0，1，2等，这里会给模型一个误导，就是这个特征存在大小的关系，但实际上并不存在，所以标签编码更适合只有两个取值的情况；
- 第二种自定义二分类的方式，局限性就更大了，必须是只需要关注某个取值的时候，但实际应用很少会这样处理。

因此，这里介绍最常用的处理方法--**One-hot 编码**。

实现 One-hot 编码有以下 3 种方法：

- Pandas 的 `get_dummies`
- Sklearn 的 `DictVectorizer`
- Sklearn 的 `LabelEncoder`+`OneHotEncoder`

#### Pandas 的 `get_dummies`

首先介绍第一种--Pandas 的 `get_dummies`，这个方法使用非常简单了：

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1572769832432-a5773d2e-76f7-4742-b1cd-1b9f6ef91287.png#align=left&display=inline&height=468&originHeight=468&originWidth=2066&size=0&status=done&style=none&width=2066)

#### Sklearn 的`DictVectorizer`

第二种方法--Sklearn 的 `DictVectorizer`，这首先需要将 `dataframe` 转化为 `dict` 类型，这可以通过 `to_dict` ，并设置参数 `orient=records`，实现代码如下所示：

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1572769832529-4308abbf-7192-400f-814b-51bec80866c4.png#align=left&display=inline&height=1048&originHeight=1048&originWidth=1242&size=0&status=done&style=none&width=1242)

#### Sklearn 的 `LabelEncoder`+`OneHotEncoder`

第三种方法--Sklearn 的 `LabelEncoder`+`OneHotEncoder`

首先是定义 `LabelEncoder`，实现代码如下，可以发现其实它就是将字符串进行了标签编码，将字符串转换为数值，这个操作很关键，因为 `OneHotEncoder` 是不能处理字符串类型的，所以需要先做这样的转换操作：

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1572769833295-ca18c29b-4d9e-40bd-8379-414d11886f94.png#align=left&display=inline&height=628&originHeight=628&originWidth=982&size=0&status=done&style=none&width=982)

接着自然就是进行 one-hot 编码了，实现代码如下所示：

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1572769832555-8598fc03-00af-40a3-a0d7-6df3154cb775.png#align=left&display=inline&height=1232&originHeight=1232&originWidth=1336&size=0&status=done&style=none&width=1336)

此外，采用 `OneHotEncoder` 的一个好处就是**可以指定特征的维度**，这种情况适用于，如果训练集和测试集的某个特征的取值数量不同的情况，比如训练集的样本包含这个特征的所有可能的取值，但测试集的样本缺少了其中一种可能，那么如果直接用 pandas 的`get_dummies`方法，会导致训练集和测试集的特征维度不一致了。

实现代码如下所示：

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1572769832558-06e16bcf-03a8-4c8c-bbed-d5e63144ad19.png#align=left&display=inline&height=578&originHeight=578&originWidth=1166&size=0&status=done&style=none&width=1166)

---


参考文章：

1. [https://mlln.cn/2018/09/18/pandas%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE%E8%BD%AC%E6%95%B4%E6%95%B0%E5%88%86%E7%B1%BB%E7%BC%96%E7%A0%81%E7%9A%84%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5/](https://mlln.cn/2018/09/18/pandas%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE%E8%BD%AC%E6%95%B4%E6%95%B0%E5%88%86%E7%B1%BB%E7%BC%96%E7%A0%81%E7%9A%84%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5/)
1. [https://blog.csdn.net/selous/article/details/72457476](https://blog.csdn.net/selous/article/details/72457476)
1. [https://towardsdatascience.com/encoding-categorical-features-21a2651a065c](https://towardsdatascience.com/encoding-categorical-features-21a2651a065c)
1. [https://www.cnblogs.com/zhoukui/p/9159909.html](https://www.cnblogs.com/zhoukui/p/9159909.html)



---

# 参考

1. [官方文档](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)
1.  [https://towardsdatascience.com/10-python-pandas-tricks-that-make-your-work-more-efficient-2e8e483808ba](https://towardsdatascience.com/10-python-pandas-tricks-that-make-your-work-more-efficient-2e8e483808ba)
