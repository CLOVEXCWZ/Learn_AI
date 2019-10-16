<font size=15>Pandas</font>

- 时间版本信息
  - 时间 2019-09
  - python版本：3.7.0
  - pandas版本： 0.25.1

主要参考网站：[易百教程-Pandas教程](https://www.yiibai.com/pandas)



<font size=5>**目录**</font>



- [1 Pandas数据结构](#1)
  - [1.1 Series](#1.1)
  - [1.2 DataFrame](#1.2)
  - [1.3 Panel（面板）](#1.3)
- [2 Pandas操作](#2)
  - [2.1 基本操作](#2.1)
  - [2.2 描述性统计](#2.2)
  - [2.3 函数应用](#2.3)
  - [2.4 重建索引](#2.4)
  - [2.5 迭代](#2.5)
  - [2.6 排序](#2.6)
  - [2.7 索引和选择数据](#2.7)
  - [2.8 缺失值](#2.8)
  - [2.9 分组(GroupBy)](#2.9)
  - [2.10 并和/连接](#2.10)
  - [2.11 级联](#2.11)
  - [2.12 日期功能](#2.12)
  - [2.13 时间差](#2.13)



​	Pandas是一款开放源码的BSD许可的Python库，为Python编程语言提供了高性能，易于使用的数据结构和数据分析工具。Pandas用于广泛的领域，包括金融，经济，统计，分析等学术和商业领域。 

##  <a name="1">1 Pandas数据结构</a>

​	Pandas处理一下三个数据结构(Serie、DataFrame、Panel )这些数据结构构建在*Numpy*数组之上，这意味着它们很快。

​	考虑这些数据结构的最好方法是，较高维数据结构是其较低维数据结构的容器。 例如，`DataFrame`是`Series`的容器，`Panel`是`DataFrame`的容器。

| 数据结构 | 维数 | 描述                                                 |
| -------- | ---- | ---------------------------------------------------- |
| 系列     | 1    | `1`D标记均匀数组，大小不变。                         |
| 数据帧   | 2    | 一般`2`D标记，大小可变的表结构与潜在的异质类型的列。 |
| 面板     | 3    | 一般`3`D标记，大小可变数组。                         |

​		

###  <a name="1.1">1.1 Series</a>

系列是具有均匀数据的一维数组结构。例如，以下系列是整数：`10`,`23`,`56`，`...`的集合。

![img](http://www.yiibai.com/uploads/images/201710/3110/493141059_40874.png)

- 创建	
  - 构造函数创建
  - 从list创建
  - 从字典创建
  - 从标量创建

```python
"""
pandas.Series( data, index, dtype, copy)
- data 数据采取各种形式，如：ndarray，list，constants
- index 索引值必须是唯一的和散列的，与数据的长度相同。 默认np.arange(n)如果没有索引被传递。
- dtype dtype用于数据类型。如果没有，将推断数据类型
- copy 复制数据，默认为false
"""
import pandas as pd
import numpy as np

s11 = pd.Series()  # 创建一个空 Series
s21 = pd.Series(['a','b','c','d']) # 从数组中创建
s22 = pd.Series(['a','b','c','d'],index=[100,101,102,103]) # 从数组中创建，并自定义索引
s31 = pd.Series({'a' : 0., 'b' : 1., 'c' : 2.}) # 从字典中创建
s32 = pd.Series({'a' : 0., 'b' : 1., 'c' : 2.}, index=['b','c','d','a'])
s41 = pd.Series(5, index=[0, 1, 2, 3]) # 从标量中创建 (0~3的索引都会5)
```

- 访问
  - 从具有位置访问
  - 使用标签检索数据(索引)

```python
import pandas as pd
s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

# 从具有位置访问
print(s[0]) # 检索第一个元素
print(s[:3]) # 检索系列中的前三个元素
print(s[-3: ]) # 检索最后三个元素

# 使用标签检索数据(索引)
print(s['a']) # 使用索引标签值检索单个元素
print(s[['a','c','d']] ) # 使用索引标签值列表检索多个元素
print(s['f']) # 如果不包含标签，则会出现异常
```

###   <a name="1.2">1.2 DataFrame</a>

数据帧(DataFrame)是一个具有异构数据的二维数组。 例如，

| 姓名  | 年龄 | 性别 | 等级 |
| ----- | ---- | ---- | ---- |
| Maxsu | 25   | 男   | 4.45 |
| Katie | 34   | 女   | 2.78 |
| Vina  | 46   | 女   | 3.9  |
| Lia   | 32   | 女   | 4.6  |

上表表示具有整体绩效评级组织的销售团队的数据。数据以行和列表示。每列(column)表示一个属性，每行(row)代表一个人。 

- 创建DataFrame
  - 创建空DataFrame
  - 从ndarray/list中创建DataFrame
  - 从列表中创建DataFrame
  - 从字典中创建DataFrame

```python
"""
pandas.DataFrame( data, index, columns, dtype, copy)
  - data 数据采取各种形式，如:ndarray，series，map，lists，dict，constant和另一个DataFrame。 
  - index 对于行标签，要用于结果帧的索引是可选缺省值np.arrange(n)，如果没有传递索引值。
  - colums 对于列标签，可选的默认语法是 - np.arange(n)。 这只有在没有索引传递的情况下才是这样。
  - dtype 每列的数据类型。
  - copy  如果默认值为False，则此命令(或任何它)用于复制数据。
"""
import pandas as pd

#-------------  创建一个空的DataFrame
df11 = pd.DataFrame()

#-------------- 从列表中创建DataFrame

# 实例-1 可以使用单个列表或列表列表创建数据帧(DataFrame)。
df21 = pd.DataFrame([1,2,3,4,5])

# 实例-2
data = [['Alex',10],['Bob',12],['Clarke',13]]
df22 = pd.DataFrame(data,columns=['Name','Age'])

# 实例-3
data = [['Alex',10],['Bob',12],['Clarke',13]]
df23 = pd.DataFrame(data,columns=['Name','Age'],dtype=float)

#---------------- 从ndarrays/Lists的字典来创建DataFrame  

# 实例-1
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df31 = pd.DataFrame(data)

# 示例-2 使用数组创建一个索引的数据帧(DataFrame)
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df32 = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])

#---------------------- 从列表创建数据帧DataFrame

# 实例-1 以下示例显示如何通过传递字典列表来创建数据帧(DataFrame)。
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)

# 示例-2 以下示例显示如何通过传递字典列表和行索引来创建数据帧(DataFrame)。
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data, index=['first', 'second'])

# 实例-3  以下示例显示如何使用字典，行索引和列索引列表创建数据帧(DataFrame)。
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}] 
df1 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b']) 

#----------------------从系列的字典来创建DataFrame

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
```



- 列操作
  - 列选择
  - 列添加
  - 列删除

```python
import pandas as pd

#---------------- 选择列

# 下面将通过从数据帧(DataFrame)中选择一列
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)

#---------------- 列添加 
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)  
df['three']=pd.Series([10,20,30],index=['a','b','c'])
df['four']=df['one']+df['three']

#--------------- 列删除

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
     'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']), 
     'three' : pd.Series([10,20,30], index=['a','b','c'])}
df = pd.DataFrame(d) 
#==using del function 
del df['one']
#==using pop function 
df.pop('two')
```

- 行操作
  - 行选择
    - 行切片
  - 行添加
  - 行删除

```python
import pandas as pd

#--------------行选择

# 标签选择 
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
     'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)

# 按整数位置选择 
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)

##-------------- 行切片
# 可以使用:运算符选择多行 
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
    'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)

#-------------- 增加行

# 使用append()函数将新行添加到DataFrame
df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])
df = df.append(df2)

#-------------- 删除行

# 使用索引标签从DataFrame中删除或删除行。 如果标签重复，则会删除多行。
df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])
df = df.append(df2) 
df = df.drop(0)
```

###  <a name="1.3">1.3 Panel（面板）</a>

​	面板是具有异构数据的三维数据结构。在图形表示中很难表示面板。但是一个面板可以说明为`DataFrame`的容器。

##  <a name="2">2 Pandas操作</a>

###  <a name="2.1">2.1 基本操作</a>

- Series

| 编号 | 属性或方法 | 描述                                |
| ---- | ---------- | ----------------------------------- |
| 1    | `axes`     | 返回行轴标签列表。                  |
| 2    | `dtype`    | 返回对象的数据类型(`dtype`)。       |
| 3    | `empty`    | 如果系列为空，则返回`True`。        |
| 4    | `ndim`     | 返回底层数据的维数，默认定义：`1`。 |
| 5    | `size`     | 返回基础数据中的元素数。            |
| 6    | `values`   | 将系列作为`ndarray`返回。           |
| 7    | `head()`   | 返回前`n`行。                       |
| 8    | `tail()`   | 返回最后`n`行。                     |

```python
import pandas as pd
import numpy as np 
s = pd.Series(np.random.randn(4)) 

print(s.axes) # 返回行轴标签列表
print(s.empty) # 返回布尔值，表示对象是否为空
print(s.ndim) # 返回对象的维数。根据定义，一个系列是一个1D数据结构
print(s.size) # 回系列的大小(长度)
print(s.values) # 以数组形式返回系列中的实际数据值。
print(s.head(2)) # head()返回前n行
print(s.tail(2)) # tail()返回最后n行
```



- DataFrame

| 编号 | 属性或方法 | 描述                                                         |
| ---- | ---------- | ------------------------------------------------------------ |
| 1    | `T`        | 转置行和列。                                                 |
| 2    | `axes`     | 返回一个列，行轴标签和列轴标签作为唯一的成员。               |
| 3    | `dtypes`   | 返回此对象中的数据类型(`dtypes`)。                           |
| 4    | `empty`    | 如果`NDFrame`完全为空[无项目]，则返回为`True`; 如果任何轴的长度为`0`。 |
| 5    | `ndim`     | 轴/数组维度大小。                                            |
| 6    | `shape`    | 返回表示`DataFrame`的维度的元组。                            |
| 7    | `size`     | `NDFrame`中的元素数。                                        |
| 8    | `values`   | NDFrame的Numpy表示。                                         |
| 9    | `head()`   | 返回开头前`n`行。                                            |
| 10   | `tail()`   | 返回最后`n`行。                                              |

```python
import pandas as pd
import numpy as np 
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Minsu','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}
df = pd.DataFrame(d) 

print(df.T) # 返回DataFrame的转置。行和列将交换
print(df.axes) # 返回行轴标签和列轴标签列表
print(df.dtypes) # 返回每列的数据类型
print(df.empty) # 返回布尔值，表示对象是否为空; 返回True表示对象为空。
print(df.ndim) # 返回对象的维数
print(df.shape) # 返回表示DataFrame的维度的元组
print(df.size) # 返回DataFrame中的元素数
print(df.values) # 将DataFrame中的实际数据作为NDarray返回
print(df.head(2)) # head()返回前n行
print(df.tail(2)) # tail()返回最后n行
```



###  <a name="2.2">2.2 描述性统计</a>

- sum()  返回所请求轴的值的总和
- mean() 返回平均值
- std()      返回数字列的Bressel标准偏差
- describe() 函数是用来计算有关DataFrame列的统计信息的摘要。

下面来了解 Python Pandas中描述性统计信息的函数，下表列出了重要函数 

| 编号 | 函数        | 描述             |
| ---- | ----------- | ---------------- |
| 1    | `count()`   | 非空观测数量     |
| 2    | `sum()`     | 所有值之和       |
| 3    | `mean()`    | 所有值的平均值   |
| 4    | `median()`  | 所有值的中位数   |
| 5    | `mode()`    | 值的模值         |
| 6    | `std()`     | 值的标准偏差     |
| 7    | `min()`     | 所有值中的最小值 |
| 8    | `max()`     | 所有值中的最大值 |
| 9    | `abs()`     | 绝对值           |
| 10   | `prod()`    | 数组元素的乘积   |
| 11   | `cumsum()`  | 累计总和         |
| 12   | `cumprod()` | 累计乘积         |

###  <a name="2.3">2.3 函数应用</a>

- pipe(): 表合理函数应用

```python
import pandas as pd
import numpy as np

def adder(ele1, ele2, ele3): 
    return ele1+ele2+ele3
df1 = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
print("df1:\n", df1)
df2=df1.pipe(adder, 2, 3) 
print("df2:\n", df2)
"""
df1:
        col1      col2      col3
0 -0.170694  1.719652 -0.136084
1  0.025139  0.663993 -1.083569
2 -0.217043  0.465953 -0.076968
3  1.294821 -1.237323 -0.766703
4 -1.570477  0.502310 -0.540119
df2:
        col1      col2      col3
0  4.829306  6.719652  4.863916
1  5.025139  5.663993  3.916431
2  4.782957  5.465953  4.923032
3  6.294821  3.762677  4.233297
4  3.429523  5.502310  4.459881
"""

def adder(ele1): 
    return ele1+ele1
df1 = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
print("df1:\n", df1) 
df2=df1.pipe(adder)
print("df2:\n", df2)
"""
df1:
        col1      col2      col3
0  0.657631 -0.748022  0.479552
1  0.103155  0.828946  0.200918
2 -0.765607 -0.791743 -1.761628
3  1.822568  0.147871  0.718759
4 -0.600519  0.587664 -1.720643
df2:
        col1      col2      col3
0  1.315261 -1.496043  0.959104
1  0.206310  1.657892  0.401836
2 -1.531214 -1.583487 -3.523257
3  3.645136  0.295741  1.437518
4 -1.201038  1.175328 -3.441285
"""
```

- apply(): 行或列函数应用
  - 可以使用apply()方法沿DataFrame或Panel的轴应用任意函数，它与描述性统计方法一样，采用可选的axis参数。 默认情况下，操作按列执行，将每列列为数组。

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
print(df)
print(df.apply(np.mean))
"""
       col1      col2      col3
0  0.699171  1.174304  1.808056
1  0.371191  1.106797 -0.349805
2  0.906245 -1.044522 -0.450919
3  0.480870  0.870196  0.073125
4  0.381928 -1.508164 -0.384090

col1    0.567881
col2    0.119722
col3    0.139273
dtype: float64
"""

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
print(df)
print(df.apply(np.mean, axis=1))
"""
       col1      col2      col3
0 -0.072673  0.715110 -0.592150
1 -1.216105  0.098471 -0.920143
2 -1.195518  0.536300 -1.038083
3 -0.628611 -1.098576 -0.761081
4 -0.832026 -0.248135  0.139155
0    0.016763
1   -0.679259
2   -0.565767
3   -0.829423
4   -0.313669
dtype: float64
"""

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
print(df)
print(df.apply(lambda x: x.max() - x.min()))
"""
       col1      col2      col3
0  0.276372 -0.348530 -0.912841
1 -0.819007 -0.467740 -0.298918
2  1.607816  0.424580  0.158974
3  0.100884  0.873865  0.775902
4 -1.025005 -0.534617  0.892838
col1    2.632821
col2    1.408482
col3    1.805679
dtype: float64
"""
```



- applymap() : 元素函数应用
  - 并不是所有的函数都可以向量化(也不是返回另一个数组的NumPy数组，也不是任何值)，在DataFrame上的方法applymap()和类似于在Series上的map()接受任何Python函数，并且返回单个值。

```python
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3']) 

print("df:\n", df)
print("\nSeries map:\n", df['col1'].map(lambda x:x*100))
print("\ndataformat:\n", df.applymap(lambda x:x*100))

"""
df:
        col1      col2      col3
0 -0.423348 -0.926960  1.028099
1 -0.401481  0.805968 -0.657708
2 -1.802866  1.225432  2.084175
3  0.589757  0.269814 -1.294617
4  0.628785 -0.144890 -1.060085

Series map:
 0    -42.334849
1    -40.148116
2   -180.286637
3     58.975712
4     62.878452
Name: col1, dtype: float64

dataformat:
          col1        col2        col3
0  -42.334849  -92.695992  102.809885
1  -40.148116   80.596832  -65.770781
2 -180.286637  122.543225  208.417542
3   58.975712   26.981405 -129.461665
4   62.878452  -14.489015 -106.008526
"""
```

###  <a name="2.4">2.4 重建索引</a>

```python
import pandas as pd
import numpy as np 
N=20 
df = pd.DataFrame({
   'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
   'x': np.linspace(0,stop=N-1,num=N),
   'y': np.random.rand(N),
   'C': np.random.choice(['Low','Medium','High'],N).tolist(),
   'D': np.random.normal(100, 10, size=(N)).tolist()
}) 
print(df)
df_reindexed = df.reindex(index=[0,2,5], columns=['A', 'C', 'B'])
print (df_reindexed)

"""
            A     x         y       C           D
0  2016-01-01   0.0  0.442713    High  111.722146
1  2016-01-02   1.0  0.698932    High   90.258637
2  2016-01-03   2.0  0.064254     Low  107.429447
3  2016-01-04   3.0  0.556987     Low   80.436316
4  2016-01-05   4.0  0.879583  Medium   92.343476
5  2016-01-06   5.0  0.200288    High   85.114561
6  2016-01-07   6.0  0.349371    High  109.959658
7  2016-01-08   7.0  0.590062     Low   99.075823
8  2016-01-09   8.0  0.415122  Medium  101.204360
9  2016-01-10   9.0  0.594882  Medium   94.290754
10 2016-01-11  10.0  0.003399    High   99.368224
11 2016-01-12  11.0  0.936685     Low   82.517894
12 2016-01-13  12.0  0.282026    High   86.651163
13 2016-01-14  13.0  0.182096  Medium  104.525852
14 2016-01-15  14.0  0.761361     Low  105.588561
15 2016-01-16  15.0  0.252168    High   98.667607
16 2016-01-17  16.0  0.317600     Low   93.904366
17 2016-01-18  17.0  0.144913  Medium   97.380534
18 2016-01-19  18.0  0.329985  Medium   93.251733
19 2016-01-20  19.0  0.393104     Low   98.169141
           A     C   B
0 2016-01-01  High NaN
2 2016-01-03   Low NaN
5 2016-01-06  High NaN
"""
```



- 重建索引与其他对象对齐

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(10,3),columns=['col1','col2','col3'])
df2 = pd.DataFrame(np.random.randn(7,3),columns=['col1','col2','col3'])
print("df1.shape:", df1.shape)
print("df2.shape:", df2.shape)
df1 = df1.reindex_like(df2)
print(df1)
print("df1.shape:", df1.shape)

"""
df1.shape: (10, 3)
df2.shape: (7, 3)
       col1      col2      col3
0  1.130328  0.225797 -1.086386
1  0.163961 -1.302032  0.515123
2 -0.017108 -2.618230 -0.332424
3 -0.149488 -1.392738  0.176868
4  2.311437  0.820681  0.194800
5  0.863296 -0.016309 -0.264341
6 -0.775899  0.296823  0.465187
df1.shape: (7, 3)
"""
```



- 填充时重新加注

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(6,3),columns=['col1','col2','col3'])
df2 = pd.DataFrame(np.random.randn(2,3),columns=['col1','col2','col3'])
print(df2.reindex_like(df1)) 
print(df2.reindex_like(df1,method='ffill')) 

"""
       col1      col2      col3
0 -1.511791  0.189542 -0.643108
1  0.875902  0.290877 -0.698173
2       NaN       NaN       NaN
3       NaN       NaN       NaN
4       NaN       NaN       NaN
5       NaN       NaN       NaN
       col1      col2      col3
0 -1.511791  0.189542 -0.643108
1  0.875902  0.290877 -0.698173
2  0.875902  0.290877 -0.698173
3  0.875902  0.290877 -0.698173
4  0.875902  0.290877 -0.698173
5  0.875902  0.290877 -0.698173
"""
```



- 重建索引时的填充限制

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(6,3),columns=['col1','col2','col3'])
df2 = pd.DataFrame(np.random.randn(2,3),columns=['col1','col2','col3'])
# Padding NAN's
print(df2.reindex_like(df1)) 
print(df2.reindex_like(df1,method='ffill',limit=1))

"""
       col1      col2      col3
0  0.665770 -0.670541  1.823377
1  0.064197  0.401259 -1.160411
2       NaN       NaN       NaN
3       NaN       NaN       NaN
4       NaN       NaN       NaN
5       NaN       NaN       NaN
       col1      col2      col3
0  0.665770 -0.670541  1.823377
1  0.064197  0.401259 -1.160411
2  0.064197  0.401259 -1.160411
3  0.064197  0.401259 -1.160411
4  0.064197  0.401259 -1.160411
5  0.064197  0.401259 -1.160411
"""
```

- 重命名

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(6,3),columns=['col1','col2','col3'])
print(df1)

print ("After renaming the rows and columns:")
print(df1.rename(columns={'col1' : 'c1', 'col2' : 'c2'}, index = {0 : 'apple', 1 : 'banana', 2 : 'durian'}))

"""
       col1      col2      col3
0 -0.491026  0.232165 -0.029898
1 -0.237460  1.152158  1.978778
2  0.091988 -0.585323  0.338671
3 -1.133684  0.777426  2.194401
4 -0.448795 -0.983682 -0.151114
5  1.631571  0.904795  0.791048
After renaming the rows and columns:
              c1        c2      col3
apple  -0.491026  0.232165 -0.029898
banana -0.237460  1.152158  1.978778
durian  0.091988 -0.585323  0.338671
3      -1.133684  0.777426  2.194401
4      -0.448795 -0.983682 -0.151114
5       1.631571  0.904795  0.791048
"""
```

###  <a name="2.5">2.5 迭代</a>

- 迭代DataFrame
  - for in 迭代DataFrame提供列名 
  - for in  迭代DataFrame提供列名 
  - iteritems()  列  迭代(key，value)对
    - 将每个列作为键，将值与值作为键和列值迭代为Series对象。
  - iterrows()  行   将行迭代为(索引，系列)对
    - iterrows()返回迭代器，产生每个索引值以及包含每行数据的序列。
  - itertuples() 行 以namedtuples的形式迭代行
    - itertuples()方法将为DataFrame中的每一行返回一个产生一个命名元组的迭代器。元组的第一个元素将是行的相应索引值，而剩余的值是行值。

```python
# 迭代DataFrame提供列名

import pandas as pd
import numpy as np
N=20
df = pd.DataFrame({
    'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
    'x': np.linspace(0,stop=N-1,num=N),
    'y': np.random.rand(N),
    'C': np.random.choice(['Low','Medium','High'],N).tolist(),
    'D': np.random.normal(100, 10, size=(N)).tolist()
    })
print(df)
for col in df:
   print(col) 
    
"""
            A     x         y       C           D
0  2016-01-01   0.0  0.577244    High  109.975602
1  2016-01-02   1.0  0.665014  Medium  113.394454
2  2016-01-03   2.0  0.982564    High   94.549445
3  2016-01-04   3.0  0.851160  Medium  100.231794
4  2016-01-05   4.0  0.538774    High   87.453212
5  2016-01-06   5.0  0.844544     Low  107.448078
6  2016-01-07   6.0  0.820369    High  115.640149
7  2016-01-08   7.0  0.277820  Medium   99.226653
8  2016-01-09   8.0  0.077948  Medium  111.503528
9  2016-01-10   9.0  0.497468     Low   85.159039
10 2016-01-11  10.0  0.223452    High  120.865425
11 2016-01-12  11.0  0.725847  Medium   83.862872
12 2016-01-13  12.0  0.358913  Medium   92.891491
13 2016-01-14  13.0  0.672671    High   92.261664
14 2016-01-15  14.0  0.891195    High   96.139938
15 2016-01-16  15.0  0.318164  Medium  103.441272
16 2016-01-17  16.0  0.872782  Medium  101.092868
17 2016-01-18  17.0  0.367514     Low   93.585234
18 2016-01-19  18.0  0.993005     Low   99.181664
19 2016-01-20  19.0  0.969207  Medium  102.383043
A
x
y
C
D
"""
```

```python
#------------- iteritems()示例
"""
将每个列作为键，将值与值作为键和列值迭代为Series对象。
"""
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(4,3),columns=['col1','col2','col3'])
for key,value in df.iteritems():
   print (key, value)
"""
col1 0    0.851255
    1    0.965122
    2    0.393984
    3    0.607448
Name: col1, dtype: float64
col2 0   -0.325713
    1    0.535768
    2   -0.158361
    3    2.036761
Name: col2, dtype: float64
col3 0    0.363819
    1    0.732609
    2   -0.788404
    3    0.118169
Name: col3, dtype: float64
"""
```

```python
#---------------- iterrows()示例
"""
iterrows()返回迭代器，产生每个索引值以及包含每行数据的序列。
"""
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(4,3),columns = ['col1','col2','col3'])
for row_index,row in df.iterrows():
   print (row_index,row) 
    
"""
0 col1   -2.141848
  col2   -1.080591
  col3    0.016402
Name: 0, dtype: float64
1 col1    0.562631
  col2    0.782613
  col3    0.312021
Name: 1, dtype: float64
2 col1    0.214056
  col2   -0.941972
  col3    0.249979
Name: 2, dtype: float64
3 col1   -0.126826
  col2    0.811581
  col3   -0.274224
Name: 3, dtype: float64
"""
```

```python
# itertuples()示例
"""
itertuples()方法将为DataFrame中的每一行返回一个产生一个命名元组的迭代器。
元组的第一个元素将是行的相应索引值，而剩余的值是行值。 
"""
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(4,3),columns = ['col1','col2','col3'])
for row in df.itertuples():
    print (row) 
    
"""
Pandas(Index=0, col1=0.5025492795882459, col2=-0.5298381707299722, col3=1.3939868813609084)
Pandas(Index=1, col1=-0.9956693819463067, col2=0.4979543732257999, col3=-1.0555851162850065)
Pandas(Index=2, col1=-1.476966722137543, col2=-0.20229783735299992, col3=-0.4034280948728957)
Pandas(Index=3, col1=-0.5497550150207252, col2=-0.0924983193288906, col3=1.4724032859606115)
"""
```

###  <a name="2.6">2.6 排序</a>

- 按标签排序

```python
#---------------- 按标签排序
"""
使用sort_index()方法，通过传递axis参数和排序顺序，可以对DataFrame进行排序。
默认情况下，按照升序对行标签进行排序。 
"""
import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame(np.random.randn(5,2),index=[5,9,8,0,7],columns = ['col2','col1'])
sorted_df=unsorted_df.sort_index()
print (sorted_df) 
"""
       col2      col1
0  1.263412 -0.090153
5  0.435609 -1.312084
7 -0.043339 -0.299793
8 -1.220893  1.063311
9  0.377686 -1.051608
"""
```

- 排序顺序

```python
#------------------排序顺序

"""
通过将布尔值传递给升序参数，可以控制排序顺序。 来看看下面的例子来理解一下。
"""

import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame(np.random.randn(5,2),index=[5,9,8,0,7],columns = ['col2','col1'])
sorted_df = unsorted_df.sort_index(ascending=False)
print (sorted_df)
""" 
       col2      col1
9  0.072576 -0.454951
8 -0.078238  0.889726
7  0.128667 -1.092788
5 -0.699209 -0.470586
0 -0.424473  0.118444
"""
```

- 按列排列

```python
#--------------- 按列排列
"""
通过传递axis参数值为0或1，可以对列标签进行排序。 
默认情况下，axis = 0，逐行排列。
"""
import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame(np.random.randn(5,2),index=[5,9,8,0,7],columns = ['col2','col1'])
sorted_df=unsorted_df.sort_index(axis=1)
print (sorted_df) 
"""
       col1      col2
5  0.817850  1.021765
9  0.767597  1.350208
8  0.866164  0.717240
0 -1.427572 -1.072589
7 -0.274216  0.968537
"""
```



- 按值排序

```python
#---------------------- 按值排序

"""
像索引排序一样，sort_values()是按值排序的方法。
它接受一个by参数，它将使用要与其排序值的DataFrame的列名称。 
"""
import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
sorted_df = unsorted_df.sort_values(by='col1')
print (sorted_df) 
"""
   col1  col2
1     1     3
2     1     2
3     1     4
0     2     1
"""

sorted_df2 = unsorted_df.sort_values(by=['col1','col2'])
print (sorted_df2) 
"""
   col1  col2
2     1     2
1     1     3
3     1     4
0     2     1
"""
```



- 排序算法

```python
#-------------------- 排序算法

"""
sort_values()提供了从mergeesort，heapsort和quicksort中选择算法的一个配置。
Mergesort是唯一稳定的算法。 
"""

import pandas as pd
import numpy as np
unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
sorted_df = unsorted_df.sort_values(by='col1' ,kind='mergesort')
print (sorted_df)
"""
   col1  col2
1     1     3
2     1     2
3     1     4
0     2     1
""" 
```



###  <a name="2.7">2.7 索引和选择数据</a>

| 编号 | 索引      | 描述     |
| ---- | --------- | -------- |
| 1    | `.loc()`  | 基于标签 |
| 2    | `.iloc()` | 基于整数 |



```python
#------------- loc需要两个单/列表/范围运算符，用","分隔。第一个表示行，第二个表示列。 

"""
Pandas提供了各种方法来完成基于标签的索引。 切片时，也包括起始边界。整数是有效的标签，但它们是指标签而不是位置。 .loc()具有多种访问方式，如
"""
 
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(8, 4),
index = ['a','b','c','d','e','f','g','h'], columns = ['A', 'B', 'C', 'D']) 

print (df.loc[:,'A'])   # 只去出  'A' 这一列  以及 所有的行
print (df.loc[:,['A','C']])  # 取出所有的行 以及 'A' 'C' 列
print (df.loc[['a','b','f','h'],['A','C']])   # 取出a,b,f,h行 以及 A,C列
print (df.loc['a':'g'])  # 取出a~g的行 以及所有的列
print (df.loc['a']>0)  # 获取行a是否大于0 (布尔值列表)
```



```python
#----------------- .iloc()

"""
Pandas提供了各种方法，以获得纯整数索引。像python和numpy一样，第一个位置是基于0的索引。 各种访问方式如下
- 整数
- 整数列表
- 系列值
"""
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D']) 

print (df.iloc[:4])  # [0, 4）的行 以及所有的列
print (df.iloc[1:5, 2:4]) # [1, 5)的行 以及 [2, 4)的列
print (df.iloc[[1, 3, 5], [1, 3]]) # 第1，3，5行  1，3列
print (df.iloc[1:3, :]) # [1, 3)行 以及所有列
print (df.iloc[:,1:3])  # 所有行，[1, 3)列
```



- 直接访问

```python
#--------------------- 直接访问

import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])
print(df)
print (df['A'])  # 直接返回 A 的数据
print (df[['A','B']]) # 返回A、B 列的数据
print (df[0:2]) #返回0~2行
```



- 属性访问  
  - 可以使用属性运算符.来选择列

```python
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])
print(df.A) 
```



###  <a name="2.8">2.8 缺失值</a>

- 检查缺失值

```python
"""
为了更容易地检测缺失值(以及跨越不同的数组dtype)，Pandas提供了isnull()和notnull()函数，它们也是Series和DataFrame对象的方法
"""

import pandas as pd
import numpy as np 
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three']) 
df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) 
print (df['one'].isnull())  # 返回一个列表(布尔值的列表)
print (df['one'].notnull()) 
```



- 缺少数据的计算

```python
"""
在求和数据时，NA将被视为0
"""

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])
df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print (df['one'].sum())  
```



- 清理/填充缺少数据
  - 用标量值替换NaN
  - 填写NA前进和后退
  - 丢失缺少的值

```python
"""
Pandas提供了各种方法来清除缺失的值。fillna()函数可以通过几种方法用非空数据“填充”NA值，在下面的章节中将学习和使用。 
- 用标量值替换NaN
- 填写NA前进和后退
- 丢掉缺少的值
"""

#--------------------- 用标量值替换NaN

"""
以下程序显示如何用0替换NaN
"""
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(3, 3), index=['a', 'c', 'e'],columns=['one',
'two', 'three'])
df = df.reindex(['a', 'b', 'c']) 
print (df.fillna(0))  # 用0来填充缺失值

#------------------ 填写NA前进和后退
"""
使用重构索引章节讨论的填充概念，来填补缺失的值。

- pad/fill        填充方法向前
- bfill/backfill  填充方法向后
"""

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])
df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print (df.fillna(method='pad'))  # 填充前向
print (df.fillna(method='backfill')) # 填充后向

#-------------------- 丢失缺少的值
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])
df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print (df.dropna())  # 丢弃掉有缺失值的行
print (df.dropna(axis=1)) # 丢弃掉有缺失值的列
```



- 替换通用值

```python
#------------------ 替换通用值
import pandas as pd
import numpy as np
df = pd.DataFrame({'one':[10,20,30,40,50,2000],
'two':[1000,0,30,40,50,60]})
print(df)
print (df.replace({1000:10,2000:60})) 
```



###  <a name="2.9">2.9 分组(GroupBy)</a>

- 拆分

```python
"""
将数据拆分成组

Pandas对象可以分成任何对象。有多种方式来拆分对象，如 -
- obj.groupby(‘key’)
- obj.groupby([‘key1’,’key2’])
- obj.groupby(key,axis=1)
"""
import pandas as pd
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data) 
group_team = df.groupby('Team')
print(group_team) 
print(group_team.groups) 
# 按照多个列进行分组 
print (df.groupby(['Team','Year']).groups) 
```

- 迭代遍历分组

```python
#------------ 迭代遍历分组
import pandas as pd
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
grouped = df.groupby('Year')
print(df)
for name,group in grouped:
    print (name)
    print (group) 
```



- 选择一个分组

```python
"""
使用get_group()方法，可以选择一个组
"""

import pandas as pd
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
grouped = df.groupby('Year')
print (grouped.get_group(2014)) 
```



- 聚合

```python
"""
聚合函数为每个组返回单个聚合值。当创建了分组(group by)对象，就可以对分组数据执行多个聚合操作。 一个比较常用的是通过聚合或等效的agg方法聚合
一次应用多个聚合函数
通过分组系列，还可以传递函数的列表或字典来进行聚合，并生成DataFrame作为输出
"""
import pandas as pd
import numpy as np

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
grouped = df.groupby('Year')
print (grouped['Points'].agg(np.mean)) 
print (grouped.agg(np.size)) # 另一种查看每个分组的大小的方法是应用size()函数

"""
一次应用多个聚合函数
通过分组系列，还可以传递函数的列表或字典来进行聚合，并生成DataFrame作为输出
"""
agg = grouped['Points'].agg([np.sum, np.mean, np.std])
print (agg) 
agg = grouped.agg([np.sum, np.mean, np.std])
print (agg) 
```



- 转换

```python
"""
分组或列上的转换返回索引大小与被分组的索引相同的对象。因此，转换应该返回与组块大小相同的结果。
"""
import pandas as pd
import numpy as np

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)

grouped = df.groupby('Team')
score = lambda x: (x - x.mean()) / x.std()*10
print(grouped.transform(score)) 
```



- 过滤

```python
"""
过滤根据定义的标准过滤数据并返回数据的子集。filter()函数用于过滤数据。
"""
import pandas as pd
import numpy as np
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
filter = df.groupby('Team').filter(lambda x: len(x) >= 3) # 在上述过滤条件下，要求返回三次以上参加IPL的队伍。
print (filter) 
```



###  <a name="2.10">2.10并和/连接</a>

```python
"""
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
left_index=False, right_index=False, sort=True) 

- left - 一个DataFrame对象。
- right - 另一个DataFrame对象。
- on - 列(名称)连接，必须在左和右DataFrame对象中存在(找到)。
- left_on - 左侧DataFrame中的列用作键，可以是列名或长度等于DataFrame长度的数组。
- right_on - 来自右的DataFrame的列作为键，可以是列名或长度等于DataFrame长度的数组。
- left_index - 如果为True，则使用左侧DataFrame中的索引(行标签)作为其连接键。 在具有MultiIndex(分层)的DataFrame的情况下，级别的数量必须与来自右DataFrame的连接键的数量相匹配。
- right_index - 与右DataFrame的left_index具有相同的用法。
- how - 它是left, right, outer以及inner之中的一个，默认为内inner。 下面将介绍每种方法的用法。
- sort - 按照字典顺序通过连接键对结果DataFrame进行排序。默认为True，设置为False时，在很多情况下大大提高性能。
"""
```



- 并和数据
  - 在一个键上合并两个数据帧
  - 合并多个键上的两个数据框

```python
import pandas as pd
left = pd.DataFrame({
         'id':[1,2,3,4,5],
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame(
         {'id':[1,2,3,4,5],
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5']})
print (left)
print("========================================")
print (right) 
#-------------- 在一个键上合并两个数据帧
rs = pd.merge(left, right,on='id')
print(rs) 
# ---------  合并多个键上的两个数据框
rs = pd.merge(left,right,on=['id', 'subject_id'])
print(rs) 
"""
   id    Name subject_id
0   1    Alex       sub1
1   2     Amy       sub2
2   3   Allen       sub4
3   4   Alice       sub6
4   5  Ayoung       sub5
========================================
   id   Name subject_id
0   1  Billy       sub2
1   2  Brian       sub4
2   3   Bran       sub3
3   4  Bryce       sub6
4   5  Betty       sub5

#-------------- 在一个键上合并两个数据帧 ('id')
   id  Name_x subject_id_x Name_y subject_id_y
0   1    Alex         sub1  Billy         sub2
1   2     Amy         sub2  Brian         sub4
2   3   Allen         sub4   Bran         sub3
3   4   Alice         sub6  Bryce         sub6
4   5  Ayoung         sub5  Betty         sub5

# ---------  合并多个键上的两个数据框 (['id', 'subject_id'])
   id  Name_x subject_id Name_y
0   4   Alice       sub6  Bryce
1   5  Ayoung       sub5  Betty
"""
```



- 合并使用“how”的参数
  - Left Join示例
  - Right Join示例
  - Outer Join示例
  - Inner Join示例	

| 合并方法 | SQL等效            | 描述             |
| -------- | ------------------ | ---------------- |
| `left`   | `LEFT OUTER JOIN`  | 使用左侧对象的键 |
| `right`  | `RIGHT OUTER JOIN` | 使用右侧对象的键 |
| `outer`  | `FULL OUTER JOIN`  | 使用键的联合     |
| `inner`  | `INNER JOIN`       | 使用键的交集     |

```python
import pandas as pd
left = pd.DataFrame({
         'id':[1,2,3,4,5],
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame(
         {'id':[1,2,3,4,5],
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5']})

#-------------- Left Join示例
rs = pd.merge(left, right, on='subject_id', how='left')
print (rs) 

#-------------- Right Join示例
rs = pd.merge(left, right, on='subject_id', how='right')
print (rs) 

#----------------- Outer Join示例
rs = pd.merge(left, right, how='outer', on='subject_id')
print (rs)

#----------------- Inner Join示例
rs = pd.merge(left, right, on='subject_id', how='inner')
print (rs) 

"""
#-------------- Left Join示例
   id_x  Name_x subject_id  id_y Name_y
0     1    Alex       sub1   NaN    NaN
1     2     Amy       sub2   1.0  Billy
2     3   Allen       sub4   2.0  Brian
3     4   Alice       sub6   4.0  Bryce
4     5  Ayoung       sub5   5.0  Betty

#-------------- Right Join示例
   id_x  Name_x subject_id  id_y Name_y
0   2.0     Amy       sub2     1  Billy
1   3.0   Allen       sub4     2  Brian
2   4.0   Alice       sub6     4  Bryce
3   5.0  Ayoung       sub5     5  Betty
4   NaN     NaN       sub3     3   Bran

#----------------- Outer Join示例
   id_x  Name_x subject_id  id_y Name_y
0   1.0    Alex       sub1   NaN    NaN
1   2.0     Amy       sub2   1.0  Billy
2   3.0   Allen       sub4   2.0  Brian
3   4.0   Alice       sub6   4.0  Bryce
4   5.0  Ayoung       sub5   5.0  Betty
5   NaN     NaN       sub3   3.0   Bran

#----------------- Inner Join示例
   id_x  Name_x subject_id  id_y Name_y
0     2     Amy       sub2     1  Billy
1     3   Allen       sub4     2  Brian
2     4   Alice       sub6     4  Bryce
3     5  Ayoung       sub5     5  Betty
"""
```



###  <a name="2.11">2.11 级联</a>

​	**Pandas**提供了各种工具(功能)，可以轻松地将`Series`，`DataFrame`和`Panel`对象组合在一起

```python
"""
pd.concat(objs,axis=0,join='outer',join_axes=None, ignore_index=False)
- objs - 这是Series，DataFrame或Panel对象的序列或映射。
- axis - {0，1，...}，默认为0，这是连接的轴。
- join - {'inner', 'outer'}，默认inner。如何处理其他轴上的索引。联合的外部和交叉的内部。
- ignore_index − 布尔值，默认为False。如果指定为True，则不要使用连接轴上的索引值。结果轴将被标记为：0，...，n-1。
- join_axes - 这是Index对象的列表。用于其他(n-1)轴的特定索引，而不是执行内部/外部集逻辑。
"""
```



- 连接对象(concat)

```python
#----------------- 连接对象

import pandas as pd
one = pd.DataFrame({
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5'],
         'Marks_scored':[98,90,87,69,78]},
         index=[1,2,3,4,5])
two = pd.DataFrame({
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5'],
         'Marks_scored':[89,80,79,97,88]},
         index=[1,2,3,4,5])
rs = pd.concat([one,two])
"""
print(rs) 

     Name subject_id  Marks_scored
1    Alex       sub1            98
2     Amy       sub2            90
3   Allen       sub4            87
4   Alice       sub6            69
5  Ayoung       sub5            78
1   Billy       sub2            89
2   Brian       sub4            80
3    Bran       sub3            79
4   Bryce       sub6            97
5   Betty       sub5            88
"""

#  假设想把特定的键与每个碎片的DataFrame关联起来。可以通过使用键参数来实现这一点 
rs = pd.concat([one,two],keys=['x','y'])
"""
print(rs) 

       Name subject_id  Marks_scored
x 1    Alex       sub1            98
  2     Amy       sub2            90
  3   Allen       sub4            87
  4   Alice       sub6            69
  5  Ayoung       sub5            78
y 1   Billy       sub2            89
  2   Brian       sub4            80
  3    Bran       sub3            79
  4   Bryce       sub6            97
  5   Betty       sub5            88
"""

# 结果的索引是重复的; 每个索引重复。如果想要生成的对象必须遵循自己的索引，请将ignore_index设置为True。
rs = pd.concat([one,two],keys=['x','y'],ignore_index=True)
"""
print(rs) 

     Name subject_id  Marks_scored
0    Alex       sub1            98
1     Amy       sub2            90
2   Allen       sub4            87
3   Alice       sub6            69
4  Ayoung       sub5            78
5   Billy       sub2            89
6   Brian       sub4            80
7    Bran       sub3            79
8   Bryce       sub6            97
9   Betty       sub5            88
"""

# 观察，索引完全改变，键也被覆盖。如果需要沿axis=1添加两个对象，则会添加新列。
rs = pd.concat([one,two],axis=1)
"""
print(rs) 

     Name subject_id  Marks_scored   Name subject_id  Marks_scored
1    Alex       sub1            98  Billy       sub2            89
2     Amy       sub2            90  Brian       sub4            80
3   Allen       sub4            87   Bran       sub3            79
4   Alice       sub6            69  Bryce       sub6            97
5  Ayoung       sub5            78  Betty       sub5            88
""" 
```



- 使用附加连接(append)

```python
#---------------- 使用附加连接
"""
连接的一个有用的快捷方式是在Series和DataFrame实例的append方法。
这些方法实际上早于concat()方法。 它们沿axis=0连接，即索引 
"""
import pandas as pd
one = pd.DataFrame({
         'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
         'subject_id':['sub1','sub2','sub4','sub6','sub5'],
         'Marks_scored':[98,90,87,69,78]},
         index=[1,2,3,4,5])
two = pd.DataFrame({
         'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
         'subject_id':['sub2','sub4','sub3','sub6','sub5'],
         'Marks_scored':[89,80,79,97,88]},
         index=[1,2,3,4,5])
rs = one.append(two)
"""
print(rs) 

     Name subject_id  Marks_scored
1    Alex       sub1            98
2     Amy       sub2            90
3   Allen       sub4            87
4   Alice       sub6            69
5  Ayoung       sub5            78
1   Billy       sub2            89
2   Brian       sub4            80
3    Bran       sub3            79
4   Bryce       sub6            97
5   Betty       sub5            88
"""

# append()函数也可以带多个对象 
rs = one.append([two,one,two])
"""
print(rs)

     Name subject_id  Marks_scored
1    Alex       sub1            98
2     Amy       sub2            90
3   Allen       sub4            87
4   Alice       sub6            69
5  Ayoung       sub5            78
1   Billy       sub2            89
2   Brian       sub4            80
3    Bran       sub3            79
4   Bryce       sub6            97
5   Betty       sub5            88
1    Alex       sub1            98
2     Amy       sub2            90
3   Allen       sub4            87
4   Alice       sub6            69
5  Ayoung       sub5            78
1   Billy       sub2            89
2   Brian       sub4            80
3    Bran       sub3            79
4   Bryce       sub6            97
5   Betty       sub5            88
""" 
```



###  <a name="2.12">2.12 日期功能</a>

- 创建一个日期范围

```python
import pandas as pd
datelist = pd.date_range('2020/11/21', periods=5)
print(datelist) 
"""
DatetimeIndex(['2020-11-21', '2020-11-22', '2020-11-23', '2020-11-24',
               '2020-11-25'],
              dtype='datetime64[ns]', freq='D')
"""
```



- 更改日期频率

```python
import pandas as pd
datelist = pd.date_range('2020/11/21', periods=5,freq='M')
print(datelist) 
"""
DatetimeIndex(['2020-11-30', '2020-12-31', '2021-01-31', '2021-02-28',
               '2021-03-31'],
              dtype='datetime64[ns]', freq='M')
"""
```



- bdate_range()函数

```python
"""
bdate_range()用来表示商业日期范围，不同于date_range()，它不包括星期六和星期天。
"""
import pandas as pd
datelist = pd.bdate_range('2019/09/27', periods=5)
print(datelist) # 2019-09-28  2019-09-29 是周末 所以没有打印出来
"""
DatetimeIndex(['2019-09-27', '2019-09-30', '2019-10-01', '2019-10-02',
               '2019-10-03'],
              dtype='datetime64[ns]', freq='B')
"""
```



- 偏移别名

大量的字符串别名被赋予常用的时间序列频率。我们把这些别名称为偏移别名。

| 别名     | 描述说明         |
| -------- | ---------------- |
| `B`      | 工作日频率       |
| `BQS`    | 商务季度开始频率 |
| `D`      | 日历/自然日频率  |
| `A`      | 年度(年)结束频率 |
| `W`      | 每周频率         |
| `BA`     | 商务年底结束     |
| `M`      | 月结束频率       |
| `BAS`    | 商务年度开始频率 |
| `SM`     | 半月结束频率     |
| `BH`     | 商务时间频率     |
| `SM`     | 半月结束频率     |
| `BH`     | 商务时间频率     |
| `BM`     | 商务月结束频率   |
| `H`      | 小时频率         |
| `MS`     | 月起始频率       |
| `T, min` | 分钟的频率       |
| `SMS`    | SMS半开始频率    |
| `S`      | 秒频率           |
| `BMS`    | 商务月开始频率   |
| `L, ms`  | 毫秒             |
| `Q`      | 季度结束频率     |
| `U, us`  | 微秒             |
| `BQ`     | 商务季度结束频率 |
| `N`      | 纳秒             |
| `BQ`     | 商务季度结束频率 |
| `QS`     | 季度开始频率     |

### <a name="2.13">2.13 时间差 </a>

​	时间差(Timedelta)是时间上的差异，以不同的单位来表示。例如：日，小时，分钟，秒。它们可以是正值，也可以是负值。

- 时间差创建
  - 字符串
  - 整数
  - 数据偏移 

```python
#-------------  字符串
"""
通过传递字符串，可以创建一个timedelta对象。
"""

import pandas as pd
timediff = pd.Timedelta('2 days 2 hours 15 minutes 30 seconds')
"""
print(timediff) 

2 days 02:15:30
"""

#--------------- 整数
"""
通过传递一个整数值与指定单位，这样的一个参数也可以用来创建Timedelta对象。
"""
import pandas as pd
timediff = pd.Timedelta(6,unit='h')
"""
print(timediff) 

0 days 06:00:00
"""

#--------------- 数据偏移
"""
例如 - 周，天，小时，分钟，秒，毫秒，微秒，纳秒的数据偏移也可用于构建。
"""
import pandas as pd
timediff = pd.Timedelta(days=2)
"""
print(timediff) 

2 days 00:00:00
""" 
```



- 运算操作
  - 相加操作
  - 相减操作

```python
import pandas as pd

s = pd.Series(pd.date_range('2018-1-1', periods=3, freq='D'))
td = pd.Series([ pd.Timedelta(days=i) for i in range(3) ])
df = pd.DataFrame(dict(A = s, B = td))
"""
print(df) 

           A      B
0 2018-01-01 0 days
1 2018-01-02 1 days
2 2018-01-03 2 days
"""

#--------------------- 相加操作
df['C']=df['A']+df['B']
"""
print(df) 

           A      B          C
0 2018-01-01 0 days 2018-01-01
1 2018-01-02 1 days 2018-01-03
2 2018-01-03 2 days 2018-01-05
"""

#--------------------- 相减操作
df['D']=df['C']-df['B']
"""
print(df) 

0 2018-01-01 0 days 2018-01-01 2018-01-01
1 2018-01-02 1 days 2018-01-03 2018-01-02
2 2018-01-03 2 days 2018-01-05 2018-01-03
""" 
```

