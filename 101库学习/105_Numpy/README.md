<font size=15>**Numpy**</font>





主要参考网站：[易百教程-
Numpy教程](https://www.yiibai.com/numpy)



<font size = 5>**目录**</font>



- [1
Numpy数据结构](#1)
  - [1.1 Ndarray ](#1.1)
- [2 Numpy 操作](#2)
  - [2.1 创建数组](#2.1)
- [2.2 索引与切片](#2.2)
  - [2.3 广播](#2.3)
  - [2.4 迭代](#2.4)
  - [2.5 数组操作](#2.5)
- [2.6函数](#2.6)
    - [2.6.1 数学函数](#2.6.1)
    - [2.6.2 算术运算](#2.6.2)
    -
[2.6.3 统计函数](#2.6.3)
  - [2.7 排序搜索和计数 ](#2.7)
  - [2.8 线性代数](#2.8)



NumPy 是一个
Python 包。 它代表 “Numeric Python”。 它是一个由多维数组对象和用于处理数组的例程集合组成的库。

**Numeric**，即
NumPy 的前身，是由 Jim Hugunin 开发的。 也开发了另一个包 Numarray ，它拥有一些额外的功能。 2005年，Travis
Oliphant 通过将 Numarray 的功能集成到 Numeric 包中来创建 NumPy 包。 这个开源项目有很多贡献者。



# <a
name="1">1 Numpy数据结构</a>

## <a name="1.1">1.1 Ndarray</a>

​	NumPy
中定义的最重要的对象是称为 `ndarray`的 N 维数组类型。 它描述相同类型的元素集合。 可以使用基于零的索引访问集合中的项目。
`ndarray`中的每个元素在内存中使用相同大小的块。 `ndarray`中的每个元素是数据类型对象的对象(称为 `dtype`)。 

# <a
name="2">2 Numpy 操作</a>

Numpy数组的属性

|       属性       |
说明                             |
| :--------------: |
:----------------------------------------------------------: |
|  ndarray.shape
| 这一数组属性返回一个包含数组维度的元组，它也可以用于调整数组大小 |
|   ndarray.ndim   |
这一数组属性返回数组的维数。                 |
| ndarray.itemsize |
这一数组属性返回数组中每个元素的字节单位长度。        |

## <a name="2.1">2.1 创建数组</a>

### 2.1.1 空数组
- empty 空数组
- zeros 以0为填充的数组
- ones 以1为填充的数组

```{.python .input}
#---------------------- empty 空数组
"""
numpy.empty(shape, dtype = float, order = 'C')

- Shape 空数组的形状，整数或整数元组
- Dtype 所需的输出数组类型，可选
- Order 'C'为按行的 C 风格数组，'F'为按列的 Fortran 风格数组 
"""
import numpy as np 
x = np.empty([3,2], dtype =  int)  # 创建 3x2 的二维数组

#--------------------- numpy.zeros
"""
返回特定大小，以 0 填充的新数组。

numpy.zeros(shape, dtype = float, order = 'C')

- Shape 空数组的形状，整数或整数元组
- Dtype 所需的输出数组类型，可选
- Order 'C'为按行的 C 风格数组，'F'为按列的 Fortran 风格数组 
"""
x = np.zeros(5)  #  [0. 0. 0. 0. 0.]
x = np.zeros((5,), dtype = np.int) # [0 0 0 0 0] 
x = np.zeros((2,2), dtype =  [('x',  'i4'),  ('y',  'i4')])   # 2x2的0填充的数组

#----------------------- numpy.ones
"""
返回特定大小，以 1 填充的新数组。

numpy.ones(shape, dtype = None, order = 'C')

- Shape 空数组的形状，整数或整数元组
- Dtype 所需的输出数组类型，可选
- Order 'C'为按行的 C 风格数组，'F'为按列的 Fortran 风格数组 
"""
x = np.ones(5) # [1. 1. 1. 1. 1.]

import numpy as np 
x = np.ones([2,2], dtype =  int)  # 2x2以1填充的数组
```

### 2.1.2 现有数据的数组

- asarray （列表、列表的元组、元组、元组的元组、元组的列表）

```{.python .input}
#--------------------- asarray
"""
此函数类似于numpy.array，除了它有较少的参数。 这个例程对于将 Python 序列转换为ndarray非常有用。 

numpy.asarray(a, dtype = None, order = None)

- a 任意形式的输入参数，比如列表、列表的元组、元组、元组的元组、元组的列表
- dtype 通常，输入数据的类型会应用到返回的ndarray
- order 'C'为按行的 C 风格数组，'F'为按列的 Fortran 风格数组
"""
import numpy as np 
print(np.asarray([1, 2, 3])) # 把列表转化ndarray
print(np.asarray([1, 2, 3], dtype=float)) # 把列表转化ndarray (类型转化为float)
print(np.asarray((1, 2, 3))) # 把元组转化ndarray
print(np.asarray([(1,2,3),(4,5)] )) # 把元组列表转化为ndarray
```

### 2.1.3 来自数据范围

- arange
- linspace
- logspace

```{.python .input}
#--------------------- arange
"""
这个函数返回ndarray对象，包含给定范围内的等间隔值。

numpy.arange(start, stop, step, dtype)

- start 范围的起始值，默认为0
- stop 范围的终止值(不包含)
- step 两个值的间隔，默认为1
- dtype 返回ndarray的数据类型，如果没有提供，则会使用输入数据的类型。 
"""
import numpy as np

print(np.arange(5))  # [0 1 2 3 4]
print(np.arange(5, dtype=float)) # [0. 1. 2. 3. 4.]
print(np.arange(10,20,2)) # [10 12 14 16 18]

#------------------------ linspace
"""
此函数类似于arange()函数。 在此函数中，指定了范围之间的均匀间隔数量，而不是步长。 此函数的用法如下。

numpy.linspace(start, stop, num, endpoint, retstep, dtype)
- start 序列的起始值
- stop 序列的终止值，如果endpoint为true，该值包含于序列中
- num 要生成的等间隔样例数量，默认为50
- endpoint 序列中是否包含stop值，默认为ture
- retstep 如果为true，返回样例，以及连续数字之间的步长
= dtype 输出ndarray的数据类型 
"""
print(np.linspace(10,20,5)) # [10.  12.5 15.  17.5 20. ]
print(np.linspace(10,20,  5, endpoint =  False) ) # [10. 12. 14. 16. 18.]
print(np.linspace(1,2,5, retstep =  True) ) # (array([1.  , 1.25, 1.5 , 1.75, 2.  ]), 0.25)

#---------------------------- logspace
"""
此函数返回一个ndarray对象，其中包含在对数刻度上均匀分布的数字。 刻度的开始和结束端点是某个底数的幂，通常为 10。

numpy.logscale(start, stop, num, endpoint, base, dtype)

- start 起始值是base ** start
- stop 终止值是base ** stop
- num 范围内的数值数量，默认为50
- endpoint 如果为true，终止值包含在输出数组当中
- base 对数空间的底数，默认为10
- dtype 输出数组的数据类型，如果没有提供，则取决于其它参数 
"""
print(np.logspace(1.0,  2.0, num =  5)) # [ 10. 17.7827941 31.6227766   56.23413252 100. ]
print(np.logspace(1,10,num =  5,  base  =  2) ) # [2. 9.51365692   45.254834    215.2694823  1024.        ]
```

### 2.1.4 random

- rand(d0, d1, ..., dn) 生成一个0~1之间的随机浮点数或N维浮点数组。
- randn(d0,
d1, ..., dn)：生成一个浮点数或N维浮点数组，取数范围：正态分布的随机样本数。
-
standard_normal(size=None)：生产一个浮点数或N维浮点数组，取数范围：标准正态分布随机样本
- randint(low,
high=None, size=None,
dtype='l')：生成一个整数或N维整数数组，取数范围：若high不为None时，取low,high之间随机整数，否则取值0,low之间随机整数。 
-
random_sample(size=None)：生成一个0,1之间随机浮点数或N维浮点数组。
- choice(a, size=None,
replace=True, p=None)：从序列中获取元素，若a为整数，元素取值为np.range(a)中随机数；若a为数组，取值为a数组元素中随机元素。
-
shuffle(x)：对X进行重排序，如果X为多维数组，只沿第一条轴洗牌，输出为None。
-
permutation(x)：与numpy.random.shuffle(x)函数功能相同，两者区别：peumutation(x)不会修改X的顺序。

```{.python .input}
#-------------- rand(d0, d1, ..., dn) 生成一个0~1之间的随机浮点数或N维浮点数组。
#numpy.random.rand(d0, d1, ..., dn)
import numpy as np

np.random.rand()#生成生成[0,1)之间随机浮点数 type  float 
np.random.rand(3)#以为数组  numpy.ndarray 
np.random.rand(2,3)#生成2x3的二维数组 

#---------------- randn(d0, d1, ..., dn)：生成一个浮点数或N维浮点数组，取数范围：正态分布的随机样本数。
np.random.randn()#1.4872544578730051，不一定是[0,1)之间的随机数
np.random.randn(5)#生成形状为5的一维数组 
np.random.randn(2,3)#生成2x3数组 


#------------- standard_normal(size=None)：生产一个浮点数或N维浮点数组，取数范围：标准正态分布随机样本
np.random.standard_normal(2)#array([-2.04606393, -1.05720303])
np.random.standard_normal((2,3))
np.random.standard_normal([2,3]).shape#(2, 3)

#----------------- randint(low, high=None, size=None, dtype='l')：生成一个整数或N维整数数组
"""
取数范围：若high不为None时，取low,high之间随机整数，否则取值0,low之间随机整数。
"""
#numpy.random.randint(low, high=None, size=None, dtype='l')
np.random.randint(2)#生成一个[0,2)之间随机整数 
np.random.randint(2, size=5)#array([0, 1, 1, 0, 1]) 
np.random.randint(2, 6)#生成一个[2,6)之间随机整数 
np.random.randint(2, 6,size=5)#生成形状为5的一维整数数组 
np.random.randint(2, size=(2,3))#生成一个2x3整数数组,取数范围：[0,2)随机整数
np.random.randint(2, 6, (2,3))#生成一个2x3整数数组,取值范围：[2,6)随机整数 
np.random.randint(2, dtype='int32')
np.random.randint(2, dtype=np.int32)

#---------------------- random_sample(size=None)
"""
random_sample(size=None)：生成一个0,1之间随机浮点数或N维浮点数组。
"""
#numpy.random.random_sample(size=None)
np.random.random_sample()#生成一个[0,1)之间随机浮点数 
np.random.random_sample(2)#生成shape=2的一维数组 
np.random.random_sample((2,))#等同np.random.random_sample(2) 
np.random.random_sample((2,3))#生成2x3数组
np.random.random_sample((3,2,2))#3x2x2数组


#-------------------------- choice(a, size=None, replace=True, p=None)
"""
choice(a, size=None, replace=True, p=None)：从序列中获取元素，若a为整数，
元素取值为np.range(a)中随机数；若a为数组，取值为a数组元素中随机元素。
"""
#numpy.random.choice(a, size=None, replace=True, p=None)
np.random.choice(2)#生成一个range(2)中的随机数 
np.random.choice(2,2)#生成一个shape=2一维数组 
np.random.choice(5,(2,3))#生成一个2x3数组 
np.random.choice(np.array(['a','b','c','f']))#生成一个np.array(['a','b','c','f']中随机元素 
np.random.choice(5,(2,3))#生成2x3数组 
np.random.choice(np.array(['a','b','c','f']),(2,3))#生成2x3数组 
np.random.choice(5,p=[0,0,0,0,1])#生成的始终是4
np.random.choice(5,3,p=[0,0.5,0.5,0,0])#生成shape=3的一维数组，元素取值为1或2的随机数

#------------------------ shuffle(x)：对X进行重排序，如果X为多维数组，只沿第一条轴洗牌，输出为None
"""
shuffle(x)：对X进行重排序，如果X为多维数组，只沿第一条轴洗牌，输出为None
"""
#numpy.random.shuffle(x)
list1 = [1,2,3,4,5]
np.random.shuffle(list1)#输出None
list1#[1, 2, 5, 3, 4],原序列的顺序也被修改
arr = np.arange(9).reshape(3,3)
np.random.shuffle(arr)#对于多维数组，只沿着第一条轴打乱顺序

#--------------------- permutation(x)：
"""
与numpy.random.shuffle(x)函数功能相同，两者区别：peumutation(x)不会修改X的顺序。
"""
#numpy.random.permutation(x)
np.random.permutation(5)#生成一个range(5)随机顺序的数组 
list1 = [1,2,3,4]
np.random.permutation(list1)#array([2, 1, 4, 3]) 
arr = np.arange(9)
np.random.permutation(arr)
arr2 = np.arange(9).reshape(3,3)
np.random.permutation(arr2)#对于多维数组，只会沿着第一条轴打乱顺序
```

## <a name="2.2">2.2 索引与切片</a>

### 2.2.1 基本访问

```{.python .input}
import numpy as np
a = np.arange(10)
a[3]
```

### 2.2.2 切片

- slice (start, stop, step)
- :
- ...

```{.python .input}
import numpy as np

# ----------------- slice
a = np.arange(10)
print(a[slice(2, 7, 2)]) # slice [2 4 6]

# ----------------- start:stop:step 

print(a[2: 7: 2]) # start:stop:step  [2 4 6]
print(a[2:]) # [2 3 4 5 6 7 8 9]
print(a[-2:])


#------------------- ...
"""
切片还可以包括省略号(...)，来使选择元组的长度与数组的维度相同。
如果在行位置使用省略号，它将返回包含行中元素的ndarray。 
"""
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print(a[...,1]) # 第二列  [2 4 5]
print(a[1,...]) # 第二行的元素
print(a[...,1:]) # 第二列及其剩余元素
```

### 2.2.3 索引

- 整数索引
- 布尔值索引

```{.python .input}
import numpy as np 

#----------------- 整数索引

x = np.array([[1,  2],  [3,  4],  [5,  6]]) 
y = x[[0,1,2],  [0,1,0]]   # [1 4 5]
 
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]]) 
rows = np.array([[0,0],[3,3]]) 
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols] 
"""
x：                                                                 
[[ 0  1  2]                                                                   
 [ 3  4  5]                                                                   
 [ 6  7  8]                                                                   
 [ 9 10 11]]

y：                                      
[[ 0  2]                                                                      
 [ 9 11]] 
"""

x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
z = x[1:4,1:3] 
y = x[1:4,[1,2]] 
"""
x：
[[ 0  1  2] 
 [ 3  4  5] 
 [ 6  7  8]
 [ 9 10 11]]
z:
[[ 4  5]
 [ 7  8]
 [10 11]]

y:
[[ 4  5]
 [ 7  8]
 [10 11]]

"""

#------------------------ 布尔索引
"""
当结果对象是布尔运算(例如比较运算符)的结果时，将使用此类型的高级索引。
"""
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print(x[x >  5])  # [ 6  7  8  9 10 11]

#  这个例子使用了~(取补运算符)来过滤NaN。
a = np.array([np.nan,  1,2,np.nan,3,4,5]) 
print(a[~np.isnan(a)]) # [ 1.   2.   3.   4.   5.]

# 以下示例显示如何从数组中过滤掉非复数元素。
a = np.array([1,  2+6j,  5,  3.5+5j])  
print(a[np.iscomplex(a)])  # [2.0+6.j  3.5+5.j]
```

## <a name="2.3">2.3 广播</a>

术语**广播**是指 NumPy 在算术运算期间处理不同形状的数组的能力。
对数组的算术运算通常在相应的元素上进行。 如果两个阵列具有完全相同的形状，则这些操作被无缝执行。

```{.python .input}
import numpy as np 

a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b 
print(c)  # [ 10  40  90 160]

a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([1.0,2.0,3.0])  
print(a+b)
"""
a:
[[ 0. 0. 0.]
 [ 10. 10. 10.]
 [ 20. 20. 20.]
 [ 30. 30. 30.]]
 
b:
[ 1. 2. 3.]

a+b:
[[ 1. 2. 3.]
 [ 11. 12. 13.]
 [ 21. 22. 23.]
 [ 31. 32. 33.]]
 
 
b数组被处理成：
[[ 1. 2. 3.],
 [ 1. 2. 3.],
 [ 1. 2. 3.],
 [ 1. 2. 3.] ]
 然后在和a相加
"""
```

## <a name="2.4">2.4 迭代</a>

NumPy 包包含一个迭代器对象`numpy.nditer`。
它是一个有效的多维迭代器对象，可以用于在数组上进行迭代。 数组的每个元素可使用 Python 的标准`Iterator`接口来访问。

###
2.4.1nditer

```{.python .input}
import numpy as np 
a = np.arange(0,60,5).reshape(3, -1)
for x in np.nditer(a):  
    print(x) # 遍历每一个元素
"""
0 5 10 15 20 25 30 35 40 45 50 55
"""
```

### 2.4.2 迭代顺序

- 如果相同元素使用 F 风格顺序存储，则迭代器选择以更有效的方式对数组进行迭代。

```{.python .input}
import numpy as np 
a = np.arange(0,60,5).reshape(3, -1)
b = a.T 
c = b.copy(order='C') #  '以 C 风格顺序排序：' 
print(c)
for x in np.nditer(c):   
    print (x)
c = b.copy(order='F')  
print(c)
for x in np.nditer(c):   
    print(x)

"""
原始数组是：
[[ 0 5 10 15]
 [20 25 30 35]
 [40 45 50 55]]

原始数组的转置是：
[[ 0 20 40]
 [ 5 25 45]
 [10 30 50]
 [15 35 55]]

以 C 风格顺序排序：
[[ 0 20 40]
 [ 5 25 45]
 [10 30 50]
 [15 35 55]]
0 20 40 5 25 45 10 30 50 15 35 55

以 F 风格顺序排序：
[[ 0 20 40]
 [ 5 25 45]
 [10 30 50]
 [15 35 55]]
0 5 10 15 20 25 30 35 40 45 50 55
"""

for x in np.nditer(a, order =  'C'):   # 以 C 风格顺序排序：
    print(x)
for x in np.nditer(a, order =  'F'):  # 以 F 风格顺序排序：' 
    print(x)
"""
原始数组是：
[[ 0 5 10 15]
 [20 25 30 35]
 [40 45 50 55]]

以 C 风格顺序排序：
0 5 10 15 20 25 30 35 40 45 50 55

以 F 风格顺序排序：
0 20 40 5 25 45 10 30 50 15 35 55 
""" 
```

### 2.4.3 修改数组的值

- `nditer`对象有另一个可选参数`op_flags`。 其默认值为只读，但可以设置为读写或只写模式。
这将允许使用此迭代器修改数组元素。

```{.python .input}
import numpy as np
a = np.arange(0,60,5).reshape(3,4)   
print(a)
for x in np.nditer(a, op_flags=['readwrite']): 
    x[...]=2*x  
print(a)
"""
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]
[[  0  10  20  30]
 [ 40  50  60  70]
 [ 80  90 100 110]]
"""
```

### 2.4.4 外部循环

```{.python .input}
"""
nditer类的构造器拥有flags参数，它可以接受下列值：
- c_index 可以跟踪 C 顺序的索引
- f_index 可以跟踪 Fortran 顺序的索引
- multi-index 每次迭代可以跟踪一种索引类型
- external_loop 给出的值是具有多个值的一维数组，而不是零维数组
"""
import numpy as np 
a = np.arange(0,60,5).reshape(3,4)    
print(a)
for x in np.nditer(a, flags =  ['external_loop'], order =  'F'):  
    print(x)
"""
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]
[ 0 20 40]
[ 5 25 45]
[10 30 50]
[15 35 55]
"""
```

### 2.4.5 广播迭代

- 如果两个数组是**可广播的**，`nditer`组合对象能够同时迭代它们。 假设数组`a`具有维度 3X4，并且存在维度为
1X4 的另一个数组`b`，则使用以下类型的迭代器(数组`b`被广播到`a`的大小)

```{.python .input}
import numpy as np 
a = np.arange(0,60,5).reshape(3,4)     
print(a) 
b = np.array([1,  2,  3,  4], dtype =  int)  
print(b)   
for x,y in np.nditer([a,b]):  
    print("%d:%d"  %  (x,y))
    
"""
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]
[1 2 3 4]
0:1
5:2
10:3
15:4
20:1
25:2
30:3
35:4
40:1
45:2
50:3
55:4
"""
```

## <a name="2.5">2.5 数组操作</a>

### 2.5.1 修改形状

- reshape 不改变数据的条件下修改形状
- flat
数组上的一维迭代器
- flatten 返回折叠为一维的数组副本
- ravel 返回连续的展开数组

```{.python .input}
import numpy as np 
    
#---------------------------- reshape 不改变数据的条件下修改形状
"""
# numpy.reshape(arr, newshape, order')
这个函数在不改变数据的条件下修改形状，它接受如下参数：
"""
a = np.arange(8)
b = a.reshape(4, 2) 

#---------------------------- flat 数组上的一维迭代器
"""
numpy.ndarray.flat
该函数返回数组上的一维迭代器，行为类似 Python 内建的迭代器。
"""
a = np.arange(8).reshape(2,4) 
b = a.flat[1] 

#---------------------------- flatten 返回折叠为一维的数组副本
"""
ndarray.flatten(order) 
order：'C' — 按行，'F' — 按列，'A' — 原顺序，'k' — 元素在内存中的出现顺序。 

该函数返回折叠为一维的数组副本，函数接受下列参数：
"""
a = np.arange(8).reshape(2,4) 
print(a.flatten()) # [0 1 2 3 4 5 6 7]
print(a.flatten(order = 'F')) # [0 4 1 5 2 6 3 7]


#---------------------------- ravel 返回连续的展开数组
"""
numpy.ravel(a, order) 
order：'C' — 按行，'F' — 按列，'A' — 原顺序，'k' — 元素在内存中的出现顺序。 

这个函数返回展开的一维数组，并且按需生成副本。返回的数组和输入数组拥有相同数据类型。这个函数接受两个参数。
"""
a = np.arange(8).reshape(2,4) 
print(a.ravel()) # [0 1 2 3 4 5 6 7]
print(a.ravel(order = 'F')) # [0 4 1 5 2 6 3 7]
```

### 2.5.2 翻转操作

- transpose 翻转数组的维度
- ndarray.T和self.transpose()相同
- rollaxis
向后滚动指定的轴
- swapaxes 互换数组的两个轴

```{.python .input}
import numpy as np

#---------------------------- transpose 翻转数组的维度
"""
numpy.transpose(arr, axes) 
- arr：要转置的数组
- axes：整数的列表，对应维度，通常所有维度都会翻转。

这个函数翻转给定数组的维度。如果可能的话它会返回一个视图。函数接受下列参数：
"""
a = np.arange(12).reshape(3,4)
"""
print(a)
print(np.transpose(a))

[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[[ 0  4  8]
 [ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]]
"""

#---------------------------- ndarray.T和self.transpose()相同
"""
该函数属于ndarray类，行为类似于numpy.transpose。
"""
a = np.arange(12).reshape(3,4) 
"""
print(a)
print(a.T)

[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[[ 0  4  8]
 [ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]]
"""

#---------------------------- rollaxis 向后滚动指定的轴
"""
numpy.rollaxis(arr, axis, start)
- arr：输入数组
- axis：要向后滚动的轴，其它轴的相对位置不会改变
- start：默认为零，表示完整的滚动。会滚动到特定位置。

该函数向后滚动特定的轴，直到一个特定位置。
"""

a = np.arange(8).reshape(2,2,2) 
"""
print(a)
print(np.rollaxis(a, 2)) # 将轴 2 滚动到轴 0(宽度到深度)

[[[0 1]
  [2 3]]

 [[4 5]
  [6 7]]]
[[[0 2]
  [4 6]]

 [[1 3]
  [5 7]]]
"""

#---------------------------- swapaxes 互换数组的两个轴
"""
numpy.swapaxes(arr, axis1, axis2) 
- arr：要交换其轴的输入数组
- axis1：对应第一个轴的整数
- axis2：对应第二个轴的整数 

该函数交换数组的两个轴。
"""
a = np.arange(4).reshape(2,2)
"""
print(a)
print(np.swapaxes(a, 1, 0))

[[0 1]
 [2 3]]
[[0 2]
 [1 3]]
""" 
```

### 2.5.3 修改维度

- broadcast 产生模仿广播的对象
- broadcast_to 将数组广播到新形状
- expand_dims
扩展数组的形状
- squeeze 从数组的形状中删除单维条目

```{.python .input}
import numpy as np

#---------------------------- broadcast 产生模仿广播的对象
"""
broadcast

如前所述，NumPy 已经内置了对广播的支持。
此功能模仿广播机制。 它返回一个对象，
该对象封装了将一个数组广播到另一个数组的结果。 
"""
x = np.array([[1], [2], [3]])
y = np.array([4, 5, 6])  
b = np.broadcast(x, y) 
print(b.shape)   # (3, 3)
 
#---------------------------- broadcast_to 将数组广播到新形状
"""
numpy.broadcast_to(array, shape, subok) 

此函数将数组广播到新形状。 它在原始数组上返回只读视图。
它通常不连续。 如果新形状不符合 NumPy 的广播规则，该函数可能会抛出ValueError。 
"""
a = np.arange(4).reshape(1,4) 
"""
print(a)
print(np.broadcast_to(a, (4, 4)))

[[0 1 2 3]]
[[0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]]
"""

#---------------------------- expand_dims 扩展数组的形状
"""
numpy.expand_dims(arr, axis)
- arr：输入数组
- axis：新轴插入的位置

函数通过在指定位置插入新的轴来扩展数组形状
"""
x = np.array(([1,2],[3,4])) 
"""
print(x)
print(np.expand_dims(x, axis=2))

[[1 2]
 [3 4]]
[[[1]
  [2]]

 [[3]
  [4]]]
"""

#---------------------------- squeeze 从数组的形状中删除单维条目
"""
numpy.squeeze(arr, axis) 
- arr：输入数组
- axis：整数或整数元组，用于选择形状中单一维度条目的子集

函数从给定数组的形状中删除一维条目
"""
x = np.arange(9).reshape(1,3,3) 
"""
print(x.shape)
print(np.squeeze(x).shape)

(1, 3, 3)
(3, 3)
""" 
```

### 2.5.4 数组连接

- concatenate 沿着现存的轴连接数据序列
- srack 沿着新轴连接数组序列
- hstack
水平堆叠序列中的数组(列方向)
- vastack 竖直堆叠序列中的数组(行方向)

```{.python .input}
import numpy as np

#---------------------------- sconcatenate 沿着现存的轴连接数据序列
"""
numpy.concatenate((a1, a2, ...), axis) 
- a1, a2, ...：相同类型的数组序列
- axis：沿着它连接数组的轴，默认为 0

数组的连接是指连接。 此函数用于沿指定轴连接相同形状的两个或多个数组
"""
a = np.array([[1,2],[3,4]]) 
b = np.array([[5,6],[7,8]]) 
"""
print(np.concatenate((a, b)))
print(np.concatenate((a, b), axis=1))

[[1 2]
 [3 4]
 [5 6]
 [7 8]]
[[1 2 5 6]
 [3 4 7 8]]
"""

#---------------------------- ssrack 沿着新轴连接数组序列
"""
numpy.stack(arrays, axis) 
- arrays：相同形状的数组序列
- axis：返回数组中的轴，输入数组沿着它来堆叠

此函数沿新轴连接数组序列。
"""
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
"""
print(np.stack((a, b), 0))
print(np.stack((a, b), 1))

第一个数组：
[[1 2]
 [3 4]]

第二个数组：
[[5 6]
 [7 8]]

沿轴 0 堆叠两个数组：
[[[1 2]
 [3 4]]
 [[5 6]
 [7 8]]]

沿轴 1 堆叠两个数组：
[[[1 2]
 [5 6]]
 [[3 4]
 [7 8]]] 
"""

#---------------------------- shstack 水平堆叠序列中的数组(列方向)
"""
numpy.stack函数的变体，通过堆叠来生成水平的单个数组。
"""
a = np.array([[1,2],[3,4]]) 
b = np.array([[5,6],[7,8]]) 
"""
print(np.hstack((a, b)))

[[1 2 5 6]
 [3 4 7 8]]
"""

#---------------------------- svastack 竖直堆叠序列中的数组(行方向)
"""
numpy.stack函数的变体，通过堆叠来生成竖直的单个数组。
"""
a = np.array([[1,2],[3,4]]) 
b = np.array([[5,6],[7,8]]) 
"""
print(np.vstack((a, b)))

[[1 2]
 [3 4]
 [5 6]
 [7 8]]
"""
```

### 2.5.5 数组分割

- split 将一个数组分割为多个子数组
- hsplit 将一个数组水平分割为多个子数组(按列)
- vsplit
将一个数组竖直分割为多个子数组(按行)

```{.python .input}
import numpy as np

#---------------------------- split 将一个数组分割为多个子数组
"""
numpy.split(ary, indices_or_sections, axis)
- ary：被分割的输入数组
- indices_or_sections：可以是整数，表明要从输入数组创建的，等大小的子数组的数量。 如果此参数是一维数组，则其元素表明要创建新子数组的点。
- axis：默认为 0 

该函数沿特定的轴将数组分割为子数组
"""
a = np.arange(9) 
# '将数组分为三个大小相等的子数组：'
print(np.split(a, 3)) # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
# 将数组在一维数组中表明的位置分割：
print(np.split(a, [4, 7])) # [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8])]

#---------------------------- hsplit 将一个数组水平分割为多个子数组(按列)
"""
numpy.hsplit是split()函数的特例，其中轴为 1 表示水平分割，无论输入数组的维度是什么。
"""
a = np.arange(16).reshape(4, 4)
print(np.hsplit(a, 2)) # 水平分割
"""
[array([[ 0,  1],
       [ 4,  5],
       [ 8,  9],
       [12, 13]]), 
       
array([[ 2,  3],
       [ 6,  7],
       [10, 11],
       [14, 15]])]
"""

#---------------------------- vsplit 将一个数组竖直分割为多个子数组(按行)
"""
numpy.vsplit是split()函数的特例，其中轴为 0 表示竖直分割，无论输入数组的维度是什么 
"""
a = np.arange(16).reshape(4, 4)
print(np.vsplit(a, 2))
"""
[array([[0, 1, 2, 3],
       [4, 5, 6, 7]]), 
       
array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])]
"""
```

### 2.5.6 添加/删除元素

- resize 返回指定形状的新数组
- append 将值添加到数组末尾
- insert
沿指定轴将值插入到指定下标之前
- delete 返回删掉某个轴的子数组的新数组
- unique 寻找数组内的唯一元素

```{.python .input}
import numpy as np

#---------------------------- resize 返回指定形状的新数组
"""
numpy.resize(arr, shape)
- arr：要修改大小的输入数组
- shape：返回数组的新形状

此函数返回指定大小的新数组。 如果新大小大于原始大小，则包含原始数组中的元素的重复副本。 
"""
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape) # (2, 3)
"""
print(np.resize(a, (2, 3)))
[[1 2]
 [3 4]
 [5 6]]
 
print(np.resize(a, (2, 2)))
 [[1 2]
 [3 4]]
 
print(np.resize(a, (3, 3)))
 [[1 2 3]
 [4 5 6]
 [1 2 3]]
"""

#---------------------------- append 将值添加到数组末尾
"""
numpy.append(arr, values, axis)
- arr：输入数组
- values：要向arr添加的值，比如和arr形状相同(除了要添加的轴)
- axis：沿着它完成操作的轴。如果没有提供，两个参数都会被展开。 

此函数在输入数组的末尾添加值。 附加操作不是原地的，而是分配新的数组。
此外，输入数组的维度必须匹配否则将生成ValueError。 
"""
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.append(a, [7, 8, 9]))  # [1 2 3 4 5 6 7 8 9] 向数组添加元素
"""
print(np.append(a, [[7, 8, 9]], axis=0))  # 沿轴 0 添加元素

[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""
"""
print(np.append(a, [[5],[7]], axis=1))  # 沿轴 1 添加元素

[[1 2 3 5]
 [4 5 6 7]]
"""

#---------------------------- insert 沿指定轴将值插入到指定下标之前
"""
numpy.insert(arr, obj, values, axis) 
- arr：输入数组
- obj：在其之前插入值的索引
- values：要插入的值
- axis：沿着它插入的轴，如果未提供，则输入数组会被展开 

此函数在给定索引之前，沿给定轴在输入数组中插入值。
如果值的类型转换为要插入，则它与输入数组不同。 插入没有原地的，函数会返回一个新数组。
此外，如果未提供轴，则输入数组会被展开。 
"""
a = np.array([[1, 2], [3, 4], [5, 6]])
# '未传递 Axis 参数。 在插入之前输入数组会被展开。'
print(np.insert(a, 3, [11, 12])) # [ 1  2  3 11 12  4  5  6]
"""
print(np.insert(a, 1, [11], axis=0)) # 沿轴 0 广播

[[ 1  2]
 [11 11]
 [ 3  4]
 [ 5  6]]
"""

"""
print(np.insert(a, 1, 11, axis=1)) # 沿轴 1 广播：

[[ 1 11  2]
 [ 3 11  4]
 [ 5 11  6]]
"""


#---------------------------- delete 返回删掉某个轴的子数组的新数组
"""
Numpy.delete(arr, obj, axis) 
- arr：输入数组
- obj：可以被切片，整数或者整数数组，表明要从输入数组删除的子数组
- axis：沿着它删除给定子数组的轴，如果未提供，则输入数组会被展开 

此函数返回从输入数组中删除指定子数组的新数组。 
与insert()函数的情况一样，如果未提供轴参数，则输入数组将展开。 
"""
a = np.arange(12).reshape(3, 4)

# 未传递 Axis 参数。 在删除之前输入数组会被展开。'
print(np.delete(a, 5)) # [ 0  1  2  3  4  6  7  8  9 10 11]

"""
# 删除第二列
print(np.delete(a,1,axis = 1))

[[ 0  2  3]
 [ 4  6  7]
 [ 8 10 11]]
"""

#---------------------------- unique 寻找数组内的唯一元素
"""
numpy.unique(arr, return_index, return_inverse, return_counts) 
- arr：输入数组，如果不是一维数组则会展开
- return_index：如果为true，返回输入数组中的元素下标
- return_inverse：如果为true，返回去重数组的下标，它可以用于重构输入数组
- return_counts：如果为true，返回去重数组中的元素在原数组中的出现次数

此函数返回输入数组中的去重元素数组。 
该函数能够返回一个元组，包含去重数组和相关索引的数组。
索引的性质取决于函数调用中返回参数的类型。 
"""
a = np.array([5,2,6,2,7,5,6,8,2,9, 20]) 
print(np.unique(a)) # 去重数
# 去重数组的索引数组
print(np.unique(a, return_index=True)) # (array([ 2,  5,  6,  7,  8,  9, 20]), array([ 1,  0,  2,  4,  7,  9, 10]))
# 去重数组的下标
print(np.unique(a, return_inverse=True)) # (array([ 2,  5,  6,  7,  8,  9, 20]), array([1, 0, 2, 0, 3, 1, 2, 4, 0, 5, 6]))
```

## <a name="2.6">2.6 函数</a>

### <a name="2.6.1">2.6.1 数学函数</a>

- 三角函数
  -
numpy.sin
  - numpy.cos
  - numpy.tan
- 舍入函数
  - numpy.around(a, decimals)
这个函数返回四舍五入到所需精度的值
    - decimals 小数位数

### <a name="2.6.2">2.6.2 算术运算</a>

-
算术函数
  - add(a, b) 加
  - substract(a, b) 减
  - mutiply(a, b) 乘
  - divide(a, b)
除
- 其他函数
  - power(a, n) 幂 
  - mod(a, b) 此函数返回输入数组中相应元素的除法余数

### <a
name="2.6.3">2.6.3 统计函数</a>

- numpy.amin() 最小值
- numpy.amax() 最大值 （np.max(a,
1)）
- numpy.ptp() 返回沿轴最大值 （np.ptp(a), np.ptp(a, axis=0), np.ptp(a, axis=1)）
-
numpy.percentile(a, q, axis)  百分位数是统计中使用的度量，表示小于这个值得观察值占某个百分比。
- numpy.median()
**中值**定义为将数据样本的上半部分与下半部分分开的值
- numpy.mea() 算术平均值是沿轴的元素的总和除以元素的数量
-
numpy.average() 加权平均值是由每个分量乘以反映其重要性的因子得到的平均值
- sqrt(mean((x -x.mean())**2)) 标准差
- numpy.std 方差

## <a name="2.7">2.7 排序搜索和计数</a>

### 2.7.1 排序

- sort
函数返回输入数组的排序副本。
- argsort 并使用指定排序类型返回数据的索引数组
- lexsort 该函数返回一个索引数组

```{.python .input}
import numpy as np

#--------------------------- sort
"""
numpy.sort(a, axis, kind, order) 
- a 要排序的数组 
- axis 沿着它排序数组的轴 
- kind 默认为'quicksort'(快速排序)
- order 如果数组包含字段，则是要排序的字段 

sort()函数返回输入数组的排序副本。
"""

a = np.array([[3, 7], [9, 1]])
"""
print(np.sort(a))
print(np.sort(a, axis=0))

[[3 7]
 [1 9]]
[[3 1]
 [9 7]]
"""

dt = np.dtype([('name',  'S10'),('age',  int)]) 
a = np.array([("raju",21),("anil",25),("ravi",  17),  ("amar",27)], dtype = dt)  
"""
print(np.sort(a, order='name'))

[(b'amar', 27) (b'anil', 25) (b'raju', 21) (b'ravi', 17)]
"""

#--------------------------- argsort
"""
numpy.argsort()函数对输入数组沿给定轴执行间接排序，
并使用指定排序类型返回数据的索引数组。 这个索引数组用于构造排序后的数组。 
"""
x = np.array([3, 1, 2])
y = np.argsort(x)
"""
print(y)
print(x[y])

[1 2 0]
[1 2 3]
"""

#--------------------------- lexsort
"""
函数使用键序列执行间接排序。 键可以看作是电子表格中的一列。
该函数返回一个索引数组，使用它可以获得排序数据。 注意，最后一个键恰好是 sort 的主键。 
"""
nm =  ('raju','anil','ravi','amar') 
dv =  ('f.y.',  's.y.',  's.y.',  'f.y.')  

ind = np.lexsort((dv, nm))
print(ind)
print([nm[i]+","+dv[i] for i in ind])
```

### 2.7.2 搜索

- argmax 这两个函数分别沿给定轴返回最大元素的索引。
- argmin  这两个函数分别沿给定轴返回最小元素的索引。
-
nonzero 函数返回输入数组中非零元素的索引。
- where  函数返回输入数组中满足给定条件的元素的索引。
- extract
函数返回满足任何条件的元素。

```{.python .input}
import numpy as np

#--------------------------- argmax
a = np.array([[30, 40, 70], [80, 20, 10], [50, 90, 60]])
"""
print(np.argmax(a))
print(np.argmax(a, axis=0))
print(np.argmax(a, axis=1))

7
[1 2 0]
[2 0 1]
"""

#--------------------------- argmin
"""
print(np.argmin(a))
print(np.argmin(a, axis=0))
print(np.argmin(a, axis=1))

5
[0 1 1]
[0 2 0]
"""

#--------------------------- nonzero
"""
numpy.nonzero()函数返回输入数组中非零元素的索引。
"""
a = np.array([[30, 40, 0], [0, 20, 10], [50, 0, 60]])
"""
print(np.nonzero(a))

# 两个数组分别代表2个轴的坐标
(array([0, 0, 1, 1, 2, 2]), array([0, 1, 1, 2, 0, 2]))
"""

#--------------------------- where
"""
where()函数返回输入数组中满足给定条件的元素的索引。
"""
x = np.arange(9.).reshape(3, 3)
y = np.where(x > 3)
"""
print(x)
print(y)
print(x[y])

[[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]
(array([1, 1, 2, 2, 2]), array([1, 2, 0, 1, 2]))
[4. 5. 6. 7. 8.]
"""

#--------------------------- extract
"""
extract()函数返回满足任何条件的元素。
"""
x = np.arange(9.).reshape(3, 3)
condition = np.mod(x, 2)==0
"""
print(x)
print(condition)
print(np.extract(condition, x))

[[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]
[[ True False  True]
 [False  True False]
 [ True False  True]]
[0. 2. 4. 6. 8.]
""" 
```

## <a name="2.8">2.8 线性代数</a>

| 函数        | 描述                 |
| -----------
| -------------------- |
| dot         | 两个数组的点积       |
| vdot        | 两个向量的点积
|
| inner       | 两个数组的内积       |
| determinant | 数组的行列式         |
| solv
| 求解线性矩阵方程     |
| inv         | 寻找矩阵的乘法逆矩阵 |

```{.python .input}
import numpy as np

#----------------------- dot 两个数组的点积
"""
numpy.dot()

此函数返回两个数组的点积。 
对于二维向量，其等效于矩阵乘法。
对于一维数组，它是向量的内积。 
对于 N 维数组，它是a的最后一个轴上的和与b的倒数第二个轴的乘积。 
"""
a = np.array([[1, 2], [3, 4]])
b = np.array([[11, 12], [13, 14]])
"""
print(np.dot(a, b))
注意点积的计算为
[[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]


[[37 40]
 [85 92]]
"""

#----------------------- vdot 两个向量的内积
"""
numpy.vdot()

此函数返回两个向量的点积。
如果第一个参数是复数，那么它的共轭复数会用于计算。 
如果参数id是多维数组，它会被展开。 
"""
a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 
"""
print(np.vdot(a,b))

1*11 + 2*12 + 3*13 + 4*14 = 130

130
"""

#----------------------- inner 两个数组的矩阵积
"""
numpy.inner()

此函数返回一维数组的向量内积。 对于更高的维度，它返回最后一个轴上的和的乘积。
"""
print(np.inner(np.array([1,2,3]),np.array([0,1,0])) )
# 等价于 1*0+2*1+3*0 = 2

a = np.array([[1,2], [3,4]])  
b = np.array([[11, 12], [13, 14]])  
"""
print(np.inner(a, b))

1*11+2*12, 1*13+2*14 
3*11+4*12, 3*13+4*14 

[[35 41]
 [81 95]]
"""

#----------------------- matmul 两个数组的矩阵积
"""
numpy.matmul

numpy.matmul()函数返回两个数组的矩阵乘积。 
虽然它返回二维数组的正常乘积，但如果任一参数的维数大于2，
则将其视为存在于最后两个索引的矩阵的栈，并进行相应广播。 
"""
a = [[1,0],[1,1]] 
b = [[4,1],[2,2]] 
"""
print(np.matmul(a, b) )

[[1, 0],
 [1, 1]]
[[4, 1],
 [2, 2]]
# 过程
1*4+0*2 1*1+0*2
1*4+1*2 1*1+1*2

[[4 1]
 [6 3]]
"""
a = [[1,0],[0,1]] 
b = [1,2] 
"""
print(np.matmul(a,b))
print(np.matmul(b,a))

[1 2]
[1 2]
""" 
```
