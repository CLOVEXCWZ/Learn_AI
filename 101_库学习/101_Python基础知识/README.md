<font size=10>**Python 基础知识**</font>

​	 Python基础知识不会全部都涉及到，如果需要系统学习Python3可到[菜鸟教程](https://www.runoob.com/python3/python3-tutorial.html)进行比较系统的学习和练习。

 	这里的Python基础知识，主要介绍字符串的处理、文件操作部分



<font size=5>**目录**</font>

- [1 字符串操作](#1)
  - [ 1.1 字符串常用操作](#1.1)
  - [1.2 字符串检测](#1.2)
  - [ 1.3 字符串转化](#1.3)
  - [ 1.4 字符串格式化](#1.4)
- [ 2 文件操作](#2)
  - [ 2.1 基本操作](#2.1)
  - [ 2.2 read&write](#2.2)
  - [ 2.3 line](#2.3)
  - [2.4 二进制的读写](#2.4)
  - [ 2.5 pickle](#2.5)
  - [ 2.6 文件路径](#2.6)



## <a name="1">1 字符串操作</a>

### <a name="1.1">1.1 字符串常用操作</a>

- 初始化
  - s = "abc"  
- 读取截取
  - s[n]  获取字符串s索引为n处的字符
  - s[start : end : step]  截取字符start到end(步长为step)的字符串，其中step默认为1
    - s[1:]   s[-1:]
    - s[:2]   s[:-2]
    - s[1:3]  s[-3:-1]
    - s[: 8 : 2 ]  
  - ord(s)  函数获取字符的整数表示
  - chr(s) 函数把编码转换为对应的字符
- 拼接
  - s1+s2 把s1和s2两段字符串拼接起来
  - s1.join(iterable)  (iterable是字符串元组字符串数组等)
- 读取长度
  - len(s)  获取字符串长度
  - len(s.encode())  获取字节数
- 统计
  - s.count('ch') 统计字符串s中ch的出现次数
- 分割
  - s.split(sep, maxsplit) 以sep作为分割符号，最多分割成maxsplit（默认为-1，没有次数限制）
  - s.split('ch') 以ch为分隔符，将s进行分割，并返回分割后的列表
  - s.rsplit('ch') 以ch为分隔符，将s进行分割，并返回分割后的列表,从s的右端开始搜索ch
  - s.partition('ch') 将字符串s按照ch字符串分成三个部分，从左到右的顺序
  - s.rpartition('ch') 将字符串s按照ch字符串分成三个部分，从右到左的顺序
- 替换
  - s.replace(old,new) 将字符串s中的old子字符串全部替换成new
  - s.expandtabs(n) 将字符串s中的每个制表符替换成n个空格

### <a name="1.2">1.2 字符串检测</a>

- 子串的位置
  - s.find(t) 如果没有找到子字符串t，返回-1，否则返回t在s中的起始位置
  - s.rfind(t) 如果没有找到子字符串t，返回-1，否则返回t在s中的最后一个的位置
  - s.index(t) 如果没有找到子字符串t，返回ValueError异常，否则返回t在s中的起始位
  - s.rindex(t) 如果没有找到子字符串t，返回ValueError异常，否则返回t在s中的最后一个的位置
- 开始检测 （s.startswith(t) s以字符串t开头，区分大小写）
- 结尾结束检测 （s.endswith(t) s以字符串t结尾，区分大小写）
- 是否只包含字符串或数字（s.isalnum()）
- 是否只包含字符串（s.isalpha()）
- 是否只包含数字字符（s.isdigit()）
- 是否合法标识符（s.isidentifier() ）
- 是否只包含小写字符（s.islower()）
- 是否只包含数字（ s.isnumeric()）
- 是否只包含可打印字符（ s.isprintable() ）
- 是否只包含空白符（ s.isspace()）
- 是否符合头衔大小写（ s.istitle() ）
- 是否只包含大写字母（s.isupper()）

### <a name="1.3">1.3 字符串转化</a>

- 大小写转化
  - s.capitalize() 将s[0]变成大写
  - s.lower() 将字母全部改成小写
  - s.upper() 将字母全部改成大写
  - s.swapcase() 交换大小写
  - s.title() 将字符串改成符合头衔大小写
- 填充
  - s.center(n,ch) 包含n个字符串，其中s位于中间，两边使用ch填充
  - s.ljust(n,ch) 包含n个字符串，其中s位于右边，左边使用ch填充
  - s.rjust(n,ch) 包含n个字符串，其中s位于左边，右边使用ch填充
  - s.format(ch1,ch2) s中包含{0}{1}的位置被ch1、ch2填充
- 过滤
  - s.strip(ch) 去除字符串s中开头和结尾的ch字符串
  - s.lstrip(ch) 去除字符串s中开头的ch字符串
  - s.rstrip(ch) 去除字符串s中结尾的ch字符串

### <a name="1.4">1.4 字符串格式化</a>

- 占位符 (例如 %%  %c  %d 等)
- %-formatting
- format方法
- f-strings

## <a name="2">2 文件操作</a>

| 方式 | 说明                                                         |
| ---- | ------------------------------------------------------------ |
| r    | 以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式。 |
| w    | 打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。 |
| a    | 打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。 |
| rb   | 以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。 |
| wb   | 以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。 |
| ab   | 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。 |
| r+   | 打开一个文件用于读写。文件指针将会放在文件的开头。           |
| w+   | 打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。 |
| a+   | 打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。 |
| rb+  | 以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。 |
| wb+  | 以二进制格式打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。 |
| ab+  | 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写。 |

### <a name="2.1">2.1 基本操作</a>

- open()       打开文件 
- close()        关闭文件

- readline()   读取行
- readlines() 
- seek(n)        按照位置访问
- tell()             读取当前位置

### <a name="2.2">2.2 read&write</a>

### <a name="2.3">2.3 line</a>

### <a name="2.4">2.4 二进制的读写</a>

### <a name="2.5">2.5 pickle</a>

- pickle.dump (data, fileHandle ) 写入文件
- pickle.load(fileHandle) 加载数据

### <a name="2.6">2.6 文件路径</a>

- os.getcwd()  获取当前目录
- os.listdir() 获取当前目录下的文件名列表
- os.remove() 删除一个文件
- os.path.isfile() 判断是否是一个文件
- os.path.isdir() 判断是否是一个目录
- os.path.isabs() 判断是否为绝对路径
- os.path.islink ( filename ) 检查是否为会快捷方式
- os.path.exists() 检查路径是否存在
- os.path.splitext() 分离扩展名
- os.path.dirname() 获取路径名
- os.path.basename() 获取文件名
- os.rename（old， new） 重命名
- os.makedirs（r“c：\python\test”） 创建多级目录
- os.mkdir（“test”）创建单个目录
- os.stat（file） 获取文件属性
- os.chmod（file）修改文件权限与时间戳
- os.exit（）终止当前进程
- os.path.getsize（filename）获取文件大小





