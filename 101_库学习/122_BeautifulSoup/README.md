<font size=10>**Beautiful Soup**</font>

Beautiful Soup是一个可以从HTML或XML文件中提取数据的Python库。它能够通过你喜欢的转换器实现惯用的文档导航，查找、修改文档的方式。



- [1 快速开始](#1)
  - [1.1 快速开始](#1.1)
  - [1.2 如何使用](#1.2)
- [2 对象的种类](#2)
  - [2.1 tag](#2.1)
  - [2.2 可以遍历的字符串](#2.2)
  - [2.3 BeautifulSoup](#2.3)
  - [2.4 注释以及特殊字符串](#2.4)
- [3 遍历文档树](#3)
  - [3.1 子节点](#3.1)
  - [3.2父节点](#3.2)
  - [3.3兄弟节点](#3.3)
  - [3.4 回退和前进](#3.4)
  - [3.5搜索文档树](#3.5)
  - [3.6 过滤器](#3.6)
  - [3.7 find_all](#3.7)
  - [3.8 像调用find_all一样调用tag](#3.8)
  - [3.9 find()](#3.9)
  - [3.10 find_parents() 和 find_parent()](#3.10)
  - [3.11 find_next_siblings() 和 find_next_sibling()](#3.11)
  - [3.12 find_previous_siblings() 和 find_previous_siblingx](#2.12)
  - [3.13 find_all_next() 和 find_next()](#3.13)
  - [3.14 find_all_previous() 和 find_previous()](#3.14)
  -  [3.15 CSS选择器](#3.15)
- [4 修改文档树](#4)
  - [4.1 修改tag的名称和属性](#4.1)
  - [4.2 修改 .string](#4.2)
  - [4.3 append()](#4.3)
  - [4.4 NavigableString() 和 .new_tag()](#4.4)
  - [4.5 insert()](#4.5)
  - [4.6 insert_before() 和 insert_after()](#4.7)
  - [4.7 clear()](#4.7)
  - [4.8 extract()](#4.8)
  - [4.9 decompose()](#4.9)
  - [4.10 replace_with()](#4.10)
  - [4.11 wrap()](#4.11)
  - [4.12 unwrap()](#4.12)
- [5 输出](#5)
  - [5.1 格式化输出](#5.1)
  - [5.2 压缩输出](#5.2)
  - [5.3 输出格式](#5.3)
  - [5.4 get_text()](#5.4)



 # <a name="1"> 1 快速开始</a>



 ## <a name="1.1"> 1.1 快速开始</a>

```python
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

"""
使用BeautifulSoup解析这段代码,能够得到一个 BeautifulSoup 的对象,
并能按照标准的缩进格式的结构输出:
"""
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')
# print(soup.prettify()) # 输出标准缩进的结构

# 几个简单的结构化数据的方法
print(soup.title) # <title>The Dormouse's story</title>
print(soup.title.name) # title
print(soup.title.string) # The Dormouse's story
print(soup.title.parent.name) # head
print(soup.p) # <p class="title"><b>The Dormouse's story</b></p>
print(soup.p['class']) # ['title']
print(soup.a) # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
print(soup.find_all('a'))
"""
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, 
<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
"""
print(soup.find(id="link3"))
"""
<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
"""

# 从文档中找到所有<a>标签的链接:
for link in soup.find_all('a'):
    print(link.get('href'))
"""
http://example.com/elsie
http://example.com/lacie
http://example.com/tillie
"""

# 从文档中获取所有文字内容:
print(soup.get_text())
"""
The Dormouse's story

The Dormouse's story
Once upon a time there were three little sisters; and their names were
Elsie,
Lacie and
Tillie;
and they lived at the bottom of a well.
...
"""
```



 ## <a name="1.2"> 1.2 如何使用</a>

我们将一段文档传入BeautifulSoup的构造方法，就能得到一个文档的对象，可以传入一段字符串或一个文件句柄

```python
from bs4 import BeautifulSoup
soup = BeautifulSoup(open('data/101.txt'))
print(soup)
soup =BeautifulSoup("<html>data</html>")
print(soup)
```



 # <a name="2"> 2 对象的种类</a>

Beautiful Soup将复杂HTML文档转化成一个复杂的树形结构，每个节点都是Python对象，所有对象可以归纳为4中：Tag、NavigableString、BeautifulSoup、Comment

 ## <a name="2.1"> 2.1 tag</a>

Tag 对象与XML或HTML原文档中的tag相同

```python
from bs4 import BeautifulSoup

#-------------------------- tag
"""
tag 的基本属性
- name
- attributes
"""

soup = BeautifulSoup('<b class="boldest">Extremely bold</b>')
tag = soup.b
type(tag) # bs4.element.Tag

#------------------------- Name

print(tag.name) # b

# 修改tag name
tag.name = "blockquote" 
print(tag)

#------------------------- Attributes
"""
一个tag中可能有很多属性 
tag <b class="boldest"> 有一个 “class” 的属性,值为 “boldest” .
tag的属性的操作方法与字典相同:
"""
print(tag['class']) # 像字典一样访问
print(tag.attrs) # 返回属性列表
tag['class'] = 'vergbold' # 修改属性值
tag['id'] = 1 # 增加属性值
print(tag) # <blockquote class="vergbold" id="1">Extremely bold</blockquote>

del tag['id'] # 删除属性

#------------------------ 多值属性
"""
HTML 4定义了一系列可以包含多个值的属性.在HTML5中移除了一些,
却增加更多.最常见的多值的属性是 class (一个tag可以有多个CSS的class). 
还有一些属性 rel , rev , accept-charset , headers , accesskey . 在Beautiful Soup中多值属性的返回类型是list:
"""
css_soup = BeautifulSoup('<p class="body strikeout"></p>')
print(css_soup.p['class']) # ['body', 'strikeout']
```



 ## <a name="2.2"> 2.2 可以遍历的字符串</a>
字符串常被包含在tag内，Beautiful Soup用NavigableString类来包装tag中字符串

```python
from bs4 import BeautifulSoup 
soup = BeautifulSoup('<b class="boldest">Extremely bold</b>')
tag = soup.b

print(tag.string)  # Extremely bold
print(type(tag.string)) # <class 'bs4.element.NavigableString'>

#--- 替换 string
tag.string.replace_with('No longer bold')
print(tag) # <b class="boldest">No longer bold</b>
```



 ## <a name="2.3"> 2.3 BeautifulSoup</a>

BeautifulSoup对象表示的是一个文档的全部内容，大部分时候，可以吧它当作Tag对象，它支持方法比较多。

因为BeautifulSoup对象并不是真正的HTML或XML的tag所以它并没有name和attribute属性。但有时查看它的 .name 属性是很方便的,所以 BeautifulSoup 对象包含了一个值为 “[document]” 的特殊属性 .name

```python
from bs4 import BeautifulSoup 
soup = BeautifulSoup('<b class="boldest">Extremely bold</b>')

print(soup.name) # [document]
```



 ## <a name="2.4"> 2.4 注释以及特殊字符串</a>

Tag , NavigableString , BeautifulSoup 几乎覆盖了html和xml中的所有内容,但是还有一些特殊对象.容易让人担心的内容是文档的注释部分:

```python
from bs4 import BeautifulSoup 
markup = "<b><!--Hey, buddy. Want to buy a used parser?--></b>"

soup = BeautifulSoup(markup)
comment = soup.b.string
print(type(comment))
```



 # <a name="3"> 3 遍历文档树</a>

```python
html_doc = """
<html><head><title>The Dormouse's story</title></head>
    <body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')
```



 ## <a name="3.1"> 3.1 子节点</a>

一个tag可能包含多个字符串或其他的Tag这些都是这个Tag节点子节点。Beautiful Soup提供了许多操作和遍历子节点的属性.

```python
#------------------- tag的名字
"""
操作文档树最简单的方法就是告诉它你想获取的tag的name.如果想获取 <head> 标签,只要用 soup.head :
"""
print(soup.head) # <head><title>The Dormouse's story</title></head>
print(soup.title) # <title>The Dormouse's story</title>
print(soup.body.b) # <b>The Dormouse's story</b>

# 通过点取属性的方式只能获得当前名字的第一个tag:
print(soup.a) # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
print(soup.find_all('a'))
"""
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>, 
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
"""

#------------------------- .contents 和 .children
"""
tag的 .contents 属性可以将tag的子节点以列表的方式输出:

通过tag的 .children 生成器,可以对tag的子节点进行循环:
"""
head_tag = soup.head
head_tag # <head><title>The Dormouse's story</title></head>

head_tag.contents # [<title>The Dormouse's story</title>]
title_tag = head_tag.contents[0] 
title_tag # <title>The Dormouse's story</title>
title_tag.contents # ["The Dormouse's story"]


for child in title_tag.children:
    print(child)
    # The Dormouse's story
    
    
#--------------------- .descendants
""" 
descendants 属性可以对所有tag的子孙节点进行递归循环 [5] :

head_tag # <head><title>The Dormouse's story</title></head>
"""
for child in head_tag.descendants:
    print(child)
    # <title>The Dormouse's story</title>
    # The Dormouse's story

#----------------------------- .string
"""
如果tag只有一个 NavigableString 类型子节点,那么这个tag可以使用 .string 得到子节点
"""

#----------------------------- .strings 和 stripped_strings
"""
如果tag中包含多个字符串 [2] ,可以使用 .strings 来循环获取:
输出的字符串中可能包含了很多空格或空行,使用 .stripped_strings 可以去除多余空白内容:
"""
# for string in soup.strings:
#     print(repr(string))

# for string in soup.stripped_strings:
#     print(repr(string))
```



 ## <a name="3.2"> 3.2 父节点</a>

继续分析文档树,每个tag或字符串都有父节点:被包含在某个tag中

```python
#----------------------- .parent
"""
通过 .parent 属性来获取某个元素的父节点.在例子“爱丽丝”的文档中,<head>标签是<title>标签的父节点:
"""
title_tag = soup.title
title_tag
# <title>The Dormouse's story</title>
title_tag.parent
# <head><title>The Dormouse's story</title></head>

#---- 文档title的字符串也有父节点:<title>标签
title_tag.string.parent
# <title>The Dormouse's story</title>

#-------------------------- .parents
"""
通过元素的 .parents 属性可以递归得到元素的所有父辈节点,下面的例子使用了
.parents 方法遍历了<a>标签到根节点的所有节点.
"""
link = soup.a
link
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
for parent in link.parents:
    if parent is None:
        print(parent)
    else:
        print(parent.name)
"""
p
body
html
[document]
"""
```



 ## <a name="3.3"> 3.3 兄弟节点</a>

```python
sibling_soup = BeautifulSoup("<a><b>text1</b><c>text2</c></a>")
# print(sibling_soup.prettify())
"""
因为<b>标签和<c>标签是同一层:他们是同一个元素的子节点,所以<b>和<c>可以被称为兄弟节点.
"""

#------------------------------------ .next_sibling 和 .previous_sibling
sibling_soup.b.next_sibling
# <c>text2</c>
sibling_soup.c.previous_sibling
# <b>text1</b>

"""
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>

如果以为第一个<a>标签的
.next_sibling 结果是第二个<a>标签,那就错了,
真实结果是第一个<a>标签和第二个<a>标签之间的顿号和换行符:
"""
link = soup.a
link
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
link.next_sibling
# ',\n'
link.next_sibling.next_sibling
# <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>

#------------------------------- .next_siblings 和 .previous_siblings
for sibling in soup.a.next_siblings:
    print(repr(sibling))
"""
',\n'
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
' and\n'
<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
';\nand they lived at the bottom of a well.'
"""
    
for sibling in soup.find(id="link3").previous_siblings:
    print(repr(sibling))
"""
' and\n'
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
',\n'
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
'Once upon a time there were three little sisters; and their names were\n'
"""
```



 ## <a name="3.4"> 3.4 回退和前进</a>

.next_element 属性指向解析过程中下一个被解析的对象(字符串或tag),结果可能与 
.next_sibling 相同,但通常是不一样的.

```python
"""
看一下“爱丽丝” 文档:

<html><head><title>The Dormouse's story</title></head>
<p class="title"><b>The Dormouse's story</b></p>
"""

#----------------------------------- .next_element 和 .previous_element
"""
.next_element 属性指向解析过程中下一个被解析的对象(字符串或tag),结果可能与 
.next_sibling 相同,但通常是不一样的.
"""
#------------------------------------ .next_elements 和 .previous_elements
"""
通过 .next_elements 和 .previous_elements 的迭代器就可以向前或向后访问文档的解析内容,
就好像文档正在被解析一样:
"""
```



 ## <a name="3.5"> 3.5 搜索文档树</a>

Beautiful Soup定义了很多搜索方法,这里着重介绍2个: find() 和 find_all() .其它方法的参数和用法类似,请读者举一反三.

```python
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')
```



 ## <a name="3.6"> 3.6 过滤器</a>

介绍 find_all() 方法前,先介绍一下过滤器的类型 [3] ,这些过滤器贯穿整个搜索的API.过滤器可以被用在tag的name中,节点的属性中,字符串中或他们的混合中.

- 字符串
- 正则表达式
- 列表
- True
- 方法

```python
#----------------------------------- 字符串
"""
最简单的过滤器是字符串.在搜索方法中传入一个字符串参数,Beautiful Soup会查找与字符串完整匹配的内容,
下面的例子用于查找文档中所有的<b>标签:
"""
soup.find_all('b')
# [<b>The Dormouse's story</b>]

#----------------------------------- 正则表达式
"""
如果传入正则表达式作为参数,Beautiful Soup会通过正则表达式的 match() 来匹配内容.
下面例子中找出所有以b开头的标签,这表示<body>和<b>标签都应该被找到:
"""
import re
for tag in soup.find_all(re.compile("^b")):
    print(tag.name)
"""
body
b
"""
    
#----------------------------------- 列表
"""
如果传入列表参数,Beautiful Soup会将与列表中任一元素匹配的内容返回.
下面代码找到文档中所有<a>标签和<b>标签:
"""
soup.find_all(["a", "b"])
"""
[<b>The Dormouse's story</b>,
 <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
"""

#----------------------------------- True
"""
True 可以匹配任何值,下面代码查找到所有的tag,但是不会返回字符串节点
"""
for tag in soup.find_all(True):
    print(tag.name)
""" 
html
head
title
body
p
b
p
a
a
a
p
""" 

#---------------------------------- 方法
"""
如果没有合适过滤器,那么还可以定义一个方法,方法只接受一个元素参数 [4] ,
如果这个方法返回 True 表示当前元素匹配并且被找到,如果不是则反回 False
"""
def has_class_but_no_id(tag):
    return tag.has_attr('class') and not tag.has_attr('id')
soup.find_all(has_class_but_no_id)
```



 ## <a name="3.7"> 3.7 find_all</a>

find_all() 方法搜索当前tag的所有tag子节点,并判断是否符合过滤器的条件

- name参数
- keyword 参数
- 按CSS搜索

```python
soup.find_all("title")
# [<title>The Dormouse's story</title>]

soup.find_all("p", "title")
# [<p class="title"><b>The Dormouse's story</b></p>]

soup.find_all("a")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.find_all(id="link2")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

import re
soup.find(string=re.compile("sisters"))
# u'Once upon a time there were three little sisters; and their names were\n'

#----------------------- name参数
"""
name 参数可以查找所有名字为 name 的tag,字符串对象会被自动忽略掉.
"""
soup.find_all("title")
# [<title>The Dormouse's story</title>]


#------------------------ keyword 参数
"""
如果一个指定名字的参数不是搜索内置的参数名,搜索时会把该参数当作指定名字tag的属性来搜索,
如果包含一个名字为 id 的参数,Beautiful Soup会搜索每个tag的”id”属性.
"""
soup.find_all(id='link2')
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

soup.find_all(href=re.compile("elsie"))
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

soup.find_all(id=True)
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.find_all(href=re.compile("elsie"), id='link1')
# [<a class="sister" href="http://example.com/elsie" id="link1">three</a>]

# data_soup = BeautifulSoup('<div data-foo="value">foo!</div>')
# data_soup.find_all(data-foo="value")
# SyntaxError: keyword can't be an expression

# data_soup.find_all(attrs={"data-foo": "value"})
# # [<div data-foo="value">foo!</div>]

#----------------------- 按CSS搜索
"""
按照CSS类名搜索tag的功能非常实用,但标识CSS类名的关键字 class 在Python中是保留字,使用 class 做参数会导致语法错误.

string 参数
limit 参数
recursive 参数
"""
soup.find_all("a", class_="sister")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.find_all(class_=re.compile("itl"))
# [<p class="title"><b>The Dormouse's story</b></p>]

def has_six_characters(css_class):
    return css_class is not None and len(css_class) == 6
soup.find_all(class_=has_six_characters)
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.find_all(string="Elsie")
# [u'Elsie']

soup.find_all(string=["Tillie", "Elsie", "Lacie"])
# [u'Elsie', u'Lacie', u'Tillie']

soup.find_all(string=re.compile("Dormouse"))
# [u"The Dormouse's story", u"The Dormouse's story"]

soup.find_all("a", limit=2)
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]
```



- 按CSS搜索

 ## <a name="3.8"> 3.8 像调用find_all一样调用tag</a>

find_all() 几乎是Beautiful Soup中最常用的搜索方法,所以我们定义了它的简写方法. BeautifulSoup 对象和 tag 对象可以被当作一个方法来使用,这个方法的执行结果与调用这个对象的 find_all() 方法相同,下面两行代码是等价的:

 ## <a name="3.9"> 3.9 find()</a>

find_all() 方法将返回文档中符合条件的所有tag,尽管有时候我们只想得到一个结果.比如文档中只有一个标签,那么使用 find_all() 方法来查找标签就不太合适, 使用 find_all 方法并设置 limit=1 参数不如直接使用 find() 方法.下面两行代码是等价的:

 ## <a name="3.10"> 3.10 find_parents() 和 find_parent()</a>

我们已经用了很大篇幅来介绍 find_all() 和 find() 方法,Beautiful Soup中还有10个用于搜索的API.它们中的五个用的是与 find_all() 相同的搜索参数,另外5个与 find() 方法的搜索参数类似.区别仅是它们搜索文档的不同部分.

 ```python
a_string = soup.find(string="Lacie")
a_string
# u'Lacie'

a_string.find_parents("a")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

a_string.find_parent("p")
# <p class="story">Once upon a time there were three little sisters; and their names were
#  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
#  and they lived at the bottom of a well.</p>


 ```



 ## <a name="3.11"> 3.11 find_next_siblings() 和 find_next_sibling()</a>

这2个方法通过 .next_siblings 属性对当tag的所有后面解析 [5] 的兄弟tag节点进行迭代, find_next_siblings() 方法返回所有符合条件的后面的兄弟节点, find_next_sibling() 只返回符合条件的后面的第一个tag节点.

 ```python
first_link = soup.a
first_link
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

first_link.find_next_siblings("a")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

first_story_paragraph = soup.find("p", "story")
first_story_paragraph.find_next_sibling("p")
# <p class="story">...</p>
 ```



 ## <a name="3.12"> 3.12 find_previous_siblings() 和 find_previous_sibling()</a>

这2个方法通过 .previous_siblings 属性对当前tag的前面解析 [5] 的兄弟tag节点进行迭代, find_previous_siblings() 方法返回所有符合条件的前面的兄弟节点, find_previous_sibling() 方法返回第一个符合条件的前面的兄弟节点

```python
last_link = soup.find("a", id="link3")
last_link
# <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

last_link.find_previous_siblings("a")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

first_story_paragraph = soup.find("p", "story")
first_story_paragraph.find_previous_sibling("p")
# <p class="title"><b>The Dormouse's story</b></p>
```



 ## <a name="3.13"> 3.13 find_all_next() 和 find_next()</a>

这2个方法通过 .next_elements 属性对当前tag的之后的 [5] tag和字符串进行迭代, find_all_next() 方法返回所有符合条件的节点, find_next() 方法返回第一个符合条件的节点:

```python
first_link = soup.a
first_link
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

first_link.find_all_next(string=True)
# [u'Elsie', u',\n', u'Lacie', u' and\n', u'Tillie',
#  u';\nand they lived at the bottom of a well.', u'\n\n', u'...', u'\n']

first_link.find_next("p")
# <p class="story">...</p>
```



 ## <a name="3.14"> 3.14 find_all_previous() 和 find_previous()</a>

这2个方法通过 .previous_elements 属性对当前节点前面 [5] 的tag和字符串进行迭代, find_all_previous() 方法返回所有符合条件的节点, find_previous() 方法返回第一个符合条件的节点.

```python
first_link = soup.a
first_link
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

first_link.find_all_previous("p")
# [<p class="story">Once upon a time there were three little sisters; ...</p>,
#  <p class="title"><b>The Dormouse's story</b></p>]

first_link.find_previous("title")
# <title>The Dormouse's story</title>
```



 ## <a name="3.15">  3.15 CSS选择器</a>

Beautiful Soup支持大部分的CSS选择器 http://www.w3.org/TR/CSS2/selector.html [6] , 在 Tag 或 BeautifulSoup 对象的 .select() 方法中传入字符串参数, 即可使用CSS选择器的语法找到tag:

- 通过tag标签逐层查找
- 找到某个tag标签下的直接子标签
- 找到兄弟节点标签
- 通过CSS的类名查找
- 通过tag的id查找
- 同时用多种CSS选择器查询元素
- 通过是否存在某个属性来查找:
- 通过属性的值来查找
- 返回查找到的元素的第一个

```python
soup.select("title")
# [<title>The Dormouse's story</title>]

soup.select("p:nth-of-type(3)")
# [<p class="story">...</p>]

#--------------------- 通过tag标签逐层查找:

soup.select("body a")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie"  id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select("html head title")
# [<title>The Dormouse's story</title>]

#--------------------- 找到某个tag标签下的直接子标签
soup.select("head > title")
# [<title>The Dormouse's story</title>]

soup.select("p > a")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie"  id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select("p > a:nth-of-type(2)")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

soup.select("p > #link1")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

soup.select("body > a")
# []

#--------------------- 找到兄弟节点标签:
soup.select("#link1 ~ .sister")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie"  id="link3">Tillie</a>]

soup.select("#link1 + .sister")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

#---------------------- 通过CSS的类名查找:

soup.select(".sister")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select("[class~=sister]")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

#------------------------ 通过tag的id查找:
soup.select("#link1")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

soup.select("a#link2")
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]


#---------------------------- 同时用多种CSS选择器查询元素:
soup.select("#link1,#link2")
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

#---------------------------- 通过是否存在某个属性来查找:
soup.select('a[href]')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

#----------------------------- 通过属性的值来查找:
soup.select('a[href="http://example.com/elsie"]')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

soup.select('a[href^="http://example.com/"]')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select('a[href$="tillie"]')
# [<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select('a[href*=".com/el"]')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

#---------------------------------- 通过语言设置来查找:
multilingual_markup = """
 <p lang="en">Hello</p>
 <p lang="en-us">Howdy, y'all</p>
 <p lang="en-gb">Pip-pip, old fruit</p>
 <p lang="fr">Bonjour mes amis</p>
"""
multilingual_soup = BeautifulSoup(multilingual_markup)
multilingual_soup.select('p[lang|=en]')
# [<p lang="en">Hello</p>,
#  <p lang="en-us">Howdy, y'all</p>,
#  <p lang="en-gb">Pip-pip, old fruit</p>]

#---------------------------------- 返回查找到的元素的第一个
soup.select_one(".sister")
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
```



 # <a name="4"> 4 修改文档树</a>

 ## <a name="4.1"> 4.1 修改tag的名称和属性</a>

修改tag的属性和名称 

- tag.name = "blockquote"   
- tag['class'] = 'verybold'

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup('<b class="boldest">Extremely bold</b>')
tag = soup.b

tag.name = "blockquote"
tag['class'] = 'verybold'
tag['id'] = 1
tag
# <blockquote class="verybold" id="1">Extremely bold</blockquote>

del tag['class']
del tag['id']
tag
# <blockquote>Extremely bold</blockquote>
```



 ## <a name="4.2"> 4.2 修改 .string</a>

- 给tag的 .string 属性赋值,就相当于用当前的内容替代了原来的内容:

```python
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)

tag = soup.a
tag.string = "New link text."
tag
# <a href="http://example.com/">New link text.</a>
```



 ## <a name="4.3"> 4.3 append()</a>

Tag.append() 方法想tag中添加内容,就好像Python的列表的 .append() 方法:

```python
soup = BeautifulSoup("<a>Foo</a>")
soup.a.append("Bar")

soup
# <html><head></head><body><a>FooBar</a></body></html>
soup.a.contents
# [u'Foo', u'Bar']
```



 ## <a name="4.4"> 4.4 NavigableString() 和 .new_tag()</a>

如果想添加一段文本内容到文档中也没问题,可以调用Python的 append() 方法 或调用 NavigableString 的构造方法:

```python
from bs4 import NavigableString

soup = BeautifulSoup("<b></b>")
tag = soup.b
tag.append("Hello")
new_string = NavigableString(" there")
tag.append(new_string)
tag
# <b>Hello there.</b>
tag.contents
# [u'Hello', u' there']

from bs4 import Comment
new_comment = soup.new_string("Nice to see you.", Comment)
tag.append(new_comment)
tag
# <b>Hello there<!--Nice to see you.--></b>
tag.contents
# [u'Hello', u' there', u'Nice to see you.']


soup = BeautifulSoup("<b></b>")
original_tag = soup.b

new_tag = soup.new_tag("a", href="http://www.example.com")
original_tag.append(new_tag)
original_tag
# <b><a href="http://www.example.com"></a></b>

new_tag.string = "Link text."
original_tag
# <b><a href="http://www.example.com">Link text.</a></b>
```



 ## <a name="4.5"> 4.5 insert()</a>

Tag.insert() 方法与 Tag.append() 方法类似,区别是不会把新元素添加到父节点 .contents 属性的最后,而是把元素插入到指定的位置.与Python列表总的

```python
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
tag = soup.a

tag.insert(1, "but did not endorse ")
tag
# <a href="http://example.com/">I linked to but did not endorse <i>example.com</i></a>
tag.contents
# [u'I linked to ', u'but did not endorse', <i>example.com</i>]
```



 ## <a name="4.6"> 4.6 insert_before() 和 insert_after()</a>

insert_before() 方法在当前tag或文本节点前插入内容:

```python
soup = BeautifulSoup("<b>stop</b>")
tag = soup.new_tag("i")
tag.string = "Don't"
soup.b.string.insert_before(tag)
soup.b
# <b><i>Don't</i>stop</b>

soup.b.i.insert_after(soup.new_string(" ever "))
soup.b
# <b><i>Don't</i> ever stop</b>
soup.b.contents
# [<i>Don't</i>, u' ever ', u'stop']
```



 ## <a name="4.7"> 4.7 clear()</a>

Tag.clear() 方法移除当前tag的内容:

```python
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
tag = soup.a

tag.clear()
tag
# <a href="http://example.com/"></a>
```



 ## <a name="4.8"> 4.8 extract()</a>

PageElement.extract() 方法将当前tag移除文档树,并作为方法结果返回:

```python
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
a_tag = soup.a

i_tag = soup.i.extract()

a_tag
# <a href="http://example.com/">I linked to</a>

i_tag
# <i>example.com</i>

print(i_tag.parent)
None
```



 ## <a name="4.9"> 4.9 decompose()</a>

Tag.decompose() 方法将当前节点移除文档树并完全销毁:

```python
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
a_tag = soup.a

soup.i.decompose()

a_tag
# <a href="http://example.com/">I linked to</a>
```



 ## <a name="4.10"> 4.10 replace_with()</a>

PageElement.replace_with() 方法移除文档树中的某段内容,并用新tag或文本节点替代它:

```python
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
a_tag = soup.a

new_tag = soup.new_tag("b")
new_tag.string = "example.net"
a_tag.i.replace_with(new_tag)

a_tag
# <a href="http://example.com/">I linked to <b>example.net</b></a>
```



 ## <a name="4.11"> 4.11 wrap()</a>

PageElement.wrap() 方法可以对指定的tag元素进行包装

```python
soup = BeautifulSoup("<p>I wish I was bold.</p>")
soup.p.string.wrap(soup.new_tag("b"))
# <b>I wish I was bold.</b>

soup.p.wrap(soup.new_tag("div"))
# <div><p><b>I wish I was bold.</b></p></div>
```



 ## <a name="4.12"> 4.12 unwrap()</a>

Tag.unwrap() 方法与 wrap() 方法相反.将移除tag内的所有tag标签,该方法常被用来进行标记的解包:

```python
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
a_tag = soup.a

a_tag.i.unwrap()
a_tag
# <a href="http://example.com/">I linked to example.com</a>
```



 # <a name="5"> 5 输出</a>
 ## <a name="5.1"> 5.1 格式化输出</a>

prettify() 方法将Beautiful Soup的文档树格式化后以Unicode编码输出,每个XML/HTML标签都独占一行

```python
from bs4 import BeautifulSoup

markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
soup.prettify()
# '<html>\n <head>\n </head>\n <body>\n  <a href="http://example.com/">\n...'

print(soup.prettify())
# <html>
#  <head>
#  </head>
#  <body>
#   <a href="http://example.com/">
#    I linked to
#    <i>
#     example.com
#    </i>
#   </a>
#  </body>
# </html>


print(soup.a.prettify())
# <a href="http://example.com/">
#  I linked to
#  <i>
#   example.com
#  </i>
# </a>
```



 ## <a name="5.2"> 5.2 压缩输出</a>



 ## <a name="5.3"> 5.3 输出格式</a>



 ## <a name="5.4"> 5.4 get_text()</a>

得到tag里面包含文档呢内容

```python
markup = '<a href="http://example.com/">\nI linked to <i>example.com</i>\n</a>'
soup = BeautifulSoup(markup)

soup.get_text()
# '\nI linked to example.com\n'
soup.i.get_text()
# 'example.com'


#---------------------- 可以通过参数指定tag的文本内容的分隔符:
soup.get_text("|")
u'\nI linked to |example.com|\n'

#---------------------- 还可以去除获得文本内容的前后空白:
soup.get_text("|", strip=True)
# 'I linked to|example.com'

[text for text in soup.stripped_strings]
# [u'I linked to', u'example.com']
```





