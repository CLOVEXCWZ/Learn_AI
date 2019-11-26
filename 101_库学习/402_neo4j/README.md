<font size=10>**neo4j**</font>



# 1 安装(for mac)



## 1.1 下载 neo4j

首先在 https://neo4j.com/download/ 下载 Neo4j。





## 1.2 启动neo4j

在 bin 目录下用终端执行

```python
./neo4j start
```

进入网址

```python
http://localhost:7474/
```



# 2 基本操作

- 删除节点和关系

```python
MATCH (n)
OPTIONAL MATCH (n)-[r]-()
DELETE n,r
```





# 3 CQL

[Neo4j - CQL 语法简介 ](./201_CQL.md)