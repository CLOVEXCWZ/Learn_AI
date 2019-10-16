<font size=15>**mkdocs**</font>



<font color='purper' size=5>**Kears离线文档说明**</font>



原因：

​	在网上找keras中文文档的时候，找了一圈发现有个不错的文档，但是目录层次那边不能拖动，弄了好一会发现的确是不能拖动。在网上查了一下发展一个博客 [Keras中文官方文档(离线版)](https://blog.csdn.net/u010299280/article/details/82705336) 介绍说离线看keras离线文档，于是就跟着做了一遍，结果的确和他所说的一样。甚是满意



## mkdoc步骤

下载Keras中文文档，并且使用mkdocs在离线查看的步骤：

- 1 github上下载包

```
git clone https://github.com/keras-team/keras-docs-zh
```

- 2 安装mkdocs

```
pip install mkdocs

并且在文档目录下(keras-docs-zh文件夹)执行命令：
mkdocs build
```

- 3 启动服务

```
mkdocs serve // 启动本地服务

访问网址：
http://localhost:8000
```



