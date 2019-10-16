# Keras Dense层输入大于数据为3维

描述：

​	在seq2seq中，decoder层中，最后一个Dense（全连接层）输入数据维3维，产生的疑惑。因为在印象里全连接层输入都是2维的数据。



[博客： [keras基础-Dense的输入的秩大于2时的注意事项](https://www.mlln.cn/2018/08/30/keras基础-Dense的输入的秩大于2时的注意事项/)]

我们可以看到, 输出`output`的形状是`(batch_size, 2, 4)`, 为什么会是这样的? 我查阅了keras中的文档, 文档中说:`if the input to the layer has a rank greater than 2, then it is flattened prior to the initial dot product with kernel`,  

