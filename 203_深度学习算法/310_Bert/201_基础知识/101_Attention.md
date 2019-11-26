<font  size = 10>**Attention 机制**</font>



目的：

​	虽然之前对Attention有了一个基本的了解，但是并不算深入，并没有很好的理解Attention的完整的原理，以至于在看Attention实现的源码时会有不少的障碍。此次主要目的就是尽可能的去理解Attention机制，能理解源码，进而理解Transformer以及Bert等相关模型。

​	本次学习主要以看博客为主，内容以抄写为主，主要的目的就是能再深入理解Attention机制。



[主要参考文章----知乎-- nlp中的Attention注意力机制+Transformer详解](https://zhuanlan.zhihu.com/p/53682800)



**目录**

- 一、Attention机制剖析
  - 1、为什么要引入Attentiin机制
  - 2、Attention机制有哪些？（怎么分类？）
  - 3、Attention机制的计算流程是怎样的？
  - 4、Attention机制的变种有哪些？
  - 5、一种强大的Attention机制：为什么自主注意力模型（self-Attention model）在长距离序列中如此强大？
    - （1）卷积或循环卷积网络难道不能处理长距离序列吗？
    - （2）要解决这种短距离依赖的“局部编码”问题，从而对输入序列建立长距离依赖关系，有哪些办法呢？
    - （3）自注意力模型（self-Attention model）具体的计算流程是怎样的呢？
- 二、Transformer（Attention Is All You Need）详解
  - 1、Transformer的整体架构是怎样的？由哪些部分组成？
  - 2、Transformer Encoder 与 Transformer Decoder有哪些不同？
  - 3、Encoder-Decoder attention 与self-attention mechanism有哪些不同？
  - 4、multi-head self-attension mechanism具体的计算过程是怎样的？
  - 5、Transformer在GPT和Bert等词向量预训练模型中具有怎样的应用？有什么变化？

# 一、Attention机制剖析

## 1、为什么要引进Attention机制

根据通用近似定理，前馈网络和循环网络有很强的能力。但为什么还要引入注意力机制呢？

- **计算能力的限制**：当要记住很多‘信息’，模型就要变得更复杂，然而目前计算能力依然是限制神经网络发展的瓶颈。
- **优化算法的限制**：虽然局部连接、权重共享以及pooling等优化操作可以让神经网络变得简单一些，有效缓解模型复杂度和表达能力之间的矛盾；但是，如循环神经网络中的长距离依赖问题，信息‘记忆’能力并不高。

**可以借助人脑处理信息过载的方式，例如Attention机制可以提高神经网络处理信息的能力**

## 2、Attention机制有哪些？（怎么分类？）

当用升级网络处理大量的输入信息时，也可以简介人喃注意力机制，值选择一些关键的信息输入进行处理，来提高神经网络的效率。按照认知神经学中的注意力，可以总体上分为两类：

- **聚焦式（focus）注意力**：自上而下的有意识的注意力，主动注意力------是指有预定目的、依赖任务的、主动有意识地聚焦于某一对象的注意力；
- **显著性（saliency-based）注意力**：自下而上的有意识的注意力，被动注意----基于显著性的注意力是有外界刺激驱动的注意力，不需要主动干预，也和任务无关；可以将max-pooling和门控制（gating）机制来近似的看作是自下而上的基于显著性的注意力机制。

在人工神经网络中，注意力机制一般就特指聚焦式注意力。

## 3、Attention机制的计算流程是怎样的？

![img](https://pic1.zhimg.com/80/v2-54fe529ded98721f35277a5bfa79febc_hd.jpg)

​													Attention机制的实质：寻址（addressing）

**Attention机制的实质就是一个寻址（addressing）的过程，**如上图所示：给定一个任务和相关的查询Query向量q，通过计算与key的注意力分布并附加在Value上，从而计算Attention Value，这个过程实际上是Attention机制缓解神经网络复杂度的体现：不需要将所有的N个输入信息都输入到神经网络进行计算，只需要从X中选择一些和任务相关的信息输入给神经网络。

​	注意力机制可以分为三步：一是信息输入；二是计算注意力分布α；三是根据注意力分布α来计算输入信息的加权平均。

**step1-信息输入:**用 X = [x1, x2, ..., xn]表示N个输入信息；

**step2-注意力分布计算：**令key=Value=X，则可以给出注意力分布

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_i%3Dsoftmax%28s%28key_i%2Cq%29%29%3Dsoftmax%28s%28X_i%2Cq%29%29)

我们将![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_i) 称之为注意力分布（概率分布）， ![[公式]](https://www.zhihu.com/equation?tex=s%28X_i%2Cq%29)为注意力打分机制，有几种打分机制：

![img](https://pic3.zhimg.com/80/v2-981a0c9ab01531c7139e4701574cb056_hd.jpg)

**step3-信息加权平均：**注意力分布![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_i) 可以解释为在上下文查询q时，第i个信息受到关注的程度，采用一直"软性"的信息选择机制对输入信息X进行编码为：

![[公式]](https://www.zhihu.com/equation?tex=att%28q%2CX%29%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%5Calpha_iX_i%7D)

这种编码方式为**软性注意力机制(soft Attention)**，软性注意力机制有两种：普通模式(key=value=X)和键值对模式(key != value)。

![img](https://pic3.zhimg.com/80/v2-aa371755dc73b7137149b8d2905fc4ba_hd.jpg)

## 4、Attention机制的变种有哪些？

与普通的Attention机制（上图左）相比，Attention机制有哪些变种呢？

- **变种1-硬性注意力：**之前提到注意力是软性注意力，其选择的信息是所有输入信息的注意力分布下的期望。还有一种注意力是只关注某一位置上的信息，叫做硬性注意力（hard-attention）。硬性注意力有两种实现方式：（1）一种是选取最高概率的输入信息；（2）另一种硬性注意力可以通过在注意力分布式上才去随机采样的方式。

  **硬性注意的缺点：**

  硬性注意力的一个缺点就是基于最大采样或者随机采样的方式来选择信息。因此最终的损失函数与注意力分布之间的函数关系不可导，因此无法使用在反向传播进行训练。为了使用方向传播算法，一般使用软性注意力来代替硬性注意力。硬性注意力需要通过强化学习来进行训练。------《神经网络与深度学习》

- **变种2-键值对注意力：**即上图右边的键值对模式，此时key != value，注意力函数变为：

  ![img](https://pic3.zhimg.com/80/v2-49a8acda3757ea21218b2f7ecca6e9ae_hd.jpg)

- **变种3-多头注意力：**多头注意力（muti-head-attention）是利用多个查询Q=[q1....1m]，来平行地计算从输入信息中选取多个信息。每个多头注意力关注输入信息的不同部分，然后再进行拼接。

![img](https://pic4.zhimg.com/80/v2-27673fff36241d6ef163c9ac1cedcce7_hd.png)

## 5、一种强大的Attention机制：为什么自注意力模型(self-Attention model)在长距离序列中如此强大？

### （1）卷积或循环神经网络难道不能处理长距离序列吗？

当使用神经网络来处理一个变长的向量序列时，我们通常可以使用卷积神经网络或循环神经网络进行编码，来得到一个相同长度的输出序列，如图所示：
![img](https://pic3.zhimg.com/80/v2-8b369281a66bea6920962f45660a9f0a_hd.jpg)

从上图可以看出，无论卷积还是循环神经网络其实都是对变长序列的一种”局部编码“：卷积神经网络显然是基于N-gram的局部编码；而对于循环神经网络由于梯度消失等问题也只能建立短距离依赖。

### （2）要解决这种短距离以来的”局部编码“问题，从而对输入序列建立长距离依赖关系，有哪些办法呢？

​	如果要建立输入序列之间的长距离依赖关系，可以使用一下两种方法：一种方法是增加网络的层数，通过一个深度网络来获取远距离的信息交互，另一种方法是使用全连接网络-------《深度网络与深度学习》

![img](https://pic1.zhimg.com/80/v2-cd2d7f0961c669d983b73db4e93ccbdc_hd.jpg)

由上图可以看出，全连接网络虽然是非常直接的建模远距离依赖模型，但是无法处理变长的输入序列。不同的输入长度，其连接权重的大小也是不同的。

这时我们就可以利用注意力机制来”动态“地生成不同连接的权值，这就是自注意力模型（self-attention model）。由于自注意力模型的权值是动态生成的，因此我们可以处理变成的信息序列。

总体来说，为什么自注意力模型（self-Attention model）如此强大：利用注意力机制来”动态“地生成不同连接的权重，从而处理变长的信息序列。

（3）自注意力模型（self-Attention model）具体计算流程是怎样的呢？

同样，给出信息输入：用X=[x1,..., xn]表示N个输入信息，通过线性变换得到为查询向量序列，键向量序列和值向量序列：

![img](https://pic1.zhimg.com/80/v2-ab400406cf423842e4274527dc5a7074_hd.png)

上面公式可以看出来，self-Attention中的Q是对自身（self）输入的变换，而传统Attention中，Q来自于外部。

![img](https://pic4.zhimg.com/80/v2-fcc2df696966a9c6700d1476690cff9f_hd.jpg)self-Attention计算过程剖解（来自《细讲 | Attention Is All You Need 》）

注意力计算公式为：

![img](https://pic2.zhimg.com/80/v2-72093f153e59cfdc851e2ac1fbf5c03d_hd.jpg)

**自注意力模型(self-Attention model)**中，通常使用缩放点击来作为注意力打分函数，输出向量序列可以写为：

![img](https://pic2.zhimg.com/80/v2-2f76af60c24ba75e37f2f5df8edfdb71_hd.jpg)



# 二、Transformer·（Attention Is All You Need）详解

从Transformer这篇论文的题目可以看出，Transformer的核心就是Attention,这也就是为什么本文会在剖析玩Attention机制之后会引出Transformer，如果对上面的Attention机制别特是自注意力模型（self-Attention model）理解后，Transformer就很容易理解了。

## 1、Transformer的整体架构是怎样的？由哪些部分组成？

![img](https://pic1.zhimg.com/80/v2-7f8b460cd617fedc822064c4230302b0_hd.jpg)

Transformer其实这就是一个Seq2Seq模型，左边一个encoder把输入读进去，右边一个decoder得到输出：

![img](https://pic4.zhimg.com/80/v2-846cf91009c44c6e479bada42bfc437f_hd.jpg)

Transformer=Transformer Encoder+Transformer Decoder

### (1) Transformer Encoder (N=6 层，每层包括2个sub-layers):

![img](https://pic3.zhimg.com/80/v2-3b97d37951078856097069778293230a_hd.jpg)

- sub-layer-1:mutil-head self-attention mechanism, 用来进行self-attention。

- sub-layer-2:Posision-wise Feed-forward Networks, 简单的全连接网络，对每个position的向量分别进行相同的操作，包括两个线性变换和一个Relu激活输出（输入输出层的维度都是512，中间层为2048）

  ![img](https://pic3.zhimg.com/80/v2-5236351e3efd93d567ac1fceea7716ee_hd.png)

- 每个sub-layer都使用了残差网络： ![[公式]](https://www.zhihu.com/equation?tex=LayerNorm%28X%2Bsublayer%28X%29%29)



（2）Transformer Decoder(N=6, 每层包括三个sub-layers):

![img](https://pic1.zhimg.com/80/v2-4dc71fe78c4752645de1f1ba8dd762a4_hd.jpg)

- sub-layer-1:Masked multi-head self-attention mechanism,用来进行self-attention，与Encoder不同：由于序列生成过程所以在I的时候，大于i的时候都没有结果，只有小于i的时刻有结果，因此需要做Mask。
- sub-layer-2:Posion-wise Feed-forward Networks, 同Encoder。
- sub-layer-3:Encoder-Decoder attention 计算。

## 2 Transformer Encoder与Transformer Decoder有哪些不同？

（1）multi-head self-attention mechanism不同，Encoder中不需要使用Masked，而Decoder中需要使用Masked;

（2）Decoder中多了一层Endcoder-Decoder attention,这与self-attention mechanise不同。

## 3、Encoder-Decoder attention 与 self-attention mechanism有哪些不同？

他们都用multi-head计算，不过Encoder-Decoder attention采用传统的attention机制，其中的Query是self-attention mechanism已经计算出的上一时间i处的编码值，key和value都是Encoder的输出，这与self-attention mechanism不同。具体代码中具体体现：

```python
 ## Multihead Attention ( self-attention)
            self.dec = multihead_attention(queries=self.dec,
                                           keys=self.dec,
                                           num_units=hp.hidden_units,
                                           num_heads=hp.num_heads,
                                           dropout_rate=hp.dropout_rate,
                                           is_training=is_training,
                                           causality=True,
                                           scope="self_attention")

## Multihead Attention ( Encoder-Decoder attention)
            self.dec = multihead_attention(queries=self.dec,
                                           keys=self.enc,
                                           num_units=hp.hidden_units,
                                           num_heads=hp.num_heads,
                                           dropout_rate=hp.dropout_rate,
                                           is_training=is_training,
                                           causality=False,
                                           scope="vanilla_attention")
```

## 4、muti-head self attention mechanism具体的计算过程是怎样的？

![img](https://pic3.zhimg.com/80/v2-392692c19c57f5bfa116f7b505dfde7a_hd.jpg)multi-

Transformer中的Attention机制由Scaled Dot-Product Attention和Mutlti-Head Attention组成，上图给出整个流程。下面具体介绍各个环节：

- **Expand:** 实际上是经过线性变换，生成Q、K、V三个向量
- **Split heads:** 进行分头操作，在原文中将原来每个位置的512维度分成8个head，每个head维度变成64；
- **Self Attention:**对每个head进行Self Attention，具体过程和第一部分介绍的一致；

- **Concat heads:** 对进行完Self Attention每个head进行拼接

上述的公式为：

![img](https://pic3.zhimg.com/80/v2-c7100e268bcefaa7ff0a1344acc15e7e_hd.jpg)

5.Transformer在GPT和Beart等词向量预训练模型中具体是怎样应用的？有什么变化？

- GPT中训练的是单向语言模型，其实就是直接应用Transformer Decoder；
- Bert中训练的是双向的语言模型，应用了Transformer Encoder部分，不过在Encoder基础上做了Masked操作

BERT Transformer使用双向self-attention, 而GPT Transformer使用受限制的self-attention，其中每个Token只能处理器左侧的上下文。双向Transformer通常被称为“Transformer encoder”,而左侧上下文被称为“Transformer decoder”，decoder是不能获取预测信息的。









