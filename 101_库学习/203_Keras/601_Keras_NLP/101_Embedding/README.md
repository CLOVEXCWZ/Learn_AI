<font size=5>**Embedding - keras**</font>



时间 2019-11-27



**摘要：在github项目 [Keras-TextClassification](https://github.com/yongzhuo/Keras-TextClassification) 的启发下把Embedding部分抽取出来，进行学习理解。根据项目采用random、word2vec、bert、xlnet四种方式进行Embedding，并把整个过程记录下来。**



(这段是废话 start) Embedding是NLP必不可少的一个步骤，也是困扰着我的问题，终于有那么一个如此好的机会让我能去理解并且能手动去实现，这也算是雪中送炭了！当然也少不了前两天对keras的比较深入的理解咯，这才促成了这一些。内心有点小激动======= (这段是废话 end)



以下是对探索的记录(探索一下午，记录一晚上 [手动哭笑]])



# Embedding random

**摘要:**随机Embedding其实就是在Embedding层的参数进行随机初始化的过程,本次主要对随机Embedding的整个过程进行记录，以便以后有需要的时候可以很快的复现。

[参考源码地址--------](https://github.com/yongzhuo/Keras-TextClassification)

采用随机生成参数的形式进行Embedding操作，可在训练中更新参数以达到训练的目的，在word2vec对Embedd的初始化过程都是采用随机生成的，然后在训练中去不断更新参数，以达到训练的目的。

随机Embedding其实就是在Embedding层的参数进行随机初始化的过程

**注意:本次实例是字符级别的Embedding**



```python
from keras.layers import Embedding 
from keras.models import Input, Model

import numpy as np  
import re

# 字符字典文件
path_embedding_term_char = "/Users/zhouwencheng/Desktop/Grass/data/model" \
                           "/ImportModel/Word2Vec/term_char.txt"
  
class RandomEmbedding(object):
    def __init__(self,
                 len_max=50,  # 文本最大长度, 建议25-50
                 embed_size=300,  # 嵌入层尺寸
                 vocab_size=30000,  # 字典大小, 这里随便填的，会根据代码里修改
                 trainable=True,  # 是否训练参数
                 path_char=path_char,
                ):
        self.len_max = len_max
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.trainable = trainable
        self.path_char = path_char
        
        self.input = None
        self.output = None
        self.model = None
        self.token2idx = {}
        self.idx2token = {}
        
        # 定义符号
        self.ot_dict = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[BOS]': 2,
            '[EOS]': 3, }
        self.deal_corpus()
        self.build()
        
    def deal_corpus(self):
        token2idx = self.ot_dict.copy()
        count = 3
        with open(file=self.path_char, mode='r', encoding='utf-8') as fd:
            while True:
                term_one = fd.readline()
                if not term_one:
                    break
                term_one = term_one.strip()
                if term_one not in token2idx:
                    count = count + 1
                    token2idx[term_one] = count
        self.token2idx = token2idx
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key
    
    def build(self, **kwargs):
        self.vocab_size = len(self.token2idx)
        self.input = Input(shape=(self.len_max, ), dtype='int32')
        self.output = Embedding(input_dim=self.vocab_size,
                                output_dim=self.embed_size,
                                input_length=self.len_max,
                                trainable=self.trainable,
                                )(self.input)
        self.model = Model(inputs=self.input, outputs=self.output)
    
    def sentence2idx(self, text):
        text = self.extract_chinese(str(text)).upper()
        text = list(text)
        text = [text_one for text_one in text]
        len_leave = self.len_max - len(text)

        # 转换和填充处理
        if len_leave >= 0:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for text_char in text] + [self.token2idx['[PAD]'] for i in range(len_leave)]
        else:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text[0:self.len_max]]
        return text_index
    
    def idx2sentence(self, idx):
        assert type(idx) == list
        text_idx = [self.idx2token[id] if id in self.idx2token else self.idx2token['[UNK]'] for id in idx]
        return "".join(text_idx)
    
    def extract_chinese(self, text):
        """
              只提取出中文、字母和数字
            :param text: str, input of sentence
            :return:
            """
        chinese_exttract = ''.join(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@._])", text))
        return chinese_exttract   
      
#===================== 测试 ==============
texts = ["今天天气不错",
                 "明天天气也不错"]
eb = RandomEmbedding()
x = []
for t in texts:
    x.append(eb.sentence2idx(t))
x = np.array(x)
print(x.shape)
print(x)

model = eb.model
p = model.predict(x)
print(p.shape)
print(p)
print(p.shape)
```



# Embedding word2vec

**[摘要] word2vec是NLP文字转向量很重要的组成，此次采用 gensim 加载训练好的vec变成参数加载到Embedding的初始化参数中（vec向量是由维基百科训练得到的向量，具体没有过多介绍---）。这边便能达到embeddin操作的目的，同时加载到Embedding的参数还可以选择再训练**

[参考源码地址==========](https://github.com/yongzhuo/Keras-TextClassification)

word2vec是在NLP向量化非常普遍的形式，由于计算资源等因素能采用别人训练好的向量既可以达到向量化的目的，又可以不用再花太多时间这可以说是一举两得的！如果不是领域行非常强的话可以做一些微调即可很好的使用，如果领域性很强则需要考虑是否需要自己重先训练。

这里记录下今天（2019-11-27）探索Word2Vec的实现代码，以便于以后有需要的时候能快速实现。

```python
from keras.layers import Add, Embedding
from gensim.models import KeyedVectors
from keras.models import Input, Model

import numpy as np
import codecs
import os
import re

# 通过维基百科训练出来的词向量，维度为300W（字符级）
path_embedding_w2v_wiki = "/Users/zhouwencheng/Desktop/Grass/data/model" \
                          "/ImportModel/Word2Vec/w2v_model_wiki_char.vec"


class WordEmbedding(object):
    def __init__(self,
                 len_max=50,  # 文本最大长度, 建议25-50
                 embed_size=300,  # 嵌入层尺寸
                 vocab_size=30000,  # 字典大小, 这里随便填的，会根据代码里修改
                 trainable=True,  # 是否训练参数
                 path_vec=path_embedding_w2v_wiki,
                ):
        self.len_max = len_max
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.trainable = trainable
        self.path_vec = path_vec
        
        self.input = None
        self.output = None
        self.model = None
        self.token2idx = {}
        self.idx2token = {}
        
        # 定义符号
        self.ot_dict = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[BOS]': 2,
            '[EOS]': 3, }
        self.deal_corpus()
        self.build()
        
    def deal_corpus(self): 
        pass
    
    def build(self, **kwargs):
        print(f"load word2vec start!")
        self.key_vector = KeyedVectors.load_word2vec_format(self.path_vec, **kwargs)
        print(f"load word2vec end!")
        self.embed_size = self.key_vector.vector_size
        self.token2idx = self.ot_dict.copy()
        embedding_matrix = []
        # 首先加self.token2idx中的四个[PAD]、[UNK]、[BOS]、[EOS]
        embedding_matrix.append(np.zeros(self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        for word in self.key_vector.index2entity:
            self.token2idx[word] = len(self.token2idx)
            embedding_matrix.append(self.key_vector[word])

        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key

        self.vocab_size = len(self.token2idx)
        embedding_matrix = np.array(embedding_matrix)
        self.input = Input(shape=(self.len_max, ), dtype='int32')
        self.output = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_size,
            input_length=self.len_max,
            weights=[embedding_matrix],
            trainable=self.trainable
        )(self.input)
        self.model = Model(inputs=self.input, outputs=self.output)
    
    def sentence2idx(self, text):
        text = self.extract_chinese(str(text)).upper()
        text = list(text)
        text = [text_one for text_one in text]
        len_leave = self.len_max - len(text)

        # 转换和填充处理
        if len_leave >= 0:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for text_char in text] + [self.token2idx['[PAD]'] for i in range(len_leave)]
        else:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text[0:self.len_max]]
        return text_index
    
    def idx2sentence(self, idx):
        assert type(idx) == list
        text_idx = [self.idx2token[id] if id in self.idx2token else self.idx2token['[UNK]'] for id in idx]
        return "".join(text_idx)
    
    def extract_chinese(self, text):
        """
              只提取出中文、字母和数字
            :param text: str, input of sentence
            :return:
            """
        chinese_exttract = ''.join(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@._])", text))
        return chinese_exttract   
      
#================= 测试 ===============#
texts = ["今天天气不错",
             "明天天气也不错"]
eb = WordEmbedding()
x = []
for t in texts:
    x.append(eb.sentence2idx(t))
x = np.array(x)
print(x.shape)

model = eb.model
p = model.predict(x)
print(p.shape)
print(p)
print(p.shape)
```



# Embedding BERT

**摘要: BERT的出现是令人兴奋的，它足以方NLP走上一个台阶。BERT作为语言模型在它出现的时候表现可谓是惊艳，在一段时间内统领NLP的多个领域，它的预训练模型便是促使这一些的主要原因。BERT本质是一个语言模型，于是用它来做向量化也就是常理之中了。本次探索加载预训练模型(chinese_L-12_H-768_A-12)进行文字的向量化**

[参考源码地址==========](https://github.com/yongzhuo/Keras-TextClassification)

BERT的强大就不做赘述了，不过BERT需要耗费的算力也是非常让人头疼的事情了！也至于我知道现在有也没有能真正去好好是使用BERT等一些比较大型的网络，是在是有些遗憾。等哥以后有GPU了(可以用到GPU资源，不是说自己买哈-------=-=-=)哥一定好好去跑这些大型网络，想想还有点开心。

这里记录下今天（2019-11-27）探索BERT Embedding的实现代码，以便于以后有需要的时候能快速实现。

```python
from keras.layers import Add, Embedding
from gensim.models import KeyedVectors
from keras.models import Input, Model

import numpy as np
import codecs
import os
import re

path_embedding_bert = "/Users/zhouwencheng/Desktop/Grass/data/model" \
                      "/ImportModel/BERT/chinese_L-12_H-768_A-12"
  
from __future__ import print_function, division
from keras.engine import Layer


class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
      
class BertEmbedding(object):
    def __init__(self,
                 len_max=50,  # 文本最大长度, 建议25-50
                 embed_size=300,  # 嵌入层尺寸
                 vocab_size=30000,  # 字典大小, 这里随便填的，会根据代码里修改
                 trainable=True,  # 是否训练参数
                 path_mode=path_embedding_bert,
                 layer_indexes=[24] # 默认取最后一层的输出 大于13则取最后一层的输出
                ):
        self.len_max = len_max
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.trainable = trainable 
        self.path_mode = path_mode
        self.layer_indexes = layer_indexes
        
        self.input = None
        self.output = None
        self.model = None 
        self.build()
        
 

    def build(self):
        import keras_bert

        config_path = os.path.join(self.path_mode, 'bert_config.json')
        check_point_path = os.path.join(self.path_mode, 'bert_model.ckpt')
        dict_path = os.path.join(self.path_mode, 'vocab.txt')
        print('load bert model start!')
        model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                              checkpoint_file=check_point_path,
                                                              seq_len=self.len_max,
                                                              trainable=self.trainable)
        print('load bert model end!')

        layer_dict = [6]
        layer_0 = 7
        for i in range(12):
            layer_0 = layer_0 + 8
            layer_dict.append(layer_0)
        print(layer_dict)

        # 输出他本身
        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
        # 分类如果只有一层，就只取最后那一层的weight；取得不正确，就默认取最后一层
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in [i + 1 for i in range(13)]:
                encoder_layer = model.get_layer(index=layer_dict[self.layer_indexes[0]-1]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-1]).output
        # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
        else:
            # layer_indexes must be [1,2,3,......12]
            # all_layers = [model.get_layer(index=lay).output if lay is not 1 else model.get_layer(index=lay).output[0] for lay in layer_indexes]
            all_layers = [model.get_layer(index=layer_dict[lay - 1]).output if lay in [i + 1 for i in range(13)]
                          else model.get_layer(index=layer_dict[-1]).output  # 如果给出不正确，就默认输出最后一层
                          for lay in self.layer_indexes]
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
        self.output = NonMaskingLayer()(encoder_layer)
        self.input = model.inputs
        self.model = Model(inputs=self.input, outputs=self.output)
        self.embedding_size = self.model.output_shape[-1]

        self.token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        self.vocab_size = len(self.token_dict)
        self.tokenizer = keras_bert.Tokenizer(self.token_dict)

    
    def sentence2idx(self, text, second_text=None):
        text = self.extract_chinese(str(text)).upper()
        input_id, input_type_id = self.tokenizer.encode(first=text,
                                                        second=second_text,
                                                        max_len=self.len_max)
        return [input_id, input_type_id]
    
    
    def extract_chinese(self, text):
        """
          只提取出中文、字母和数字
        :param text: str, input of sentence
        :return:
        """
        chinese_exttract = ''.join(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@._])", text))
        return chinese_exttract  
      
      
#===================== 测试 ==================#
texts = ["今天天气不错",
             "明天天气也不错"]
eb = BertEmbedding()
x = []
x_type = []
for t in texts:
    x_buff, x_type_buff = eb.sentence2idx(t)
    x.append(x_buff)
    x_type.append(x_type_buff)
x = np.array(x)
x_type = np.array(x_type)

print(x.shape)
print(x_type.shape)

model = eb.model
p = model.predict([x, x_type])
print(p.shape)
print(p)
print(p.shape)
```



# Embedding XLNET

**摘要：首先呢，需要承认一件事情，就是我还没有对XLNET去很好的了解=====。XLNET也是基于transfromer网络的，是google在推出BERT后再推出的网络，性能比BERT会更好。由于XLNET也是语言模型，所以也是可以做文本向量化的,本文采用XLNET中文预训练模型(chinese_xlnet_mid_L-24_H-768_A-12)进行向量化**

[参考源码地址==========](https://github.com/yongzhuo/Keras-TextClassification)

- 预训练模型说明
  - 来源：哈工大进行中文预训练的
  - [github原地址======](https://github.com/ymcui/Chinese-PreTrained-XLNet)
  - 网络结构：XLNet-mid：24-layer, 768-hidden, 12-heads, 209M parameters
  - 下载tensorflow版本(具体请打开链接查看)

XLNET以后会去探索，暂时呢就不知道了------- 不过呢，不知道也可以先用着嘛=======【大笑】

这里记录下今天（2019-11-27）探索BXLNET Embedding的实现代码，以便于以后有需要的时候能快速实现。



```python
from keras.layers import Add, Embedding
from gensim.models import KeyedVectors
from keras.models import Input, Model

import numpy as np
import codecs
import os
import re

path_embedding_xlnet = "/Users/zhouwencheng/Desktop/Grass/data/model" \
                       "/ImportModel/XLNET/chinese_xlnet_mid_L-24_H-768_A-12"


 from __future__ import print_function, division
from keras.engine import Layer


class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
      
class XlnetEmbedding(object):
    def __init__(self,
                 len_max=50,  # 文本最大长度, 建议25-50
                 embed_size=300,  # 嵌入层尺寸
                 vocab_size=30000,  # 字典大小, 这里随便填的，会根据代码里修改
                 trainable=False,  # 是否训练参数
                 path_mode=path_embedding_xlnet,
                 layer_indexes=[24], # 默认取最后一层的输出 大于13则取最后一层的输出
                 xlnet_embed={},
                 batch_size=2
                ):
        self.len_max = len_max
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.trainable = trainable 
        self.corpus_path = path_mode
        self.layer_indexes = layer_indexes
        self.xlnet_embed = xlnet_embed
        self.batch_size = batch_size
        
        self.input = None
        self.output = None
        self.model = None 
        self.build()
        
 
    def build(self):
        from keras_xlnet import Tokenizer, ATTENTION_TYPE_BI, ATTENTION_TYPE_UNI
        from keras_xlnet import load_trained_model_from_checkpoint

        self.checkpoint_path = os.path.join(self.corpus_path, 'xlnet_model.ckpt')
        self.config_path = os.path.join(self.corpus_path, 'xlnet_config.json')
        self.spiece_model = os.path.join(self.corpus_path, 'spiece.model')

        self.attention_type = 'bi'
        self.attention_type = ATTENTION_TYPE_BI if self.attention_type == 'bi' else ATTENTION_TYPE_UNI
        self.memory_len = 0
        self.target_len = 5
        print('load xlnet model start!')
        model = load_trained_model_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                   attention_type=self.attention_type,
                                                   in_train_phase=self.trainable,
                                                   config_path=self.config_path,
                                                   memory_len=self.memory_len,
                                                   target_len=self.target_len,
                                                   batch_size=self.batch_size,
                                                   mask_index=0)
#         model.summary()
        print('load xlnet model finish!')
        # 字典加载
        self.tokenizer = Tokenizer(self.spiece_model)
        self.model_layers = model.layers
        len_layers = self.model_layers.__len__()
        print(len_layers)
        len_couche = int((len_layers - 6) / 10)
        # 一共246个layer
        # 每层10个layer（MultiHeadAttention,Dropout,Add,LayerNormalization）,第一是9个layer的输入和embedding层
        # 一共24层
        layer_dict = [5]
        layer_0 = 6
        for i in range(len_couche):
            layer_0 = layer_0 + 10
            layer_dict.append(layer_0-2)
        # 输出它本身
        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
            # 分类如果只有一层，取得不正确的话就取倒数第二层
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in [i + 1 for i in range(len_couche + 1)]:
                encoder_layer = model.get_layer(index=layer_dict[self.layer_indexes[0]]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-1]).output
        # 否则遍历需要取的层，把所有层的weight取出来并加起来shape:768*层数
        else:
            # layer_indexes must be [0, 1, 2,3,......24]
            all_layers = [model.get_layer(index=layer_dict[lay]).output
                          if lay in [i + 1 for i in range(len_couche + 1)]
                          else model.get_layer(index=layer_dict[-1]).output  # 如果给出不正确，就默认输出倒数第一层
                          for lay in self.layer_indexes]
            print(self.layer_indexes)
            print(all_layers)
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
            print(encoder_layer.shape)
        self.output = NonMaskingLayer()(encoder_layer)
        self.input = model.inputs
        self.model = Model(inputs=self.input, outputs=self.output)

        self.embedding_size = self.model.output_shape[-1]
        self.vocab_size = len(self.tokenizer.sp)

    def sentence2idx(self, text):
        text = self.extract_chinese(str(text).upper())
        tokens = self.tokenizer.encode(text)
        tokens = tokens + [0] * (self.target_len - len(tokens)) \
            if len(tokens) < self.target_len \
            else tokens[0:self.target_len]
        token_input = np.expand_dims(np.array(tokens), axis=0)
        segment_input = np.zeros_like(token_input)
        memory_length_input = np.zeros((1, 1))
        return [token_input, segment_input, memory_length_input]
    
    
    def extract_chinese(self, text):
        """
          只提取出中文、字母和数字
        :param text: str, input of sentence
        :return:
        """
        chinese_exttract = ''.join(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@._])", text))
        return chinese_exttract   
      
#========================== 测试 ===================#
texts = ["今天天气不错",
             "明天天气也不错"]
eb = XlnetEmbedding()
x = []
x_segment = []
x_memory = []
for t in texts:
    x_buff, x_segment_buff, x_memory_buff = eb.sentence2idx(t)
    x.append(x_buff[0])
    x_segment.append(x_segment_buff[0])
    x_memory.append(x_memory_buff[0])

x = np.array(x)
x_segment = np.array(x_segment)
x_memory = np.array(x_memory)

print(x.shape)
print(x_segment.shape)
print(x_memory.shape)

print(x)
print(x_segment)
print(x_memory)

model = eb.model
p = model.predict([x, x_segment, x_memory])
print(p.shape)
print(p)
print(p.shape)
```









