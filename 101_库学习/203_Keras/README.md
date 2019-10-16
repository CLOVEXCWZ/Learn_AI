<font size=15>**Keras: 基于Python深度学习库**</font>

​	Keras是一个用Python编写的高级神经网络API，它能够以 [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), 或者 [Theano](https://github.com/Theano/Theano) 作为后端运行。Keras 的开发重点是支持快速的实验。能够以最小的时延把你的想法转换为实验结果，是做好研究的关键。



[Keras中文离线文档说明](../801Other/101mkdocs(keras中文文档).md)



# 目录

- [1模型](#1)
- [2 Layers](#2)
  - [2.1 核心网络](#2.1)
  - [ 2.2 卷积层](#2.2)
  - [ 2.3 池化层](#2.2)
  - [2.4 局部连接层](#2.4)
  - [ 2.5 循环层](#2.5)
  - [ 2.6 嵌入层](#2.6)
  - [ 2.7 融合层](#2.7)
  - [ 2.8 高级激活函数层](#2.8)
  - [ 2.9 标准化层](#2.9)
  - [ 2.10 噪声层](#2.10)
  - [ 2.12 编写自己的层](#2.12)

- [ 3 数据预处理](#3)
  - [ 3.1 序列预处理](#3.1)
  - [3.2 文本预处理](#3.2)
  - [ 3.3 图像预处理](#3.3)

- [ 4 组件](#4)
  - [ 4.1 损失函数](#4.1)
  - [ 4.2 标准评估](#4.1)
  - [ 4.3 优化器](#4.3)
  - [4.4 激活函数](#4.4)
  - [ 4.5 正则化](#4.5)
  - [ 4.6 初始化](#4.6)
  - [ 4.7 回调函数](#4.7)
- [ 5 数据集](#5)
- [ 6 应用 Applications](#6)
  - [6.1 可用模型](#6.1)
  - [6.2 模型概览](#6.2)
- [ 7 经典案例](#7)



# <a name="1"> 1 模型 </a>

在Keras中有两类主要模型： Sequential顺序模型和使用函数式API的Model类模型

这些模型有许多共同的方法和属性：

- model.layer 是包含模型网络层的展平列表
- model.inputs 是模型输入张量的列表
- model.outputs 是模型输出张量的列表
- model.summary() 打印出模型概述信息
- model.get_config() 返回包含模型配置信息的字典
- model.get_weights() 返回模型中所有权重张量的列表
- model.set_weights(weights) 从Numpy数组中为模型设置权重
- model.to_json() 以JSON字符串形式返回模型的表示
- model.to_yaml() 以YAML字符串的形式返回模型的表示。
- model.save_weights(filepath) 将模型权重存储为HDF5文件
- model.load_weights(filepath, by_name=False) 从HDF5文件中加载权重

## <a name="2">2 Layers</a>

所有Keras网络层都有很多共同的函数：

- layer.get_weights() 以含有Numpy矩阵的列表形式返回层的权重
- layer.set_weights() 从含有Numpy矩阵的列表中设置层的权重
- layer.get_config() 返回包含层配置的字典

输入输出

- layer.input()
- layer.output()
- layer.input_shape()
- layer.output_shape()

多层节点；

- layer.get_input_at(node_index)
- layer.get_output_at(node_index)
- layer.get_input_shape_at(node_index)
- layer.get_output_shape_at(node_index)

## <a name="2.1">2.1 核心网络层</a>

- Dense 全连接层

- Activation 激活函数
- Dropout 
- Flatten 展平
- Input 实例化Keras张量
- Reshape 重先调整尺寸
- Permute 根据给的的模式置换输入的维度
- RepeatVector 将输入重复n次
- Lambda 将任意表达式封装为Layer对象
- ActivityRegularization 网络层，对基于代价函数的输入活动应用一个更新。
- Masking 使用覆盖系列，以跳过时间步
- SpatialDropout1D Dropout 的 Spatial 1D 版本
- SpatialDropout2D Dropout 的 Spatial 2D 版本
- SpatialDropout3D Dropout 的 Spatial 3D 版本

```python
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.Activation(activation)

keras.layers.Dropout(rate, noise_shape=None, seed=None)

keras.layers.Flatten(data_format=None)

keras.engine.input_layer.Input()

keras.layers.Reshape(target_shape)

keras.layers.Permute(dims)

keras.layers.RepeatVector(n)

keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)

keras.layers.ActivityRegularization(l1=0.0, l2=0.0)

keras.layers.Masking(mask_value=0.0)

keras.layers.SpatialDropout1D(rate)

keras.layers.SpatialDropout2D(rate, data_format=None)

keras.layers.SpatialDropout3D(rate, data_format=None)
```



## <a name="2.2">2.2 卷积层</a>

- Conv1D 1D 卷积层 (例如时序卷积)。
- Conv2D 2D 卷积层 (例如对图像的空间卷积)。
- SeparableConv1D 深度方向的可分离 1D 卷积。
- SeparableConv2D 深度方向的可分离 2D 卷积。
- DepthwiseConv2D 深度可分离 2D 卷积。
- Conv2DTranspose 转置卷积层 (有时被成为反卷积)。
- Conv3D 3D 卷积层 (例如立体空间卷积)。
- Conv3DTranspose 转置卷积层 (有时被成为反卷积)。
- Cropping1D 1D 输入的裁剪层（例如时间序列）。
- Cropping2D 2D 输入的裁剪层（例如图像）。
- Cropping3D 3D 数据的裁剪层（例如空间或时空）。
- UpSampling1D 1D 输入的上采样层。
- UpSampling2D 2D 输入的上采样层。
- UpSampling3D 3D 输入的上采样层。
- ZeroPadding1D 1D 输入的零填充层（例如，时间序列）。
- ZeroPadding2D 2D 输入的零填充层（例如图像）。
- ZeroPadding3D 3D 数据的零填充层(空间或时空)。

## <a name="2.3">2.3 池化层</a>

- MaxPooling1D 对于时序数据的最大池化。
- MaxPooling2D 对于空间数据的最大池化。
- MaxPooling3D 对于 3D（空间，或时空间）数据的最大池化。
- AveragePooling1D 对于时序数据的平均池化。
- AveragePooling2D 对于空间数据的平均池化。
- AveragePooling3D 对于 3D （空间，或者时空间）数据的平均池化。
- GlobalMaxPooling1D 对于时序数据的全局最大池化。
- GlobalAveragePooling1D 对于时序数据的全局平均池化。
- GlobalMaxPooling2D 对于空域数据的全局最大池化。
- GlobalAveragePooling2D 对于空域数据的全局平均池化。
- GlobalMaxPooling3D 对于 3D 数据的全局最大池化。
- GlobalAveragePooling3D 对于 3D 数据的全局平均池化。

## <a name="2.4">2.4 局部连接层</a>

- LocallyConnected1D 1D 输入的局部连接层。
- LocallyConnected2D 2D 输入的局部连接层。

## <a name="2.5">2.5 循环层</a>

- RNN 循环神经网络层基类。
- SimpleRNN 全连接的 RNN，其输出将被反馈到输入。
- GRU 门限循环单元网络
- LSTM 长短期记忆网络层
- ConvLSTM2D 卷积 LSTM
- SimpleRNNCell SimpleRNN 的单元类
- GRUCell GRU 层的单元类
- LSTMCell LSTM 层的单元类
- CuDNNGRU 由 CuDNN 支持的快速 GRU 实现
- CuDNNLSTM 由CuDNN 支持的快速 LSTM 实现

## <a name="2.6">2.6 嵌入层</a>

- Embedding  将正整数（索引值）转换为固定尺寸的稠密向量

## <a name="2.7">2.7 融合层</a>

- Add 计算输入张量列表的和
- Subtract 计算两个输入张量的差
- Multiply 计算输入张量列表的（逐元素间的）乘积
- Average 计算输入张量列表的平均值
- Maximum 计算输入张量列表的（逐元素间的）最大值
- Concatenate 连接一个输入张量的列表
- Dot 计算两个张量之间样本的点积
- add `Add` 层的函数式接口
- subtract `Subtract` 层的函数式接口
- Multiply `Multiply` 层的函数式接口
- average ` Average` 层的函数式接口
- maximum `Maximum` 层的函数式接口
- concatenate `Concatenate` 层的函数式接口
- dot `Dot` 层的函数式接口

## <a name="2.8">2.8 高级激活函数层</a>

- LeakyReLU 带泄漏的 ReLU
- PReLU 参数化的 ReLU
- ELU 指数线性单元
- ThresholdedReLU 带阈值的修正线性单元
- Softmax Softmax 激活函数
- ReLU 

```python
keras.layers.LeakyReLU(alpha=0.3)

keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

keras.layers.ELU(alpha=1.0)

keras.layers.ThresholdedReLU(theta=1.0)

keras.layers.Softmax(axis=-1)

keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
```



## <a name="2.9">2.9 标准化层</a>

- BatchNormalization 批量标准化层

```python
keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```



## <a name="2.10">2.10 噪声层</a>

- GaussianNoise 应用以 0 为中心的加性高斯噪声
- GaussianDropout 应用以 1 为中心的 乘性高斯噪声
- AlphaDropout 将 Alpha Dropout 应用到输入

```python
keras.layers.GaussianNoise(stddev)

keras.layers.GaussianDropout(rate)

keras.layers.AlphaDropout(rate, noise_shape=None, seed=None)
```



## <a name="2.11">2.11 层封装器</a>

- TimeDistributed 这个封装器将一个层应用于输入的每个时间片
- Bidirectional RNN 的双向封装器，对序列进行前向和后向计算

```python
keras.layers.TimeDistributed(layer)

keras.layers.Bidirectional(layer, merge_mode='concat', weights=None)
```

## <a name="2.12">2.12 编写自己的层</a>

对于简单、无状态的自定义操作，你也许可以通过layers.core.Lambda层来实现。但对于那些包含了可训练权重的自定义层，你应该自己实现这些层。

你需要实现三个方法

- build(input_shape): 这是你定义权重的地方，这个方法必须设 self.built = Ture,可以通过调用 super([Layer], self).build完成
- call(x):这里是编写层的功能逻辑的地方。你只需要关注传入call的第一个参数：输入张量，除非你希望你的层支持msking。
- compute_output_shape(input_shape) : 如果你的层更改了输入张量的形状，你应该在这里定义形状变化，这让Keras能够自动推断各层的形状。

```python
from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```



# <a name="3">3 数据预处理</a>

## <a name="3.1">3.1 序列预处理</a>

- TimeseriesGenerator
- pad_sequence
- skipgrams
- make_sampling_table

## <a name="3.2">3.2 文本预处理</a>

- Tokenizer
- hashing_trick
- one_hot
- text_to_word_sequence

## <a name="3.3">3.3 图像预处理</a>

ImageDataGenerator

# <a name="4">4 组件</a>

深度学习一整个流程包括很多流程，正向传播以及反向传播，还有评价指标等，每个流程，每个步骤都有着不同的影响。

## <a name="4.1">4.1 损失函数</a>

损失函数（或目标函数，优化评分函数）是编译模型时所需的两个参数之一

```python
model.compile(loss='mean_squared_error', optimizer='sgd')

from keras import losses
model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

### 可用损失函数

- mean_squared_error
- mean_absolute_error
- mean_absolute_percentage_error
- mean_squared_logarithmic_error
- squared_hinge
- hinge
- categorical_hinge
- logcosh
- categorical_crossentropy
- sparse_categorical_crossentropy
- binary_crossentropy
- kullback_leibler_divergence
- poisson
- cosine_proximity

```python
mean_squared_error(y_true, y_pred)

mean_absolute_error(y_true, y_pred)

mean_absolute_percentage_error(y_true, y_pred)

mean_squared_logarithmic_error(y_true, y_pred)

squared_hinge(y_true, y_pred)

hinge(y_true, y_pred)

categorical_hinge(y_true, y_pred)

logcosh(y_true, y_pred)

categorical_crossentropy(y_true, y_pred)

sparse_categorical_crossentropy(y_true, y_pred)

binary_crossentropy(y_true, y_pred)

kullback_leibler_divergence(y_true, y_pred)

poisson(y_true, y_pred)

cosine_proximity(y_true, y_pred)
```



## <a name="4.2">4.2 标准评估</a>

评价函数用于评估当前训练模型的性能。

评价函数和损失函数相似，只不过评价函数的结果不会用到训练过程中。

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])

from keras import metrics
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

- binary_accuracy
- categorical_accuracy
- sparse_categorical_accuracy
- top_k_categorical_accuracy
- sparse_top_k_categorical_accuracy

```python
binary_accuracy(y_true, y_pred)

categorical_accuracy(y_true, y_pred)

sparse_categorical_accuracy(y_true, y_pred)

top_k_categorical_accuracy(y_true, y_pred, k=5)

sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
```





## <a name="4.3">4.3 优化器</a>

优化器（optimizer）是编译Keras模型的所需两个参数之一

```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

Keras 优化器的公共参数

参数 `clipnorm` 和 `clipvalue` 能在所有的优化器中使用，用于控制梯度裁剪（Gradient Clipping）：

```python
from keras import optimizers

# 所有参数梯度将被裁剪，让其l2范数最大为1：g * 1 / max(1, l2_norm)
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

# 所有参数d 梯度将被裁剪到数值范围内：
# 最大值0.5
# 最小值-0.5
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```



### 优化器类型

- SGD 随机梯度下降优化器
- RMSprop
- Adagrad
- Adadelta
- Adam
- Adamax
- Nadam

```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
```





## <a name="4.4">4.4 激活函数</a>

激活函数可以通过设置单独的激活层来实现，也可以对构造对象时通过传递activation参数实现：

```python
from keras.layers import Activation, Dense
model.add(Dense(64))
model.add(Activation('tanh'))

# 等价于
model.add(Dense(64, activation='tanh'))
```

你也可以通过传递一个逐元素运算的 Theano/TensorFlow/CNTK 函数来作为激活函数：

```python
form keras import backend as K
model.add(Dense(64, activation=K.tanh))
model.add(Activation(K.tank))
```



- softmax
- elu 指数线性单元
- selu 可伸缩的指数线性单元
- softplus

- softsign
- relu 整流线性单元
- tanh 双曲正切
- sigmoid
- hard_sigmoid 
- exponential 自然数指数
- linear 线性激活函数

```python
keras.activations.softmax(x, axis=-1)

keras.activations.elu(x, alpha=1.0)

keras.activations.selu(x)

keras.activations.softplus(x)

keras.activations.softsign(x)

keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)

keras.activations.tanh(x)

sigmoid(x)

hard_sigmoid(x)

keras.activations.exponential(x)

keras.activations.linear(x)
```



## <a name="4.5">4.5 正则化</a>

正则化器允许在优化过程中对层的参数的激活情况进行惩罚。网络优化的损失函数也包括这些惩罚项。

惩罚是以层为对象进行的。具体的API因层而异，但Dense, Conv1D, Conv2D和Conv3D这些层具有统一的API



正则化开放油3个关键字参数：

- `kernel_regularizer`: `keras.regularizers.Regularizer` 的实例
- `bias_regularizer`: `keras.regularizers.Regularizer` 的实例
- `activity_regularizer`: `keras.regularizers.Regularizer` 的实例

例子

```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

可用的正则化器

```python
keras.regularizers.l1(0.)

keras.regularizers.l2(0.)

keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```



## <a name="4.6">4.6 初始化</a>

初始化定义了设置Keras各层权值随机初始化的方法

用来将初始化器传入Keras层的参数名取决于具体的层。通常光健字为 kernel_initializer和bias_initializer

- Initializer 初始化基类：所有初始器继承这个类
- Zeros 将张量初始值设为0
- Ones 将张量初始值设为1
- Constant 将张量初始值设为常数
- RandomNormal 按照正态分布随机生成张量
- RandomUniform 按照均匀分布生成随机张量
- TruncatedNormal 按照截断=尾正态分布生成随机张量
- VarianceScaling 初始化能够根据权值的尺寸调整其规模
- Orthogonal 生成一个随机正交矩阵
- Identity 生成单位矩阵的初始化器
- lecun_uniform LeCnn均匀初始化
- glorot_normal Glorot正态分布初始化
- glorot_uniform Glorot均匀分布初始化
- he_normal He正态分布初始化
- lecun_normal LeCun正态分布初始化器
- he_uniform He均方差缩放初始化

```python
keras.initializers.Initializer()

keras.initializers.Zeros()

keras.initializers.Ones()

keras.initializers.Constant(value=0)

keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)

keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)

keras.initializers.Orthogonal(gain=1.0, seed=None)

keras.initializers.Identity(gain=1.0)

keras.initializers.lecun_uniform(seed=None)

keras.initializers.glorot_normal(seed=None)

keras.initializers.glorot_uniform(seed=None)

keras.initializers.he_normal(seed=None)

keras.initializers.lecun_normal(seed=None)

keras.initializers.he_uniform(seed=None)
```



## <a name="4.7">4.7 回调函数</a>

回调函数式一个函数的集合，会在训练的阶段中所使用。你可以使用回调函数来查看训练模型的内在状态和统计。你可以传递一个列表的回调函数(作为callbacks 光健字参数)到sequential 或 Model 类型的 .fit() 方法。在训练的时候，响应的回调函数的方法就会被在各自的阶段被调用。



- Callback 用来组件新的回调函数的抽象类
- BaseLogger 会累积训练平均评估的回调函数
- TerminateOnNaN 当遇到NaN损失会停止训练的回调函数
- ProgbarLogger 会把评估以标准输出打印的回调函数
- History 把所有时间都记录到History对象的回调函数
- ModelCheckpoint 每个训练期之后保存模型
- EarlyStopping 当被检测的数量不在提升，则停止训练
- RemoteMonitor 将事件数据量到服务器的回调函数
- LearningRateScheduler学习速率定时器
- TensorBoard Tensorboard可视化
- ReducelLROnPlateau 当标准评估提升时，降低学习率
- CSVLogger 把训练结果数据流到csv文件的回调函数
- LambdaCallback 在训练进行中创建简单、自定义的回调函数
- 创建一个回调函数

```python
keras.callbacks.Callback()

keras.callbacks.BaseLogger(stateful_metrics=None)

keras.callbacks.TerminateOnNaN()

keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)

keras.callbacks.History()

keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None, send_as_json=False)

keras.callbacks.LearningRateScheduler(schedule, verbose=0)

keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

keras.callbacks.CSVLogger(filename, separator=',', append=False)

keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```



# <a name="5">5 数据集</a>

- CIFAR10小图像分类数据集
- CIFAR100小图像分类数据集
- IMDB电影评论情感分类数据集
- 路途社新闻主题分类
- MNIST手写数字数据集
- Fashion-MINST时尚物品数据集
- Boston房价回归数据集

# <a name="6">6 应用 Applications</a>

## <a name="6.1">6.1 可用模型</a>

在ImageNet上预训练过的用于图像分类的模型：

- Xception 

- VGG16
- VGG19
- RestNet, RestNetV2, RestNetXt
- IncetionV3
- InceptionResNetV2
- MobileNet
- MobileNetV2
- DenseNet
- NASNet

## <a name="6.2">6.2 模型概览</a>

模型概览

| 模型                                                         | 大小   | Top-1 准确率 | Top-5 准确率 | 参数数量    | 深度 |
| ------------------------------------------------------------ | ------ | ------------ | ------------ | ----------- | ---- |
| Xception        | 88 MB  | 0.790        | 0.945        | 22,910,480  | 126  |
| VGG16              | 528 MB | 0.713        | 0.901        | 138,357,544 | 23   |
| VGG19             | 549 MB | 0.713        | 0.900        | 143,667,240 | 26   |
| ResNet50          | 98 MB  | 0.749        | 0.921        | 25,636,712  | -    |
| ResNet101         | 171 MB | 0.764        | 0.928        | 44,707,176  | -    |
| ResNet152         | 232 MB | 0.766        | 0.931        | 60,419,944  | -    |
| ResNet50V2        | 98 MB  | 0.760        | 0.930        | 25,613,800  | -    |
| ResNet101V2                                                  | 171 MB | 0.772        | 0.938        | 44,675,560  | -    |
| ResNet152V2                                                  | 232 MB | 0.780        | 0.942        | 60,380,648  | -    |
| ResNeXt50                                                    | 96 MB  | 0.777        | 0.938        | 25,097,128  | -    |
| ResNeXt101     | 170 MB | 0.787        | 0.943        | 44,315,560  | -    |
| InceptionV3  | 92 MB  | 0.779        | 0.937        | 23,851,784  | 159  |
| InceptionResNetV2  | 215 MB | 0.803        | 0.953        | 55,873,736  | 572  |
| MobileNet      | 16 MB  | 0.704        | 0.895        | 4,253,864   | 88   |
| MobileNetV2    | 14 MB  | 0.713        | 0.901        | 3,538,984   | 88   |
| DenseNet121     | 33 MB  | 0.750        | 0.923        | 8,062,504   | 121  |
| DenseNet169     | 57 MB  | 0.762        | 0.932        | 14,307,880  | 169  |
| DenseNet201     | 80 MB  | 0.773        | 0.936        | 20,242,984  | 201  |
| NASNetMobile    | 23 MB  | 0.744        | 0.919        | 5,326,716   | -    |
| NASNetLarge     | 343 MB | 0.825        | 0.960        | 88,949,818  | -    |

# <a name="7">7 经典案例</a>

- 1 Addition RNN

- 2 Baby RNN

- 3 Baby MemNN

- 4 CIFAR-10CNN

- 5 CIFAR-10CNN-Capsule

- 6 CIFAR-10CNN with augmentation

- 7 CIFAR-10ResNet

- 8 Convolution fiter visualization

- 9 Image OCR

- 10 Bidirection LSTM







