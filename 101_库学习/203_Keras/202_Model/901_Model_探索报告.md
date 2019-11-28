**目录**

- [1 探索过程记录](#1 探索过程记录)
	- [1.1 创建Model](#1.1 创建Model)
		- [1.1.1 **Sequential**](#1.1.1 **Sequential**)
		- [1.1.2 **Model**](#1.1.2 **Model**)
	- [1.2 编译Model](#1.2 编译Model)
		- [1.2.1 **整体代码**](#1.2.1 **整体代码**)
		- [1.2.2 **损失函数**](#1.2.2 **损失函数**)
		- [1.2.3 **优化器**](#1.2.3 **优化器**)
		- [1.2.4 **评价函数**](#1.2.4 **评价函数**)
	- [1.3 训练Model](#1.3 训练Model)
		- [1.3.1 **fit**](#1.3.1 **fit**)
		- [1.3.2 **fit_generator**](#1.3.2 **fit_generator**)
		- [1.3.3 **train_on_batch**](#1.3.3 **train_on_batch**)
	- [1.4 重先整合Model](#1.4 重先整合Model)




<font size=5>**Keras Model 探索报告**</font>



**时间：2019-11-26**



**摘要：**Keras Model是Keras神经网络的一个模型，是Layer的容器，于是想要真正去理解和灵活运用Keras实现的各种神经网络就必须去理解Model。理解Model就需要从创建、编译、训练模型这些步骤去一一探索。通过对Model的创建、编译、训练以及Layer的获取和重先整合，已经对Model有了一个初步的了解，往后可以再这个基础上继续更深入的了解。



# <a name="1 探索过程记录">1 探索过程记录</a>

​	本次探索主要从网络搜索相应的教程，去理解整个过程同时配合keras的源码来加深这种理解。

## <a name="1.1 创建Model">1.1 创建Model</a>

​	创建Model有两种方法，一种是通过Sequential。另一种是Model()。



### <a name="1.1.1 **Sequential**">1.1.1 **Sequential**</a>

```python
from keras.models import Sequential
from keras.layers import Dense, Activation 

model = Sequential()
model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
model.summary()

""" 网络结构
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 64)                6464      
_________________________________________________________________
activation_1 (Activation)    (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                650       
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0         
=================================================================
Total params: 7,114
Trainable params: 7,114
Non-trainable params: 0
_________________________________________________________________
"""
```

```python
from keras.models import Sequential
from keras.layers import Dense, Activation 

model = Sequential([
    (Dense(units=64, input_dim=100)),
    (Activation('relu')),
    (Dense(units=10)),
    (Activation('softmax'))
])
model.name = "CL"
model.summary()

"""网络结构
Model: "CL"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_5 (Dense)              (None, 64)                6464      
_________________________________________________________________
activation_5 (Activation)    (None, 64)                0         
_________________________________________________________________
dense_6 (Dense)              (None, 10)                650       
_________________________________________________________________
activation_6 (Activation)    (None, 10)                0         
=================================================================
Total params: 7,114
Trainable params: 7,114
Non-trainable params: 0
_________________________________________________________________
"""
```

### <a name="1.1.2 **Model**">1.1.2 **Model**</a>

```python
from keras.layers import Input, Dense
from keras.models import Model

# this returen a tensor
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.name = "MINST-Dense"

model.summary()

"""网络结构
Model: "MINST-Dense"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 64)                50240     
_________________________________________________________________
dense_11 (Dense)             (None, 64)                4160      
_________________________________________________________________
dense_12 (Dense)             (None, 10)                650       
=================================================================
Total params: 55,050
Trainable params: 55,050
Non-trainable params: 0
_________________________________________________________________
"""
```



## <a name="1.2 编译Model">1.2 编译Model</a>

​	编译Model主要是对神经网络损失函数、优化器、评价函数的选择。

### <a name="1.2.1 **整体代码**">1.2.1 **整体代码**</a>

```python
from keras.layers import Input, Dense
from keras.models import Model

# this returen a tensor
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.name = "CL-Dense"

model.compile(loss='categorical_crossentropy',
             optimizer='sgd', metrics=['accuracy']) 
```



### <a name="1.2.2 **损失函数**">1.2.2 **损失函数**</a>

- mean_squared_error
  - mean_squared_error(y_true, y_pred)
- mean_absolute_error
  - mean_absolute_error(y_true, y_pred)
- mean_absolute_percentage_error
  - mean_absolute_percentage_error(y_true, y_pred)
- mean_squared_logarithmic_error
  - mean_squared_logarithmic_error(y_true, y_pred)
- squared_hinge
  - squared_hinge(y_true, y_pred)
- hinge
  - hinge(y_true, y_pred)
- categorical_hinge
  - categorical_hinge(y_true, y_pred)
- logcosh
  - logcosh(y_true, y_pred)
- categorical_crossentropy
  - categorical_crossentropy(y_true, y_pred)
- sparse_categorical_crossentropy
  - sparse_categorical_crossentropy(y_true, y_pred)
- binary_crossentropy
  - binary_crossentropy(y_true, y_pred)
- kullback_leibler_divergence
  - kullback_leibler_divergence(y_true, y_pred)
- poisson
  - poisson(y_true, y_pred)
- cosine_proximity
  - cosine_proximity(y_true, y_pred)



### <a name="1.2.3 **优化器**">1.2.3 **优化器**</a>

- sgd
  - SGD
- rmsprop
  - RMSprop
- adagrad
  - Adagrad
- adadelta
  - Adadelta
- adam
  - Adam
- adamax
  - Adamax
- nadam
  - Nadam



### <a name="1.2.4 **评价函数**">1.2.4 **评价函数**</a>

- binary_accuracy
  - binary_accuracy(y_true, y_pred)
- categorical_accuracy
  - categorical_accuracy(y_true, y_pred)
- sparse_categorical_accuracy
  - sparse_categorical_accuracy(y_true, y_pred)
- top_k_categorical_accuracy
  - top_k_categorical_accuracy(y_true, y_pred, k=5)
- sparse_top_k_categorical_accuracy
  - sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)

**从loss引入的**

- mse = MSE = mean_squared_error
- mae = MAE = mean_absolute_error
- mape = MAPE = mean_absolute_percentage_error
- msle = MSLE = mean_squared_logarithmic_error
- cosine = cosine_proximity

## <a name="1.3 训练Model">1.3 训练Model</a>

​	训练Model有fit、fit_generator、train_on_batch三种方式，其中常用的方式为fit、fit_generator，train_on_batch方式很少用到。

   fit在数据量小的时候比较合适，在数据量大的时候用fit_generator的方式会比较合适



以下的代码展示：（数据是随机生成的）



### <a name="1.3.1 **fit**">1.3.1 **fit**</a>

```python
# this returen a tensor
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.name = "CL-Dense"

model.compile(loss='categorical_crossentropy',
             optimizer='sgd', metrics=['accuracy'])

# model.summary()

import numpy as np
from keras.utils import np_utils

train_data = np.random.random((1000, 784))
train_label = np.random.randint(0, 10, (1000,))
test_data = np.random.random((200, 784))
test_label = np.random.randint(0, 10, (200,)) 

test_label = np_utils.to_categorical(test_label, num_classes=10)
train_label = np_utils.to_categorical(train_label, num_classes=10)

model.fit(x=train_data, 
          y=train_label, 
          batch_size=32,
          epochs=1,
          validation_data=(test_data, test_label))
```



### <a name="1.3.2 **fit_generator**">1.3.2 **fit_generator**</a>

```python
from keras.layers import Input, Dense
from keras.models import Model

# this returen a tensor
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.name = "CL-Dense"

model.compile(loss='categorical_crossentropy',
             optimizer='sgd', metrics=['accuracy'])

# model.summary()

import numpy as np
from keras.utils import np_utils

def get_generate(num=1000, batch_size=32):
    num_batch = 10000//32
    for _ in range(num_batch):
        data = np.random.random((batch_size, 784))
        label = np.random.randint(0, 10, (batch_size,))
        label = np_utils.to_categorical(label, num_classes=10)
        yield (data, label) 

        
v_data = np.random.random((32, 784))
v_label = np.random.randint(0, 10, (32,))
v_label = np_utils.to_categorical(v_label, num_classes=10)
        
model.fit_generator(get_generate(),
                    steps_per_epoch=10, 
                    epochs=2, 
                    max_queue_size=1,
                    validation_data=(v_data, v_label)) 
```



### <a name="1.3.3 **train_on_batch**">1.3.3 **train_on_batch**</a>

```python
from keras.layers import Input, Dense
from keras.models import Model

# this returen a tensor
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.name = "CL-Dense"

model.compile(loss='categorical_crossentropy',
             optimizer='sgd', metrics=['accuracy'])

# model.summary()

import numpy as np
from keras.utils import np_utils

for n_b in range(1000):
    data = np.random.random((32, 784))
    label = np.random.randint(0, 10, (32,))
    label = np_utils.to_categorical(label, num_classes=10)
    cost = model.train_on_batch(data, label)
    if n_b % 100 == 0:
        print(f'loss: {cost}')
    
print('----Testing----' ) 
        
v_data = np.random.random((32, 784))
v_label = np.random.randint(0, 10, (32,))
v_label = np_utils.to_categorical(v_label, num_classes=10)
cost = model.evaluate(v_data, v_label, batch_size = 32) 
print(f'loss: {cost}')
```



## <a name="1.4 重先整合Model">1.4 重先整合Model</a>

​	模型整合主要的目的就是应对迁移学习，还有模型融合。在很多任务中是需要去修改一些已经成型的模型，比如Bert这些有预训练的模型；有些时候就需要在一些成熟的模型上进行修改，比如实现一些论文上的模型，这时候就需要拥有模型的修改能力了。

​	不过此次探索只是初步的探索，并没有太深入，结合之前对Layer的理解应该能应付基础的处理，再深入则需要往后的学习和理解。



```python
from keras.layers import Input, Dense
from keras.models import Model

# 原模型
inputs = Input(shape=(784,), name="Input")
x = Dense(64, activation='relu', name="D-1")(inputs)
x = Dense(64, activation='relu', name="D-2")(x)
outputs = Dense(10, activation='softmax', name="D-3")(x)
model = Model(inputs=inputs, outputs=outputs)
model.name = "D-E"

# 查看模型的Layer
for layer in model.layers:
    print(f"layer name: {layer.name}")
"""
layer name: Input
layer name: D-1
layer name: D-2
layer name: D-3
"""

# 重先 整合 整个模型
i_i = model.get_layer(name="Input").input
d1_o = model.get_layer(name="D-1").output
d2_o = model.get_layer(name="D-2").output
d3_o = model.get_layer(name="D-3").output

model_ = Model(inputs=i_i, outputs=d2_o)
model_.summary()
"""
Model: "model_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Input (InputLayer)           (None, 784)               0         
_________________________________________________________________
D-1 (Dense)                  (None, 64)                50240     
_________________________________________________________________
D-2 (Dense)                  (None, 64)                4160      
=================================================================
Total params: 54,400
Trainable params: 54,400
Non-trainable params: 0
_________________________________________________________________
"""

i_i = model.get_layer(name="Input").input 
d2_o = model.get_layer(name="D-2").output 
d3_o = Dense(20, activation='softmax', name="D-3")(d2_o)

model_2 = Model(inputs=i_i, outputs=d3_o)
model_2.summary()
"""
Model: "model_9"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Input (InputLayer)           (None, 784)               0         
_________________________________________________________________
D-1 (Dense)                  (None, 64)                50240     
_________________________________________________________________
D-2 (Dense)                  (None, 64)                4160      
_________________________________________________________________
D-3 (Dense)                  (None, 20)                1300      
=================================================================
Total params: 55,700
Trainable params: 55,700
Non-trainable params: 0
_________________________________________________________________
"""
```

