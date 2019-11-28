**目录**

- [1 创建模型](#1 创建模型)
	- [1.1 直接初始化 Layer(torch里面都是继承至Module的)](#1.1 直接初始化 Layer(torch里面都是继承至Module的))
	- [1.2 通过 Sequential 快速添加](#1.2 通过 Sequential 快速添加)
	- [1.3 通过 Sequential 的 add 方法一个个组件添加](#1.3 通过 Sequential 的 add 方法一个个组件添加)
	- [1.4 通过 Sequential 添加 dict 来添加](#1.4 通过 Sequential 添加 dict 来添加)
- [2 模型编译](#2 模型编译)
	- [2.1 损失函数](#2.1 损失函数)
	- [2.2 优化器Optimizer](#2.2 优化器Optimizer)
- [3 数据处理](#3 数据处理)
	- [3.1 直接用torch生成tensor -> Variable](#3.1 直接用torch生成tensor -> Variable)
	- [3.2 从numpy -> torch tensor -> Variable](#3.2 从numpy -> torch tensor -> Variable)
	- [3.3 采用自带的 dataset](#3.3 采用自带的 dataset)
	- [3.4 自定义的数据集](#3.4 自定义的数据集)




<font size=5>**torch Module 探索学习报告**</font>

**摘要：** 在Github搜索项目的时候，会看到很多torch的身影，torch的性能在之前也是有所耳闻，所以想能更深入的了解torch这个框架。学习的路径依然是按照对模型的学习、对一些层的内部逻辑的了解(torch里面其实都是Module)，然后能对Torch有一个基本的了解。通过对Module搭建方法、优化器、损失函数、训练方法的实践，基本了解了torch进行神经网络搭建已经训练的整个流程。



**探索学习记录**

本次探索主要从网络搜索相应的教程，然后去理解同时配合torch源码去理解



# <a name="1 创建模型">1 创建模型</a>

​	torch搭建模型主要通过四种方法，每种方法差别不大，但也都有着一些小的区别。



## <a name="1.1 直接初始化 Layer(torch里面都是继承至Module的)">1.1 直接初始化 Layer(torch里面都是继承至Module的)</a>

```python
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

class Net_1(torch.nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.pool_1 = torch.nn.MaxPool2d(2)
        self.dense_1 = torch.nn.Linear(32*3*3, 128) # 之所以是32*3*3是因为会经过一个池化层
        self.dense_2 = torch.nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.pool_1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)
        return x

summary(Net_1(), (3, 6, 6))

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1             [-1, 32, 6, 6]             896
         MaxPool2d-2             [-1, 32, 3, 3]               0
            Linear-3                  [-1, 128]          36,992
            Linear-4                   [-1, 10]           1,290
================================================================
Total params: 39,178
Trainable params: 39,178
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.01
Params size (MB): 0.15
Estimated Total Size (MB): 0.16
----------------------------------------------------------------
"""
```



## <a name="1.2 通过 Sequential 快速添加">1.2 通过 Sequential 快速添加</a>

```python
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

# 通过 Sequential 快速添加
class Net_2(torch.nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1),
                                        torch.nn.ReLU(), 
                                        torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(32*3*3, 128),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(128, 10))
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x
    
summary(Net_2(), (3, 6, 6))   
```



## <a name="1.3 通过 Sequential 的 add 方法一个个组件添加">1.3 通过 Sequential 的 add 方法一个个组件添加</a>

```python
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

# 通过 Sequential 的 add 方法一个个组件添加
class Net_3(torch.nn.Module):
    def __init__(self):
        super(Net_3, self).__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module('conv_1', torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv.add_module('relu_1', torch.nn.ReLU())
        self.conv.add_module('pool_1', torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential()
        self.dense.add_module('dense_1', torch.nn.Linear(32*3*3, 128))
        self.dense.add_module('relu_2', torch.nn.ReLU())
        self.dense.add_module('dense_2', torch.nn.Linear(128, 10))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x
    
summary(Net_3(), (3, 6, 6))  
print(Net_3())
```



## <a name="1.4 通过 Sequential 添加 dict 来添加">1.4 通过 Sequential 添加 dict 来添加</a>

```python
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

# 通过 Sequential 添加 dict 来添加

class Net_4(torch.nn.Module):
    def __init__(self):
        super(Net_4, self).__init__()
        self.conv = torch.nn.Sequential(
            OrderedDict([
                ("conv_1", torch.nn.Conv2d(3, 32, 3, 1, 1)),
                ("rule_1", torch.nn.ReLU()),
                ("pool", torch.nn.MaxPool2d(2))
            ])
        )
        self.dense = torch.nn.Sequential(
            OrderedDict([
                ("dense_1", torch.nn.Linear(32*3*3, 128)),
                ("rule_2", torch.nn.ReLU()),
                ("desnse_2", torch.nn.Linear(128, 10))
            ])
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

summary(Net_4(), (3, 6, 6))  
print(Net_4())
```



# <a name="2 模型编译">2 模型编译</a>

模型编译主要是设置损失函数、设置优化器的过程(torch后续还需要进行反向反向传播等操作)



**例**

```python
import torch 
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # xdata shape = (100,1) 
y = x.pow(2) + 0.2*torch.rand(x.size())
x, y = Variable(x), Variable(y)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 包含的层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        # 层连接
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x 

net = Net(1, 10, 1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%200 == 0:
        print('loss:', loss) 
```



## <a name="2.1 损失函数">2.1 损失函数</a>

```python
# 基本用法

criterion = LossCriterion() # 构建函数自己的参数
loss = criterion(x, y) # 调用
```

- class torch.nn.L1Loss(size_average=True)
  - 创建一个衡量输入x(模型预测输出)和目标y之间差的绝对值的平均值的标准。
- class torch.nn.MSELoss(size_average=True)
  - 创建一个衡量输入x(模型预测输出)和目标y之间均方误差标准。
- class torch.nn.CrossEntropyLoss(weight=None, size_average=True)
  - 此标准将LogSoftMax和NLLLoss集成到一个类中
- class torch.nn.NLLLoss(weight=None, size_average=True)
  - 负的log likelihood loss损失。用于训练一个n类分类器。
- class torch.nn.NLLLoss2d(weight=None, size_average=True)
  - 对于图片的 negative log likehood loss。计算每个像素的 NLL loss。
- class torch.nn.KLDivLoss(weight=None, size_average=True)
  - 计算 KL 散度损失
- class torch.nn.BCELoss(weight=None, size_average=True)
  - 计算 target 与 output 之间的二进制交叉熵
- class torch.nn.MarginRankingLoss(margin=0, size_average=True)
  - 创建一个标准，给定输入 𝑥1x1,𝑥2x2两个1-D mini-batch Tensor's，和一个𝑦y(1-D mini-batch tensor) ,𝑦y里面的值只能是-1或1。
- class torch.nn.HingeEmbeddingLoss(size_average=True) - 
- class torch.nn.MultiLabelMarginLoss(size_average=True)
- class torch.nn.SmoothL1Loss(size_average=True)
  - 平滑版L1 loss
- class torch.nn.SoftMarginLoss(size_average=True)
  - 创建一个标准，用来优化2分类的logistic loss。
- class torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=True)
  - 创建一个标准，基于输入x和目标y的 max-entropy，优化多标签 one-versus-all 的损失。
- class torch.nn.CosineEmbeddingLoss(margin=0, size_average=True)
  - 给定 输入 Tensors，x1, x2 和一个标签Tensor y(元素的值为1或-1)。
- class torch.nn.MultiMarginLoss(p=1, margin=1, weight=None, size_average=True)
  - 用来计算multi-class classification的hinge loss（magin-based loss）



## <a name="2.2 优化器Optimizer">2.2 优化器Optimizer</a>



```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```



- class torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
  - 实现Adadelta算法
- class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)
  - 实现Adagrad算法
- class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
  - 实现Adam算法
- class torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
  - 实现Adamax算法（Adam的一种基于无穷范数的变种）
- class torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
  - 实现平均随机梯度下降算法。
- class torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
  - 实现L-BFGS算法
- class torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
  - 实现RMSprop算法
- class torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
  - 实现弹性反向传播算法。
- class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
  - 实现随机梯度下降算法（momentum可选）



# <a name="3 数据处理">3 数据处理</a>

数据处理可以用多少种方式，但是最终都需要处理成torch需要的数据类型，否则会报错



主要探索的处理方式:

- 直接用torch生成tensor -> Variable
- 从numpy -> torch tensor -> Variable
- 采用自带的 dataset
- 自定义的数据集



## <a name="3.1 直接用torch生成tensor -> Variable">3.1 直接用torch生成tensor -> Variable</a>

```python
import torch 
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # xdata shape = (100,1) 
y = x.pow(2) + 0.2*torch.rand(x.size())
x, y = Variable(x), Variable(y)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 包含的层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        # 层连接
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x 

net = Net(1, 10, 1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%200 == 0:
        print('loss:', loss) 
```



## <a name="3.2 从numpy -> torch tensor -> Variable">3.2 从numpy -> torch tensor -> Variable</a>

```python
import torch 
from torch.autograd import Variable
import numpy as np

x = np.linspace(-1, 1, 100).reshape((100, -1)).astype(np.float32)
y = x**2 + 0.2*np.random.random(x.shape).astype(np.float32)  

x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y)) 

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 包含的层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        # 层连接
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x 

net = Net(1, 10, 1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%200 == 0:
        print('loss:', loss) 
```



## <a name="3.3 采用自带的 dataset">3.3 采用自带的 dataset</a>

```python
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
%matplotlib inline

torch.manual_seed(1)

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data

mnist_base_path="/Users/zhouwencheng/Desktop/Grass/data/picture/mnist"
train_data = dsets.MNIST(
    root = mnist_base_path,
    train=True,
    transform=transforms.ToTensor(), # (0, 1)
    download=DOWNLOAD_MNIST
)

print(train_data.train_data.size())
print(train_data.targets.size())

train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_data = dsets.MNIST(root=mnist_base_path, 
                        train=False, 
                        transform=transforms.ToTensor())
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets.numpy().squeeze()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(               # if use nn.RNN(), it hardly learns
            input_size = INPUT_SIZE,
            hidden_size = 64,     # rnn hidden unit
            num_layers = 1,        # number of rnn layer
            batch_first = True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 10)
        
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None) # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1,:])
        return out
    
rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR) # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss() # the target label is not one-hotted

# train and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28))
        b_y = Variable(y)
        
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y)/float(test_y.size)
            print(f'Epoch:  {epoch} | step:{step} | train loss: {loss.data} | test accuracy: {accuracy}')
#             print(loss.data)
    print("OK")
            
```



## <a name="3.4 自定义的数据集">3.4 自定义的数据集</a>

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms 
import numpy as np


class CustomDataset(Dataset):#需要继承data.Dataset
    
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass
    
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        
        data = np.random.random(10).astype(np.float32)
        label = np.random.randint(0, 2)  

        data = Variable(torch.from_numpy(data)) 
        return data, label 
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 1000
    
a = 100
train_data = CustomDataset()
train_loader = DataLoader(dataset=train_data, batch_size=20, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 包含的层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        # 层连接
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x 

net = Net(10, 10, 2)

print(net)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR) # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss() # the target label is not one-hotted

# train and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = Variable(x)
        y = Variable(y)
        output = net(x) 
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if step%10 == 0 :
            print(loss)
    


```

