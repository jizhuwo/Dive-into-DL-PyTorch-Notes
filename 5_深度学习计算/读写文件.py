#!/usr/bin/en
# encoding: utf-8

"""
@author: kirk
@software: PyCharm
@file: 读写文件.py
@time: 2023/5/18 20:30
"""

import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)

y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
print((x2, y2))

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# state_dict：是一个字典，包含只有可学习层的参数（权重和偏差）

torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))

# model.train()和model.eval()的区别主要在于Batch Normalization和Dropout两层:
#     - model.train(): 模型中有BN层(Batch Normalization)和Dropout，在训练时添加model.train()。
#       保证BN层能够用到每一批数据的均值和方差，对于Dropout，model.train()是随机舍弃神经元更新参数。
#     - model.eval(): 模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。
#       保证BN层能够用到全部训练数据的均值和方差，对于Dropout，model.eval()是不随机舍弃神经元。
#       在model(test)之前，需要加上model.eval()。否则，有输入数据，即使不训练，它也会改变权值。
#       这是model中含有BN层和Dropout所带来的的性质。
clone.eval()
Y_clone = clone(X)
print(Y_clone == Y)

