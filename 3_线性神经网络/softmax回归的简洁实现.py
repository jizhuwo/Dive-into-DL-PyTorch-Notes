#!/usr/bin/en
# encoding: utf-8

"""
@author: kirk
@software: PyCharm
@file: softmax回归的简洁实现.py
@time: 2023/3/17 1:44
"""
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# torch.nn.Flatten()默认从第二维开始平坦化;
# torch.Flatten()默认将张量拉成一维的向量，也就是说从第一维开始平坦化。

def init_weights(m):            # 以均值0和标准差0.01随机初始化权重
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)         # 将init_weight递归地应用到该模块以及该模块的每一个子模块

loss = nn.CrossEntropyLoss(reduction='none')
# reduction='None': 不减少;
# reduction='mean': 取输出的加权平均值;
# reduction='sum': 输出将被求和;

trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# torch.nn.Parameter():
# 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
# 并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
# 所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
# 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.draw()

