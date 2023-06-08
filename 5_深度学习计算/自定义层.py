#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: kirk
@software: PyCharm
@project: PytorchLearning
@file: 自定义层.py
@time: 2023/6/2 7:52
"""
import torch
import torch.nn.functional as F
from torch import nn

# 自定义层（与自定义网络没区别，自定义层也是继承Module类）
# 构造一个没有任何参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
# print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

# 将层作为组件合并到构建更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
# print(Y)

# 带参数的图层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
linear = MyLinear(5, 3)
print(linear.weight)
print(linear(torch.rand(2, 5)))

# net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
# net(torch.rand(2, 64))


