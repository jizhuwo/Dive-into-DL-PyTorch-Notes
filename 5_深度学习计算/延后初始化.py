#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: kirk
@software: PyCharm
@project: PytorchLearning
@file: 延后初始化.py
@time: 2023/6/2 7:41
"""
import torch
from torch import nn

"""延后初始化"""
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
# print(net[0].weight)  # 尚未初始化
print(net)

X = torch.rand(2, 20)
net(X)
print(net)

