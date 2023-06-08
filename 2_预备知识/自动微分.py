#!/usr/bin/en
# encoding: utf-8

"""
@author: kirk
@software: PyCharm
@file: 自动微分.py
@time: 2023/3/13 15:52
"""

import torch

x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
# torch.dot(input, other): 计算两个一维张量的点积。
#   intput: 点积中的第一个张量，必须是一维的。
#   other: 点积中的第二个张量，必须是一维的。
y.backward()    # 通过调用backward()自动计算y关于x每个分量的梯度。
print(x.grad)   # y = 2x^2

# 非标量变量的反向传播
x.grad.zero_()  # 在默认情况下，PyTorch会累积梯度，需要清除之前的值
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

# 分离计算
