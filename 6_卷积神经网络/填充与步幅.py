#!/usr/bin/en
# encoding: utf-8

"""
@author: kirk
@software: PyCharm
@file: 填充与步幅.py
@time: 2022/9/13 20:39
"""
import torch
from torch import nn

# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数均为1
    # print('+++++++++++++++')
    # print(X.shape)
    X = X.view((1, 1) + X.shape)
    # print('--------------')
    # print(X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding=1)

X = torch.rand(8, 8)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)


