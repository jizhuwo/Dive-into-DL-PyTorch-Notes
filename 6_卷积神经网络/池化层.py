#!/usr/bin/en
# encoding: utf-8

"""
@author: kirk
@software: PyCharm
@file: 池化层.py
@time: 2022/12/28 13:11
"""

import torch
from torch import nn


# def pool2d(X, pool_size, mode='max'):
#     X = X.float()
#     p_h, p_w = pool_size
#     Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
#     for i in range(Y.shape[0]):
#         for j in range(Y.shape[1]):
#             if mode == 'max':
#                 Y[i, j] = X[i: i + p_h, j: j + p_w].max()
#             elif mode == 'avg':
#                 Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
#     return Y
#
# X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# print(pool2d(X, (2, 2)))
# print(pool2d(X, (2, 2), 'avg'))


# 定义输入
# 四个参数分别表示 (batch_size, C_in, H_in, W_in)
# 分别对应，批处理大小，输入通道数，图像高度（像素），图像宽度（像素）
# 为了简化表示，我们只模拟单张图片输入，单通道图片，图片大小是4x4
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X)

# 仅定义一个 3x3 的池化层窗口
pool2d = nn.MaxPool2d(3)
print(pool2d(X))

# 定义一个 3x3 的池化层窗口;
# 周围填充了一圈 0;
# 步长为 2。
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

# 定义一个 2x4 的池化层窗口;
# 上下方向填充两行 0, 左右方向填充一行 0;
# 窗口将每次向右滑动 3 个元素位置，或者向下滑动 2 个元素位置。
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
print(pool2d(X))



# X = torch.cat((X, X + 1), dim=1)
# print(X)
#
# pool2d = nn.MaxPool2d(3, padding=1, stride=2)
# print(pool2d(X))

