#!/usr/bin/en
# encoding: utf-8

"""
@author: kirk
@software: PyCharm
@file: GPU.py
@time: 2023/5/18 13:51
"""
import torch
from torch import nn
import time

# torch.device('cuda:0')
# # print(torch.cuda.device_count())
#
# def try_gpu(i=0):  #@save
#     """如果存在，则返回gpu(i)，否则返回cpu()"""
#     if torch.cuda.device_count() >= i + 1:
#         return torch.device(f'cuda:{i}')
#     return torch.device('cpu')
#
# def try_all_gpus():  #@save
#     """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
#     devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
#     return devices if devices else [torch.device('cpu')]
# # print(try_gpu())
# # print(try_gpu(10))
# # print(try_all_gpus())
#
# X = torch.ones(2, 3, device='cuda:0')
# # print(X)
# Z = X.cuda(0)
# # print(X+Z)
# # print(X.cuda(0) is Z)
#
# net = nn.Sequential(nn.Linear(3, 1))
# net = net.to(device=try_gpu())
# # print(net(X))
# # print(net[0])           # Linear(in_features=3, out_features=1, bias=True)
# # print(net[0].weight.data.device)
# print(torch.cuda.get_device_name(0))


# 计算量较大的任务
X = torch.rand((10000, 10000))
Y = X.cuda(0)
time_start = time.time()
Z = torch.matmul(X, X)
time_end = time.time()
print(f'cpu time cost: {round((time_end - time_start) * 1000, 2)}ms')
time_start = time.time()
Z = torch.matmul(Y, Y)
time_end = time.time()
print(f'gpu time cost: {round((time_end - time_start) * 1000, 2)}ms')

# 计算量很小的任务
X = torch.rand((100, 100))
Y = X.cuda(0)
time_start = time.time()
Z = torch.matmul(X, X)
time_end = time.time()
print(f'cpu time cost: {round((time_end - time_start) * 1000)}ms')
time_start = time.time()
Z = torch.matmul(Y, Y)
time_end = time.time()
print(f'gpu time cost: {round((time_end - time_start) * 1000)}ms')