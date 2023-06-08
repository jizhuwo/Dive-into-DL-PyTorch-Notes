#!/usr/bin/en
# encoding: utf-8

"""
@author: kirk
@software: PyCharm
@file: 线性回归的简洁实现.py
@time: 2023/3/14 15:25
"""
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn                    # nn是神经网络的缩写

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)      # TensorDataset本质上与python zip方法类似，对数据进行打包整合。
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
# next(iter(data_iter))


net = nn.Sequential(nn.Linear(2, 1))    # 顺序容器
net[0].weight.data.normal_(0, 0.01)     # 初始化模型参数
net[0].bias.data.fill_(0)
loss = nn.MSELoss()                     # 损失函数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)    # 优化算法

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')     # f'': 格式化字符串常量,大括号{}表示被替换字段

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

