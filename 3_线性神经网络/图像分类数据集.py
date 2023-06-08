#!/usr/bin/en
# encoding: utf-8

"""
@author: kirk
@software: PyCharm
@file: 图像分类数据集.py
@time: 2023/2/5 19:45
"""

import torch
import torchvision
import sys
import time
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


# # 读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
feature, label = mnist_train[0]
print(feature.shape, label)             # 通过下标来访问任意一个样本
print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape)


def get_fashion_mnist_labels(labels):  #@save
    """将数值标签转换成相应的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    '''在一行里画出多张图像和对应标签的函数'''
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))    # plt.subplots(行数, 列数, Figure显示框大小)
    # https://www.runoob.com/matplotlib/matplotlib-subplots.html
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view(28, 28).numpy())      # imshow: 将数据显示为图像
        f.set_title(lbl)    # 设置标题
        f.axis('off')       # 关闭坐标轴
    plt.show()


# 训练集中前10个样本的图像内容和文本标签
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))


# 训练数据集中前18个样本的图像内容和文本标签
# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
#     """绘制图像列表"""
#     d2l.use_svg_display()
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         if torch.is_tensor(img):
#             # 图片张量
#             ax.imshow(img.numpy())
#         else:
#             # PIL图片
#             ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     plt.show()
#
# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))


# 多进程来加速数据读取
# 在Windows上，使用代码的所有multiprocessing必须由if __name__ == "__main__":保护

# if __name__ == "__main__":
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0         # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))


