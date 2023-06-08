import numpy as np
import torch
import math
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x  # f(x) = x * x的导数为f'(x) = 2 * x
        results.append(x)
    print('epoch 10, x:', x)
    return results

res = gd(0.2)
print(res)

