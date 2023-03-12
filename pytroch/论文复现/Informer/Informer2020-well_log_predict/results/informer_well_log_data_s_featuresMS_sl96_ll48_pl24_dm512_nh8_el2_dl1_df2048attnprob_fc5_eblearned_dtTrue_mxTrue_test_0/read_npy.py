# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 22:08
# @Author  : Dreamstar
# @File    : read_npy.py
# @Desc    :

import numpy as np

# file = np.load('metrics.npy')
# print(['mae', 'mse', 'rmse', 'mape', 'mspe'])
# print(file.tolist())
# np.savetxt('metrics.txt',file)


file = np.load('pred.npy')
print(['pred'])
print(file.tolist())
np.savetxt('pred.txt', file)


file = np.load('true.npy')
print(['true'])
print(file.tolist())
np.savetxt('true.txt', file)