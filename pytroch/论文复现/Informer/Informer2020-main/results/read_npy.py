# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 22:08
# @Author  : Dreamstar
# @File    : read_npy.py
# @Desc    :

import numpy as np

file = np.load('pred.npy')
print(['mae', 'mse', 'rmse', 'mape', 'mspe'])
print(file)
# np.savetxt('metrics.txt',file)