# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 21:05
# @Author  : Dreamstar
# @File    : test_numba_cuda.py
# @Desc    : 对numba官网中的CUDA进行测试
import math
import torch
import numpy as np
# 1，GPU内核通常以下面的方式启动
from numba import cuda

# 1. 生成数据
# (1)数据转换
an_array = np.random.random(size=(1000, 1000))
an_array = cuda.to_device(an_array)
# (2)创建空的数组
an_empty_array = cuda.device_array(shape=(1000, 1000))  # numpy.empty()
#
an_array = cuda.device_array_like(an_array)
# 2.内核调用
threadsperblock = (16, 16)
blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

# 3.内核声明
# (1) 绝对定位
# 一维向量
@cuda.jit
def increment_by_one(an_array, out):
    pos = cuda.grid(1)
    if pos < an_array.size:
        an_array[pos] += 1
        out[pos] = an_array[pos] + 1

# 二维数组
@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
        an_array[x, y] += 1


increment_a_2D_array[blockspergrid, threadsperblock](an_array)

hary = an_array.copy_to_host()
pass

# (2) 手动计算定位
@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1
