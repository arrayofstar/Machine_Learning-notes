#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-05-18 17:43
# @Author  : Dreamstar
# @File    : test.py
# @Desc    :
import time

import numba
import numpy as np
from numba import cuda

print(np.__version__)
print(numba.__version__)

cuda.detect()

# 1. 向量相加
def vecAdd(n, a, b, c):
    for i in range(n):
        c[i] = a[i] + b[i]
    return out

@cuda.jit
def vecAdd_gpu(x, y, out):
    tx = cuda.threadIdx.x  # 当前线程在block中的索引值
    ty = cuda.blockIdx.x  # 当前线程所在block在grid中的索引值

    block_size = cuda.blockDim.x  # 每个block有多少个线程
    grid_size = cuda.gridDim.x  # 每个grid有多少个线程块

    start = tx + ty * block_size
    stride = block_size * grid_size

    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]


if __name__ == '__main__':
    x = np.random.random(size=(100000,)).astype(np.float32)
    y = np.random.random(size=(100000,)).astype(np.float32)
    out = np.empty_like(x)
    out1 = np.empty_like(x)
    # n = 100000
    # x = np.arange(n).astype(np.float32)
    # y = 2 * x
    # out = np.empty_like(x)
    # out1 = np.empty_like(x)

    threads_per_block = 128
    blocks_per_grid = 30

    # 1.向量相加
    t1 = time.time()
    vec_add = vecAdd(x.shape[0], x, y, out)
    print('cpu cost time is:', time.time() - t1)
    t2 = time.time()
    vecAdd_gpu[blocks_per_grid, threads_per_block](x, y, out1)
    print('gpu cost time is:', time.time() - t2)



    pass