#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-05-18 15:13
# @Author  : Dreamstar
# @File    : tools.py
# @Desc    :
import time
import numpy as np


# 用于计时的函数
class Timer:  # @save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

if __name__ == '__main__':
    ######################
    # Test : class Timer #
    ######################
    n = 1000000
    a = np.ones(n)
    b = np.ones(n)

    c = np.zeros(n)
    timer = Timer()  # 1.定义函数
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'{timer.stop():.6f} sec')  # 2.计时

    timer.start()  # 3.第二次开始计时
    d = a + b
    print(f'{timer.stop():.6f} sec')  # 4.第二次计时结束
