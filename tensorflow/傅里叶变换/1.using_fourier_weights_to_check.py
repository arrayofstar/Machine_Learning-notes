# -*- coding: utf-8 -*-
# @Time    : 2022/12/29 15:47
# @Author  : Dreamstar
# @File    : 1.using_fourier_weights.py
# @Desc    : 使用傅里叶权重计算傅里叶变换来验证方法可靠性 - 体现了傅里叶权重矩阵和numpy中调用的快速傅里叶变换的结果的rmse是很接近的

import numpy as np
from matplotlib import pyplot as plt


def create_fourier_weights(signal_length):
    "Create weights, as described above."
    k_vals, n_vals = np.mgrid[0:signal_length, 0:signal_length]
    theta_vals = 2 * np.pi * k_vals * n_vals / signal_length
    return np.hstack([np.cos(theta_vals), -np.sin(theta_vals)])

# Generate data:
signal_length = 64
x = np.random.random(size=[1, signal_length]) - 0.5  # 创建一个随机信号

# Compute Fourier transform using method described above:
W_fourier = create_fourier_weights(signal_length)
y = np.matmul(x, W_fourier)  # 使用傅里叶权重与随机信号相乘

# Compute Fourier transform using the fast Fourier transform:
fft = np.fft.fft(x)
y_fft = np.hstack([fft.real, fft.imag])

# Compare the results:
print('rmse: ', np.sqrt(np.mean((y - y_fft)**2)))

plt.subplot(2, 1, 1)
plt.plot(y[0,:])
plt.title('Original y signal')
plt.subplot(2, 1, 2)
plt.plot(y_fft[0,:])
plt.title('y_fft signal after DFT')
plt.tight_layout()
plt.show()