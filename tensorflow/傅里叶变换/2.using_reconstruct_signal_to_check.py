# -*- coding: utf-8 -*-
# @Time    : 2022/12/29 15:47
# @Author  : Dreamstar
# @File    : 2.using_reconstruct_signal_to_check.py
# @Desc    : 使用重构原始信号的方式来验证傅里叶权重矩阵的可靠性: 重构回去的x信号rmse和曲线形态均很接近

import matplotlib.pyplot as plt
import numpy as np

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

y_real = y[:, :signal_length]
y_imag = y[:, signal_length:]
tvals = np.arange(signal_length).reshape([-1, 1])
freqs = np.arange(signal_length).reshape([1, -1])
arg_vals = 2 * np.pi * tvals * freqs / signal_length
sinusoids = (y_real * np.cos(arg_vals) - y_imag * np.sin(arg_vals)) / signal_length
reconstructed_signal = np.sum(sinusoids, axis=1)

print('rmse:', np.sqrt(np.mean((x - reconstructed_signal)**2)))

plt.subplot(2, 1, 1)
plt.plot(x[0,:])
plt.title('Original signal')
plt.subplot(2, 1, 2)
plt.plot(reconstructed_signal)
plt.title('Signal reconstructed from sinusoids after DFT')
plt.tight_layout()
plt.show()