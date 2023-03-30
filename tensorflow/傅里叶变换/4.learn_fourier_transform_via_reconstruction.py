# -*- coding: utf-8 -*-
# @Time    : 2022/12/29 15:47
# @Author  : Dreamstar
# @File    : 4.learn_fourier_transform_via_reconstruction.py
# @Desc    : 使用重构输入信号来训练一个神经网络来执行离散傅里叶变换DFT

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def create_fourier_weights(signal_length):
    "Create weights, as described above."
    k_vals, n_vals = np.mgrid[0:signal_length, 0:signal_length]
    theta_vals = 2 * np.pi * k_vals * n_vals / signal_length
    return np.hstack([np.cos(theta_vals), -np.sin(theta_vals)])


import tensorflow as tf

signal_length = 32

# Initialise weight vector to train:
W_learned = tf.Variable(np.random.random([signal_length, 2 * signal_length]) - 0.5)

# Expected weights, for comparison:
W_expected = create_fourier_weights(signal_length)

tvals = np.arange(signal_length).reshape([-1, 1])
freqs = np.arange(signal_length).reshape([1, -1])
arg_vals = 2 * np.pi * tvals * freqs / signal_length
cos_vals = tf.cos(arg_vals) / signal_length
sin_vals = tf.sin(arg_vals) / signal_length

losses = []
rmses = []

for i in tqdm(range(10000)):
    x = np.random.random([1, signal_length]) - 0.5

    with tf.GradientTape() as tape:
        y_pred = tf.matmul(x, W_learned)
        y_real = y_pred[:, 0:signal_length]
        y_imag = y_pred[:, signal_length:]
        sinusoids = y_real * cos_vals - y_imag * sin_vals
        reconstructed_signal = tf.reduce_sum(sinusoids, axis=1)
        loss = tf.reduce_sum(tf.square(x - reconstructed_signal))

    W_gradient = tape.gradient(loss, W_learned)
    W_learned = tf.Variable(W_learned - 0.5 * W_gradient)

    losses.append(loss)
    rmses.append(np.sqrt(np.mean((W_learned - W_expected) ** 2)))

reconstructed_signal = np.sum(sinusoids, axis=1)

loss = np.average(losses)
rmse = np.average(rmses)
print(f'Final loss value:{loss}')
print(f"Final weights' rmse value:{rmse}")

plt.rcParams.update({'font.size': 8})  # 全局设置
plt.subplot(3, 2, 1)
plt.plot(losses)
plt.title('Loss', fontsize=10)
plt.xlabel('Iterations', fontsize=8)
plt.ylabel('Loss value', fontsize=8)
plt.xticks(fontsize=8)  # 设置标签
plt.yticks(fontsize=8)
plt.subplot(3, 2, 2)
plt.plot(rmses)
plt.title('Learned weights vs expected weights', fontsize=10)
plt.xlabel('Iterations', fontsize=8)
plt.ylabel('RMSE', fontsize=8)
plt.subplot(3, 2, 3)
plt.imshow(W_expected, vmin=-1, vmax=1, cmap='gray')
plt.title('Expected Fourier weights', fontsize=10)
plt.subplot(3, 2, 4)
plt.imshow(W_learned, vmin=-1, vmax=1, cmap='gray')
plt.title('Learned weights', fontsize=10)
plt.subplot(3, 2, 5)
plt.plot(x[0,:])
plt.title('Original signal')
plt.subplot(3, 2, 6)
plt.plot(reconstructed_signal)
plt.title('Signal reconstructed from sinusoids')
plt.tight_layout()
plt.show()