# -*- coding: utf-8 -*-
# @Time    : 2022/12/29 15:47
# @Author  : Dreamstar
# @File    : 3.learn_fourier_transform_via_gradient_descent.py
# @Desc    : 使用快速傅里叶变换FFT来训练一个神经网络来执行离散傅里叶变换DFT

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

losses = []
rmses = []

for i in tqdm(range(1000)):
    # Generate a random signal each iteration:
    x = np.random.random([1, signal_length]) - 0.5

    # Compute the expected result using the FFT:
    fft = np.fft.fft(x)
    y_true = np.hstack([fft.real, fft.imag])

    with tf.GradientTape() as tape:
        y_pred = tf.matmul(x, W_learned)
        loss = tf.reduce_sum(tf.square(y_pred - y_true))

    # Train weights, via gradient descent:
    W_gradient = tape.gradient(loss, W_learned)
    W_learned = tf.Variable(W_learned - 0.1 * W_gradient)

    losses.append(loss)
    rmses.append(np.sqrt(np.mean((W_learned - W_expected) ** 2)))

loss = np.average(losses)
rmse = np.average(rmses)
print(f'Final loss value:{loss}')
print(f"Final weights' rmse value:{rmse}")

plt.rcParams.update({'font.size': 8})  # 全局设置
plt.subplot(2, 2, 1)
plt.plot(losses)
plt.title('Loss', fontsize=10)
plt.xlabel('Iterations', fontsize=8)
plt.ylabel('Loss value', fontsize=8)
plt.xticks(fontsize=8)  # 设置标签
plt.yticks(fontsize=8)
plt.subplot(2, 2, 2)
plt.plot(rmses)
plt.title('Learned weights vs expected weights', fontsize=10)
plt.xlabel('Iterations', fontsize=8)
plt.ylabel('RMSE', fontsize=8)
plt.subplot(2, 2, 3)
plt.imshow(W_expected, vmin=-1, vmax=1, cmap='gray')
plt.title('Expected Fourier weights', fontsize=10)
plt.subplot(2, 2, 4)
plt.imshow(W_learned, vmin=-1, vmax=1, cmap='gray')
plt.title('Learned weights', fontsize=10)
plt.tight_layout()
plt.show()