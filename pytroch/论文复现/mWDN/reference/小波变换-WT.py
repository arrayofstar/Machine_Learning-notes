# -*- coding: utf-8 -*-
# @Time    : 2023/5/13 22:22
# @Author  : Dreamstar
# @File    : 小波变换-wavelet transform.py
# @Link    : https://blog.csdn.net/abc1234abcdefg/article/details/123517320 - 小波变换python实现
# @Link    : https://blog.csdn.net/weixin_46713695/article/details/127049520 - 傅里叶变换及小波变换（深入浅出）
# @Link    : https://blog.csdn.net/weixin_42341666/article/details/107026123 -
# @Link    : https://blog.csdn.net/weixin_39107270/article/details/129627802 - 小波分解、小波重构、小波去噪
# @Desc    : pywt的安装方式（！！很重要！！） - pip install PyWavelets


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

# # 小波
# sampling_rate = 1024
# t = np.arange(0, 1.0, 1.0 / sampling_rate)
# f1 = 100
# f2 = 200
# f3 = 300
# f4 = 400
# data = np.piecewise(t, [t < 1, t < 0.8, t < 0.5, t < 0.3],
#                     [lambda t: 400*np.sin(2 * np.pi * f4 * t),
#                      lambda t: 300*np.sin(2 * np.pi * f3 * t),
#                      lambda t: 200*np.sin(2 * np.pi * f2 * t),
#                      lambda t: 100*np.sin(2 * np.pi * f1 * t)])
# wavename = 'cgau8'
# totalscal = 256
# fc = pywt.central_frequency(wavename)
# cparam = 2 * fc * totalscal
# scales = cparam / np.arange(totalscal, 1, -1)
# [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
# plt.figure(figsize=(8, 4))
# plt.subplot(211)
# plt.plot(t, data)
# plt.xlabel("t(s)")
# plt.title('shipinpu',  fontsize=20)
# plt.subplot(212)
# plt.contourf(t, frequencies, abs(cwtmatr))
# plt.ylabel(u"prinv(Hz)")
# plt.xlabel(u"t(s)")
# plt.subplots_adjust(hspace=0.4)
# plt.show()


fs = 1000
N = 200
k = np.arange(200)
frq = k * fs / N
frq1 = frq[range(int(N / 2))]

# aa = []
# for i in range(200):
#     aa.append(np.sin(0.3 * np.pi * i))
# for i in range(200):
#     aa.append(np.sin(0.13 * np.pi * i))
# for i in range(200):
#     aa.append(np.sin(0.05 * np.pi * i))
# y = aa
df = pd.read_csv("../data/stretch/aligned_well_01_1_strech_1.csv")
y = df['RHOB'][4000:8000].values


wavename = 'db5'
cA, cD = pywt.dwt(y, wavename)
ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component
x = df['DEPT'][4000:8000].values
plt.figure(figsize=(12, 9))
plt.subplot(311)
plt.plot(x, y)
plt.title('original signal')
plt.subplot(312)
plt.plot(x, ya)
plt.title('approximated component')
plt.subplot(313)
plt.plot(x, yd)
plt.title('detailed component')
plt.tight_layout()
plt.show()

# 图像单边谱
plt.figure(figsize=(12, 9))
plt.subplot(311)
data_f = abs(np.fft.fft(cA)) / N
data_f1 = data_f[range(int(N / 2))]
plt.plot(frq1, data_f1, 'red')

plt.subplot(312)
data_ff = abs(np.fft.fft(cD)) / N
data_f2 = data_ff[range(int(N / 2))]
plt.plot(frq1, data_f2, 'k')

plt.xlabel('pinlv(hz)')
plt.ylabel('amplitude')

plt.show()