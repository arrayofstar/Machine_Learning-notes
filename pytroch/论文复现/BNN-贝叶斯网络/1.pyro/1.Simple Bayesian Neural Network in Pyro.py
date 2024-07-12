# -*- coding: utf-8 -*-
# @Time    : 2024/7/10 16:30
# @Author  : Dreamstar
# @File    : 1.Simple Bayesian Neural Network in Pyro.py
# @Desc    : 基于Pyro库的贝叶斯网络实现
# @Link    : https://www.kaggle.com/code/carlossouza/simple-bayesian-neural-network-in-pyro/comments

# Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. ArXiv, abs/1505.05424.
# Uber Technologies, Inc. (2018). Bayesian Regression Tutorial. Pyro.Ai. http://pyro.ai/examples/bayesian_regression.html

#
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 0.5, 1000)
ε = 0.02 * np.random.randn(x.shape[0])
y = x + 0.3 * np.sin(2 * np.pi * (x + ε)) + 0.3 * np.sin(4 * np.pi * (x + ε)) + ε

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, y, 'o', markersize=1)

plt.show()

# part2 - Model
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange, tqdm


class Model(PyroModule):
    def __init__(self, h1=20, h2=20):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](1, h1)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([h1, 1]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([h1]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](h1, h2)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([h2, h1]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([h2]).to_event(1))
        self.fc3 = PyroModule[nn.Linear](h2, 1)
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([1, h2]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc3(x).squeeze()  # 将结果挤压成一维（移除多余的维度），得到均值 mu。
        beta_test = 1234
        sigma = pyro.sample("sigma", dist.Uniform(0., 1.))  # 从均匀分布中采样标准差 sigma
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)  # 用于采样和观测数据
        return mu

# Training
model = Model()
guide = AutoDiagonalNormal(model)  # 自动生成一个引导函数，这个引导函数用于近似后验分布。
adam = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(model, guide, adam, loss=Trace_ELBO())  # 创建一个 Stochastic Variational Inference (SVI) 对象，用于进行变分推断。损失函数使用 Trace_ELBO()，这是一个常用的变分推断损失函数。

pyro.clear_param_store()  # 清除 Pyro 的参数存储，确保每次训练都是从头开始
bar = trange(2000)  # 使用 trange 创建一个进度条，迭代次数为 20000 次
x_train = torch.from_numpy(x).float()
y_train = torch.from_numpy(y).float()
for epoch in bar:
    loss = svi.step(x_train, y_train)
    bar.set_postfix(loss=f'{loss / x.shape[0]:.3f}')


# Prediction
predictive = Predictive(model, guide=guide, num_samples=500)  # 创建一个 Predictive 对象，用于生成预测。这里 num_samples=500 表示将从后验分布中采样 500 次。
x_test = torch.linspace(-0.5, 1, 3000)  # 生成一组测试数据，从 -0.5 到 1 之间等间隔的 3000 个点。
preds = predictive(x_test)

y_pred = preds['obs'].T.detach().numpy().mean(axis=1)  # 获取预测的均值
y_std = preds['obs'].T.detach().numpy().std(axis=1)  # 获取预测的不确定性（标准差）。

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, y, 'o', markersize=1)  # 绘制原始数据点
ax.plot(x_test, y_pred)  # 绘制预测的均值曲线
ax.fill_between(x_test, y_pred - y_std, y_pred + y_std,
                alpha=0.5, color='#ffcd3c')  # 绘制预测的不确定性区域（均值 ± 标准差）。
plt.show()
