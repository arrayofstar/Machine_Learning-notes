# -*- coding: utf-8 -*-
# @Time    : 2024/7/12 下午5:39
# @Author  : Dreamstar
# @File    : 02. Dropout as a Bayesian Approximation.py
# @Desc    : 很重要很重要
# @Link    : https://github.com/cpark321/uncertainty-deep-learning/blob/master/02.%20Dropout%20as%20a%20Bayesian%20Approximation.ipynb


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


class SimpleModel(torch.nn.Module):
    def __init__(self, dropout_rate, decay):
        super(SimpleModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.decay = decay
        self.f = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_rate),
            torch.nn.Linear(20, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_rate),
            torch.nn.Linear(20, 1)
        )

    def forward(self, X):
        return self.f(X)


def uncertainity_estimate(x, model, num_samples, l2):
    outputs = np.hstack(
        [model(x).cpu().detach().numpy() for i in range(num_samples)])  # n번 inference, output.shape = [20, N]
    y_mean = outputs.mean(axis=1)
    y_variance = outputs.var(axis=1)
    tau = l2 * (1. - model.dropout_rate) / (2. * N * model.decay)
    y_variance += (1. / tau)
    y_std = np.sqrt(y_variance)
    return y_mean, y_std


N = 200  # number of points
min_value = -10
max_value = 10

x_obs = np.linspace(min_value, max_value, N)
noise = np.random.normal(loc=10, scale=80, size=N)
y_obs = x_obs ** 3 + noise

x_test = np.linspace(min_value - 10, max_value + 10, N)
y_test = x_test ** 3 + noise

# Normalise data:
x_mean, x_std = x_obs.mean(), x_obs.std()
y_mean, y_std = y_obs.mean(), y_obs.std()
x_obs = (x_obs - x_mean) / x_std
y_obs = (y_obs - y_mean) / y_std
x_test = (x_test - x_mean) / x_std
y_test = (y_test - y_mean) / y_std

plt.figure(figsize=(12, 6))
plt.plot(x_obs, y_obs)
plt.grid()
plt.show()

model = SimpleModel(dropout_rate=0.5, decay=1e-6).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=model.decay)

for iter in range(2000):
    y_pred = model(torch.Tensor(x_obs).view(-1, 1).to(device))
    y_true = Variable(torch.Tensor(y_obs).view(-1, 1).to(device))
    optimizer.zero_grad()
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()

    if iter % 200 == 0:
        print("Iter: {}, Loss: {:.4f}".format(iter, loss.item()))

plt.figure(figsize=(12, 6))
y_pred = model(torch.Tensor(x_obs).view(-1, 1).to(device))
plt.plot(x_obs, y_obs, ls="none", marker="o", color="0.1", alpha=0.8, label="observed")
plt.plot(x_obs, y_pred.cpu().detach().numpy(), ls="-", color="b", label="mean")
plt.grid()
plt.show()

iters_uncertainty = 200

lengthscale = 0.01
n_std = 2  # number of standard deviations to plot
y_mean, y_std = uncertainity_estimate(torch.Tensor(x_test).view(-1, 1).to(device), model, iters_uncertainty,
                                      lengthscale)

plt.figure(figsize=(12, 6))
plt.plot(x_obs, y_obs, ls="none", marker="o", color="0.1", alpha=0.8, label="observed")
plt.plot(x_test, y_mean, ls="-", color="b", label="mean")
plt.plot(x_test, y_test, ls='--', color='r', label='true')
for i in range(n_std):
    plt.fill_between(x_test,
                     y_mean - y_std * (i + 1.),
                     y_mean + y_std * (i + 1.),
                     color="b",
                     alpha=0.1)
plt.legend()
plt.grid()
plt.show()
